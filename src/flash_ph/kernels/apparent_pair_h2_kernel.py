"""Triton GPU apparent pair scan for H2 cohomology reduction.

Classifies triangles as apparent or non-apparent by performing 3-way CSR
merge-intersection on vertex neighbor lists. A triangle (a,b,c) with
max_edge_rank r_t is apparent if its youngest tet cofacet has
max_edge_rank == r_t (zero persistence) AND the triangle is the oldest
face of that tet (d < s, where s is the vertex NOT on the max edge).

Architecture:
  - Vectorized Triton kernel: BLOCK_T triangles per program
  - 3-way CSR merge-intersection: advance smallest pointer(s)
  - Pure masking pattern: bounded for-loop, no break/continue
  - Outputs: is_apparent (bool) + pivot_colex (tet colex for apparent)
  - CPU fallback via Numba for non-CUDA inputs
  - False negatives safe (MAX_DEGREE truncation), false positives impossible
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from numba import njit

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Numba CPU fallback (same logic as persistence.py _h2_apparent_pair_prepass)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _csr_rank_lookup_cpu(src, tgt, adj_ptr, adj_idx, adj_rank):
    """Binary search for edge rank of (src, tgt) in CSR adjacency."""
    lo = adj_ptr[src]
    hi = adj_ptr[src + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        if adj_idx[mid] < tgt:
            lo = mid + 1
        elif adj_idx[mid] > tgt:
            hi = mid
        else:
            return adj_rank[mid]
    return np.int32(-1)


@njit(cache=True)
def _tet_colex_cpu(a, b, c, d):
    """Colex index for tet (a<b<c<d). Overflow-safe."""
    a64, b64, c64, d64 = np.int64(a), np.int64(b), np.int64(c), np.int64(d)
    c4_hi = d64 * (d64 - 1) // 2
    c4_lo = (d64 - 2) * (d64 - 3) // 2
    return c4_hi * c4_lo // 6 + c64 * (c64 - 1) * (c64 - 2) // 6 + b64 * (b64 - 1) // 2 + a64


@njit(cache=True)
def _numba_h2_apparent_scan(
    tri_v0, tri_v1, tri_v2, tri_max_rank,
    adj_ptr, adj_idx, adj_rank, T,
):
    """CPU Numba: classify all triangles as apparent or not for H2."""
    is_apparent = np.zeros(T, dtype=np.int32)
    pivot_colex = np.zeros(T, dtype=np.int64)

    for i in range(T):
        a = tri_v0[i]
        b = tri_v1[i]
        c = tri_v2[i]
        r_t = tri_max_rank[i]

        a_start = adj_ptr[a]
        a_end = adj_ptr[a + 1]
        b_start = adj_ptr[b]
        b_end = adj_ptr[b + 1]
        c_start = adj_ptr[c]
        c_end = adj_ptr[c + 1]
        ia, ib, ic = a_start, b_start, c_start

        best_rank = np.int32(2147483647)
        best_colex = np.int64(9223372036854775807)
        best_d = np.int32(-1)

        while ia < a_end and ib < b_end and ic < c_end:
            na = adj_idx[ia]
            nb = adj_idx[ib]
            nc = adj_idx[ic]
            max_v = na
            if nb > max_v:
                max_v = nb
            if nc > max_v:
                max_v = nc
            if na == nb == nc:
                d = na
                r_ad = adj_rank[ia]
                r_bd = adj_rank[ib]
                r_cd = adj_rank[ic]
                max_r = r_t
                if r_ad > max_r:
                    max_r = r_ad
                if r_bd > max_r:
                    max_r = r_bd
                if r_cd > max_r:
                    max_r = r_cd
                if d < a:
                    v0, v1, v2, v3 = d, a, b, c
                elif d < b:
                    v0, v1, v2, v3 = a, d, b, c
                elif d < c:
                    v0, v1, v2, v3 = a, b, d, c
                else:
                    v0, v1, v2, v3 = a, b, c, d
                colex = _tet_colex_cpu(v0, v1, v2, v3)
                if max_r < best_rank or (max_r == best_rank and colex < best_colex):
                    best_rank = max_r
                    best_colex = colex
                    best_d = d
                ia += 1
                ib += 1
                ic += 1
            else:
                if na < max_v:
                    ia += 1
                if nb < max_v:
                    ib += 1
                if nc < max_v:
                    ic += 1

        if best_d >= 0 and best_rank == r_t:
            d = best_d
            if d < a:
                is_apparent[i] = 1
                pivot_colex[i] = best_colex
            elif d < c:
                r_bc = _csr_rank_lookup_cpu(b, c, adj_ptr, adj_idx, adj_rank)
                if r_bc == r_t:
                    pass  # max edge = (b,c), s = a, need d < a (failed)
                else:
                    r_ac = _csr_rank_lookup_cpu(a, c, adj_ptr, adj_idx, adj_rank)
                    if r_ac == r_t:
                        if d < b:
                            is_apparent[i] = 1
                            pivot_colex[i] = best_colex
                    else:
                        if d < c:
                            is_apparent[i] = 1
                            pivot_colex[i] = best_colex

    return is_apparent, pivot_colex


# ---------------------------------------------------------------------------
# Triton kernel: vectorized H2 apparent pair scan
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _h2_apparent_pair_kernel(
        tri_v0_ptr, tri_v1_ptr, tri_v2_ptr, tri_max_rank_ptr,
        adj_ptr_ptr, adj_idx_ptr, adj_rank_ptr,
        is_apparent_ptr, pivot_colex_ptr,
        T,
        BLOCK_T: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Classify BLOCK_T triangles per program as apparent or not for H2.

        3-way CSR merge-intersection: for triangle (a,b,c), finds youngest
        tet cofacet via N(a) ∩ N(b) ∩ N(c), then checks apparent pair
        condition with oldest-face check.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        active = offs < T

        a = tl.load(tri_v0_ptr + offs, mask=active, other=0)
        b = tl.load(tri_v1_ptr + offs, mask=active, other=0)
        c = tl.load(tri_v2_ptr + offs, mask=active, other=0)
        r_t = tl.load(tri_max_rank_ptr + offs, mask=active, other=0)

        # CSR range for each vertex
        a_start = tl.load(adj_ptr_ptr + a, mask=active, other=0)
        a_end = tl.load(adj_ptr_ptr + a + 1, mask=active, other=0)
        b_start = tl.load(adj_ptr_ptr + b, mask=active, other=0)
        b_end = tl.load(adj_ptr_ptr + b + 1, mask=active, other=0)
        c_start = tl.load(adj_ptr_ptr + c, mask=active, other=0)
        c_end = tl.load(adj_ptr_ptr + c + 1, mask=active, other=0)

        ia = a_start
        ib = b_start
        ic = c_start

        # Best cofacet tracking (youngest = smallest rank, then colex)
        best_rank = tl.full([BLOCK_T], 2147483647, dtype=tl.int32)
        best_colex = tl.full([BLOCK_T], 9223372036854775807, dtype=tl.int64)
        best_d = tl.full([BLOCK_T], -1, dtype=tl.int32)

        # 3-way merge: bounded loop, max 3*MAX_DEGREE steps
        for _step in range(3 * MAX_DEGREE):
            still_going = active & (ia < a_end) & (ib < b_end) & (ic < c_end)

            na = tl.load(adj_idx_ptr + ia, mask=still_going, other=2147483647)
            nb = tl.load(adj_idx_ptr + ib, mask=still_going, other=2147483647)
            nc = tl.load(adj_idx_ptr + ic, mask=still_going, other=2147483647)

            # 3-way max
            max_v = tl.where(na > nb, na, nb)
            max_v = tl.where(nc > max_v, nc, max_v)

            # Match: na == nb == nc
            match = still_going & (na == nb) & (nb == nc)

            # Load edge ranks for matching entries
            r_ad = tl.load(adj_rank_ptr + ia, mask=match, other=0)
            r_bd = tl.load(adj_rank_ptr + ib, mask=match, other=0)
            r_cd = tl.load(adj_rank_ptr + ic, mask=match, other=0)

            # max_edge_rank of tet cofacet
            max_r = tl.where(match, r_t, tl.zeros([BLOCK_T], dtype=tl.int32))
            max_r = tl.where(match & (r_ad > max_r), r_ad, max_r)
            max_r = tl.where(match & (r_bd > max_r), r_bd, max_r)
            max_r = tl.where(match & (r_cd > max_r), r_cd, max_r)

            # Sort 4 vertices for colex: {a, b, c, d=na} → (v0<v1<v2<v3)
            d_val = na  # na == nb == nc when match
            # Determine sorted position of d among a, b, c
            d_lt_a = match & (d_val < a)
            d_lt_b = match & (d_val < b) & (~d_lt_a)
            d_lt_c = match & (d_val < c) & (~d_lt_a) & (~d_lt_b)
            # d >= c case is the else
            va = tl.where(d_lt_a, d_val, a).to(tl.int64)
            vb = tl.where(d_lt_a, a, tl.where(d_lt_b, d_val, b)).to(tl.int64)
            vc = tl.where(d_lt_a, b, tl.where(d_lt_b, b, tl.where(d_lt_c, d_val, c))).to(tl.int64)
            vd = tl.where(d_lt_a, c, tl.where(d_lt_b, c, tl.where(d_lt_c, c, d_val))).to(tl.int64)

            # Tet colex: C(vd,4) + C(vc,3) + C(vb,2) + va
            c4_hi = vd * (vd - 1) // 2
            c4_lo = (vd - 2) * (vd - 3) // 2
            colex = c4_hi * c4_lo // 6 + vc * (vc - 1) * (vc - 2) // 6 + vb * (vb - 1) // 2 + va

            # Update best if this cofacet is younger
            is_better = match & (
                (max_r < best_rank) |
                ((max_r == best_rank) & (colex < best_colex))
            )
            best_rank = tl.where(is_better, max_r, best_rank)
            best_colex = tl.where(is_better, colex, best_colex)
            best_d = tl.where(is_better, d_val, best_d)

            # Advance pointers: advance any pointer whose value < max_v, or all on match
            adv_a = still_going & ((na < max_v) | match)
            adv_b = still_going & ((nb < max_v) | match)
            adv_c = still_going & ((nc < max_v) | match)
            ia += tl.where(adv_a, 1, 0)
            ib += tl.where(adv_b, 1, 0)
            ic += tl.where(adv_c, 1, 0)

        # --- Oldest face check ---
        # Apparent if best_rank == r_t AND d < s (vertex not on max edge)
        has_best = active & (best_d >= 0) & (best_rank == r_t)

        # Case 1: d < a → always apparent (d < a ≤ b ≤ c)
        case_d_lt_a = has_best & (best_d < a)

        # Cases where a ≤ d < c: need to look up which edge is max
        need_lookup = has_best & (~case_d_lt_a) & (best_d < c)

        # Binary search for r_bc = rank of edge (b, c)
        # We need to search adj_idx[adj_ptr[b]..adj_ptr[b+1]) for c
        bc_lo = tl.load(adj_ptr_ptr + b, mask=need_lookup, other=0)
        bc_hi = tl.load(adj_ptr_ptr + b + 1, mask=need_lookup, other=0)
        # Unrolled binary search (log2(MAX_DEGREE) iterations)
        for _bs in range(16):  # Enough for MAX_DEGREE up to 65536
            bc_valid = need_lookup & (bc_lo < bc_hi)
            bc_mid = (bc_lo + bc_hi) // 2
            bc_mid_val = tl.load(adj_idx_ptr + bc_mid, mask=bc_valid, other=0)
            bc_lo = tl.where(bc_valid & (bc_mid_val < c), bc_mid + 1, bc_lo)
            bc_hi = tl.where(bc_valid & (bc_mid_val > c), bc_mid, bc_hi)
            # On exact match: bc_lo == bc_mid, bc_hi == bc_mid (will converge)
            # Break condition: bc_mid_val == c → both sides converge to bc_mid
            bc_lo = tl.where(bc_valid & (bc_mid_val == c), bc_mid, bc_lo)
            bc_hi = tl.where(bc_valid & (bc_mid_val == c), bc_mid, bc_hi)

        # r_bc is the rank at position bc_lo (where c was found)
        r_bc = tl.load(adj_rank_ptr + bc_lo, mask=need_lookup, other=-1)
        max_is_bc = need_lookup & (r_bc == r_t)

        # For those where max edge != (b,c), check (a,c)
        need_ac = need_lookup & (~max_is_bc)
        ac_lo = tl.load(adj_ptr_ptr + a, mask=need_ac, other=0)
        ac_hi = tl.load(adj_ptr_ptr + a + 1, mask=need_ac, other=0)
        for _bs in range(16):
            ac_valid = need_ac & (ac_lo < ac_hi)
            ac_mid = (ac_lo + ac_hi) // 2
            ac_mid_val = tl.load(adj_idx_ptr + ac_mid, mask=ac_valid, other=0)
            ac_lo = tl.where(ac_valid & (ac_mid_val < c), ac_mid + 1, ac_lo)
            ac_hi = tl.where(ac_valid & (ac_mid_val > c), ac_mid, ac_hi)
            ac_lo = tl.where(ac_valid & (ac_mid_val == c), ac_mid, ac_lo)
            ac_hi = tl.where(ac_valid & (ac_mid_val == c), ac_mid, ac_hi)

        r_ac = tl.load(adj_rank_ptr + ac_lo, mask=need_ac, other=-1)
        max_is_ac = need_ac & (r_ac == r_t)

        # max edge = (a,c) → s = b → need d < b
        case_ac = max_is_ac & (best_d < b)
        # max edge = (a,b) → s = c → need d < c (already guaranteed by need_lookup condition)
        case_ab = need_ac & (~max_is_ac) & (best_d < c)

        # Combine all apparent cases
        is_app = case_d_lt_a | case_ac | case_ab

        tl.store(is_apparent_ptr + offs, tl.where(is_app, 1, 0), mask=active)
        tl.store(pivot_colex_ptr + offs, tl.where(is_app, best_colex, tl.zeros([BLOCK_T], dtype=tl.int64)), mask=active)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_MAX_DEGREE_CAP = 2048
_BLOCK_T = 64  # Triangles per Triton program


def h2_apparent_pair_scan(
    tri_v0: Tensor,
    tri_v1: Tensor,
    tri_v2: Tensor,
    tri_max_rank: Tensor,
    adj_ptr: Tensor,
    adj_idx: Tensor,
    adj_rank: Tensor,
    max_degree: int,
) -> tuple[Tensor, Tensor]:
    """Classify triangles as apparent pairs for H2 via 3-way CSR intersection.

    For each triangle (a,b,c) with max_edge_rank r_t, finds the youngest tet
    cofacet d in N(a) ∩ N(b) ∩ N(c). Apparent if max_edge_rank(tet) == r_t
    AND d < s (oldest face condition).

    Uses GPU Triton kernel when available, with CPU Numba fallback.

    Parameters
    ----------
    tri_v0, tri_v1, tri_v2 : (T,) int32 -- triangle vertices (v0 < v1 < v2)
    tri_max_rank : (T,) int32 -- max edge rank per triangle
    adj_ptr : (n+1,) int32 -- CSR vertex adjacency pointers
    adj_idx : (2E,) int32 -- CSR neighbor indices (sorted per vertex)
    adj_rank : (2E,) int32 -- CSR edge ranks
    max_degree : int -- maximum vertex degree in the graph

    Returns
    -------
    is_apparent : (T,) int32 -- 1 if apparent, 0 otherwise
    pivot_colex : (T,) int64 -- tet colex for apparent pairs (0 for non-apparent)
    """
    T = tri_v0.shape[0]
    device = tri_v0.device
    use_gpu = (
        _HAS_TRITON
        and device.type == 'cuda'
        and torch.cuda.is_available()
    )

    if T == 0:
        return (
            torch.zeros(T, dtype=torch.int32, device=device),
            torch.zeros(T, dtype=torch.int64, device=device),
        )

    if use_gpu:
        return _triton_h2_scan(
            tri_v0, tri_v1, tri_v2, tri_max_rank,
            adj_ptr, adj_idx, adj_rank, max_degree,
        )
    else:
        return _cpu_h2_scan(
            tri_v0, tri_v1, tri_v2, tri_max_rank,
            adj_ptr, adj_idx, adj_rank,
        )


def _triton_h2_scan(
    tri_v0, tri_v1, tri_v2, tri_max_rank,
    adj_ptr, adj_idx, adj_rank, max_degree,
):
    """GPU path: Triton vectorized H2 apparent pair scan."""
    T = tri_v0.shape[0]
    device = tri_v0.device

    tri_v0 = tri_v0.contiguous().to(torch.int32)
    tri_v1 = tri_v1.contiguous().to(torch.int32)
    tri_v2 = tri_v2.contiguous().to(torch.int32)
    tri_max_rank = tri_max_rank.contiguous().to(torch.int32)
    adj_ptr = adj_ptr.contiguous().to(torch.int32)
    adj_idx = adj_idx.contiguous().to(torch.int32)
    adj_rank = adj_rank.contiguous().to(torch.int32)

    is_apparent = torch.zeros(T, dtype=torch.int32, device=device)
    pivot_colex = torch.zeros(T, dtype=torch.int64, device=device)

    md = max(64, 1 << (max_degree - 1).bit_length())
    md = min(md, _MAX_DEGREE_CAP)

    grid = ((T + _BLOCK_T - 1) // _BLOCK_T,)
    _h2_apparent_pair_kernel[grid](
        tri_v0, tri_v1, tri_v2, tri_max_rank,
        adj_ptr, adj_idx, adj_rank,
        is_apparent, pivot_colex,
        T,
        BLOCK_T=_BLOCK_T,
        MAX_DEGREE=md,
    )

    return is_apparent, pivot_colex


def _cpu_h2_scan(
    tri_v0, tri_v1, tri_v2, tri_max_rank,
    adj_ptr, adj_idx, adj_rank,
):
    """CPU path: Numba H2 apparent pair scan."""
    device = tri_v0.device
    T = tri_v0.shape[0]
    v0 = tri_v0.cpu().to(torch.int32).numpy()
    v1 = tri_v1.cpu().to(torch.int32).numpy()
    v2 = tri_v2.cpu().to(torch.int32).numpy()
    mr = tri_max_rank.cpu().to(torch.int32).numpy()
    ptr = adj_ptr.cpu().to(torch.int32).numpy()
    idx = adj_idx.cpu().to(torch.int32).numpy()
    rank = adj_rank.cpu().to(torch.int32).numpy()

    is_app, piv_colex = _numba_h2_apparent_scan(v0, v1, v2, mr, ptr, idx, rank, T)

    return (
        torch.from_numpy(is_app).to(device),
        torch.from_numpy(piv_colex).to(device),
    )
