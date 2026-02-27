"""Public API for Vietoris-Rips persistent homology (H0 + H1 + H2).

Pipeline:
- GPU (CUDA): direct Triton edge enumeration -> Boruvka MST (H0)
- CPU (Numba): cohomology-based reduction for H1 (host transfer of CSR/edges)
- CPU (Numba): lazy H2 cohomology — triangles materialized, tet cofacets
  computed on-the-fly via triple CSR intersection (same pattern as H1)

H2 design notes (lazy cofacet enumeration):
  Like Ripser, cofacets are computed implicitly rather than materializing
  all tetrahedra.  For each triangle (a,b,c), tet cofacets are found by
  N(a) ∩ N(b) ∩ N(c) with d > c.  Apparent pair check: if the youngest
  cofacet has max_edge_rank == tri_max_rank, it's zero-persistence.
  This drops memory from O(T+Q) to O(T).
"""
from __future__ import annotations


import numpy as np
import torch
from torch import Tensor
from numba import njit

from flash_ph.rips import rips_filtration
from flash_ph.boruvka import gpu_boruvka_mst
from flash_ph.cohomology_h1 import (
    cohomology_reduction_h1,
    cohomology_reduction_h1_gpu,
    _sorted_merge_xor,
)
from flash_ph._numba_utils import (
    _hm_get, _hm_set, MAX_COL, MAX_POOL_ENTRIES,
)


# ---------------------------------------------------------------------------
# Rips simplex enumeration (Numba JIT)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _enumerate_rips_triangles(edge_i, edge_j, adj_ptr, adj_idx, adj_rank, E):
    """Enumerate all triangles in the Rips complex from CSR adjacency.

    For each edge (u,v) with u<v, finds common neighbors w > v via CSR
    intersection. Each triangle (u,v,w) with u<v<w is counted exactly once.

    Returns
    -------
    tri_v0, tri_v1, tri_v2 : (T,) int32 -- triangle vertices (v0 < v1 < v2)
    tri_max_rank : (T,) int32 -- max edge rank (filtration rank of triangle)
    """
    # Pass 1: count
    count = 0
    for e in range(E):
        u, v = edge_i[e], edge_j[e]
        if u > v:
            u, v = v, u
        u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
        v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
        iu, iv = u_start, v_start
        while iu < u_end and iv < v_end:
            nu, nv = adj_idx[iu], adj_idx[iv]
            if nu == nv:
                if nu > v:
                    count += 1
                iu += 1
                iv += 1
            elif nu < nv:
                iu += 1
            else:
                iv += 1

    tri_v0 = np.empty(count, dtype=np.int32)
    tri_v1 = np.empty(count, dtype=np.int32)
    tri_v2 = np.empty(count, dtype=np.int32)
    tri_max_rank = np.empty(count, dtype=np.int32)

    # Pass 2: fill
    idx = 0
    for e in range(E):
        u, v = edge_i[e], edge_j[e]
        if u > v:
            u, v = v, u
        r_uv = np.int32(e)
        u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
        v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
        iu, iv = u_start, v_start
        while iu < u_end and iv < v_end:
            nu, nv = adj_idx[iu], adj_idx[iv]
            if nu == nv:
                w = nu
                if w > v:
                    r_uw = adj_rank[iu]
                    r_vw = adj_rank[iv]
                    max_r = r_uv
                    if r_uw > max_r:
                        max_r = r_uw
                    if r_vw > max_r:
                        max_r = r_vw
                    tri_v0[idx] = u
                    tri_v1[idx] = v
                    tri_v2[idx] = w
                    tri_max_rank[idx] = max_r
                    idx += 1
                iu += 1
                iv += 1
            elif nu < nv:
                iu += 1
            else:
                iv += 1

    return tri_v0[:idx], tri_v1[:idx], tri_v2[:idx], tri_max_rank[:idx]


@njit(cache=True)
def _enumerate_rips_tetrahedra(
    tri_v0, tri_v1, tri_v2, tri_max_rank,
    adj_ptr, adj_idx, adj_rank, T,
):
    """Enumerate all tetrahedra via triple CSR intersection.

    For each triangle (a,b,c) with a<b<c, finds vertices d > c that are
    adjacent to all three vertices. Each tetrahedron (a,b,c,d) with
    a<b<c<d is counted exactly once.

    Returns
    -------
    tet_v0..v3 : (Q,) int32 -- tetrahedron vertices (v0 < v1 < v2 < v3)
    tet_max_rank : (Q,) int32 -- max edge rank (filtration rank of tet)
    """
    # Pass 1: count
    count = 0
    for t in range(T):
        a, b, c = tri_v0[t], tri_v1[t], tri_v2[t]
        a_start, a_end = adj_ptr[a], adj_ptr[a + 1]
        b_start, b_end = adj_ptr[b], adj_ptr[b + 1]
        c_start, c_end = adj_ptr[c], adj_ptr[c + 1]
        ia, ib, ic = a_start, b_start, c_start
        while ia < a_end and ib < b_end and ic < c_end:
            na, nb, nc = adj_idx[ia], adj_idx[ib], adj_idx[ic]
            max_v = na
            if nb > max_v:
                max_v = nb
            if nc > max_v:
                max_v = nc
            if na == nb == nc:
                if na > c:
                    count += 1
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

    tet_v0 = np.empty(count, dtype=np.int32)
    tet_v1 = np.empty(count, dtype=np.int32)
    tet_v2 = np.empty(count, dtype=np.int32)
    tet_v3 = np.empty(count, dtype=np.int32)
    tet_max_rank = np.empty(count, dtype=np.int32)

    # Pass 2: fill
    idx = 0
    for t in range(T):
        a, b, c = tri_v0[t], tri_v1[t], tri_v2[t]
        r_tri = tri_max_rank[t]
        a_start, a_end = adj_ptr[a], adj_ptr[a + 1]
        b_start, b_end = adj_ptr[b], adj_ptr[b + 1]
        c_start, c_end = adj_ptr[c], adj_ptr[c + 1]
        ia, ib, ic = a_start, b_start, c_start
        while ia < a_end and ib < b_end and ic < c_end:
            na, nb, nc = adj_idx[ia], adj_idx[ib], adj_idx[ic]
            max_v = na
            if nb > max_v:
                max_v = nb
            if nc > max_v:
                max_v = nc
            if na == nb == nc:
                d = na
                if d > c:
                    r_ad = adj_rank[ia]
                    r_bd = adj_rank[ib]
                    r_cd = adj_rank[ic]
                    max_r = r_tri
                    if r_ad > max_r:
                        max_r = r_ad
                    if r_bd > max_r:
                        max_r = r_bd
                    if r_cd > max_r:
                        max_r = r_cd
                    tet_v0[idx] = a
                    tet_v1[idx] = b
                    tet_v2[idx] = c
                    tet_v3[idx] = d
                    tet_max_rank[idx] = max_r
                    idx += 1
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

    return (tet_v0[:idx], tet_v1[:idx], tet_v2[:idx], tet_v3[:idx],
            tet_max_rank[:idx])


# ---------------------------------------------------------------------------
# Lazy H2 cofacet enumeration (Numba JIT)
# ---------------------------------------------------------------------------

_VERTEX_PACK_BITS = 16


@njit(cache=True)
def _tet_colex_numba(a, b, c, d):
    """Colexicographic index for tetrahedron (a, b, c, d) with a < b < c < d.

    Computes C(d,4) + C(c,3) + C(b,2) + a with overflow-safe early division.
    """
    a64, b64, c64, d64 = np.int64(a), np.int64(b), np.int64(c), np.int64(d)
    # C(d,4) = d*(d-1)/2 * (d-2)*(d-3)/2 / 6  (split to avoid overflow)
    c4_hi = d64 * (d64 - 1) // 2
    c4_lo = (d64 - 2) * (d64 - 3) // 2
    return c4_hi * c4_lo // 6 + c64 * (c64 - 1) * (c64 - 2) // 6 + b64 * (b64 - 1) // 2 + a64


@njit(cache=True)
def _enumerate_tet_cofacets_sorted(
    tri_v0, tri_v1, tri_v2, tri_max_rank_val,
    adj_ptr, adj_idx, adj_rank,
    out_rank, out_colex, out_d,
):
    """Enumerate ALL tet cofacets of a triangle via triple CSR intersection.

    For triangle (a,b,c), finds ALL d in N(a) ∩ N(b) ∩ N(c) (d != a,b,c).
    Each cofacet tet has max_edge_rank = max(tri_max_rank, r_ad, r_bd, r_cd).
    Results written to out_rank/out_colex/out_d, sorted by (rank ASC, colex ASC).

    Returns cn: number of cofacets.
    """
    a, b, c = tri_v0, tri_v1, tri_v2
    a_start, a_end = adj_ptr[a], adj_ptr[a + 1]
    b_start, b_end = adj_ptr[b], adj_ptr[b + 1]
    c_start, c_end = adj_ptr[c], adj_ptr[c + 1]
    ia, ib, ic = a_start, b_start, c_start
    cn = 0

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
            max_r = tri_max_rank_val
            if r_ad > max_r:
                max_r = r_ad
            if r_bd > max_r:
                max_r = r_bd
            if r_cd > max_r:
                max_r = r_cd
            if cn < MAX_COL:
                # Sort 4 vertices for correct colex computation
                if d < a:
                    v0, v1, v2, v3 = d, a, b, c
                elif d < b:
                    v0, v1, v2, v3 = a, d, b, c
                elif d < c:
                    v0, v1, v2, v3 = a, b, d, c
                else:
                    v0, v1, v2, v3 = a, b, c, d
                out_rank[cn] = max_r
                out_colex[cn] = _tet_colex_numba(v0, v1, v2, v3)
                out_d[cn] = d
                cn += 1
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

    # Insertion sort by (rank ASC, colex ASC) — rearrange d alongside
    for i in range(1, cn):
        kr = out_rank[i]
        kc = out_colex[i]
        kd = out_d[i]
        j = i - 1
        while j >= 0 and (out_rank[j] > kr or (out_rank[j] == kr and out_colex[j] > kc)):
            out_rank[j + 1] = out_rank[j]
            out_colex[j + 1] = out_colex[j]
            out_d[j + 1] = out_d[j]
            j -= 1
        out_rank[j + 1] = kr
        out_colex[j + 1] = kc
        out_d[j + 1] = kd

    return cn


@njit(cache=True)
def _cohomology_reduce_h2_lazy(
    tri_v0,            # (T,) int32
    tri_v1,            # (T,) int32
    tri_v2,            # (T,) int32
    tri_max_rank,      # (T,) int32
    order,             # (T,) int32 — processing order (descending rank)
    skip_mask,         # (T,) bool
    adj_ptr,           # (n+1,) int32
    adj_idx,           # (2E,) int32
    adj_rank,          # (2E,) int32
    max_degree,        # int
):
    """Lazy H2 cohomology reduction: tet cofacets computed on-the-fly.

    Same pattern as H1 lazy reduction but one dimension up:
    - Columns = triangles, rows = tetrahedra (cofacets)
    - Cofacets via triple CSR intersection (N(a) ∩ N(b) ∩ N(c), all d)
    - Apparent pair: youngest cofacet max_rank == r_t AND d < a
      (d < a ensures triangle (a,b,c) is the oldest face of the tet)
    - Lazy hash map entries for apparent pairs (materialized on XOR hit)

    Returns
    -------
    pair_birth_ranks : (P,) int32 — max-edge-ranks of finite pair births (triangles)
    pair_death_ranks : (P,) int32 — max-edge-ranks of finite pair deaths (tets)
    ess_birth_ranks  : (M,) int32 — max-edge-ranks of essential H2 features
    """
    T = order.shape[0]

    # Output buffers
    pair_births = np.empty(T, dtype=np.int32)
    pair_deaths = np.empty(T, dtype=np.int32)
    ess_births = np.empty(T, dtype=np.int32)
    n_pairs = 0
    n_ess = 0

    # Pivot hash map: tet_colex -> tagged union value
    #   val >= 0    -> pooled column index
    #   val == -1   -> empty/missing
    #   val <= -2   -> lazy: tri_array_idx = -val - 2
    hm_cap = max(T * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Pool allocator for stored columns: parallel arrays of (rank, colex)
    pool_cap = min(max(T * max(max_degree, 16), 100000), MAX_POOL_ENTRIES)
    pool_rank = np.empty(pool_cap, dtype=np.int32)
    pool_colex = np.empty(pool_cap, dtype=np.int64)
    piv_start = np.empty(T, dtype=np.int64)
    piv_length = np.empty(T, dtype=np.int32)
    pool_ptr = 0
    num_piv = 0

    # Working buffers (ping-pong)
    buf_a_rank = np.empty(MAX_COL, dtype=np.int32)
    buf_a_colex = np.empty(MAX_COL, dtype=np.int64)
    buf_b_rank = np.empty(MAX_COL, dtype=np.int32)
    buf_b_colex = np.empty(MAX_COL, dtype=np.int64)
    lazy_rank_buf = np.empty(MAX_COL, dtype=np.int32)
    lazy_colex_buf = np.empty(MAX_COL, dtype=np.int64)
    # Buffers for 4th-vertex tracking (used only for apparent pair check)
    cofacet_d_buf = np.empty(MAX_COL, dtype=np.int32)
    lazy_d_buf = np.empty(MAX_COL, dtype=np.int32)
    use_a = True

    # Process triangles in REVERSE filtration order (largest rank first)
    for oi in range(T):
        idx = order[oi]
        if skip_mask[idx]:
            continue

        r_t = tri_max_rank[idx]
        a = tri_v0[idx]
        b = tri_v1[idx]
        c = tri_v2[idx]

        # Select current buffer
        if use_a:
            cur_rank = buf_a_rank
            cur_colex = buf_a_colex
        else:
            cur_rank = buf_b_rank
            cur_colex = buf_b_colex

        # Enumerate tet cofacets via triple CSR intersection
        cn = _enumerate_tet_cofacets_sorted(
            a, b, c, r_t,
            adj_ptr, adj_idx, adj_rank,
            cur_rank, cur_colex, cofacet_d_buf,
        )

        if cn == 0:
            # No cofacets -> essential H2 cycle
            ess_births[n_ess] = r_t
            n_ess += 1
            continue

        # --- Apparent pair check ---
        # Youngest cofacet must have max_rank == r_t (zero persistence)
        # AND d < a (triangle (a,b,c) is the oldest face = largest colex face).
        if cur_rank[0] == r_t and cofacet_d_buf[0] < a:
            piv_colex = cur_colex[0]
            # Store as lazy entry: val = -(idx + 2)
            _hm_set(hm_keys, hm_vals, hm_cap,
                     piv_colex, np.int64(-(np.int64(idx) + 2)))
            continue

        # --- Reduction loop ---
        while cn > 0:
            piv_colex = cur_colex[0]
            piv_idx = _hm_get(hm_keys, hm_vals, hm_cap, piv_colex)
            if piv_idx == -1:
                break  # Pivot is free

            # Select output buffer (opposite of current)
            if use_a:
                tmp_rank = buf_b_rank
                tmp_colex = buf_b_colex
            else:
                tmp_rank = buf_a_rank
                tmp_colex = buf_a_colex

            # Handle lazy pivot: materialize column from CSR
            if piv_idx <= -2:
                lazy_tri_idx = np.int32(-piv_idx - 2)
                lcn = _enumerate_tet_cofacets_sorted(
                    tri_v0[lazy_tri_idx], tri_v1[lazy_tri_idx],
                    tri_v2[lazy_tri_idx], tri_max_rank[lazy_tri_idx],
                    adj_ptr, adj_idx, adj_rank,
                    lazy_rank_buf, lazy_colex_buf, lazy_d_buf,
                )
                if lcn == 0 or lazy_colex_buf[0] != piv_colex:
                    break  # Invariant violated

                # Memoize in pool
                if pool_ptr + lcn <= pool_cap and num_piv < T:
                    _hm_set(hm_keys, hm_vals, hm_cap,
                            piv_colex, np.int64(num_piv))
                    piv_start[num_piv] = pool_ptr
                    piv_length[num_piv] = lcn
                    for i in range(lcn):
                        pool_rank[pool_ptr + i] = lazy_rank_buf[i]
                        pool_colex[pool_ptr + i] = lazy_colex_buf[i]
                    pool_ptr += lcn
                    piv_idx = np.int64(num_piv)
                    num_piv += 1
                else:
                    # Pool full: XOR directly without memoizing
                    cn = _sorted_merge_xor(
                        cur_rank, cur_colex, cn,
                        lazy_rank_buf, lazy_colex_buf, lcn,
                        tmp_rank, tmp_colex,
                    )
                    use_a = not use_a
                    cur_rank = tmp_rank
                    cur_colex = tmp_colex
                    continue

            # XOR with stored column
            ps = piv_start[piv_idx]
            plen = piv_length[piv_idx]
            cn = _sorted_merge_xor(
                cur_rank, cur_colex, cn,
                pool_rank[ps:ps + plen], pool_colex[ps:ps + plen], plen,
                tmp_rank, tmp_colex,
            )
            use_a = not use_a
            cur_rank = tmp_rank
            cur_colex = tmp_colex

        if cn > 0:
            # New pivot found -> persistence pair
            piv_colex = cur_colex[0]
            piv_r = cur_rank[0]

            # Store column in pool
            if pool_ptr + cn <= pool_cap and num_piv < T:
                _hm_set(hm_keys, hm_vals, hm_cap, piv_colex, np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_rank[pool_ptr + i] = cur_rank[i]
                    pool_colex[pool_ptr + i] = cur_colex[i]
                pool_ptr += cn
                num_piv += 1

            # Output pair: birth = r_t (triangle), death = piv_r (tet)
            if piv_r > r_t:
                pair_births[n_pairs] = r_t
                pair_deaths[n_pairs] = piv_r
                n_pairs += 1
        else:
            # Column zeroed -> essential H2 cycle
            ess_births[n_ess] = r_t
            n_ess += 1

    return (pair_births[:n_pairs], pair_deaths[:n_pairs], ess_births[:n_ess])


def _rips_h2_reduction(
    filt,
    edge_dist_sq: Tensor,
    h1_all_pivot_colex: Tensor,
    n: int,
    E: int,
    device: torch.device,
) -> Tensor:
    """Compute H2 persistence via lazy cohomology reduction.

    Triangles are materialized (columns); tet cofacets are computed
    on-the-fly via triple CSR intersection during reduction.
    Memory: O(T) instead of O(T+Q).

    Accepts edge_dist_sq (squared distances) and computes sqrt lazily.
    """
    # Pre-sort H1 pivots once (for skip mask)
    h1_sorted = (
        h1_all_pivot_colex.sort().values
        if h1_all_pivot_colex.numel() > 0
        else h1_all_pivot_colex
    )

    # --- 1. Enumerate triangles (GPU Triton or Numba CPU) ---
    use_gpu = device.type == "cuda" and torch.cuda.is_available()

    if use_gpu:
        from flash_ph.kernels.triangle_kernel import triangle_enumerate
        tv0_t, tv1_t, tv2_t, tri_mr_t = triangle_enumerate(
            filt.edge_i, filt.edge_j,
            filt.vert_adj_ptr, filt.vert_adj_idx, filt.vert_adj_rank,
            E,
        )
        T = tv0_t.shape[0]
        if T == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)

        # Compute tri_colex on GPU for skip mask
        a = tv0_t.long()
        b = tv1_t.long()
        c = tv2_t.long()
        tri_colex = c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a

        # Skip mask: triangles that are H1 pivots
        skip_mask = torch.zeros(T, dtype=torch.bool, device=device)
        if h1_sorted.numel() > 0:
            h1_dev = h1_sorted.to(device)
            pos = torch.searchsorted(h1_dev, tri_colex)
            pos = pos.clamp(max=max(h1_dev.numel() - 1, 0))
            skip_mask = h1_dev[pos] == tri_colex

        # Processing order: descending by (tri_max_rank, tri_colex)
        # Two-pass stable argsort: colex first (secondary), then max_rank (primary)
        order = torch.argsort(tri_colex, descending=True, stable=True)
        order = order[torch.argsort(tri_mr_t[order], descending=True, stable=True)]

        # Transfer to CPU for Numba reduction
        tv0_cpu = tv0_t.cpu().numpy()
        tv1_cpu = tv1_t.cpu().numpy()
        tv2_cpu = tv2_t.cpu().numpy()
        tri_mr_cpu = tri_mr_t.cpu().numpy()
        skip_cpu = skip_mask.cpu().numpy()
        order_cpu = order.cpu().numpy().astype(np.int32)
    else:
        ei_cpu = filt.edge_i.cpu().numpy()
        ej_cpu = filt.edge_j.cpu().numpy()
        ptr_cpu = filt.vert_adj_ptr.cpu().numpy()
        idx_cpu = filt.vert_adj_idx.cpu().numpy()
        rank_cpu = filt.vert_adj_rank.cpu().numpy()

        tv0_cpu, tv1_cpu, tv2_cpu, tri_mr_cpu = _enumerate_rips_triangles(
            ei_cpu, ej_cpu, ptr_cpu, idx_cpu, rank_cpu, E,
        )
        T = tv0_cpu.shape[0]
        if T == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)

        # Compute tri_colex for skip mask (CPU)
        a64 = tv0_cpu.astype(np.int64)
        b64 = tv1_cpu.astype(np.int64)
        c64 = tv2_cpu.astype(np.int64)
        tri_colex_np = c64 * (c64 - 1) * (c64 - 2) // 6 + b64 * (b64 - 1) // 2 + a64
        tri_colex_t = torch.from_numpy(tri_colex_np)

        skip_mask_t = torch.zeros(T, dtype=torch.bool)
        if h1_sorted.numel() > 0:
            pos = torch.searchsorted(h1_sorted, tri_colex_t)
            pos = pos.clamp(max=max(h1_sorted.numel() - 1, 0))
            skip_mask_t = h1_sorted[pos] == tri_colex_t
        skip_cpu = skip_mask_t.numpy()

        # Processing order: descending by (tri_max_rank, tri_colex)
        tri_mr_t = torch.from_numpy(tri_mr_cpu)
        order_t = torch.argsort(tri_colex_t, descending=True, stable=True)
        order_t = order_t[torch.argsort(tri_mr_t[order_t], descending=True, stable=True)]
        order_cpu = order_t.numpy().astype(np.int32)

    # --- 2. CSR adjacency (always needed, may already be on CPU) ---
    if use_gpu:
        ptr_cpu = filt.vert_adj_ptr.cpu().numpy()
        idx_cpu = filt.vert_adj_idx.cpu().numpy()
        rank_cpu = filt.vert_adj_rank.cpu().numpy()

    # Compute max vertex degree for pool sizing
    degrees = ptr_cpu[1:] - ptr_cpu[:-1]
    max_deg = int(degrees.max()) if len(degrees) > 0 else 1

    # --- 3. Run lazy H2 cohomology reduction ---
    pb_ranks, pd_ranks, eb_ranks = _cohomology_reduce_h2_lazy(
        tv0_cpu, tv1_cpu, tv2_cpu, tri_mr_cpu,
        order_cpu, skip_cpu,
        ptr_cpu, idx_cpu, rank_cpu,
        max_deg,
    )

    # --- 4. Convert edge ranks to distances ---
    dist_sq_cpu = edge_dist_sq.cpu().numpy()

    h2_parts = []
    if pb_ranks.shape[0] > 0:
        birth_dist = np.sqrt(dist_sq_cpu[pb_ranks.astype(np.int64)])
        death_dist = np.sqrt(dist_sq_cpu[pd_ranks.astype(np.int64)])
        finite_h2 = torch.tensor(
            np.stack([birth_dist, death_dist], axis=1),
            dtype=torch.float32, device=device,
        )
        h2_parts.append(finite_h2)
    if eb_ranks.shape[0] > 0:
        ess_dist = np.sqrt(dist_sq_cpu[eb_ranks.astype(np.int64)])
        inf_h2 = torch.stack([
            torch.tensor(ess_dist, dtype=torch.float32, device=device),
            torch.full(
                (eb_ranks.shape[0],), float("inf"),
                dtype=torch.float32, device=device,
            ),
        ], dim=1)
        h2_parts.append(inf_h2)

    return (
        torch.cat(h2_parts, dim=0)
        if h2_parts
        else torch.empty(0, 2, dtype=torch.float32, device=device)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rips_persistence(
    points,
    max_edge_length: float,
    max_dim: int = 1,
) -> list[Tensor]:
    """Exact Vietoris-Rips persistent homology (H0 + H1 + optional H2).

    Parameters
    ----------
    points : (n, d) array-like
        Point cloud coordinates (any d >= 1). Accepts numpy arrays
        (auto-converted to CUDA tensor) or PyTorch tensors.
        CUDA recommended for speed.
    max_edge_length : float
        Maximum edge length (strict < inequality).
        Must keep the graph sparse for efficiency.
    max_dim : int
        0 (H0 only), 1 (H0 + H1), or 2 (H0 + H1 + H2). Default 1.
        H2 uses 16-bit vertex packing internally, requiring n < 65536.

    Returns
    -------
    list[Tensor]
        [H0_diagram, ...] each (K, 2) float32 on same device.
        H0 includes one (0, inf) per surviving connected component.
        H1/H2 include essential (birth, inf) features.
        Diagrams sorted by (birth, death) for stable parity comparison.

    Notes
    -----
    - Coefficients: Z/2Z
    - Metric: Euclidean (squared internally, sqrt at output)
    - Filtration: strict ``< max_edge_length``
    - Output: ``(birth, death)`` float32, ``inf`` for essential features
    """
    # Accept numpy input
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
        if torch.cuda.is_available():
            points = points.cuda()

    if max_edge_length < 0:
        raise ValueError(
            f"max_edge_length must be non-negative, got {max_edge_length}"
        )
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n, d), got {tuple(points.shape)}"
        )
    if max_dim not in (0, 1, 2):
        raise ValueError(f"max_dim must be 0, 1, or 2, got {max_dim}")
    if max_dim == 2 and points.shape[0] >= (1 << _VERTEX_PACK_BITS):
        raise ValueError(
            f"H2 requires n < {1 << _VERTEX_PACK_BITS} (16-bit vertex packing), "
            f"got n={points.shape[0]}. Use max_dim=1 or reduce n."
        )

    device = points.device
    n = points.shape[0]

    # Edge cases
    if n == 0:
        return [
            torch.empty(0, 2, dtype=torch.float32, device=device)
            for _ in range(max_dim + 1)
        ]

    if n == 1:
        h0 = torch.tensor(
            [[0.0, float('inf')]], dtype=torch.float32, device=device,
        )
        return [h0] + [
            torch.empty(0, 2, dtype=torch.float32, device=device)
            for _ in range(max_dim)
        ]

    # Step 1: Build Rips filtration
    filt = rips_filtration(points, max_edge_length)
    E = filt.edge_i.shape[0]
    edge_rank = torch.arange(E, dtype=torch.int32, device=device)

    # Step 2: GPU Boruvka MST for H0
    mst_idx, final_comp = gpu_boruvka_mst(
        filt.edge_i, filt.edge_j, edge_rank, n,
    )

    # Build H0 diagram
    h0_parts = []

    # Finite bars: (0, sqrt(dist_sq)) for each MST edge, sorted
    if mst_idx.numel() > 0:
        mst_deaths = torch.sqrt(filt.edge_dist_sq[mst_idx]).sort().values
        finite_h0 = torch.stack([
            torch.zeros_like(mst_deaths),
            mst_deaths,
        ], dim=1)
        h0_parts.append(finite_h0)

    # Infinite bars: one per surviving component
    n_components = final_comp.unique().numel()
    if n_components > 0:
        inf_h0 = torch.zeros(
            n_components, 2, dtype=torch.float32, device=device,
        )
        inf_h0[:, 1] = float('inf')
        h0_parts.append(inf_h0)

    h0 = torch.cat(h0_parts, dim=0) if h0_parts else torch.empty(
        0, 2, dtype=torch.float32, device=device
    )

    if max_dim == 0:
        return [h0]

    # Step 3: Cohomology-based H1 reduction
    mst_mask = torch.zeros(E, dtype=torch.bool, device=device)
    if mst_idx.numel() > 0:
        mst_mask[mst_idx] = True

    _h1_reduce = (
        cohomology_reduction_h1_gpu
        if device.type == 'cuda' and torch.cuda.is_available()
        else cohomology_reduction_h1
    )
    h1_pairs, essential_births, h1_all_pivot_colex = _h1_reduce(
        filt.edge_i, filt.edge_j, filt.edge_dist_sq,
        mst_mask,
        filt.vert_adj_ptr, filt.vert_adj_idx, filt.vert_adj_rank,
        n,
    )

    # Build H1 diagram
    h1_parts = []
    if h1_pairs.numel() > 0:
        h1_parts.append(h1_pairs)
    if essential_births.numel() > 0:
        inf_h1 = torch.stack([
            essential_births,
            torch.full_like(essential_births, float('inf')),
        ], dim=1)
        h1_parts.append(inf_h1)

    h1 = torch.cat(h1_parts, dim=0) if h1_parts else torch.empty(
        0, 2, dtype=torch.float32, device=device
    )

    if max_dim == 1:
        return [h0, h1]

    # Step 4: H2 via general cohomology reduction with clearing from H1
    h2 = _rips_h2_reduction(
        filt, filt.edge_dist_sq, h1_all_pivot_colex, n, E, device,
    )

    return [h0, h1, h2]
