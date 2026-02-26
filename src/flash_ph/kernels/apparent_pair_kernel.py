"""Triton GPU apparent pair scan for H1 cohomology reduction.

Classifies non-MST edges as apparent or non-apparent by performing CSR
merge-intersection on vertex neighbor lists. An edge e=(u,v) with rank r_e
is apparent if there exists a common neighbor w with rank(u,w) < r_e AND
rank(v,w) < r_e (meaning the triangle (u,v,w) has max-edge-rank == r_e
and e is its oldest face).

Architecture:
  - Vectorized Triton kernel: BLOCK_E edges per program for full warp
    utilization (25-88x faster than scalar per-edge approach)
  - Pure masking pattern: no break/continue, bounded for-loop
  - Outputs: is_apparent (bool) + pivot_colex (triangle colex for apparent)
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
# Numba CPU fallback
# ---------------------------------------------------------------------------

@njit(cache=True)
def _numba_apparent_scan(
    edge_i, edge_j, nonmst_ranks, adj_ptr, adj_idx, adj_rank,
):
    """CPU Numba: classify all non-MST edges as apparent or not.

    Returns (is_apparent, pivot_colex) arrays.
    """
    K = nonmst_ranks.shape[0]
    is_apparent = np.zeros(K, dtype=np.int32)
    pivot_colex = np.zeros(K, dtype=np.int64)

    for ki in range(K):
        r_e = nonmst_ranks[ki]
        u = edge_i[r_e]
        v = edge_j[r_e]
        if u > v:
            u, v = v, u

        u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
        v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
        iu = u_start
        iv = v_start

        while iu < u_end and iv < v_end:
            nu = adj_idx[iu]
            nv = adj_idx[iv]
            if nu == nv:
                r_uw = adj_rank[iu]
                r_vw = adj_rank[iv]
                if r_uw < r_e and r_vw < r_e:
                    is_apparent[ki] = 1
                    # Compute colex of triangle (sorted a < b < c)
                    w = nu
                    a, b, c = u, v, w
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    a64 = np.int64(a)
                    b64 = np.int64(b)
                    c64 = np.int64(c)
                    pivot_colex[ki] = (
                        c64 * (c64 - 1) * (c64 - 2) // 6
                        + b64 * (b64 - 1) // 2
                        + a64
                    )
                    break
                iu += 1
                iv += 1
            elif nu < nv:
                iu += 1
            else:
                iv += 1

    return is_apparent, pivot_colex


# ---------------------------------------------------------------------------
# Triton kernel: vectorized, BLOCK_E edges per program
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _apparent_pair_scan_kernel(
        edge_i_ptr, edge_j_ptr,
        nonmst_ptr,
        adj_ptr_ptr, adj_idx_ptr, adj_rank_ptr,
        is_apparent_ptr,
        pivot_colex_ptr,
        K,
        BLOCK_E: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Classify BLOCK_E non-MST edges per program as apparent or not.

        Pure masking pattern: bounded for-loop, all loads gated by active mask.
        Outputs is_apparent (0/1) and pivot triangle colex (int64) per edge.
        """
        pid = tl.program_id(0)
        edge_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        active_edge = edge_offs < K

        r_e = tl.load(nonmst_ptr + edge_offs, mask=active_edge, other=0)
        u = tl.load(edge_i_ptr + r_e, mask=active_edge, other=0)
        v = tl.load(edge_j_ptr + r_e, mask=active_edge, other=0)

        # Canonical order: u < v
        u_lt_v = u < v
        u_orig = u
        v_orig = v
        u = tl.where(u_lt_v, u_orig, v_orig)
        v = tl.where(u_lt_v, v_orig, u_orig)

        u_start = tl.load(adj_ptr_ptr + u, mask=active_edge, other=0)
        u_end = tl.load(adj_ptr_ptr + u + 1, mask=active_edge, other=0)
        v_start = tl.load(adj_ptr_ptr + v, mask=active_edge, other=0)
        v_end = tl.load(adj_ptr_ptr + v + 1, mask=active_edge, other=0)

        iu = u_start
        iv = v_start
        found = tl.zeros([BLOCK_E], dtype=tl.int32)
        # Store pivot witness w (not colex) during loop; compute colex once after
        pivot_w = tl.zeros([BLOCK_E], dtype=tl.int32)

        for _step in range(2 * MAX_DEGREE):
            still_going = active_edge & (iu < u_end) & (iv < v_end) & (found == 0)

            nu = tl.load(adj_idx_ptr + iu, mask=still_going, other=0)
            nv = tl.load(adj_idx_ptr + iv, mask=still_going, other=0)

            match = still_going & (nu == nv)
            r_uw = tl.load(adj_rank_ptr + iu, mask=match, other=2147483647)
            r_vw = tl.load(adj_rank_ptr + iv, mask=match, other=2147483647)

            is_app = match & (r_uw < r_e) & (r_vw < r_e)

            # Record witness w on first hit (colex computed after loop)
            found = tl.where(is_app, 1, found)
            pivot_w = tl.where(is_app, nu, pivot_w)

            # Advance pointers
            advance_u = still_going & ((nu < nv) | (nu == nv))
            advance_v = still_going & ((nv < nu) | (nu == nv))
            iu += tl.where(advance_u, 1, 0)
            iv += tl.where(advance_v, 1, 0)

        # Compute pivot colex ONCE after loop (avoids ALU inside loop)
        # Sort (u, v, w) → (a, b, c) with a < b < c, then C(c,3)+C(b,2)+a
        app_mask = found == 1
        a64 = tl.where(app_mask, u.to(tl.int64), tl.zeros([BLOCK_E], dtype=tl.int64))
        b64 = tl.where(app_mask, v.to(tl.int64), tl.zeros([BLOCK_E], dtype=tl.int64))
        c64 = tl.where(app_mask, pivot_w.to(tl.int64), tl.zeros([BLOCK_E], dtype=tl.int64))

        # 3-element sort (vectorized)
        tmp = tl.where(a64 > b64, a64, b64)
        a64 = tl.where(a64 > b64, b64, a64)
        b64 = tmp
        tmp = tl.where(b64 > c64, b64, c64)
        b64 = tl.where(b64 > c64, c64, b64)
        c64 = tmp
        tmp = tl.where(a64 > b64, a64, b64)
        a64 = tl.where(a64 > b64, b64, a64)
        b64 = tmp

        piv_colex = (
            c64 * (c64 - 1) * (c64 - 2) // 6
            + b64 * (b64 - 1) // 2
            + a64
        )

        tl.store(is_apparent_ptr + edge_offs, found, mask=active_edge)
        tl.store(pivot_colex_ptr + edge_offs, piv_colex, mask=active_edge)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_MAX_DEGREE_CAP = 2048  # Upper bound for Triton loop iterations
_BLOCK_E = 128  # Edges per Triton program (matches warp utilization)


def apparent_pair_scan(
    edge_i: Tensor,
    edge_j: Tensor,
    nonmst_ranks: Tensor,
    adj_ptr: Tensor,
    adj_idx: Tensor,
    adj_rank: Tensor,
    max_degree: int,
) -> tuple[Tensor, Tensor]:
    """Classify non-MST edges as apparent pairs via CSR intersection.

    For each non-MST edge e=(u,v) with rank r_e, checks if there exists
    a common neighbor w with rank(u,w) < r_e AND rank(v,w) < r_e (the
    apparent pair condition for H1 cohomology).

    Uses GPU Triton kernel when available (25-88x faster than CPU),
    with CPU Numba fallback.

    Parameters
    ----------
    edge_i, edge_j : (E,) int32 -- edge endpoints
    nonmst_ranks : (K,) int32 -- ranks of non-MST edges to classify
    adj_ptr : (n+1,) int32 -- CSR vertex adjacency pointers
    adj_idx : (2E,) int32 -- CSR neighbor indices (sorted per vertex)
    adj_rank : (2E,) int32 -- CSR edge ranks
    max_degree : int -- maximum vertex degree in the graph

    Returns
    -------
    is_apparent : (K,) int32 -- 1 if apparent, 0 otherwise
    pivot_colex : (K,) int64 -- triangle colex for apparent pairs (0 for non-apparent)
    """
    K = nonmst_ranks.shape[0]
    device = edge_i.device
    use_gpu = (
        _HAS_TRITON
        and device.type == 'cuda'
        and torch.cuda.is_available()
    )

    if K == 0:
        return (
            torch.zeros(K, dtype=torch.int32, device=device),
            torch.zeros(K, dtype=torch.int64, device=device),
        )

    if use_gpu:
        return _triton_apparent_scan(
            edge_i, edge_j, nonmst_ranks,
            adj_ptr, adj_idx, adj_rank, max_degree,
        )
    else:
        return _cpu_apparent_scan(
            edge_i, edge_j, nonmst_ranks,
            adj_ptr, adj_idx, adj_rank,
        )


def _triton_apparent_scan(
    edge_i, edge_j, nonmst_ranks,
    adj_ptr, adj_idx, adj_rank, max_degree,
):
    """GPU path: Triton vectorized apparent pair scan."""
    K = nonmst_ranks.shape[0]
    device = edge_i.device

    # Ensure contiguous int32 inputs
    edge_i = edge_i.contiguous().to(torch.int32)
    edge_j = edge_j.contiguous().to(torch.int32)
    nonmst_ranks = nonmst_ranks.contiguous().to(torch.int32)
    adj_ptr = adj_ptr.contiguous().to(torch.int32)
    adj_idx = adj_idx.contiguous().to(torch.int32)
    adj_rank = adj_rank.contiguous().to(torch.int32)

    is_apparent = torch.zeros(K, dtype=torch.int32, device=device)
    pivot_colex = torch.zeros(K, dtype=torch.int64, device=device)

    # Bucket MAX_DEGREE to next power of 2, capped
    md = max(64, 1 << (max_degree - 1).bit_length())
    md = min(md, _MAX_DEGREE_CAP)

    grid = ((K + _BLOCK_E - 1) // _BLOCK_E,)
    _apparent_pair_scan_kernel[grid](
        edge_i, edge_j, nonmst_ranks,
        adj_ptr, adj_idx, adj_rank,
        is_apparent, pivot_colex,
        K,
        BLOCK_E=_BLOCK_E,
        MAX_DEGREE=md,
    )

    return is_apparent, pivot_colex


def _cpu_apparent_scan(
    edge_i, edge_j, nonmst_ranks,
    adj_ptr, adj_idx, adj_rank,
):
    """CPU path: Numba apparent pair scan."""
    device = edge_i.device
    ei = edge_i.cpu().to(torch.int32).numpy()
    ej = edge_j.cpu().to(torch.int32).numpy()
    nm = nonmst_ranks.cpu().to(torch.int32).numpy()
    ptr = adj_ptr.cpu().to(torch.int32).numpy()
    idx = adj_idx.cpu().to(torch.int32).numpy()
    rank = adj_rank.cpu().to(torch.int32).numpy()

    is_app, piv_colex = _numba_apparent_scan(ei, ej, nm, ptr, idx, rank)

    return (
        torch.from_numpy(is_app).to(device),
        torch.from_numpy(piv_colex).to(device),
    )
