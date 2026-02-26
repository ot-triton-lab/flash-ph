"""Vietoris-Rips filtration construction.

Direct Triton 2D-grid edge enumeration → packed key sort → per-vertex CSR
adjacency.  No spatial sorting or AABB block-sparse path — simplified for
the flash-ph standalone package.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class RipsFiltration(NamedTuple):
    """Result of Rips filtration construction.

    All edge arrays are sorted by filtration order (distance², then colex).
    """
    edge_i: Tensor        # (E,) int32 — source vertices (i < j)
    edge_j: Tensor        # (E,) int32 — target vertices (i < j)
    edge_dist_sq: Tensor  # (E,) float32 — squared edge lengths
    vert_adj_ptr: Tensor  # (n+1,) int32 — CSR row pointers
    vert_adj_idx: Tensor  # (2E,) int32 — neighbor vertex indices
    vert_adj_rank: Tensor # (2E,) int32 — edge_rank per adjacency entry
    vert_adj_dist_sq: Tensor  # (2E,) float32 — dist² per adjacency entry
    n: int                # number of original points


def _pack_sort_key(dist_sq: Tensor, edge_i: Tensor, edge_j: Tensor) -> Tensor:
    """Pack (dist²_bits << 32 | colex) as uint64 for one-time edge sorting.

    Colex index for edge (i, j) with i < j: j*(j-1)//2 + i.

    Note: colex is truncated to 32 bits (``& 0xFFFFFFFF``). For n > ~65K
    the colex index overflows 32 bits, but this only affects tie-breaking
    order among equal-distance edges and does not affect correctness of
    the persistence computation.
    """
    # Float bits: reinterpret float32 as uint32
    dist_bits = dist_sq.view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    # Colex index: j*(j-1)//2 + i (with i < j guaranteed)
    j_long = edge_j.long()
    i_long = edge_i.long()
    colex = j_long * (j_long - 1) // 2 + i_long

    # Pack: dist_bits in upper 32 bits, colex in lower 32 bits
    key = (dist_bits << 32) | (colex & 0xFFFFFFFF)
    return key


def _build_vertex_csr(
    edge_i: Tensor,
    edge_j: Tensor,
    edge_rank: Tensor,
    edge_dist_sq: Tensor,
    n: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build per-vertex CSR adjacency with (neighbor, edge_rank, dist_sq).

    Creates two directed half-edges per undirected edge.
    Within each row, entries are sorted by neighbor vertex index.
    """
    device = edge_i.device
    E = edge_i.shape[0]

    if E == 0:
        ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        return ptr, empty_i32, empty_i32, empty_f32

    # Two directed half-edges: (i→j) and (j→i)
    src = torch.cat([edge_i, edge_j])       # (2E,)
    dst = torch.cat([edge_j, edge_i])       # (2E,)
    rank = torch.cat([edge_rank, edge_rank]) # (2E,)
    dsq = torch.cat([edge_dist_sq, edge_dist_sq])  # (2E,)

    # Sort by (src, dst) for CSR with sorted neighbors
    sort_key = src.long() * n + dst.long()
    perm = sort_key.argsort(stable=True)
    src = src[perm]
    dst = dst[perm]
    rank = rank[perm]
    dsq = dsq[perm]

    # Build CSR row pointers
    # Count entries per vertex
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    counts.scatter_add_(0, src.long(), torch.ones(2 * E, dtype=torch.int32, device=device))
    ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    ptr[1:] = counts.cumsum(dim=0)

    return ptr, dst.to(torch.int32), rank.to(torch.int32), dsq


def rips_filtration(
    points: Tensor,
    max_edge_length: float,
) -> RipsFiltration:
    """Construct a Vietoris-Rips filtration on GPU.

    Uses direct Triton 2D-grid edge enumeration kernel on CUDA,
    falls back to torch.cdist on CPU.

    Parameters
    ----------
    points : (n, d) Tensor
        Point cloud coordinates (any d >= 1).
    max_edge_length : float
        Maximum edge length (strict < inequality). Must keep graph sparse.

    Returns
    -------
    RipsFiltration
        Named tuple with sorted edges and per-vertex CSR adjacency.
    """
    if max_edge_length < 0:
        raise ValueError(
            f"max_edge_length must be non-negative, got {max_edge_length}"
        )
    n, d = points.shape
    device = points.device
    threshold_sq = max_edge_length * max_edge_length

    # Edge cases
    if n == 0:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        ptr = torch.zeros(1, dtype=torch.int32, device=device)
        return RipsFiltration(
            edge_i=empty_i32, edge_j=empty_i32,
            edge_dist_sq=empty_f32,
            vert_adj_ptr=ptr, vert_adj_idx=empty_i32,
            vert_adj_rank=empty_i32, vert_adj_dist_sq=empty_f32,
            n=0,
        )
    if n == 1:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        ptr = torch.zeros(2, dtype=torch.int32, device=device)
        return RipsFiltration(
            edge_i=empty_i32, edge_j=empty_i32,
            edge_dist_sq=empty_f32,
            vert_adj_ptr=ptr, vert_adj_idx=empty_i32,
            vert_adj_rank=empty_i32, vert_adj_dist_sq=empty_f32,
            n=1,
        )

    _BLOCK_SIZE = 32

    if points.is_cuda:
        # Direct Triton: no sort, no AABB, no perm mapping
        from flash_ph.kernels.edge_kernel import direct_edge_enumerate

        lo_ij, hi_ij, dist_sq = direct_edge_enumerate(
            points.float().contiguous(), n, threshold_sq, _BLOCK_SIZE,
        )
    else:
        # CPU fallback on unsorted points
        lo_ij, hi_ij, dist_sq = _cpu_edge_enumerate(
            points.float(), n, threshold_sq,
        )

    E = lo_ij.shape[0]

    if E == 0:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
        return RipsFiltration(
            edge_i=empty_i32, edge_j=empty_i32,
            edge_dist_sq=empty_f32,
            vert_adj_ptr=ptr, vert_adj_idx=empty_i32,
            vert_adj_rank=empty_i32, vert_adj_dist_sq=empty_f32,
            n=n,
        )

    orig_i = lo_ij
    orig_j = hi_ij

    # Guard: verify dist_sq finite and non-negative (debug only)
    if __debug__ and E < 100_000:
        assert dist_sq.isfinite().all(), "Non-finite dist_sq detected"
        assert (dist_sq >= 0).all(), "Negative dist_sq detected"

    # Pack key = (float_as_uint32(dist_sq) << 32) | colex
    sort_key = _pack_sort_key(dist_sq, orig_i, orig_j)
    sort_perm = sort_key.argsort(stable=True)

    # Apply sort permutation
    orig_i = orig_i[sort_perm]
    orig_j = orig_j[sort_perm]
    dist_sq = dist_sq[sort_perm]

    # Verify i < j (debug only)
    if __debug__ and E < 100_000:
        assert (orig_i < orig_j).all(), "Edge canonicalization failed: found i >= j"

    # edge_rank = position in sorted list
    edge_rank = torch.arange(E, dtype=torch.int32, device=device)

    # Build per-vertex CSR adjacency
    adj_ptr, adj_idx, adj_rank, adj_dsq = _build_vertex_csr(
        orig_i, orig_j, edge_rank, dist_sq, n,
    )

    return RipsFiltration(
        edge_i=orig_i,
        edge_j=orig_j,
        edge_dist_sq=dist_sq,
        vert_adj_ptr=adj_ptr,
        vert_adj_idx=adj_idx,
        vert_adj_rank=adj_rank,
        vert_adj_dist_sq=adj_dsq,
        n=n,
    )


def _cpu_edge_enumerate(
    pts: Tensor, n: int, threshold_sq: float
) -> tuple[Tensor, Tensor, Tensor]:
    """CPU fallback: brute-force pairwise edge enumeration."""
    device = pts.device
    # Pairwise squared distances
    dists_sq = torch.cdist(pts, pts, p=2.0).pow(2)
    # Upper triangle, strict < threshold
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    mask = mask & (dists_sq < threshold_sq)
    ii, jj = mask.nonzero(as_tuple=True)
    dsq = dists_sq[ii, jj]
    return ii.to(torch.int32), jj.to(torch.int32), dsq.float()
