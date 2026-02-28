"""GPU-native H1 persistence reduction for Rips complexes.

Pipeline (cohomology approach — columns = edges, cofacets = triangles):
1. Triton triangle enumeration → all triangles with max_edge_rank
2. GPU face rank lookup → triangle face edge ranks
3. GPU cofacet CSR construction → edge-to-triangle adjacency
4. GPU apparent pair detection + Numba residual reduction
   (via _general_cohomology_reduce_gpu_prepass)
5. Assembly → H1 persistence diagram

Uses cohomology (Ripser convention): process edges in reverse filtration
order, pivot = SMALLEST cofacet. This processes E edge columns (~33K for
o3_1024) instead of T triangle columns (~354K), with ~98% apparent pairs
leaving only ~600 residual columns for Numba.

Replaces giotto-ph C++ for H1 (max_dim=1) with GPU-native computation.
"""
from __future__ import annotations

import torch
from torch import Tensor

from flash_ph.kernels.triangle_kernel import triangle_enumerate


# ---------------------------------------------------------------------------
# Face rank lookup (GPU)
# ---------------------------------------------------------------------------

def _lookup_face_ranks_gpu(
    tri_v0: Tensor, tri_v1: Tensor, tri_v2: Tensor,
    edge_i: Tensor, edge_j: Tensor, n: int,
) -> Tensor:
    """Look up edge ranks for all triangle face edges.

    Uses packed (i*n + j) keys and searchsorted for O(E log E + 3T log E).

    Returns
    -------
    face_ranks : (T, 3) int32 — face edge ranks, sorted ascending per row
    """
    T = tri_v0.shape[0]
    E = edge_i.shape[0]
    device = tri_v0.device

    if T == 0:
        return torch.empty(0, 3, dtype=torch.int32, device=device)

    # Pack edge endpoints as unique int64 keys
    edge_key = edge_i.long() * n + edge_j.long()
    sorted_key, sort_perm = edge_key.sort()

    # Build face edge keys: (v0,v1), (v0,v2), (v1,v2) per triangle
    v0, v1, v2 = tri_v0.long(), tri_v1.long(), tri_v2.long()
    face_keys = torch.stack([v0 * n + v1, v0 * n + v2, v1 * n + v2], dim=1)
    face_keys_flat = face_keys.reshape(-1)

    # Binary search in sorted edge keys
    positions = torch.searchsorted(sorted_key, face_keys_flat)
    positions = positions.clamp(max=E - 1)

    # Map sorted position → original edge rank
    face_ranks_flat = sort_perm[positions].to(torch.int32)

    return face_ranks_flat.reshape(T, 3).sort(dim=1).values


# ---------------------------------------------------------------------------
# Cofacet CSR construction (GPU)
# ---------------------------------------------------------------------------

def _build_cofacet_csr_h1(
    face_ranks: Tensor, tri_global_rank: Tensor, E: int, T: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build edge → triangle cofacet CSR on GPU.

    For each edge (column), lists the global ranks of triangles containing
    that edge, sorted ascending within each edge.

    Parameters
    ----------
    face_ranks : (T, 3) int32 — face edge ranks per triangle
    tri_global_rank : (T,) int32 — global rank of each triangle
    E : int — number of edges
    T : int — number of triangles

    Returns
    -------
    cofacet_offsets : (E+1,) int32 — CSR pointers
    cofacet_ranks : (3*T,) int32 — triangle global ranks, sorted per edge
    """
    if T == 0:
        return (torch.zeros(E + 1, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device))

    # Expand: 3 entries per triangle (edge_rank, triangle_global_rank)
    tri_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, 3)
    tri_idx = tri_idx.reshape(-1)
    edge_flat = face_ranks.reshape(-1).long()
    trank_flat = tri_global_rank[tri_idx]

    # Two-level stable sort: primary=edge_rank, secondary=tri_global_rank
    perm = trank_flat.argsort(stable=True)
    perm = perm[edge_flat[perm].argsort(stable=True)]

    cofacet_data = trank_flat[perm].to(torch.int32)

    # CSR offsets via scatter_add
    counts = torch.zeros(E, dtype=torch.int32, device=device)
    edge_sorted = edge_flat[perm].to(torch.int32)
    counts.scatter_add_(0, edge_sorted.long(),
                        torch.ones(3 * T, dtype=torch.int32, device=device))
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0)

    return offsets, cofacet_data


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def rips_h1_gpu_native(filt, n, mst_idx, device):
    """GPU-native H1 persistence for Rips complexes.

    Uses cohomology reduction: columns = edges, cofacets = triangles.
    Delegates to _general_cohomology_reduce_gpu_prepass for the actual
    reduction (GPU apparent pairs + Numba residual).

    Parameters
    ----------
    filt : RipsFiltration
    n : int — number of points
    mst_idx : Tensor — MST edge indices (from gpu_boruvka_mst)
    device : torch.device

    Returns
    -------
    h1_diagram : (K, 2) float32 — H1 persistence diagram
    """
    from flash_ph.cohomology_general import (
        _general_cohomology_reduce_gpu_prepass,
    )

    E = filt.edge_i.shape[0]
    edge_i = filt.edge_i
    edge_j = filt.edge_j
    adj_ptr = filt.vert_adj_ptr
    adj_idx = filt.vert_adj_idx
    adj_rank = filt.vert_adj_rank

    # Edge filtration values
    edge_filt = torch.sqrt(filt.edge_dist_sq)

    # MST mask (skip these edges — they're paired in H0)
    mst_mask = torch.zeros(E, dtype=torch.bool, device=device)
    if mst_idx.numel() > 0:
        mst_mask[mst_idx] = True

    # Non-MST edge ranks (for T=0 edge case)
    nonmst_ranks = (~mst_mask).nonzero(as_tuple=True)[0].to(torch.int32)

    # --- Step 1: Enumerate triangles (GPU Triton kernel) ---
    tri_v0, tri_v1, tri_v2, tri_max_rank = triangle_enumerate(
        edge_i, edge_j, adj_ptr, adj_idx, adj_rank, E,
    )
    T = tri_v0.shape[0]

    if T == 0:
        # No triangles → all non-MST edges are essential H1
        if nonmst_ranks.numel() == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)
        births = edge_filt[nonmst_ranks.long()]
        deaths = torch.full_like(births, float('inf'))
        h1 = torch.stack([births, deaths], dim=1)
        return h1[births.argsort(stable=True)]

    # --- Step 2: Face rank lookup (GPU searchsorted) ---
    face_ranks = _lookup_face_ranks_gpu(
        tri_v0, tri_v1, tri_v2, edge_i, edge_j, n,
    )

    # --- Step 3: Sort triangles, assign global ranks ---
    # Triangle colex ID for tiebreaking
    v0, v1, v2 = tri_v0.long(), tri_v1.long(), tri_v2.long()
    tri_colex = v2 * (v2 - 1) * (v2 - 2) // 6 + v1 * (v1 - 1) // 2 + v0

    # Two-pass stable sort: colex (secondary), then max_edge_rank (primary)
    tri_sort_perm = tri_colex.argsort(stable=True)
    tri_sort_perm = tri_sort_perm[
        tri_max_rank[tri_sort_perm].long().argsort(stable=True)
    ]

    # Global ranks: edges 0..E-1, triangles E..E+T-1
    tri_global_rank = torch.empty(T, dtype=torch.int32, device=device)
    tri_global_rank[tri_sort_perm] = (
        torch.arange(T, dtype=torch.int32, device=device) + E
    )

    # --- Step 4: Build cofacet CSR (edge → triangle global ranks) ---
    cofacet_offsets, cofacet_ranks = _build_cofacet_csr_h1(
        face_ranks, tri_global_rank, E, T, device,
    )

    # --- Step 5: Build rank_to_filt ---
    # Edges: rank_to_filt[0..E-1] = edge filtration values
    # Triangles: rank_to_filt[E..E+T-1] = triangle filt in sorted order
    tri_filt_sorted = edge_filt[tri_max_rank[tri_sort_perm].long()]
    rank_to_filt = torch.cat([edge_filt, tri_filt_sorted])

    # --- Step 6: Build column inputs for cohomology reduction ---
    # Columns = ALL edges, col_ranks[i] = i (edge array index = global rank)
    col_ranks = torch.arange(E, dtype=torch.int32, device=device)
    skip_mask = mst_mask  # MST edges are skipped

    # --- Step 7: Run cohomology reduction ---
    pair_births, pair_deaths, ess_births, _ = (
        _general_cohomology_reduce_gpu_prepass(
            col_ranks, cofacet_offsets, cofacet_ranks,
            skip_mask, rank_to_filt,
        )
    )

    # --- Step 8: Convert global ranks → filtration values → H1 diagram ---
    parts = []

    if pair_births.numel() > 0:
        finite_births = rank_to_filt[pair_births.long()]
        finite_deaths = rank_to_filt[pair_deaths.long()]
        parts.append(torch.stack([finite_births, finite_deaths], dim=1))

    if ess_births.numel() > 0:
        ess_birth_filt = rank_to_filt[ess_births.long()]
        ess_death_filt = torch.full_like(ess_birth_filt, float('inf'))
        parts.append(torch.stack([ess_birth_filt, ess_death_filt], dim=1))

    if parts:
        h1 = torch.cat(parts, dim=0)
        h1 = h1[h1[:, 0].argsort(stable=True)]
    else:
        h1 = torch.empty(0, 2, dtype=torch.float32, device=device)

    return h1
