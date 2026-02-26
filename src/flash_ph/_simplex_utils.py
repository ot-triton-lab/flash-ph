"""Simplex packing, cofacet adjacency, and CSR building utilities.

Extracted from flash_tda.complexes.flood (_pack_simplex_keys,
_build_cofacet_adjacency) and flash_tda.homology.flood_persistence
(_build_cofacet_csr_for_reduction) for use in flash-ph without
depending on the flash-tda package.
"""
from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bit width for packing vertex indices into int64 keys.
# Supports up to 2^16 = 65536 vertices (landmark indices).
VERTEX_PACK_BITS = 16


# ---------------------------------------------------------------------------
# Simplex key packing
# ---------------------------------------------------------------------------

def _pack_simplex_keys(verts: Tensor) -> Tensor:
    """Pack sorted vertex tuples into int64 keys for fast lookup.

    Uses 16-bit shifts per vertex, so vertex indices must be < 65536.

    verts: (S, k) int32/int64 tensor, each row sorted ascending.
    Returns: (S,) int64 keys.
    """
    if verts.numel() > 0 and int(verts.max().item()) >= (1 << VERTEX_PACK_BITS):
        raise ValueError(
            f"Vertex index {int(verts.max().item())} exceeds "
            f"{VERTEX_PACK_BITS}-bit packing limit "
            f"({(1 << VERTEX_PACK_BITS) - 1}). "
            f"Reduce n_landmarks or widen the packing scheme."
        )
    keys = torch.zeros(verts.shape[0], dtype=torch.int64, device=verts.device)
    for i in range(verts.shape[1]):
        keys += verts[:, i].long() << (VERTEX_PACK_BITS * i)
    return keys


# ---------------------------------------------------------------------------
# Cofacet adjacency (CSR)
# ---------------------------------------------------------------------------

def _build_cofacet_adjacency(
    simplices: dict[int, Tensor], max_dim: int
) -> dict[int, tuple[Tensor, Tensor]]:
    """For each d-simplex (d <= max_dim), list its (d+1)-simplex cofacets.

    Vectorized: uses packed int64 keys and searchsorted for fast lookups.

    Parameters
    ----------
    simplices : dict from dim -> (S_d, dim+1) int32 Tensor
    max_dim : int

    Returns
    -------
    dict keyed by dim d, each value is (cofacet_offsets, cofacet_indices) CSR
    where cofacet_offsets is (S_d + 1,) and cofacet_indices is flat.
    """
    result: dict[int, tuple[Tensor, Tensor]] = {}
    device = next(iter(simplices.values())).device

    for d in range(max_dim + 1):
        if d not in simplices:
            result[d] = (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )
            continue

        faces = simplices[d].sort(dim=1).values  # (S_d, d+1)
        S_d = faces.shape[0]

        if d + 1 not in simplices or simplices[d + 1].shape[0] == 0:
            result[d] = (
                torch.zeros(S_d + 1, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )
            continue

        # Pack face keys and sort for searchsorted
        face_keys = _pack_simplex_keys(faces)
        face_perm = face_keys.argsort()
        face_keys_sorted = face_keys[face_perm]
        # Inverse: face_perm[rank] = original_idx -> need original_idx -> rank
        face_rank = torch.empty_like(face_perm)
        face_rank[face_perm] = torch.arange(S_d, dtype=torch.int64, device=device)

        cofacets = simplices[d + 1].sort(dim=1).values.long()  # (S_{d+1}, d+2)
        S_dp1 = cofacets.shape[0]
        n_faces_per_cofacet = cofacets.shape[1]  # d+2

        # For each cofacet, enumerate its faces (drop each vertex)
        # Collect all (face_idx, cofacet_idx) pairs
        all_face_indices = []
        all_cofacet_indices = []

        for drop in range(n_faces_per_cofacet):
            face_verts = torch.cat([
                cofacets[:, :drop], cofacets[:, drop + 1:]
            ], dim=1)  # (S_{d+1}, d+1)
            query_keys = _pack_simplex_keys(face_verts)

            pos = torch.searchsorted(face_keys_sorted, query_keys)
            pos = pos.clamp(max=S_d - 1)
            matched = face_keys_sorted[pos] == query_keys

            matched_idx = matched.nonzero(as_tuple=True)[0]
            if matched_idx.numel() > 0:
                # face_idx in original ordering
                face_idx = face_perm[pos[matched_idx]]
                cofacet_idx = matched_idx.to(torch.int32)
                all_face_indices.append(face_idx)
                all_cofacet_indices.append(cofacet_idx)

        if all_face_indices:
            all_fi = torch.cat(all_face_indices)  # face indices
            all_ci = torch.cat(all_cofacet_indices)  # cofacet indices

            # Sort by face index to build CSR
            sort_perm = all_fi.argsort(stable=True)
            sorted_fi = all_fi[sort_perm]
            sorted_ci = all_ci[sort_perm]

            # Build CSR offsets via bincount
            counts = torch.zeros(S_d, dtype=torch.int64, device=device)
            counts.scatter_add_(0, sorted_fi.long(), torch.ones_like(sorted_fi, dtype=torch.int64))
            offsets = torch.zeros(S_d + 1, dtype=torch.int32, device=device)
            offsets[1:] = counts.cumsum(0).to(torch.int32)

            result[d] = (offsets, sorted_ci)
        else:
            result[d] = (
                torch.zeros(S_d + 1, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

    return result


# ---------------------------------------------------------------------------
# CSR cofacet mapping for cohomology reduction
# ---------------------------------------------------------------------------

def _build_cofacet_csr_for_reduction(
    simplices: dict[int, Tensor],
    cofacet_adj: dict[int, tuple[Tensor, Tensor]],
    global_rank: dict[int, Tensor],
    target_dim: int,
    keep_device: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build CSR cofacet mapping using global ranks for reduction input.

    Parameters
    ----------
    simplices : dict of {dim: (S_d, dim+1) int32}
    cofacet_adj : dict of {dim: (offsets, indices)} from _build_cofacet_adjacency
    global_rank : dict of {dim: (S_d,) int32} from compute_global_ranks
    target_dim : int -- dimension of columns (faces)
    keep_device : bool -- if True, keep tensors on input device (GPU);
        if False (default), move to CPU for Numba compatibility.

    Returns
    -------
    col_ranks : (K,) int32 -- global ranks of target_dim simplices
    cofacet_offsets : (K+1,) int32 -- CSR offsets
    cofacet_ranks : (total,) int32 -- sorted global ranks of cofacets
    """
    d = target_dim
    col_ranks = global_rank[d]
    K = col_ranks.shape[0]

    if d not in cofacet_adj:
        offsets = torch.zeros(K + 1, dtype=torch.int32, device=col_ranks.device)
        cof_ranks = torch.empty(0, dtype=torch.int32, device=col_ranks.device)
        return col_ranks, offsets, cof_ranks

    adj_offsets, adj_indices = cofacet_adj[d]
    d1 = d + 1

    if d1 not in global_rank or adj_indices.numel() == 0:
        offsets = torch.zeros(K + 1, dtype=torch.int32, device=col_ranks.device)
        cof_ranks = torch.empty(0, dtype=torch.int32, device=col_ranks.device)
        return col_ranks, offsets, cof_ranks

    # Map cofacet indices to global ranks
    rank_d1 = global_rank[d1]
    adj_indices_long = adj_indices.long()

    if keep_device:
        # Stay on input device (e.g. CUDA) for GPU apparent pair prepass
        adj_offsets_w = adj_offsets
        cofacet_global = rank_d1[adj_indices_long]
    else:
        # Move to CPU for Numba compatibility (original behavior)
        adj_offsets_w = adj_offsets.cpu()
        cofacet_global = rank_d1.cpu()[adj_indices_long.cpu()]

    # Vectorized segment sort: sort cofacet ranks within each face's segment
    if cofacet_global.numel() > 0:
        dev = cofacet_global.device
        counts = (adj_offsets_w[1:] - adj_offsets_w[:-1]).long()
        seg_ids = torch.repeat_interleave(
            torch.arange(K, dtype=torch.int64, device=dev), counts,
        )
        sort_key = seg_ids * (cofacet_global.max().item() + 1) + cofacet_global.long()
        perm = sort_key.argsort(stable=True)
        cof_ranks = cofacet_global[perm]
    else:
        cof_ranks = torch.empty(0, dtype=torch.int32, device=cofacet_global.device)

    device = col_ranks.device
    cof_offsets = adj_offsets_w.to(torch.int32)
    return col_ranks, cof_offsets.to(device), cof_ranks.to(device)
