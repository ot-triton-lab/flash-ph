"""Public API for Vietoris-Rips persistent homology (H0 + H1 + H2).

Pipeline:
- GPU (CUDA): direct Triton edge enumeration -> Boruvka MST (H0)
- CPU (Numba): cohomology-based reduction for H1 (host transfer of CSR/edges)
- CPU: general cohomology reduction for H2 (explicit simplex enumeration +
  clearing cascade from H1, same reduction algorithm)

H2 design notes (Ripser-style clearing + general cohomology reducer):
  Ripser and Ripser++ compute H2 via implicit coboundary on-the-fly, with
  clearing (H1 death-pivots skip in H2) and apparent-pair detection (~99%
  of columns are apparent).

  Our approach: explicitly enumerate all triangles and tetrahedra in the
  Rips complex from CSR adjacency, then feed them into the general
  cohomology reducer. This gives:
  - Clearing optimization via skip-mask cascade (H1 pivot triangles skipped)
  - Apparent pair optimization via cofacet_max_face_rank precomputation
  - Exact results matching ripser
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
# H2 reduction helpers
# ---------------------------------------------------------------------------

_VERTEX_PACK_BITS = 16


def _rips_h2_build_csr_gpu(
    tri_verts: Tensor,
    tet_verts: Tensor,
    tri_mr: Tensor,
    tet_mr: Tensor,
    edge_dist: Tensor,
    h1_pivot_colex_sorted: Tensor,
    keep_device: bool = False,
):
    """Build H2 reduction inputs directly on GPU.

    Specialized for Rips H2: only builds dim 2->3 cofacet CSR,
    uses restricted global ranking (triangles + tetrahedra only),
    and computes skip-mask via pre-sorted H1 pivots.

    Returns (col_ranks, cofacet_offsets, cofacet_ranks, skip_mask,
             rank_to_filt, cofacet_max_face_rank) all on CPU ready
    for Numba reduction (unless keep_device=True).
    """
    from flash_ph.cohomology_h2 import _simplex_colex_id
    from flash_ph._simplex_utils import _pack_simplex_keys

    dev = tri_verts.device
    T = tri_verts.shape[0]
    Q = tet_verts.shape[0]

    # Guard: 16-bit vertex packing requires n < 65536
    max_vert = int(tri_verts.max().item()) if T > 0 else 0
    if Q > 0:
        max_vert = max(max_vert, int(tet_verts.max().item()))
    if max_vert >= (1 << _VERTEX_PACK_BITS):
        raise ValueError(
            f"Vertex index {max_vert} exceeds {_VERTEX_PACK_BITS}-bit "
            f"packing limit ({1 << _VERTEX_PACK_BITS}). Reduce n or "
            f"max_edge_length."
        )

    # --- 1. Restricted global ranking (T+Q only, not n+E+T+Q) ---
    # Sort key: (max_edge_rank, dim, colex) via three-pass stable sort.
    tri_mr_long = tri_mr.long()
    tet_mr_long = tet_mr.long() if Q > 0 else torch.empty(0, dtype=torch.int64, device=dev)
    all_mr = torch.cat([tri_mr_long, tet_mr_long])  # (T+Q,)
    all_dim = torch.cat([
        torch.full((T,), 2, dtype=torch.int64, device=dev),
        torch.full((Q,), 3, dtype=torch.int64, device=dev),
    ])  # (T+Q,)

    # Compute colex IDs for tiebreaking
    tri_sorted = tri_verts.long()
    a, b, c = tri_sorted[:, 0], tri_sorted[:, 1], tri_sorted[:, 2]
    tri_colex = c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a
    if Q > 0:
        tet_sorted = tet_verts.long()
        va, vb, vc, vd = tet_sorted[:, 0], tet_sorted[:, 1], tet_sorted[:, 2], tet_sorted[:, 3]
        # Overflow-safe C(vd,4): divide early
        c4_hi = vd * (vd - 1) // 2
        c4_lo = (vd - 2) * (vd - 3) // 2
        tet_colex = (c4_hi * c4_lo // 6
                     + vc * (vc - 1) * (vc - 2) // 6
                     + vb * (vb - 1) // 2 + va)
    else:
        tet_colex = torch.empty(0, dtype=torch.int64, device=dev)
    all_colex = torch.cat([tri_colex, tet_colex])

    # Three-pass stable sort (reverse priority): colex, dim, max_edge_rank
    perm = torch.argsort(all_colex, stable=True)
    perm = perm[torch.argsort(all_dim[perm], stable=True)]
    perm = perm[torch.argsort(all_mr[perm], stable=True)]

    total = T + Q
    rank_flat = torch.empty(total, dtype=torch.int32, device=dev)
    rank_flat[perm] = torch.arange(total, dtype=torch.int32, device=dev)
    tri_ranks = rank_flat[:T]      # (T,) global ranks of triangles
    tet_ranks = rank_flat[T:]      # (Q,) global ranks of tetrahedra

    # rank_to_filt: map restricted global rank -> filtration distance
    rank_to_filt = edge_dist[all_mr[perm]].cpu().float()  # (T+Q,)

    # --- 2. Direct dim 2->3 cofacet CSR on GPU ---
    tri_sorted_verts = tri_verts.long()
    tri_keys = _pack_simplex_keys(tri_sorted_verts)  # (T,)
    tri_key_perm = tri_keys.argsort()
    tri_keys_sorted = tri_keys[tri_key_perm]

    if Q > 0:
        tet_sorted_verts = tet_verts.long()
        all_face_idx = []
        all_cofacet_idx = []

        for drop in range(4):
            face = torch.cat([
                tet_sorted_verts[:, :drop],
                tet_sorted_verts[:, drop + 1:],
            ], dim=1)  # (Q, 3)
            query = _pack_simplex_keys(face)
            pos = torch.searchsorted(tri_keys_sorted, query)
            pos = pos.clamp(max=max(T - 1, 0))
            matched = tri_keys_sorted[pos] == query
            midx = matched.nonzero(as_tuple=True)[0]
            if midx.numel() > 0:
                all_face_idx.append(tri_key_perm[pos[midx]])
                all_cofacet_idx.append(midx)

        n_matched = sum(fi.numel() for fi in all_face_idx) if all_face_idx else 0
        if n_matched != 4 * Q:
            raise RuntimeError(
                f"Face matching failed: expected {4 * Q} matches "
                f"(4 faces × {Q} tets), got {n_matched}. "
                f"Possible key collision or enumeration bug."
            )

        if all_face_idx:
            all_fi = torch.cat(all_face_idx)
            all_ci = torch.cat(all_cofacet_idx)

            cofacet_global = tet_ranks[all_ci.long()]

            sort_perm = all_fi.argsort(stable=True)
            sorted_fi = all_fi[sort_perm]
            sorted_cof = cofacet_global[sort_perm]

            counts = torch.zeros(T, dtype=torch.int64, device=dev)
            counts.scatter_add_(
                0, sorted_fi.long(),
                torch.ones_like(sorted_fi, dtype=torch.int64),
            )
            offsets = torch.zeros(T + 1, dtype=torch.int32, device=dev)
            offsets[1:] = counts.cumsum(0).to(torch.int32)

            shift = (total - 1).bit_length() + 1
            seg_key = sorted_fi.long() << shift | sorted_cof.long()
            seg_perm = seg_key.argsort(stable=True)
            cofacet_ranks_sorted = sorted_cof[seg_perm]
        else:
            offsets = torch.zeros(T + 1, dtype=torch.int32, device=dev)
            cofacet_ranks_sorted = torch.empty(
                0, dtype=torch.int32, device=dev,
            )
    else:
        offsets = torch.zeros(T + 1, dtype=torch.int32, device=dev)
        cofacet_ranks_sorted = torch.empty(0, dtype=torch.int32, device=dev)

    # --- 3. Skip mask (H1 clearing) ---
    skip_mask = torch.zeros(T, dtype=torch.bool, device=dev)
    if h1_pivot_colex_sorted.numel() > 0:
        tri_colex = _simplex_colex_id(tri_sorted_verts.int(), dim=2)
        h1_sorted = h1_pivot_colex_sorted.to(dev)
        pos = torch.searchsorted(h1_sorted, tri_colex)
        pos = pos.clamp(max=max(h1_sorted.numel() - 1, 0))
        skip_mask = h1_sorted[pos] == tri_colex

    # --- 4. Compute cofacet_max_face_rank on GPU ---
    n_cof = cofacet_ranks_sorted.shape[0]
    cofacet_max_face_rank = torch.full(
        (total,), -1, dtype=torch.int32, device=dev,
    )
    if n_cof > 0:
        entry_idx = torch.arange(n_cof, device=dev)
        parent_face = torch.searchsorted(
            offsets[1:].long(), entry_idx, right=True,
        )
        face_ranks_expanded = tri_ranks[parent_face].clone()
        face_ranks_expanded[skip_mask[parent_face]] = -1
        cofacet_max_face_rank.scatter_reduce_(
            0, cofacet_ranks_sorted.long(),
            face_ranks_expanded, reduce='amax',
        )

    if keep_device:
        return (tri_ranks, offsets, cofacet_ranks_sorted, skip_mask,
                rank_to_filt, cofacet_max_face_rank)
    return (
        tri_ranks.cpu(),
        offsets.cpu(),
        cofacet_ranks_sorted.cpu(),
        skip_mask.cpu(),
        rank_to_filt,
        cofacet_max_face_rank.cpu(),
    )


def _rips_h2_build_csr_cpu(
    tv0, tv1, tv2, tri_mr_np,
    qv0, qv1, qv2, qv3, tet_mr_np,
    edge_dist: Tensor,
    h1_pivot_colex_sorted: Tensor,
    n: int,
    edge_i: Tensor,
    edge_j: Tensor,
):
    """CPU fallback: uses generic functions for H2 CSR building."""
    from flash_ph.cohomology_h2 import (
        compute_global_ranks, _simplex_colex_id,
    )
    from flash_ph._simplex_utils import (
        _build_cofacet_adjacency,
        _build_cofacet_csr_for_reduction,
    )

    E = edge_dist.shape[0]
    tri_verts = torch.stack([
        torch.from_numpy(tv0), torch.from_numpy(tv1), torch.from_numpy(tv2),
    ], dim=1)
    tet_verts = torch.stack([
        torch.from_numpy(qv0), torch.from_numpy(qv1),
        torch.from_numpy(qv2), torch.from_numpy(qv3),
    ], dim=1)

    edge_dist_cpu = edge_dist.cpu()
    tri_filt = edge_dist_cpu[torch.from_numpy(tri_mr_np).long()].float()
    tet_filt = edge_dist_cpu[torch.from_numpy(tet_mr_np).long()].float()

    # Full global ranking (all dims) for generic path
    vert_verts = torch.arange(n, dtype=torch.int32).unsqueeze(1)
    vert_filt = torch.zeros(n, dtype=torch.float32)
    edge_filt = edge_dist_cpu.float()
    ei = edge_i.cpu().to(torch.int32)
    ej = edge_j.cpu().to(torch.int32)
    edge_verts = torch.stack([torch.minimum(ei, ej), torch.maximum(ei, ej)], dim=1)

    simplices = {0: vert_verts, 1: edge_verts, 2: tri_verts, 3: tet_verts}
    filtrations = {0: vert_filt, 1: edge_filt, 2: tri_filt, 3: tet_filt}

    global_rank, rank_to_filt = compute_global_ranks(
        simplices, filtrations, device=torch.device("cpu"),
    )
    cofacet_adj = _build_cofacet_adjacency(simplices, max_dim=2)
    col_ranks, cofacet_offsets, cofacet_ranks = _build_cofacet_csr_for_reduction(
        simplices, cofacet_adj, global_rank, target_dim=2,
    )

    # Skip mask
    T = tri_verts.shape[0]
    skip_mask = torch.zeros(T, dtype=torch.bool)
    if h1_pivot_colex_sorted.numel() > 0:
        tri_colex = _simplex_colex_id(
            tri_verts.sort(dim=1).values, dim=2,
        )
        pos = torch.searchsorted(h1_pivot_colex_sorted, tri_colex)
        pos = pos.clamp(max=max(h1_pivot_colex_sorted.numel() - 1, 0))
        skip_mask = h1_pivot_colex_sorted[pos] == tri_colex

    return col_ranks, cofacet_offsets, cofacet_ranks, skip_mask, rank_to_filt, None


def _rips_h2_reduction(
    filt,
    edge_dist_sq: Tensor,
    h1_all_pivot_colex: Tensor,
    n: int,
    E: int,
    device: torch.device,
) -> Tensor:
    """Compute H2 persistence via general cohomology reduction.

    GPU path: specialized direct cofacet CSR on GPU (restricted ranking,
    dim 2->3 only). CPU path: generic functions as fallback.

    Accepts edge_dist_sq (squared distances) and computes sqrt lazily
    only for the T+Q simplex max-edge-ranks H2 actually needs.
    """
    from flash_ph.cohomology_h2 import general_cohomology_reduce

    # Pre-sort H1 pivots once (used by both paths)
    h1_sorted = (
        h1_all_pivot_colex.sort().values
        if h1_all_pivot_colex.numel() > 0
        else h1_all_pivot_colex
    )

    # Enumerate triangles and tetrahedra
    use_gpu = device.type == "cuda" and torch.cuda.is_available()

    if use_gpu:
        from flash_ph.kernels.triangle_kernel import (
            triangle_enumerate, tetrahedra_enumerate,
        )
        tv0_t, tv1_t, tv2_t, tri_mr_t = triangle_enumerate(
            filt.edge_i, filt.edge_j,
            filt.vert_adj_ptr, filt.vert_adj_idx, filt.vert_adj_rank,
            E,
        )
        T = tv0_t.shape[0]
        if T == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)

        qv0_t, qv1_t, qv2_t, qv3_t, tet_mr_t = tetrahedra_enumerate(
            tv0_t, tv1_t, tv2_t, tri_mr_t,
            filt.vert_adj_ptr, filt.vert_adj_idx, filt.vert_adj_rank,
            T,
        )

        tri_verts = torch.stack([tv0_t, tv1_t, tv2_t], dim=1)
        Q = qv0_t.shape[0]
        if Q > 0:
            tet_verts = torch.stack(
                [qv0_t, qv1_t, qv2_t, qv3_t], dim=1,
            )
        else:
            tet_verts = torch.empty(
                0, 4, dtype=tv0_t.dtype, device=device,
            )

        # Lazy sqrt
        edge_dist = torch.sqrt(edge_dist_sq)

        # Specialized GPU path: direct cofacet CSR + GPU scatter-max
        (col_ranks, cof_offsets, cof_ranks, skip_mask,
         rank_to_filt, cofacet_max_face_rank) = (
            _rips_h2_build_csr_gpu(
                tri_verts, tet_verts, tri_mr_t, tet_mr_t,
                edge_dist, h1_sorted,
                keep_device=True,
            )
        )
    else:
        # CPU path: Numba enumeration + generic CSR building
        ei_cpu = filt.edge_i.cpu().numpy()
        ej_cpu = filt.edge_j.cpu().numpy()
        ptr_cpu = filt.vert_adj_ptr.cpu().numpy()
        idx_cpu = filt.vert_adj_idx.cpu().numpy()
        rank_cpu = filt.vert_adj_rank.cpu().numpy()

        tv0, tv1, tv2, tri_mr_np = _enumerate_rips_triangles(
            ei_cpu, ej_cpu, ptr_cpu, idx_cpu, rank_cpu, E,
        )
        T = tv0.shape[0]
        if T == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)

        qv0, qv1, qv2, qv3, tet_mr_np = _enumerate_rips_tetrahedra(
            tv0, tv1, tv2, tri_mr_np, ptr_cpu, idx_cpu, rank_cpu, T,
        )

        edge_dist = torch.sqrt(edge_dist_sq)

        (col_ranks, cof_offsets, cof_ranks, skip_mask,
         rank_to_filt, _) = (
            _rips_h2_build_csr_cpu(
                tv0, tv1, tv2, tri_mr_np,
                qv0, qv1, qv2, qv3, tet_mr_np,
                edge_dist, h1_sorted, n,
                filt.edge_i, filt.edge_j,
            )
        )
        cofacet_max_face_rank = None

    # Run general cohomology reduction
    pb, pd, eb, _ = general_cohomology_reduce(
        col_ranks, cof_offsets, cof_ranks, skip_mask, rank_to_filt,
        cofacet_max_face_rank=cofacet_max_face_rank,
    )

    # Ensure rank_to_filt and rank indices are on same device
    rank_to_filt_cpu = rank_to_filt.cpu()
    pb = pb.cpu()
    pd = pd.cpu()
    eb = eb.cpu()

    # Build H2 diagram
    h2_parts = []
    if pb.numel() > 0:
        birth_filt = rank_to_filt_cpu[pb.long()]
        death_filt = rank_to_filt_cpu[pd.long()]
        finite_h2 = torch.stack(
            [birth_filt, death_filt], dim=1,
        ).float().to(device)
        h2_parts.append(finite_h2)
    if eb.numel() > 0:
        ess_filt = rank_to_filt_cpu[eb.long()]
        inf_h2 = torch.stack([
            ess_filt.float().to(device),
            torch.full(
                (eb.numel(),), float("inf"),
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
