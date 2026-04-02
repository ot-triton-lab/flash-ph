"""GPU-native H1 persistence reduction for Rips complexes.

Pipeline (cohomology approach — columns = edges, cofacets = triangles):
1. Triton triangle enumeration → all triangles with max_edge_rank
2. GPU face rank lookup → triangle face edge ranks
3. GPU cofacet CSR construction → edge-to-triangle adjacency
4. GPU apparent pair pre-detection (scatter-max + vectorized comparison)
5. Adaptive backend: chunked Numba reduction with time-budget monitoring
6. Cohomology reduction (GPU apparent pairs + Numba residual)
7. Assembly → H1 persistence diagram

Uses cohomology (Ripser convention): process edges in reverse filtration
order, pivot = SMALLEST cofacet. This processes E edge columns (~33K for
o3_1024) instead of T triangle columns (~354K), with ~98% apparent pairs
leaving only ~600 residual columns for Numba.

Adaptive backend: for topology-dependent hard cases (e.g. O(3) d=9 with
long XOR reduction chains), the Numba residual reducer can be 16,000x slower
per column than the baseline. Since no static metric can predict this,
we use time-budget monitoring: residual columns are processed in chunks of
100, with each Numba call receiving a time budget equal to the remaining
giotto-ph time estimate. The Numba function checks `time.perf_counter()`
(via objmode) every 100 XOR iterations and aborts if the budget is exceeded.
This catches catastrophically expensive single columns (e.g. 6000ms for
o3_4096 col 2196) that no external post-call check can detect.
Override via ``backend='gpu'`` or ``'giotto'``.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor

from flash_ph.kernels.triangle_kernel import triangle_enumerate


# ---------------------------------------------------------------------------
# Adaptive backend constants
# ---------------------------------------------------------------------------

# Estimated giotto-ph cost per edge (μs). Calibrated on A100-80GB.
# Actual range: 7–35 μs/edge (higher for small E due to fixed overhead).
_GIOTTO_US_PER_EDGE = 7.0

# Minimum giotto-ph estimate (ms) to trigger budget-monitored chunked
# reduction. Below this, both backends finish in <200ms and the
# overhead from chunking (multiple Numba calls, objmode time checks)
# is not worthwhile.
_MIN_GIOTTO_EST_MS = 200.0

# Number of residual columns per Numba call. Each call passes a
# per-chunk slice of the column order, so the Numba loop iterates
# over only _CHUNK_SIZE entries (not all residual columns). The
# intra-column objmode budget check catches catastrophic columns
# within the chunk.
_CHUNK_SIZE = 100


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
# GPU apparent pair pre-detection
# ---------------------------------------------------------------------------

def _gpu_apparent_pair_predetect(
    col_ranks: Tensor,
    cofacet_offsets: Tensor,
    cofacet_ranks: Tensor,
    skip_mask: Tensor,
    rank_to_filt: Tensor,
    total: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute cofacet_max_face_rank, detect apparent pairs, extract results.

    Replicates steps 1–3 of ``_general_cohomology_reduce_gpu_prepass``
    so that the fallback decision can be made *before* Numba reduction.

    Returns
    -------
    cof_max_face : (total,) int32 — scatter-max face rank per simplex
    is_apparent : (K,) bool — apparent pair mask
    ap_births : (A,) int32 — apparent pair birth ranks (positive persistence)
    ap_deaths : (A,) int32 — apparent pair death ranks (positive persistence)
    ap_pivots : (A',) int32 — ALL apparent pair pivots (for merge)
    """
    dev = col_ranks.device
    K = col_ranks.shape[0]
    n_cof = cofacet_ranks.shape[0]

    rank_to_filt_gpu = rank_to_filt.to(dev)

    # --- scatter-max: cofacet_max_face_rank ---
    entry_idx = torch.arange(n_cof, device=dev)
    parent_face = torch.searchsorted(
        cofacet_offsets[1:].long(), entry_idx, right=True,
    )
    face_ranks_expanded = col_ranks[parent_face].clone()
    face_ranks_expanded[skip_mask[parent_face]] = -1

    cof_max_face = torch.full((total,), -1, dtype=torch.int32, device=dev)
    cof_max_face.scatter_reduce_(
        0, cofacet_ranks.long(), face_ranks_expanded, reduce='amax',
    )

    # --- vectorized apparent pair detection ---
    counts = (cofacet_offsets[1:] - cofacet_offsets[:-1]).long()
    has_cofacets = counts > 0

    youngest_cof = torch.zeros(K, dtype=torch.int32, device=dev)
    has_cof_idx = has_cofacets.nonzero(as_tuple=True)[0]
    if has_cof_idx.numel() > 0:
        youngest_cof[has_cof_idx] = cofacet_ranks[
            cofacet_offsets[has_cof_idx].long()
        ]

    is_apparent = (
        has_cofacets & ~skip_mask
        & (cof_max_face[youngest_cof.long()] == col_ranks)
    )

    # --- extract apparent pair results ---
    ap_col_ranks = col_ranks[is_apparent]
    ap_pivots_all = youngest_cof[is_apparent]
    pos_pers = (rank_to_filt_gpu[ap_pivots_all.long()]
                > rank_to_filt_gpu[ap_col_ranks.long()])
    ap_births = ap_col_ranks[pos_pers]
    ap_deaths = ap_pivots_all[pos_pers]

    return cof_max_face, is_apparent, ap_births, ap_deaths, ap_pivots_all


def _fallback_giotto_h1(filt, n, max_edge_length, device):
    """Fall back to giotto-ph C++ reduction for H1.

    Reuses the existing ``_rips_h1h2_giotto`` path which builds a COO
    sparse distance matrix from GPU-enumerated edges and calls
    ``gph.ripser_parallel`` with edge collapse.
    """
    from flash_ph.persistence import _rips_h1h2_giotto
    diagrams = _rips_h1h2_giotto(filt, n, 1, max_edge_length, device)
    return diagrams[0]


# ---------------------------------------------------------------------------
# CPU transfer + Numba pool setup (inlined from gpu_prepass steps 4-5)
# ---------------------------------------------------------------------------

def _prepare_numba_state(
    col_ranks: Tensor,
    cofacet_offsets: Tensor,
    cofacet_ranks: Tensor,
    skip_mask: Tensor,
    rank_to_filt: Tensor,
    cof_max_face: Tensor,
    is_apparent: Tensor,
):
    """Transfer GPU tensors to CPU and prepopulate Numba hash map + pool.

    Returns numpy arrays ready for ``_general_cohomology_reduce_v2``.
    """
    from flash_ph._numba_utils import (
        _prepopulate_apparent_pool, MAX_POOL_ENTRIES,
    )

    K = col_ranks.shape[0]

    # Extend skip mask with apparent pairs
    extended_skip = skip_mask.clone()
    extended_skip[is_apparent] = True

    # Transfer to CPU
    col_np = col_ranks.cpu().to(torch.int32).numpy()
    off_np = cofacet_offsets.cpu().to(torch.int32).numpy()
    cof_np = cofacet_ranks.cpu().to(torch.int32).numpy()
    skip_np = extended_skip.cpu().numpy().astype(np.bool_)
    r2f_np = rank_to_filt.cpu().float().numpy()
    cmf_np = cof_max_face.cpu().to(torch.int32).numpy()

    # Apparent pair indices for pool prepopulation
    ap_indices_np = (
        is_apparent.cpu().nonzero(as_tuple=True)[0].to(torch.int32).numpy()
    )

    # Hash map sizing
    hm_cap = max(K * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Pool sizing
    total_cofacets = int(off_np[K]) if K > 0 else 0
    pool_cap = min(max(total_cofacets * 2, K * 16, 100000), MAX_POOL_ENTRIES)
    pool_data = np.empty(pool_cap, dtype=np.int32)
    piv_start = np.empty(K, dtype=np.int64)
    piv_length = np.empty(K, dtype=np.int32)

    # Prepopulate with apparent pair columns
    pool_ptr, num_piv = _prepopulate_apparent_pool(
        ap_indices_np, off_np, cof_np,
        hm_keys, hm_vals, hm_cap,
        pool_data, piv_start, piv_length, 0, 0,
    )

    return (col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv)


# ---------------------------------------------------------------------------
# Chunked reduction with time-budget monitoring
# ---------------------------------------------------------------------------

def _reduce_chunked_with_budget(
    residual_count: int,
    E: int,
    col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
    hm_keys, hm_vals, hm_cap,
    pool_data, piv_start, piv_length, pool_ptr, num_piv,
):
    """Chunked Numba reduction with intra-column time-budget monitoring.

    Processes residual columns in chunks of ``_CHUNK_SIZE``. Each Numba
    call receives a per-chunk slice of the column order (so the Numba
    loop iterates only over the chunk's columns, not all residuals) and
    a time budget equal to the remaining giotto-ph time estimate.

    Inside the Numba function, ``time.perf_counter()`` is checked via
    ``objmode`` every 100 XOR iterations. If the budget is exceeded
    mid-reduction, the function returns ``pool_ptr = -1`` as a timeout
    sentinel, and this function returns ``None`` to signal fallback.

    Returns
    -------
    results : tuple or None
        ``(pair_births, pair_deaths, ess_births)`` numpy arrays if completed
        within budget, or ``None`` to signal giotto-ph fallback.
    """
    from flash_ph.cohomology_general import (
        _general_cohomology_reduce_v2,
    )

    giotto_est_ms = E * _GIOTTO_US_PER_EDGE / 1000.0

    # For small complexes, both backends are fast — skip chunking overhead.
    # Also skip if giotto-ph is unavailable (no fallback target).
    try:
        import gph  # noqa: F401
    except ImportError:
        return _reduce_full(
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv,
        )

    if giotto_est_ms < _MIN_GIOTTO_EST_MS:
        return _reduce_full(
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv,
        )

    K = col_np.shape[0]

    # Build slim_order: only residual columns in descending rank order.
    full_col_order = np.argsort(col_np)[::-1].copy()
    residual_indices = []
    for ki in range(K):
        idx = full_col_order[ki]
        if not skip_np[idx]:
            residual_indices.append(idx)
    slim_order = np.array(residual_indices, dtype=np.intp)

    all_pb, all_pd, all_eb = [], [], []
    cumul_ms = 0.0

    for start in range(0, residual_count, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, residual_count)

        # Per-chunk order slice: Numba iterates only over these columns
        chunk_order = slim_order[start:end].copy()

        # Fair-share time budget: divide remaining time evenly across
        # remaining chunks, with a 300ms floor to absorb system jitter.
        remaining_chunks = max(1, (residual_count - start + _CHUNK_SIZE - 1) // _CHUNK_SIZE)
        remaining_s = max(0.3, (giotto_est_ms - cumul_ms) / 1000.0 / remaining_chunks)

        t0 = time.perf_counter()
        pb, pd, eb, _, pool_ptr, num_piv = _general_cohomology_reduce_v2(
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv,
            chunk_order,
            budget_s=remaining_s,
        )
        chunk_ms = (time.perf_counter() - t0) * 1000.0
        cumul_ms += chunk_ms

        # Intra-column timeout: Numba hit budget mid-XOR-loop
        if pool_ptr == -1:
            return None

        all_pb.append(pb)
        all_pd.append(pd)
        all_eb.append(eb)

        # Cumulative safety net
        if cumul_ms > giotto_est_ms:
            return None

    return (np.concatenate(all_pb) if all_pb else np.empty(0, dtype=np.int32),
            np.concatenate(all_pd) if all_pd else np.empty(0, dtype=np.int32),
            np.concatenate(all_eb) if all_eb else np.empty(0, dtype=np.int32))


def _reduce_full(
    col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
    hm_keys, hm_vals, hm_cap,
    pool_data, piv_start, piv_length, pool_ptr, num_piv,
):
    """Run full Numba reduction without budget monitoring."""
    from flash_ph.cohomology_general import (
        _general_cohomology_reduce_v2,
    )
    col_order = np.argsort(col_np)[::-1].copy()
    pb, pd, eb, _, _, _ = _general_cohomology_reduce_v2(
        col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
        hm_keys, hm_vals, hm_cap,
        pool_data, piv_start, piv_length, pool_ptr, num_piv,
        col_order,
    )
    return (pb, pd, eb)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ('auto', 'gpu', 'giotto')


def rips_h1_gpu_native(filt, n, mst_idx, max_edge_length, device, backend='auto'):
    """GPU-native H1 persistence for Rips complexes (with adaptive fallback).

    Uses cohomology reduction: columns = edges, cofacets = triangles.
    Pre-computes apparent pairs on GPU, then uses a timed trial of Numba
    residual columns to decide whether to continue GPU-native or fall back
    to giotto-ph.

    Parameters
    ----------
    filt : RipsFiltration
    n : int — number of points
    mst_idx : Tensor — MST edge indices (from gpu_boruvka_mst)
    max_edge_length : float — threshold for giotto-ph fallback
    device : torch.device
    backend : str
        ``'auto'`` (default) — timed trial selects GPU-native or giotto-ph.
        ``'gpu'`` — always use GPU-native (Numba residual).
        ``'giotto'`` — always use giotto-ph C++ reduction.

    Returns
    -------
    h1_diagram : (K, 2) float32 — H1 persistence diagram
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS}, got {backend!r}"
        )

    # Early exit: always use giotto-ph if requested
    if backend == 'giotto':
        return _fallback_giotto_h1(filt, n, max_edge_length, device)

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
    v0, v1, v2 = tri_v0.long(), tri_v1.long(), tri_v2.long()
    tri_colex = v2 * (v2 - 1) * (v2 - 2) // 6 + v1 * (v1 - 1) // 2 + v0

    tri_sort_perm = tri_colex.argsort(stable=True)
    tri_sort_perm = tri_sort_perm[
        tri_max_rank[tri_sort_perm].long().argsort(stable=True)
    ]

    tri_global_rank = torch.empty(T, dtype=torch.int32, device=device)
    tri_global_rank[tri_sort_perm] = (
        torch.arange(T, dtype=torch.int32, device=device) + E
    )

    # --- Step 4: Build cofacet CSR (edge → triangle global ranks) ---
    cofacet_offsets, cofacet_ranks = _build_cofacet_csr_h1(
        face_ranks, tri_global_rank, E, T, device,
    )

    # --- Step 5: Build rank_to_filt ---
    tri_filt_sorted = edge_filt[tri_max_rank[tri_sort_perm].long()]
    rank_to_filt = torch.cat([edge_filt, tri_filt_sorted])

    # --- Step 6: Build column inputs ---
    col_ranks = torch.arange(E, dtype=torch.int32, device=device)
    skip_mask = mst_mask

    # --- Step 7: GPU apparent pair pre-detection ---
    total = E + T
    cof_max_face, is_apparent, ap_births, ap_deaths, ap_pivots = (
        _gpu_apparent_pair_predetect(
            col_ranks, cofacet_offsets, cofacet_ranks,
            skip_mask, rank_to_filt, total,
        )
    )
    residual_count = int((~is_apparent & ~skip_mask).sum().item())

    # --- Step 8: Prepare Numba state (CPU transfer + pool prepopulation) ---
    numba_state = _prepare_numba_state(
        col_ranks, cofacet_offsets, cofacet_ranks,
        skip_mask, rank_to_filt, cof_max_face, is_apparent,
    )
    (col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
     hm_keys, hm_vals, hm_cap,
     pool_data, piv_start, piv_length, pool_ptr, num_piv) = numba_state

    # --- Step 9+10: Numba residual reduction (with budget monitoring) ---
    if backend == 'auto':
        result = _reduce_chunked_with_budget(
            residual_count, E,
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv,
        )
        if result is None:
            return _fallback_giotto_h1(filt, n, max_edge_length, device)
        numba_pb, numba_pd, numba_eb = result
    else:
        result = _reduce_full(
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, pool_ptr, num_piv,
        )
        numba_pb, numba_pd, numba_eb = result

    # --- Step 11: Merge GPU apparent pairs + Numba results ---
    all_pb = torch.cat([ap_births.cpu().to(torch.int32),
                        torch.from_numpy(numba_pb.copy())])
    all_pd = torch.cat([ap_deaths.cpu().to(torch.int32),
                        torch.from_numpy(numba_pd.copy())])
    all_eb = torch.from_numpy(numba_eb.copy())

    pair_births = all_pb.to(device)
    pair_deaths = all_pd.to(device)
    ess_births = all_eb.to(device)

    # --- Step 12: Convert global ranks → filtration values → H1 diagram ---
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
