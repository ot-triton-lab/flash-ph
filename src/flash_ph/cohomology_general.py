"""General cohomology reduction for any filtered simplicial complex over Z/2Z.

Generalizes the Rips-specific cohomology reduction to work with any
pre-computed filtered complex. Instead of enumerating cofacets on-the-fly
via CSR adjacency intersection, reads cofacets from pre-computed CSR arrays.

Key simplification vs Rips version:
- cofacet_ranks serve as BOTH sort key AND unique pivot ID (global rank is unique)
- Single array per column entry instead of parallel (rank, colex) arrays
- Apparent pair optimization: if σ is the oldest face of its youngest cofacet τ,
  the pair (σ, τ) is apparent and the reduction loop is skipped entirely
"""
from __future__ import annotations


import numpy as np
import torch
from torch import Tensor
from numba import njit

from flash_ph._numba_utils import (
    _hm_get, _hm_set, _sorted_merge_xor_single, MAX_COL, MAX_POOL_ENTRIES,
)


# ---------------------------------------------------------------------------
# Main reduction loop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _general_cohomology_reduce(
    col_ranks,              # (K,) int32 — global filtration rank of each column
    cofacet_offsets,        # (K+1,) int32 — CSR offsets into cofacet_ranks
    cofacet_ranks,          # (total_cofacets,) int32 — global rank of each cofacet
    skip_mask,              # (K,) bool — True = skip this column
    rank_to_filt,           # (total,) float32 — filtration value at each global rank
):
    """General cohomology reduction over Z/2Z.

    Processes columns in REVERSE rank order (largest rank first).
    Computes cofacet_max_face_rank inline (no external precomputation needed).

    Returns: pair_birth_ranks, pair_death_ranks, essential_birth_ranks, all_pivot_ranks
    """
    K = col_ranks.shape[0]
    total_ranks = rank_to_filt.shape[0]

    # --- Inline cof_max_face computation ---
    # For each cofacet (by global rank), track the maximum global rank
    # among its non-skipped faces.
    cofacet_max_face_rank = np.full(total_ranks, np.int32(-1))
    for k in range(K):
        if skip_mask[k]:
            continue
        r_k = col_ranks[k]
        for c in range(cofacet_offsets[k], cofacet_offsets[k + 1]):
            cr = cofacet_ranks[c]
            if r_k > cofacet_max_face_rank[cr]:
                cofacet_max_face_rank[cr] = r_k

    # Output buffers
    pair_births = np.empty(K, dtype=np.int32)
    pair_deaths = np.empty(K, dtype=np.int32)
    ess_births = np.empty(K, dtype=np.int32)
    all_pivots = np.empty(K, dtype=np.int32)
    n_pairs = 0
    n_ess = 0
    n_all_piv = 0

    # Pivot hash map: cofacet_rank -> pool index
    hm_cap = max(K * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Pool allocator (single array per entry)
    # Size based on actual total cofacet entries (not K*16 which underestimates
    # when cofacet lists are long, e.g. H2 with ~40 cofacets per triangle)
    total_cofacets = np.int64(cofacet_offsets[K])
    pool_cap = min(max(total_cofacets * 2, K * 16, 100000), MAX_POOL_ENTRIES)
    pool_data = np.empty(pool_cap, dtype=np.int32)
    piv_start = np.empty(K, dtype=np.int64)
    piv_length = np.empty(K, dtype=np.int32)
    pool_ptr = 0
    num_piv = 0

    # Working buffers
    cur = np.empty(MAX_COL, dtype=np.int32)
    tmp = np.empty(MAX_COL, dtype=np.int32)

    # Sort col_ranks indices by rank DESCENDING for processing order
    order = np.argsort(col_ranks)[::-1]

    for ki in range(K):
        idx = order[ki]
        if skip_mask[idx]:
            continue

        r_e = col_ranks[idx]

        # Load cofacets from pre-computed CSR
        c_start = cofacet_offsets[idx]
        c_end = cofacet_offsets[idx + 1]
        cn = c_end - c_start

        if cn == 0:
            ess_births[n_ess] = r_e
            n_ess += 1
            continue

        # --- Apparent pair check (before copying into cur) ---
        youngest_cof = cofacet_ranks[c_start]  # smallest rank (sorted ascending)
        if cofacet_max_face_rank[youngest_cof] == r_e:
            # σ is the oldest face of τ → apparent pair (skip reduction)
            all_pivots[n_all_piv] = youngest_cof
            n_all_piv += 1

            # Store column directly into pool for future XOR reductions
            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap,
                         np.int64(youngest_cof), np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_data[pool_ptr + i] = cofacet_ranks[c_start + i]
                pool_ptr += cn
                num_piv += 1

            # Record pair if positive persistence
            if rank_to_filt[youngest_cof] > rank_to_filt[r_e]:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = youngest_cof
                n_pairs += 1
            continue  # skip copy-into-cur + reduction loop entirely
        # --- End apparent pair check ---

        # Copy cofacet ranks into working buffer (already sorted ascending)
        if cn > MAX_COL:
            cn = MAX_COL  # truncate to avoid OOB on working buffer
        for i in range(cn):
            cur[i] = cofacet_ranks[c_start + i]

        # Reduction loop
        while cn > 0:
            piv = cur[0]  # pivot = SMALLEST (cohomology)
            piv_idx = _hm_get(hm_keys, hm_vals, hm_cap, np.int64(piv))
            if piv_idx == -1:
                break

            # XOR with stored column
            ps = piv_start[piv_idx]
            plen = piv_length[piv_idx]
            cn = _sorted_merge_xor_single(
                cur, cn,
                pool_data[ps:ps + plen], plen,
                tmp,
            )
            # Swap buffers
            for i in range(cn):
                cur[i] = tmp[i]

        if cn > 0:
            piv = cur[0]
            piv_rank = np.int32(piv)

            # Record ALL pivots (including zero-persistence) for skip-mask cascade
            all_pivots[n_all_piv] = piv_rank
            n_all_piv += 1

            # Store in pool
            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap, np.int64(piv), np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_data[pool_ptr + i] = cur[i]
                pool_ptr += cn
                num_piv += 1

            # Output pair: birth = r_e, death = piv_rank
            # Compare filtration values for positive persistence
            # (global ranks mix dimensions, so rank comparison would include
            # zero-persistence pairs where edge and triangle have same filt)
            if rank_to_filt[piv_rank] > rank_to_filt[r_e]:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = piv_rank
                n_pairs += 1
        else:
            # Column reduced to zero via XOR: σ is a cocycle not in the
            # coboundary image → essential birth (infinite bar).
            ess_births[n_ess] = r_e
            n_ess += 1

    return pair_births[:n_pairs], pair_deaths[:n_pairs], ess_births[:n_ess], all_pivots[:n_all_piv]


@njit(cache=True)
def _general_cohomology_reduce_v2(
    col_ranks,              # (K,) int32
    cofacet_offsets,        # (K+1,) int32
    cofacet_ranks,          # (total_cofacets,) int32
    skip_mask,              # (K,) bool — extended: apparent pairs already True
    rank_to_filt,           # (total,) float32
    cofacet_max_face_rank,  # (total,) int32 — pre-computed on GPU
    # Pre-populated hash map + pool from GPU apparent pairs:
    pre_hm_keys, pre_hm_vals, pre_hm_cap,
    pre_pool_data, pre_piv_start, pre_piv_length,
    pre_pool_ptr, pre_num_piv,
):
    """Cohomology reduction v2: receives pre-populated state from GPU prepass.

    GPU apparent pairs are already in skip_mask (True) and their columns are
    pre-loaded into the hash map + pool. This kernel only processes residual
    (non-apparent, non-skipped) columns.
    """
    K = col_ranks.shape[0]

    # Output buffers
    pair_births = np.empty(K, dtype=np.int32)
    pair_deaths = np.empty(K, dtype=np.int32)
    ess_births = np.empty(K, dtype=np.int32)
    all_pivots = np.empty(K, dtype=np.int32)
    n_pairs = 0
    n_ess = 0
    n_all_piv = 0

    # Use pre-populated state
    hm_keys = pre_hm_keys
    hm_vals = pre_hm_vals
    hm_cap = pre_hm_cap
    pool_data = pre_pool_data
    piv_start = pre_piv_start
    piv_length = pre_piv_length
    pool_ptr = pre_pool_ptr
    num_piv = pre_num_piv
    pool_cap = pool_data.shape[0]

    # Working buffers
    cur = np.empty(MAX_COL, dtype=np.int32)
    tmp = np.empty(MAX_COL, dtype=np.int32)

    # Sort col_ranks indices by rank DESCENDING for processing order
    order = np.argsort(col_ranks)[::-1]

    for ki in range(K):
        idx = order[ki]
        if skip_mask[idx]:
            continue

        r_e = col_ranks[idx]

        # Load cofacets from pre-computed CSR
        c_start = cofacet_offsets[idx]
        c_end = cofacet_offsets[idx + 1]
        cn = c_end - c_start

        if cn == 0:
            ess_births[n_ess] = r_e
            n_ess += 1
            continue

        # Safety-net apparent pair check (catches GPU false negatives)
        youngest_cof = cofacet_ranks[c_start]
        if cofacet_max_face_rank[youngest_cof] == r_e:
            all_pivots[n_all_piv] = youngest_cof
            n_all_piv += 1

            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap,
                         np.int64(youngest_cof), np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_data[pool_ptr + i] = cofacet_ranks[c_start + i]
                pool_ptr += cn
                num_piv += 1

            if rank_to_filt[youngest_cof] > rank_to_filt[r_e]:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = youngest_cof
                n_pairs += 1
            continue

        # Copy cofacet ranks into working buffer
        if cn > MAX_COL:
            cn = MAX_COL
        for i in range(cn):
            cur[i] = cofacet_ranks[c_start + i]

        # Reduction loop
        while cn > 0:
            piv = cur[0]
            piv_idx = _hm_get(hm_keys, hm_vals, hm_cap, np.int64(piv))
            if piv_idx == -1:
                break

            ps = piv_start[piv_idx]
            plen = piv_length[piv_idx]
            cn = _sorted_merge_xor_single(
                cur, cn,
                pool_data[ps:ps + plen], plen,
                tmp,
            )
            for i in range(cn):
                cur[i] = tmp[i]

        if cn > 0:
            piv = cur[0]
            piv_rank = np.int32(piv)

            all_pivots[n_all_piv] = piv_rank
            n_all_piv += 1

            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap, np.int64(piv), np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_data[pool_ptr + i] = cur[i]
                pool_ptr += cn
                num_piv += 1

            if rank_to_filt[piv_rank] > rank_to_filt[r_e]:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = piv_rank
                n_pairs += 1
        else:
            # Column reduced to zero via XOR: essential birth (infinite bar).
            ess_births[n_ess] = r_e
            n_ess += 1

    return pair_births[:n_pairs], pair_deaths[:n_pairs], ess_births[:n_ess], all_pivots[:n_all_piv]


# ---------------------------------------------------------------------------
# Global ranking utilities
# ---------------------------------------------------------------------------

def _simplex_colex_id(vertices: Tensor, dim: int) -> Tensor:
    """Canonical colexicographic ID from sorted vertex tuple.

    For edge (u, v) with u < v:         id = C(v, 2) + u
    For triangle (u, v, w) u<v<w:       id = C(w, 3) + C(v, 2) + u

    Parameters
    ----------
    vertices : (S, d+1) int -- sorted vertex indices
    dim : int -- simplex dimension (1 for edges, 2 for triangles, etc.)

    Returns
    -------
    ids : (S,) int64
    """
    v = vertices.to(torch.int64)
    if dim == 0:
        return v[:, 0]
    elif dim == 1:
        u, w = v[:, 0], v[:, 1]
        return w * (w - 1) // 2 + u
    elif dim == 2:
        a, b, c = v[:, 0], v[:, 1], v[:, 2]
        return c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a
    else:
        # General formula: sum of C(v_k, k+1) for k=0..dim
        result = torch.zeros(v.shape[0], dtype=torch.int64, device=v.device)
        for k in range(dim + 1):
            vk = v[:, k]
            # C(vk, k+1) = product(vk - j for j in 0..k) / (k+1)!
            c = torch.ones(v.shape[0], dtype=torch.int64, device=v.device)
            for j in range(k + 1):
                c = c * (vk - j)
            fac = 1
            for j in range(1, k + 2):
                fac *= j
            result += c // fac
        return result


def compute_global_ranks(
    simplices: dict[int, Tensor],
    filtration: dict[int, Tensor],
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[int, Tensor], Tensor]:
    """Compute global integer ranks for all simplices.

    Sort key: (filtration_value, simplex_dim).  Ties within the same
    (filtration, dim) are broken by original array index — the persistence
    diagram is invariant to this choice, only representative cycles differ.

    Two-pass stable sort: first by dim, then by filt.  The stable sort
    preserves dim ordering within each filt bucket, guaranteeing the
    face-before-coface invariant (lower dim → lower rank at equal filt).

    Parameters
    ----------
    simplices : dict of {dim: (S_d, dim+1) int32} -- vertex indices per simplex
    filtration : dict of {dim: (S_d,) float32} -- filtration values per simplex

    Returns
    -------
    global_rank : dict of {dim: (S_d,) int32} -- rank per simplex
    rank_to_filt : (total,) float32 -- filtration value at each rank
    """
    dims = sorted(simplices.keys())

    # Build flat arrays
    all_filt = torch.cat([filtration[d].cpu().float() for d in dims])
    all_dim = torch.cat([
        torch.full((simplices[d].shape[0],), d, dtype=torch.int32)
        for d in dims
    ])

    total = all_filt.shape[0]

    # Two-pass stable sort: by dim then by filt
    perm = torch.argsort(all_dim.to(torch.int64), stable=True)
    perm = perm[torch.argsort(all_filt[perm], stable=True)]

    global_rank_flat = torch.empty(total, dtype=torch.int32)
    global_rank_flat[perm] = torch.arange(total, dtype=torch.int32)

    rank_to_filt = all_filt[perm]

    # Split back into per-dimension arrays
    global_rank = {}
    offset = 0
    for d in dims:
        n_d = simplices[d].shape[0]
        global_rank[d] = global_rank_flat[offset:offset + n_d].to(device)
        offset += n_d

    return global_rank, rank_to_filt.to(device)


# ---------------------------------------------------------------------------
# GPU prepass: apparent pairs on GPU + Numba residual
# ---------------------------------------------------------------------------

def _general_cohomology_reduce_gpu_prepass(
    col_ranks: Tensor,
    cofacet_offsets: Tensor,
    cofacet_ranks: Tensor,
    skip_mask: Tensor,
    rank_to_filt: Tensor,
    cofacet_max_face_rank: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """GPU apparent pair prepass + CPU Numba residual reduction.

    1. Compute cofacet_max_face_rank via GPU scatter_reduce('amax')
       (skipped if pre-computed cofacet_max_face_rank is provided)
    2. Detect apparent pairs on GPU (vectorized)
    3. Pre-populate Numba hash map + pool with apparent pair columns
    4. Call v2 Numba kernel for residual columns only
    5. Merge GPU apparent pair results with Numba results
    """
    from flash_ph._numba_utils import _prepopulate_apparent_pool

    dev = col_ranks.device
    K = col_ranks.shape[0]
    n_cof = cofacet_ranks.shape[0]

    # Ensure rank_to_filt is on GPU
    rank_to_filt_gpu = rank_to_filt.to(dev)
    total = rank_to_filt_gpu.shape[0]

    if K == 0:
        empty = torch.empty(0, dtype=torch.int32, device=dev)
        return empty, empty.clone(), empty.clone(), empty.clone()

    if n_cof == 0:
        # No cofacets: all non-skipped columns are essential births
        ess = col_ranks[~skip_mask]
        empty = torch.empty(0, dtype=torch.int32, device=dev)
        return empty, empty.clone(), ess, empty.clone()

    # --- Step 1: GPU scatter-max for cofacet_max_face_rank ---
    if cofacet_max_face_rank is not None:
        cof_max_face = cofacet_max_face_rank.to(dev)
    else:
        # For each cofacet entry, find its parent face via CSR offsets
        entry_idx = torch.arange(n_cof, device=dev)
        parent_face = torch.searchsorted(
            cofacet_offsets[1:].long(), entry_idx, right=True,
        )

        face_ranks_expanded = col_ranks[parent_face].clone()
        # Skipped faces contribute -1 (won't be max)
        face_ranks_expanded[skip_mask[parent_face]] = -1

        cof_max_face = torch.full((total,), -1, dtype=torch.int32, device=dev)
        cof_max_face.scatter_reduce_(
            0, cofacet_ranks.long(), face_ranks_expanded, reduce='amax',
        )

    # --- Step 2: GPU vectorized apparent pair detection ---
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

    # --- Step 3: Extract apparent pair results ---
    ap_col_ranks = col_ranks[is_apparent]
    ap_pivots = youngest_cof[is_apparent]
    # Positive persistence filter (same criterion as Numba kernel)
    pos_pers = rank_to_filt_gpu[ap_pivots.long()] > rank_to_filt_gpu[ap_col_ranks.long()]
    ap_births = ap_col_ranks[pos_pers]
    ap_deaths = ap_pivots[pos_pers]

    # --- Step 4: Extend skip mask + transfer to CPU ---
    extended_skip = skip_mask.clone()
    extended_skip[is_apparent] = True

    col_np = col_ranks.cpu().to(torch.int32).numpy()
    off_np = cofacet_offsets.cpu().to(torch.int32).numpy()
    cof_np = cofacet_ranks.cpu().to(torch.int32).numpy()
    skip_np = extended_skip.cpu().numpy().astype(np.bool_)
    r2f_np = rank_to_filt.cpu().float().numpy()
    cmf_np = cof_max_face.cpu().to(torch.int32).numpy()

    # --- Step 5: Allocate + pre-populate Numba pool ---
    ap_indices_np = (
        is_apparent.cpu().nonzero(as_tuple=True)[0].to(torch.int32).numpy()
    )

    # Hash map sizing (same as original kernel)
    hm_cap = max(K * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Pool sizing: must fit all apparent pair cofacets + headroom for residual
    # reductions. K*16 underestimates when columns have ~40+ cofacets (H2).
    total_cofacets = int(off_np[K]) if K > 0 else 0
    pool_cap = min(max(total_cofacets * 2, K * 16, 100000), MAX_POOL_ENTRIES)
    pool_data = np.empty(pool_cap, dtype=np.int32)
    piv_start = np.empty(K, dtype=np.int64)
    piv_length = np.empty(K, dtype=np.int32)

    pool_ptr, num_piv = _prepopulate_apparent_pool(
        ap_indices_np, off_np, cof_np,
        hm_keys, hm_vals, hm_cap,
        pool_data, piv_start, piv_length, 0, 0,
    )

    # --- Step 6: Call v2 Numba kernel ---
    pb, pd, eb, ap = _general_cohomology_reduce_v2(
        col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
        hm_keys, hm_vals, hm_cap,
        pool_data, piv_start, piv_length, pool_ptr, num_piv,
    )

    # --- Step 7: Merge GPU apparent pair results with Numba results ---
    all_pb = torch.cat([ap_births.cpu().to(torch.int32),
                        torch.from_numpy(pb.copy())])
    all_pd = torch.cat([ap_deaths.cpu().to(torch.int32),
                        torch.from_numpy(pd.copy())])
    eb_t = torch.from_numpy(eb.copy())
    all_ap = torch.cat([ap_pivots.cpu().to(torch.int32),
                        torch.from_numpy(ap.copy())])

    return (all_pb.to(dev), all_pd.to(dev), eb_t.to(dev), all_ap.to(dev))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def general_cohomology_reduce(
    col_ranks: Tensor,
    cofacet_offsets: Tensor,
    cofacet_ranks: Tensor,
    skip_mask: Tensor,
    rank_to_filt: Tensor,
    cofacet_max_face_rank: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """General cohomology reduction for any filtered complex.

    All inputs are integer ranks from global ranking.
    rank_to_filt maps global rank -> filtration value (for positive-persistence check).
    Returns (pair_birth_ranks, pair_death_ranks, essential_birth_ranks, all_pivot_ranks).
    all_pivot_ranks includes ALL pivots (even zero-persistence) for skip-mask cascade.

    When inputs are on CUDA, uses GPU prepass (GPU scatter-max + apparent pair
    detection + pre-populated Numba v2 for residual). When cofacet_max_face_rank
    is pre-computed, uses v2 Numba kernel (skips O(n_cof) scatter-max loop).
    """
    device = col_ranks.device

    # GPU path: prepass (GPU apparent pairs + CPU Numba residual)
    if device.type == 'cuda':
        return _general_cohomology_reduce_gpu_prepass(
            col_ranks, cofacet_offsets, cofacet_ranks,
            skip_mask, rank_to_filt, cofacet_max_face_rank,
        )

    # CPU path with pre-computed cmf: use v2 kernel
    if cofacet_max_face_rank is not None:
        col_np = col_ranks.cpu().to(torch.int32).numpy()
        off_np = cofacet_offsets.cpu().to(torch.int32).numpy()
        cof_np = cofacet_ranks.cpu().to(torch.int32).numpy()
        skip_np = skip_mask.cpu().numpy().astype(np.bool_)
        r2f_np = rank_to_filt.cpu().float().numpy()
        cmf_np = cofacet_max_face_rank.cpu().to(torch.int32).numpy()
        K = col_np.shape[0]

        # Allocate empty hash map + pool (no pre-populated apparent pairs)
        hm_cap = max(K * 4, 64)
        p = 1
        while p < hm_cap:
            p *= 2
        hm_cap = p
        hm_keys = np.full(hm_cap, np.int64(-1))
        hm_vals = np.full(hm_cap, np.int64(-1))
        total_cofacets = int(off_np[K]) if K > 0 else 0
        pool_cap = min(max(total_cofacets * 2, K * 16, 100000), MAX_POOL_ENTRIES)
        pool_data = np.empty(pool_cap, dtype=np.int32)
        piv_start = np.empty(K, dtype=np.int64)
        piv_length = np.empty(K, dtype=np.int32)

        pb, pd, eb, ap = _general_cohomology_reduce_v2(
            col_np, off_np, cof_np, skip_np, r2f_np, cmf_np,
            hm_keys, hm_vals, hm_cap,
            pool_data, piv_start, piv_length, 0, 0,
        )

        return (torch.from_numpy(pb.copy()).to(device),
                torch.from_numpy(pd.copy()).to(device),
                torch.from_numpy(eb.copy()).to(device),
                torch.from_numpy(ap.copy()).to(device))

    # CPU path without cmf: original kernel
    col_np = col_ranks.cpu().to(torch.int32).numpy()
    off_np = cofacet_offsets.cpu().to(torch.int32).numpy()
    cof_np = cofacet_ranks.cpu().to(torch.int32).numpy()
    skip_np = skip_mask.cpu().numpy().astype(np.bool_)
    r2f_np = rank_to_filt.cpu().float().numpy()

    pb, pd, eb, ap = _general_cohomology_reduce(
        col_np, off_np, cof_np, skip_np, r2f_np,
    )

    return (torch.from_numpy(pb.copy()).to(device),
            torch.from_numpy(pd.copy()).to(device),
            torch.from_numpy(eb.copy()).to(device),
            torch.from_numpy(ap.copy()).to(device))
