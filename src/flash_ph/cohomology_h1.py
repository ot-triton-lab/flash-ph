"""Cohomology-based H1 reduction with apparent/emergent pair optimizations.

Implements Ripser-style persistent cohomology for H1 over Z/2Z:
- Columns = non-MST edges (processed in reverse filtration order)
- Rows = triangles (implicit, never materialized)
- Cofacets computed on-the-fly via CSR adjacency intersection
- Apparent pairs: bidirectional facet/cofacet check (skips ~99% of columns)
- Emergent pairs: detected during coboundary init (skips most remaining)
- Residual: sorted-array symmetric difference (Z/2Z XOR)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from numba import njit

from flash_ph._numba_utils import _hm_get, _hm_set, MAX_COL, MAX_POOL_ENTRIES


@njit(cache=True)
def _tri_colex_numba(a, b, c):
    """Colexicographic index for triangle (a, b, c) with a < b < c."""
    a64, b64, c64 = np.int64(a), np.int64(b), np.int64(c)
    return c64 * (c64 - 1) * (c64 - 2) // 6 + b64 * (b64 - 1) // 2 + a64


@njit(cache=True)
def _sorted_merge_xor(
    ra, ca, na,    # rank_a[], colex_a[], len_a
    rb, cb, nb,    # rank_b[], colex_b[], len_b
    ro, co,        # output buffers
):
    """Symmetric difference of two sorted (rank, colex) arrays over Z/2Z.

    Sorted by (rank ASC, colex ASC). Elements cancel iff both rank and colex
    match (each triangle has a unique colex, so this is safe).
    Returns number of elements in output. Truncates at MAX_COL for safety.
    """
    ia = 0
    ib = 0
    no = 0
    limit = MAX_COL
    while ia < na and ib < nb:
        if no >= limit:
            break
        # Compare by (rank, colex)
        if ra[ia] < rb[ib] or (ra[ia] == rb[ib] and ca[ia] < cb[ib]):
            ro[no] = ra[ia]; co[no] = ca[ia]; no += 1; ia += 1
        elif ra[ia] > rb[ib] or (ra[ia] == rb[ib] and ca[ia] > cb[ib]):
            ro[no] = rb[ib]; co[no] = cb[ib]; no += 1; ib += 1
        else:
            # Equal (same triangle) -> cancel (Z/2Z XOR)
            ia += 1; ib += 1
    while ia < na and no < limit:
        ro[no] = ra[ia]; co[no] = ca[ia]; no += 1; ia += 1
    while ib < nb and no < limit:
        ro[no] = rb[ib]; co[no] = cb[ib]; no += 1; ib += 1
    return no


@njit(cache=True)
def _cohomology_reduce_h1_loop(
    edge_i,        # (E,) int32
    edge_j,        # (E,) int32
    nonmst_ranks,  # (K,) int32 -- ranks of non-MST edges, sorted ascending
    adj_ptr,       # (n+1,) int32
    adj_idx,       # (2E,) int32
    adj_rank,      # (2E,) int32
    max_degree,    # int -- max vertex degree (for pool sizing)
):
    """Cohomology-based H1 reduction loop (Z/2Z, Numba JIT).

    Processes non-MST edges in REVERSE filtration order (largest rank first).
    For each edge, computes triangle cofacets on-the-fly from CSR adjacency.
    Detects apparent pairs (zero-persistence) and reduces residual columns
    via sorted-array symmetric difference.

    Returns
    -------
    pair_births : (P,) int32 -- edge ranks of finite H1 pair births
    pair_deaths : (P,) int32 -- max-edge-ranks of finite H1 pair deaths
    ess_births  : (M,) int32 -- edge ranks of essential (infinite) H1 features
    """
    K = nonmst_ranks.shape[0]

    # Output buffers
    pair_births = np.empty(K, dtype=np.int32)
    pair_deaths = np.empty(K, dtype=np.int32)
    ess_births = np.empty(K, dtype=np.int32)
    all_pivot_colex = np.empty(K, dtype=np.int64)
    n_pairs = 0
    n_ess = 0
    n_all_piv = 0

    # Pivot hash map: tri_colex -> pivot storage index
    hm_cap = max(K * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Pool allocator for stored columns: parallel arrays of (rank, colex)
    # Each column can have up to max_degree cofacets; growth during XOR is bounded
    pool_cap = max(K * max(max_degree, 16), 100000)
    pool_rank = np.empty(pool_cap, dtype=np.int32)
    pool_colex = np.empty(pool_cap, dtype=np.int64)
    piv_start = np.empty(K, dtype=np.int64)
    piv_length = np.empty(K, dtype=np.int32)
    pool_ptr = 0
    num_piv = 0

    # Working buffers
    cur_rank = np.empty(MAX_COL, dtype=np.int32)
    cur_colex = np.empty(MAX_COL, dtype=np.int64)
    tmp_rank = np.empty(MAX_COL, dtype=np.int32)
    tmp_colex = np.empty(MAX_COL, dtype=np.int64)

    # Process edges in REVERSE filtration order (largest rank first)
    for ki in range(K - 1, -1, -1):
        r_e = nonmst_ranks[ki]
        u = edge_i[r_e]
        v = edge_j[r_e]
        if u > v:
            u, v = v, u

        # --- Enumerate cofacets via inline CSR intersection ---
        u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
        v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
        iu = u_start
        iv = v_start
        cn = 0

        while iu < u_end and iv < v_end:
            nu = adj_idx[iu]
            nv = adj_idx[iv]
            if nu == nv:
                w = nu
                # Edge ranks from CSR (no binary search needed)
                r_uw = adj_rank[iu]
                r_vw = adj_rank[iv]

                max_r = r_e
                if r_uw > max_r:
                    max_r = r_uw
                if r_vw > max_r:
                    max_r = r_vw

                # Canonical vertex ordering for colex
                a, b, c = u, v, w
                if a > b:
                    a, b = b, a
                if b > c:
                    b, c = c, b
                if a > b:
                    a, b = b, a

                if cn < MAX_COL:
                    cur_rank[cn] = max_r
                    cur_colex[cn] = _tri_colex_numba(a, b, c)
                    cn += 1

                iu += 1
                iv += 1
            elif nu < nv:
                iu += 1
            else:
                iv += 1

        if cn == 0:
            # No cofacets -> essential cycle
            ess_births[n_ess] = r_e
            n_ess += 1
            continue

        # Sort by (rank ASC, colex ASC) -- insertion sort (cn typically < 20)
        for i in range(1, cn):
            kr = cur_rank[i]
            kc = cur_colex[i]
            j = i - 1
            while j >= 0 and (cur_rank[j] > kr or (cur_rank[j] == kr and cur_colex[j] > kc)):
                cur_rank[j + 1] = cur_rank[j]
                cur_colex[j + 1] = cur_colex[j]
                j -= 1
            cur_rank[j + 1] = kr
            cur_colex[j + 1] = kc

        # --- Apparent pair check (zero-persistence) ---
        # In cohomology, the pivot is the SMALLEST element (lowest filtration
        # cofacet), not the largest.  This is the dual of homology reduction.
        # If the youngest cofacet has max_rank == r_e, this edge forms a
        # zero-persistence apparent pair with that cofacet as pivot.
        if cur_rank[0] == r_e:
            # Zero-persistence apparent pair: store pivot but don't output
            piv_colex = cur_colex[0]  # pivot = smallest element
            all_pivot_colex[n_all_piv] = piv_colex
            n_all_piv += 1
            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap, piv_colex, np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_rank[pool_ptr + i] = cur_rank[i]
                    pool_colex[pool_ptr + i] = cur_colex[i]
                pool_ptr += cn
                num_piv += 1
            continue

        # --- Reduction loop ---
        while cn > 0:
            # Pivot = FIRST element (smallest rank, then smallest colex)
            piv_colex = cur_colex[0]
            piv_idx = _hm_get(hm_keys, hm_vals, hm_cap, piv_colex)
            if piv_idx == -1:
                break  # Pivot is free

            # XOR with stored column
            ps = piv_start[piv_idx]
            plen = piv_length[piv_idx]
            cn = _sorted_merge_xor(
                cur_rank, cur_colex, cn,
                pool_rank[ps:ps + plen], pool_colex[ps:ps + plen], plen,
                tmp_rank, tmp_colex,
            )
            # Swap buffers
            for i in range(cn):
                cur_rank[i] = tmp_rank[i]
                cur_colex[i] = tmp_colex[i]

        if cn > 0:
            # New pivot found -> persistence pair
            piv_colex = cur_colex[0]  # pivot = smallest element
            piv_rank = cur_rank[0]

            # Track ALL pivots (including zero-persistence) for H2 clearing
            all_pivot_colex[n_all_piv] = piv_colex
            n_all_piv += 1

            # Store column in pool (skip if pool full — pair still recorded)
            if pool_ptr + cn <= pool_cap and num_piv < K:
                _hm_set(hm_keys, hm_vals, hm_cap, piv_colex, np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_rank[pool_ptr + i] = cur_rank[i]
                    pool_colex[pool_ptr + i] = cur_colex[i]
                pool_ptr += cn
                num_piv += 1

            # Output pair: birth = r_e (edge creates cycle),
            # death = piv_rank (cofacet triangle kills it).
            # Positive persistence when piv_rank > r_e (triangle diam > edge diam).
            if piv_rank > r_e:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = piv_rank
                n_pairs += 1
        else:
            # Column zeroed via XOR -> unpaired cocycle (essential H1)
            ess_births[n_ess] = r_e
            n_ess += 1

    return (pair_births[:n_pairs], pair_deaths[:n_pairs], ess_births[:n_ess],
            all_pivot_colex[:n_all_piv])


def cohomology_reduction_h1(
    edge_i: Tensor,        # (E,) int32
    edge_j: Tensor,        # (E,) int32
    edge_dist_sq: Tensor,  # (E,) float32
    mst_mask: Tensor,      # (E,) bool
    adj_ptr: Tensor,       # (n+1,) int32
    adj_idx: Tensor,       # (2E,) int32
    adj_rank: Tensor,      # (2E,) int32
    n: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Cohomology-based H1 reduction.

    Replaces find_apparent_pairs() + column_reduction_h1() with a single
    cohomology pass that subsumes apparent pair detection, triangle
    enumeration, and column reduction.

    Returns
    -------
    h1_pairs : (K, 2) float32 -- finite H1 bars (birth, death)
    essential_births : (M,) float32 -- birth times of essential H1 features
    all_pivot_colex : (P,) int64 -- colex IDs of ALL pivot triangles
        (including zero-persistence) for H2 clearing cascade
    """
    device = edge_i.device
    E = edge_i.shape[0]

    # Non-MST edge ranks (sorted ascending)
    nonmst_ranks = (~mst_mask).nonzero(as_tuple=True)[0].to(torch.int32)

    if nonmst_ranks.numel() == 0:
        return (
            torch.empty(0, 2, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int64),
        )

    # Transfer to CPU for Numba
    ei_cpu = edge_i.cpu().numpy()
    ej_cpu = edge_j.cpu().numpy()
    nonmst_cpu = nonmst_ranks.cpu().numpy()
    ptr_cpu = adj_ptr.cpu().numpy()
    idx_cpu = adj_idx.cpu().numpy()
    rank_cpu = adj_rank.cpu().numpy()

    # Compute max vertex degree for pool sizing
    degrees = ptr_cpu[1:] - ptr_cpu[:-1]
    max_deg = int(degrees.max()) if len(degrees) > 0 else 1

    # Run cohomology reduction
    pair_births_r, pair_deaths_r, ess_births_r, all_piv_colex_r = (
        _cohomology_reduce_h1_loop(
            ei_cpu, ej_cpu, nonmst_cpu, ptr_cpu, idx_cpu, rank_cpu, max_deg,
        )
    )

    # Convert edge ranks to distances
    dist_sq_cpu = edge_dist_sq.cpu().numpy()

    if pair_births_r.shape[0] > 0:
        birth_dist = np.sqrt(dist_sq_cpu[pair_births_r.astype(np.int64)])
        death_dist = np.sqrt(dist_sq_cpu[pair_deaths_r.astype(np.int64)])
        h1_pairs = torch.tensor(
            np.stack([birth_dist, death_dist], axis=1),
            dtype=torch.float32, device=device,
        )
    else:
        h1_pairs = torch.empty(0, 2, dtype=torch.float32, device=device)

    if ess_births_r.shape[0] > 0:
        ess_dist = np.sqrt(dist_sq_cpu[ess_births_r.astype(np.int64)])
        essential_births = torch.tensor(ess_dist, dtype=torch.float32, device=device)
    else:
        essential_births = torch.empty(0, dtype=torch.float32, device=device)

    all_pivot_colex = torch.from_numpy(all_piv_colex_r.copy()).to(torch.int64)

    return h1_pairs, essential_births, all_pivot_colex


@njit(cache=True)
def _enumerate_cofacets_sorted(
    edge_rank, edge_i, edge_j, adj_ptr, adj_idx, adj_rank,
    out_rank, out_colex,
):
    """Enumerate all triangle cofacets of an edge, sorted by (rank, colex).

    Used for lazy column materialization in hybrid GPU+CPU reduction.

    Parameters
    ----------
    edge_rank : int -- rank of the edge
    edge_i, edge_j : (E,) int32 -- edge endpoint arrays
    adj_ptr, adj_idx, adj_rank : CSR adjacency

    Returns
    -------
    cn : int -- number of cofacets written to out_rank/out_colex
    """
    u = edge_i[edge_rank]
    v = edge_j[edge_rank]
    if u > v:
        u, v = v, u

    u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
    v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
    iu = u_start
    iv = v_start
    cn = 0

    while iu < u_end and iv < v_end:
        nu = adj_idx[iu]
        nv = adj_idx[iv]
        if nu == nv:
            w = nu
            r_uw = adj_rank[iu]
            r_vw = adj_rank[iv]

            max_r = edge_rank
            if r_uw > max_r:
                max_r = r_uw
            if r_vw > max_r:
                max_r = r_vw

            a, b, c = u, v, w
            if a > b:
                a, b = b, a
            if b > c:
                b, c = c, b
            if a > b:
                a, b = b, a

            out_rank[cn] = max_r
            out_colex[cn] = _tri_colex_numba(a, b, c)
            cn += 1
            iu += 1
            iv += 1
        elif nu < nv:
            iu += 1
        else:
            iv += 1

    # Insertion sort by (rank ASC, colex ASC)
    for i in range(1, cn):
        kr = out_rank[i]
        kc = out_colex[i]
        j = i - 1
        while j >= 0 and (out_rank[j] > kr or (out_rank[j] == kr and out_colex[j] > kc)):
            out_rank[j + 1] = out_rank[j]
            out_colex[j + 1] = out_colex[j]
            j -= 1
        out_rank[j + 1] = kr
        out_colex[j + 1] = kc

    return cn


@njit(cache=True)
def _bulk_hm_insert_lazy(
    hm_keys, hm_vals, hm_cap,
    apparent_ranks, apparent_pivots,
):
    """Bulk-insert apparent pair pivots as lazy entries into hash map.

    Encodes each as val = -(edge_rank + 2) (tagged union: <= -2 means lazy).
    Much faster than calling _hm_set from Python in a loop.
    """
    for i in range(apparent_ranks.shape[0]):
        er = np.int64(apparent_ranks[i])
        pc = np.int64(apparent_pivots[i])
        lazy_val = np.int64(-(er + 2))
        _hm_set(hm_keys, hm_vals, hm_cap, pc, lazy_val)


@njit(cache=True)
def _cohomology_reduce_h1_hybrid_loop(
    edge_i,                # (E,) int32
    edge_j,                # (E,) int32
    non_apparent_ranks,    # (K',) int32 -- ranks of NON-apparent non-MST edges
    adj_ptr,               # (n+1,) int32
    adj_idx,               # (2E,) int32
    adj_rank,              # (2E,) int32
    max_degree,            # int
    # Pre-populated from GPU apparent pair scan:
    hm_keys,               # (hm_cap,) int64 -- hash map keys (pivot colex)
    hm_vals,               # (hm_cap,) int64 -- hash map vals (tagged union)
    hm_cap,                # int
    total_non_mst,         # int -- total non-MST edges (for output sizing)
):
    """Hybrid H1 reduction: GPU apparent pairs + CPU residual.

    The hash map is pre-populated with apparent pair pivots using tagged
    union encoding:
      val >= 0        -> pooled column index (data in pool)
      val == -1       -> empty/missing
      val <= -2       -> lazy apparent: edge_rank = -val - 2
                        Column NOT stored; enumerate from CSR on first use.

    Only processes non-apparent edges. When XOR hits a lazy pivot,
    materializes the column from CSR and memoizes (updates to pooled).

    Returns
    -------
    pair_births, pair_deaths, ess_births, all_pivot_colex
    """
    K = non_apparent_ranks.shape[0]

    # Output buffers (sized for worst case: all non-MST edges)
    pair_births = np.empty(total_non_mst, dtype=np.int32)
    pair_deaths = np.empty(total_non_mst, dtype=np.int32)
    ess_births = np.empty(total_non_mst, dtype=np.int32)
    all_pivot_colex = np.empty(total_non_mst, dtype=np.int64)
    n_pairs = 0
    n_ess = 0
    n_all_piv = 0

    # Pool: must accommodate residual pivots + materialized lazy columns
    # Cap to prevent multi-GB allocations on large Rips complexes
    pool_cap = min(max(total_non_mst * max(max_degree, 16), 100000), MAX_POOL_ENTRIES)
    pool_rank = np.empty(pool_cap, dtype=np.int32)
    pool_colex = np.empty(pool_cap, dtype=np.int64)
    piv_start = np.empty(total_non_mst, dtype=np.int64)
    piv_length = np.empty(total_non_mst, dtype=np.int32)
    pool_ptr = 0
    num_piv = 0

    # Working buffers (ping-pong: swap instead of copy)
    buf_a_rank = np.empty(MAX_COL, dtype=np.int32)
    buf_a_colex = np.empty(MAX_COL, dtype=np.int64)
    buf_b_rank = np.empty(MAX_COL, dtype=np.int32)
    buf_b_colex = np.empty(MAX_COL, dtype=np.int64)
    lazy_rank = np.empty(MAX_COL, dtype=np.int32)
    lazy_colex = np.empty(MAX_COL, dtype=np.int64)
    use_a = True  # True: cur=buf_a, tmp=buf_b; False: swapped

    # Process non-apparent edges in REVERSE rank order
    for ki in range(K - 1, -1, -1):
        r_e = non_apparent_ranks[ki]
        u = edge_i[r_e]
        v = edge_j[r_e]
        if u > v:
            u, v = v, u

        # Select current buffer
        if use_a:
            cur_rank = buf_a_rank
            cur_colex = buf_a_colex
        else:
            cur_rank = buf_b_rank
            cur_colex = buf_b_colex

        # Enumerate cofacets via CSR intersection
        u_start, u_end = adj_ptr[u], adj_ptr[u + 1]
        v_start, v_end = adj_ptr[v], adj_ptr[v + 1]
        iu = u_start
        iv = v_start
        cn = 0

        while iu < u_end and iv < v_end:
            nu = adj_idx[iu]
            nv = adj_idx[iv]
            if nu == nv:
                w = nu
                r_uw = adj_rank[iu]
                r_vw = adj_rank[iv]
                max_r = r_e
                if r_uw > max_r:
                    max_r = r_uw
                if r_vw > max_r:
                    max_r = r_vw
                a, b, c = u, v, w
                if a > b:
                    a, b = b, a
                if b > c:
                    b, c = c, b
                if a > b:
                    a, b = b, a
                if cn < MAX_COL:
                    cur_rank[cn] = max_r
                    cur_colex[cn] = _tri_colex_numba(a, b, c)
                    cn += 1
                iu += 1
                iv += 1
            elif nu < nv:
                iu += 1
            else:
                iv += 1

        if cn == 0:
            ess_births[n_ess] = r_e
            n_ess += 1
            continue

        # Sort by (rank ASC, colex ASC)
        for i in range(1, cn):
            kr = cur_rank[i]
            kc = cur_colex[i]
            j = i - 1
            while j >= 0 and (cur_rank[j] > kr or (cur_rank[j] == kr and cur_colex[j] > kc)):
                cur_rank[j + 1] = cur_rank[j]
                cur_colex[j + 1] = cur_colex[j]
                j -= 1
            cur_rank[j + 1] = kr
            cur_colex[j + 1] = kc

        # Non-apparent edges should NOT have youngest cofacet with max_rank == r_e
        # (GPU already classified those as apparent). But check for false negatives:
        if cur_rank[0] == r_e:
            piv_colex = cur_colex[0]
            all_pivot_colex[n_all_piv] = piv_colex
            n_all_piv += 1
            if pool_ptr + cn <= pool_cap and num_piv < total_non_mst:
                _hm_set(hm_keys, hm_vals, hm_cap, piv_colex, np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_rank[pool_ptr + i] = cur_rank[i]
                    pool_colex[pool_ptr + i] = cur_colex[i]
                pool_ptr += cn
                num_piv += 1
            continue

        # Reduction loop with buffer ping-pong
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

            # Handle lazy apparent pivot: materialize column from CSR
            if piv_idx <= -2:
                lazy_edge_rank = np.int32(-piv_idx - 2)
                lcn = _enumerate_cofacets_sorted(
                    lazy_edge_rank, edge_i, edge_j,
                    adj_ptr, adj_idx, adj_rank,
                    lazy_rank, lazy_colex,
                )
                # Defensive check: materialized pivot must match expected
                if lcn == 0 or lazy_colex[0] != piv_colex:
                    break  # Invariant violated — treat pivot as free

                # Memoize: store in pool and update hash map
                if pool_ptr + lcn <= pool_cap and num_piv < total_non_mst:
                    _hm_set(hm_keys, hm_vals, hm_cap,
                            piv_colex, np.int64(num_piv))
                    piv_start[num_piv] = pool_ptr
                    piv_length[num_piv] = lcn
                    for i in range(lcn):
                        pool_rank[pool_ptr + i] = lazy_rank[i]
                        pool_colex[pool_ptr + i] = lazy_colex[i]
                    pool_ptr += lcn
                    piv_idx = np.int64(num_piv)
                    num_piv += 1
                else:
                    # Pool full: XOR directly without memoizing
                    cn = _sorted_merge_xor(
                        cur_rank, cur_colex, cn,
                        lazy_rank, lazy_colex, lcn,
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
            # Swap buffers
            use_a = not use_a
            cur_rank = tmp_rank
            cur_colex = tmp_colex

        if cn > 0:
            piv_colex = cur_colex[0]
            piv_r = cur_rank[0]

            all_pivot_colex[n_all_piv] = piv_colex
            n_all_piv += 1

            if pool_ptr + cn <= pool_cap and num_piv < total_non_mst:
                _hm_set(hm_keys, hm_vals, hm_cap, piv_colex, np.int64(num_piv))
                piv_start[num_piv] = pool_ptr
                piv_length[num_piv] = cn
                for i in range(cn):
                    pool_rank[pool_ptr + i] = cur_rank[i]
                    pool_colex[pool_ptr + i] = cur_colex[i]
                pool_ptr += cn
                num_piv += 1

            if piv_r > r_e:
                pair_births[n_pairs] = r_e
                pair_deaths[n_pairs] = piv_r
                n_pairs += 1
        else:
            # Column zeroed via XOR -> unpaired cocycle (essential H1)
            ess_births[n_ess] = r_e
            n_ess += 1

    return (pair_births[:n_pairs], pair_deaths[:n_pairs], ess_births[:n_ess],
            all_pivot_colex[:n_all_piv])


def cohomology_reduction_h1_gpu(
    edge_i: Tensor,
    edge_j: Tensor,
    edge_dist_sq: Tensor,
    mst_mask: Tensor,
    adj_ptr: Tensor,
    adj_idx: Tensor,
    adj_rank: Tensor,
    n: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Hybrid GPU+CPU H1 cohomology reduction.

    Phase 1: GPU Triton kernel classifies edges as apparent/non-apparent.
    Phase 2: CPU Numba reducer handles non-apparent columns with lazy
             column enumeration for apparent pivots.

    Same interface as cohomology_reduction_h1().
    """
    from flash_ph.kernels.apparent_pair_kernel import apparent_pair_scan

    device = edge_i.device
    E = edge_i.shape[0]

    nonmst_ranks = (~mst_mask).nonzero(as_tuple=True)[0].to(torch.int32)
    K = nonmst_ranks.shape[0]

    if K == 0:
        return (
            torch.empty(0, 2, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int64),
        )

    # Compute max degree
    degrees_t = adj_ptr[1:] - adj_ptr[:-1]
    max_deg = int(degrees_t.max().item()) if degrees_t.numel() > 0 else 1

    # --- Phase 1: GPU apparent pair classification ---
    is_apparent, pivot_colex = apparent_pair_scan(
        edge_i, edge_j, nonmst_ranks,
        adj_ptr, adj_idx, adj_rank, max_deg,
    )

    # Transfer to CPU
    is_app_cpu = is_apparent.cpu().numpy()
    piv_colex_cpu = pivot_colex.cpu().numpy()
    nonmst_cpu = nonmst_ranks.cpu().numpy()
    ei_cpu = edge_i.cpu().numpy()
    ej_cpu = edge_j.cpu().numpy()
    ptr_cpu = adj_ptr.cpu().numpy()
    idx_cpu = adj_idx.cpu().numpy()
    rank_cpu = adj_rank.cpu().numpy()

    # Separate apparent and non-apparent edge ranks
    apparent_mask = is_app_cpu.astype(np.bool_)
    apparent_ranks = nonmst_cpu[apparent_mask]
    apparent_pivots = piv_colex_cpu[apparent_mask]
    non_apparent_ranks = nonmst_cpu[~apparent_mask]

    # --- Phase 2: Pre-populate hash map with lazy apparent pivots ---
    hm_cap = max(K * 4, 64)
    p = 1
    while p < hm_cap:
        p *= 2
    hm_cap = p
    hm_keys = np.full(hm_cap, np.int64(-1))
    hm_vals = np.full(hm_cap, np.int64(-1))

    # Bulk-insert apparent pivots as lazy entries (single Numba call)
    _bulk_hm_insert_lazy(
        hm_keys, hm_vals, hm_cap,
        apparent_ranks.astype(np.int64),
        apparent_pivots.astype(np.int64),
    )

    # --- Phase 3: CPU residual reduction on non-apparent edges ---
    pair_births_r, pair_deaths_r, ess_births_r, residual_pivots = (
        _cohomology_reduce_h1_hybrid_loop(
            ei_cpu, ej_cpu,
            non_apparent_ranks.astype(np.int32),
            ptr_cpu, idx_cpu, rank_cpu, max_deg,
            hm_keys, hm_vals, hm_cap,
            K,
        )
    )

    # --- Merge results ---
    # All pivot colex = apparent pivots + residual pivots
    all_piv_colex = np.concatenate([apparent_pivots, residual_pivots])

    # Convert edge ranks to distances
    dist_sq_cpu = edge_dist_sq.cpu().numpy()

    if pair_births_r.shape[0] > 0:
        birth_dist = np.sqrt(dist_sq_cpu[pair_births_r.astype(np.int64)])
        death_dist = np.sqrt(dist_sq_cpu[pair_deaths_r.astype(np.int64)])
        h1_pairs = torch.tensor(
            np.stack([birth_dist, death_dist], axis=1),
            dtype=torch.float32, device=device,
        )
    else:
        h1_pairs = torch.empty(0, 2, dtype=torch.float32, device=device)

    if ess_births_r.shape[0] > 0:
        ess_dist = np.sqrt(dist_sq_cpu[ess_births_r.astype(np.int64)])
        essential_births = torch.tensor(ess_dist, dtype=torch.float32, device=device)
    else:
        essential_births = torch.empty(0, dtype=torch.float32, device=device)

    all_pivot_colex = torch.from_numpy(all_piv_colex.copy()).to(torch.int64)

    return h1_pairs, essential_births, all_pivot_colex
