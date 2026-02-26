"""Shared Numba-JIT utilities for cohomology reduction.

Used by both cohomology_h2.py (global-rank reducer) and
cohomology_h1.py (Rips H1 reducer).
"""
from __future__ import annotations

import numpy as np
from numba import njit

MAX_COL = 262144  # Maximum column length for working buffers (256K)
MAX_POOL_ENTRIES = 100_000_000  # Cap pool at 100M entries (~1.2GB)


@njit(cache=True)
def _hm_get(keys, vals, cap, key):
    """Open-addressing hash map lookup. Returns -1 if not found."""
    h = np.int64(key * np.int64(2654435761)) % np.int64(cap)
    if h < 0:
        h += cap
    for _ in range(cap):
        if keys[h] == -1:
            return np.int64(-1)
        if keys[h] == key:
            return vals[h]
        h = (h + 1) % cap
    return np.int64(-1)


@njit(cache=True)
def _hm_set(keys, vals, cap, key, val):
    """Open-addressing hash map insert/update."""
    h = np.int64(key * np.int64(2654435761)) % np.int64(cap)
    if h < 0:
        h += cap
    for _ in range(cap):
        if keys[h] == -1 or keys[h] == key:
            keys[h] = key
            vals[h] = val
            return
        h = (h + 1) % cap


@njit(cache=True)
def _sorted_merge_xor_single(a, na, b, nb, out):
    """Symmetric difference of two sorted int32 arrays over Z/2Z.

    Elements cancel iff they match (each rank is unique).
    Returns number of elements in output. Truncates at MAX_COL for safety.
    """
    ia = ib = no = 0
    limit = MAX_COL
    while ia < na and ib < nb:
        if no >= limit:
            break
        if a[ia] < b[ib]:
            out[no] = a[ia]; no += 1; ia += 1
        elif a[ia] > b[ib]:
            out[no] = b[ib]; no += 1; ib += 1
        else:
            ia += 1; ib += 1  # cancel (Z/2Z XOR)
    while ia < na and no < limit:
        out[no] = a[ia]; no += 1; ia += 1
    while ib < nb and no < limit:
        out[no] = b[ib]; no += 1; ib += 1
    return no


@njit(cache=True)
def _prepopulate_apparent_pool(
    ap_indices,        # (A,) int32 — column indices of apparent pairs
    cofacet_offsets,   # (K+1,) int32
    cofacet_ranks,     # (total_cofacets,) int32
    hm_keys, hm_vals, hm_cap,
    pool_data, piv_start, piv_length,
    pool_ptr_init, num_piv_init,
):
    """Pre-populate hash map + pool with apparent pair columns.

    For each apparent pair column, reads cofacets from CSR and stores them
    in the pool. Inserts pivot (youngest cofacet rank) into hash map.
    """
    pool_ptr = pool_ptr_init
    num_piv = num_piv_init
    pool_cap = pool_data.shape[0]
    K = piv_start.shape[0]

    for ai in range(ap_indices.shape[0]):
        idx = ap_indices[ai]
        c_start = cofacet_offsets[idx]
        c_end = cofacet_offsets[idx + 1]
        cn = c_end - c_start
        if cn == 0:
            continue
        youngest_cof = cofacet_ranks[c_start]
        if pool_ptr + cn <= pool_cap and num_piv < K:
            _hm_set(hm_keys, hm_vals, hm_cap,
                     np.int64(youngest_cof), np.int64(num_piv))
            piv_start[num_piv] = pool_ptr
            piv_length[num_piv] = cn
            for i in range(cn):
                pool_data[pool_ptr + i] = cofacet_ranks[c_start + i]
            pool_ptr += cn
            num_piv += 1
    return pool_ptr, num_piv
