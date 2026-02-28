"""Shared Numba-JIT utilities for cohomology reduction and simplex enumeration.

Used by cohomology_general.py (global-rank reducer), triangle_kernel.py
(CPU fallback), and tests.
"""
from __future__ import annotations
import numpy as np
from numba import njit

MAX_COL = 1048576  # Maximum column length for working buffers (1M)
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


# ---------------------------------------------------------------------------
# Rips simplex enumeration (Numba JIT) — used by triangle_kernel CPU fallback
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
