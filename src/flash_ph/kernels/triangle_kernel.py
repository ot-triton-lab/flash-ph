"""Triton GPU triangle and tetrahedra enumeration for Rips complexes.

Two-pass approach per kernel: count simplices, then emit.
Triangle kernel: one Triton program per BLOCK_E edges, CSR merge-intersection
of endpoint neighbor lists to find common neighbors w > max(u,v).
Tetrahedra kernel: one program per triangle, triple CSR merge-intersection.

CPU fallback via existing Numba implementations.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triangle enumeration (Triton)
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _triangle_count_kernel(
        edge_i_ptr, edge_j_ptr,
        adj_ptr_ptr, adj_idx_ptr,
        counts_ptr,
        E,  # runtime scalar (not constexpr to avoid recompilation)
        BLOCK_E: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Count triangles per edge via vectorized CSR merge-intersection.

        Each program handles BLOCK_E edges. Uses mask-based pattern
        (no Python if/break on tensor values).
        """
        pid = tl.program_id(0)
        e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        active = e_offs < E

        u = tl.load(edge_i_ptr + e_offs, mask=active, other=0)
        v = tl.load(edge_j_ptr + e_offs, mask=active, other=0)
        # Ensure u < v
        u_lo = tl.minimum(u, v)
        v_hi = tl.maximum(u, v)
        u = u_lo
        v = v_hi

        u_start = tl.load(adj_ptr_ptr + u, mask=active, other=0)
        u_end = tl.load(adj_ptr_ptr + u + 1, mask=active, other=0)
        v_start = tl.load(adj_ptr_ptr + v, mask=active, other=0)
        v_end = tl.load(adj_ptr_ptr + v + 1, mask=active, other=0)

        iu = u_start
        iv = v_start
        count = tl.zeros([BLOCK_E], dtype=tl.int32)

        for _step in range(2 * MAX_DEGREE):
            still_going = active & (iu < u_end) & (iv < v_end)

            nu = tl.load(adj_idx_ptr + iu, mask=still_going, other=0)
            nv = tl.load(adj_idx_ptr + iv, mask=still_going, other=0)

            match = still_going & (nu == nv)
            valid_tri = match & (nu > v)
            count = tl.where(valid_tri, count + 1, count)

            # Advance pointers (both advance on match)
            advance_u = still_going & (nu <= nv)
            advance_v = still_going & (nv <= nu)
            iu = tl.where(advance_u, iu + 1, iu)
            iv = tl.where(advance_v, iv + 1, iv)

        tl.store(counts_ptr + e_offs, count, mask=active)

    @triton.jit
    def _triangle_emit_kernel(
        edge_i_ptr, edge_j_ptr,
        adj_ptr_ptr, adj_idx_ptr, adj_rank_ptr,
        prefix_ptr,
        out_v0_ptr, out_v1_ptr, out_v2_ptr, out_mr_ptr,
        E,  # runtime scalar
        BLOCK_E: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Emit triangles per edge via vectorized CSR merge-intersection.

        Each program handles BLOCK_E edges. Each lane maintains its own
        write_pos (loaded from prefix), so no cross-lane coordination needed.
        """
        pid = tl.program_id(0)
        e_offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        active = e_offs < E

        u = tl.load(edge_i_ptr + e_offs, mask=active, other=0)
        v = tl.load(edge_j_ptr + e_offs, mask=active, other=0)
        u_lo = tl.minimum(u, v)
        v_hi = tl.maximum(u, v)
        u = u_lo
        v = v_hi
        r_uv = e_offs  # edge rank = position in sorted array

        u_start = tl.load(adj_ptr_ptr + u, mask=active, other=0)
        u_end = tl.load(adj_ptr_ptr + u + 1, mask=active, other=0)
        v_start = tl.load(adj_ptr_ptr + v, mask=active, other=0)
        v_end = tl.load(adj_ptr_ptr + v + 1, mask=active, other=0)

        write_pos = tl.load(prefix_ptr + e_offs, mask=active, other=0)
        iu = u_start
        iv = v_start

        for _step in range(2 * MAX_DEGREE):
            still_going = active & (iu < u_end) & (iv < v_end)

            nu = tl.load(adj_idx_ptr + iu, mask=still_going, other=0)
            nv = tl.load(adj_idx_ptr + iv, mask=still_going, other=0)

            match = still_going & (nu == nv)
            valid_tri = match & (nu > v)

            # Compute max rank for this triangle
            r_uw = tl.load(adj_rank_ptr + iu, mask=valid_tri, other=0)
            r_vw = tl.load(adj_rank_ptr + iv, mask=valid_tri, other=0)
            max_r = tl.maximum(r_uv, tl.maximum(r_uw, r_vw))

            # Store triangle
            tl.store(out_v0_ptr + write_pos, u, mask=valid_tri)
            tl.store(out_v1_ptr + write_pos, v, mask=valid_tri)
            tl.store(out_v2_ptr + write_pos, nu, mask=valid_tri)
            tl.store(out_mr_ptr + write_pos, max_r, mask=valid_tri)
            write_pos = tl.where(valid_tri, write_pos + 1, write_pos)

            # Advance pointers
            advance_u = still_going & (nu <= nv)
            advance_v = still_going & (nv <= nu)
            iu = tl.where(advance_u, iu + 1, iu)
            iv = tl.where(advance_v, iv + 1, iv)

        # No final store needed — all writes done inline


# ---------------------------------------------------------------------------
# Tetrahedra enumeration (Triton)
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _tet_count_kernel(
        tri_v0_ptr, tri_v1_ptr, tri_v2_ptr,
        adj_ptr_ptr, adj_idx_ptr,
        counts_ptr,
        T,  # runtime scalar
        BLOCK_T: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Count tetrahedra per triangle via vectorized triple CSR merge."""
        pid = tl.program_id(0)
        t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        active = t_offs < T

        a = tl.load(tri_v0_ptr + t_offs, mask=active, other=0)
        b = tl.load(tri_v1_ptr + t_offs, mask=active, other=0)
        c = tl.load(tri_v2_ptr + t_offs, mask=active, other=0)

        a_start = tl.load(adj_ptr_ptr + a, mask=active, other=0)
        a_end = tl.load(adj_ptr_ptr + a + 1, mask=active, other=0)
        b_start = tl.load(adj_ptr_ptr + b, mask=active, other=0)
        b_end = tl.load(adj_ptr_ptr + b + 1, mask=active, other=0)
        c_start = tl.load(adj_ptr_ptr + c, mask=active, other=0)
        c_end = tl.load(adj_ptr_ptr + c + 1, mask=active, other=0)

        ia = a_start
        ib = b_start
        ic = c_start
        count = tl.zeros([BLOCK_T], dtype=tl.int32)

        for _step in range(3 * MAX_DEGREE):
            still_going = active & (ia < a_end) & (ib < b_end) & (ic < c_end)

            na = tl.load(adj_idx_ptr + ia, mask=still_going, other=0)
            nb = tl.load(adj_idx_ptr + ib, mask=still_going, other=0)
            nc = tl.load(adj_idx_ptr + ic, mask=still_going, other=0)

            max_v = tl.maximum(na, tl.maximum(nb, nc))
            triple_match = still_going & (na == nb) & (nb == nc)
            valid_tet = triple_match & (na > c)
            count = tl.where(valid_tet, count + 1, count)

            # Advance: on triple match, advance all three
            # Otherwise, advance pointers whose value < max_v
            all_match_advance = triple_match
            advance_a = still_going & ((na < max_v) | all_match_advance)
            advance_b = still_going & ((nb < max_v) | all_match_advance)
            advance_c = still_going & ((nc < max_v) | all_match_advance)
            ia = tl.where(advance_a, ia + 1, ia)
            ib = tl.where(advance_b, ib + 1, ib)
            ic = tl.where(advance_c, ic + 1, ic)

        tl.store(counts_ptr + t_offs, count, mask=active)

    @triton.jit
    def _tet_emit_kernel(
        tri_v0_ptr, tri_v1_ptr, tri_v2_ptr, tri_mr_ptr,
        adj_ptr_ptr, adj_idx_ptr, adj_rank_ptr,
        prefix_ptr,
        out_v0_ptr, out_v1_ptr, out_v2_ptr, out_v3_ptr, out_mr_ptr,
        T,  # runtime scalar
        BLOCK_T: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        """Emit tetrahedra per triangle via vectorized triple CSR merge."""
        pid = tl.program_id(0)
        t_offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        active = t_offs < T

        a = tl.load(tri_v0_ptr + t_offs, mask=active, other=0)
        b = tl.load(tri_v1_ptr + t_offs, mask=active, other=0)
        c = tl.load(tri_v2_ptr + t_offs, mask=active, other=0)
        r_tri = tl.load(tri_mr_ptr + t_offs, mask=active, other=0)

        a_start = tl.load(adj_ptr_ptr + a, mask=active, other=0)
        a_end = tl.load(adj_ptr_ptr + a + 1, mask=active, other=0)
        b_start = tl.load(adj_ptr_ptr + b, mask=active, other=0)
        b_end = tl.load(adj_ptr_ptr + b + 1, mask=active, other=0)
        c_start = tl.load(adj_ptr_ptr + c, mask=active, other=0)
        c_end = tl.load(adj_ptr_ptr + c + 1, mask=active, other=0)

        write_pos = tl.load(prefix_ptr + t_offs, mask=active, other=0)
        ia = a_start
        ib = b_start
        ic = c_start

        for _step in range(3 * MAX_DEGREE):
            still_going = active & (ia < a_end) & (ib < b_end) & (ic < c_end)

            na = tl.load(adj_idx_ptr + ia, mask=still_going, other=0)
            nb = tl.load(adj_idx_ptr + ib, mask=still_going, other=0)
            nc = tl.load(adj_idx_ptr + ic, mask=still_going, other=0)

            max_v = tl.maximum(na, tl.maximum(nb, nc))
            triple_match = still_going & (na == nb) & (nb == nc)
            valid_tet = triple_match & (na > c)

            # Compute max edge rank
            r_ad = tl.load(adj_rank_ptr + ia, mask=valid_tet, other=0)
            r_bd = tl.load(adj_rank_ptr + ib, mask=valid_tet, other=0)
            r_cd = tl.load(adj_rank_ptr + ic, mask=valid_tet, other=0)
            max_r = tl.maximum(r_tri, tl.maximum(r_ad, tl.maximum(r_bd, r_cd)))

            # Store tetrahedron
            tl.store(out_v0_ptr + write_pos, a, mask=valid_tet)
            tl.store(out_v1_ptr + write_pos, b, mask=valid_tet)
            tl.store(out_v2_ptr + write_pos, c, mask=valid_tet)
            tl.store(out_v3_ptr + write_pos, na, mask=valid_tet)
            tl.store(out_mr_ptr + write_pos, max_r, mask=valid_tet)
            write_pos = tl.where(valid_tet, write_pos + 1, write_pos)

            # Advance pointers
            all_match_advance = triple_match
            advance_a = still_going & ((na < max_v) | all_match_advance)
            advance_b = still_going & ((nb < max_v) | all_match_advance)
            advance_c = still_going & ((nc < max_v) | all_match_advance)
            ia = tl.where(advance_a, ia + 1, ia)
            ib = tl.where(advance_b, ib + 1, ib)
            ic = tl.where(advance_c, ic + 1, ic)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

_BLOCK_E = 128
_BLOCK_T = 64  # Fewer lanes for tet (3-way merge is heavier)


def _next_power_of_2(x: int) -> int:
    v = 1
    while v < x:
        v *= 2
    return v


def triangle_enumerate(
    edge_i: Tensor,
    edge_j: Tensor,
    adj_ptr: Tensor,
    adj_idx: Tensor,
    adj_rank: Tensor,
    E: int,
    max_triangles: int = 10_000_000,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Enumerate all triangles in the Rips complex.

    Two-pass: count per edge, then emit. Uses Triton on CUDA, Numba on CPU.

    Parameters
    ----------
    edge_i, edge_j : (E,) int32 — edge endpoints (filtration-sorted)
    adj_ptr : (n+1,) int32 — CSR row pointers
    adj_idx : (2E,) int32 — CSR neighbor indices
    adj_rank : (2E,) int32 — edge rank per adjacency entry
    E : int — number of edges
    max_triangles : int — safety limit (default 10M)

    Returns
    -------
    tri_v0, tri_v1, tri_v2 : (T,) int32 — triangle vertices (v0 < v1 < v2)
    tri_max_rank : (T,) int32 — max edge rank (filtration rank)
    """
    device = edge_i.device

    if E == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty.clone(), empty.clone(), empty.clone()

    # CPU fallback
    if device.type != "cuda" or not _HAS_TRITON:
        from flash_ph.persistence import _enumerate_rips_triangles
        rv0, rv1, rv2, rmr = _enumerate_rips_triangles(
            edge_i.cpu().numpy(), edge_j.cpu().numpy(),
            adj_ptr.cpu().numpy(), adj_idx.cpu().numpy(),
            adj_rank.cpu().numpy(), E,
        )
        return (
            torch.from_numpy(rv0).to(device),
            torch.from_numpy(rv1).to(device),
            torch.from_numpy(rv2).to(device),
            torch.from_numpy(rmr).to(device),
        )

    edge_i = edge_i.contiguous().to(torch.int32)
    edge_j = edge_j.contiguous().to(torch.int32)
    adj_ptr = adj_ptr.contiguous().to(torch.int32)
    adj_idx = adj_idx.contiguous().to(torch.int32)
    adj_rank = adj_rank.contiguous().to(torch.int32)

    # Compute max degree for bounded loop
    degrees = adj_ptr[1:] - adj_ptr[:-1]
    max_deg = int(degrees.max().item()) if degrees.numel() > 0 else 1
    max_deg_po2 = _next_power_of_2(max_deg)

    # High-degree fallback: Triton compiles a loop of 2*MAX_DEGREE iterations
    # as constexpr; very high degree graphs cause slow JIT compilation.
    if max_deg_po2 > 4096:
        from flash_ph.persistence import _enumerate_rips_triangles
        rv0, rv1, rv2, rmr = _enumerate_rips_triangles(
            edge_i.cpu().numpy(), edge_j.cpu().numpy(),
            adj_ptr.cpu().numpy(), adj_idx.cpu().numpy(),
            adj_rank.cpu().numpy(), E,
        )
        return (
            torch.from_numpy(rv0).to(device),
            torch.from_numpy(rv1).to(device),
            torch.from_numpy(rv2).to(device),
            torch.from_numpy(rmr).to(device),
        )

    # Pass 1: count
    counts = torch.zeros(E, dtype=torch.int32, device=device)
    grid = ((_BLOCK_E + E - 1) // _BLOCK_E,)
    _triangle_count_kernel[grid](
        edge_i, edge_j, adj_ptr, adj_idx,
        counts, E,
        BLOCK_E=_BLOCK_E, MAX_DEGREE=max_deg_po2,
    )

    total_T = int(counts.to(torch.int64).sum().item())
    if total_T == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty.clone(), empty.clone(), empty.clone()

    if total_T > max_triangles:
        raise ValueError(
            f"Triangle count {total_T} exceeds max_triangles={max_triangles}. "
            f"Reduce max_edge_length or increase max_triangles."
        )

    # Exclusive prefix sum
    prefix = torch.cumsum(counts.to(torch.int64), dim=0) - counts.to(torch.int64)
    prefix = prefix.to(torch.int32)

    # Pass 2: emit
    out_v0 = torch.empty(total_T, dtype=torch.int32, device=device)
    out_v1 = torch.empty(total_T, dtype=torch.int32, device=device)
    out_v2 = torch.empty(total_T, dtype=torch.int32, device=device)
    out_mr = torch.empty(total_T, dtype=torch.int32, device=device)

    _triangle_emit_kernel[grid](
        edge_i, edge_j, adj_ptr, adj_idx, adj_rank,
        prefix, out_v0, out_v1, out_v2, out_mr, E,
        BLOCK_E=_BLOCK_E, MAX_DEGREE=max_deg_po2,
    )

    return out_v0, out_v1, out_v2, out_mr


def tetrahedra_enumerate(
    tri_v0: Tensor,
    tri_v1: Tensor,
    tri_v2: Tensor,
    tri_max_rank: Tensor,
    adj_ptr: Tensor,
    adj_idx: Tensor,
    adj_rank: Tensor,
    T: int,
    max_tetrahedra: int = 10_000_000,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Enumerate all tetrahedra via triple CSR intersection.

    Parameters
    ----------
    tri_v0, tri_v1, tri_v2 : (T,) int32 — triangle vertices (v0 < v1 < v2)
    tri_max_rank : (T,) int32 — triangle filtration ranks
    adj_ptr, adj_idx, adj_rank : CSR per-vertex adjacency
    T : int — number of triangles
    max_tetrahedra : int — safety limit (default 10M)

    Returns
    -------
    tet_v0..v3 : (Q,) int32 — tetrahedra vertices (v0 < v1 < v2 < v3)
    tet_max_rank : (Q,) int32 — max edge rank
    """
    device = tri_v0.device

    if T == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty.clone(), empty.clone(), empty.clone(), empty.clone()

    # CPU fallback
    if device.type != "cuda" or not _HAS_TRITON:
        from flash_ph.persistence import _enumerate_rips_tetrahedra
        rv = _enumerate_rips_tetrahedra(
            tri_v0.cpu().numpy(), tri_v1.cpu().numpy(),
            tri_v2.cpu().numpy(), tri_max_rank.cpu().numpy(),
            adj_ptr.cpu().numpy(), adj_idx.cpu().numpy(),
            adj_rank.cpu().numpy(), T,
        )
        return tuple(torch.from_numpy(x).to(device) for x in rv)

    tri_v0 = tri_v0.contiguous().to(torch.int32)
    tri_v1 = tri_v1.contiguous().to(torch.int32)
    tri_v2 = tri_v2.contiguous().to(torch.int32)
    tri_max_rank = tri_max_rank.contiguous().to(torch.int32)
    adj_ptr = adj_ptr.contiguous().to(torch.int32)
    adj_idx = adj_idx.contiguous().to(torch.int32)
    adj_rank = adj_rank.contiguous().to(torch.int32)

    degrees = adj_ptr[1:] - adj_ptr[:-1]
    max_deg = int(degrees.max().item()) if degrees.numel() > 0 else 1
    max_deg_po2 = _next_power_of_2(max_deg)

    # High-degree fallback to CPU Numba
    if max_deg_po2 > 4096:
        from flash_ph.persistence import _enumerate_rips_tetrahedra
        rv = _enumerate_rips_tetrahedra(
            tri_v0.cpu().numpy(), tri_v1.cpu().numpy(),
            tri_v2.cpu().numpy(), tri_max_rank.cpu().numpy(),
            adj_ptr.cpu().numpy(), adj_idx.cpu().numpy(),
            adj_rank.cpu().numpy(), T,
        )
        return tuple(torch.from_numpy(x).to(device) for x in rv)

    # Pass 1: count
    counts = torch.zeros(T, dtype=torch.int32, device=device)
    grid = ((_BLOCK_T + T - 1) // _BLOCK_T,)
    _tet_count_kernel[grid](
        tri_v0, tri_v1, tri_v2,
        adj_ptr, adj_idx, counts, T,
        BLOCK_T=_BLOCK_T, MAX_DEGREE=max_deg_po2,
    )

    total_Q = int(counts.to(torch.int64).sum().item())
    if total_Q == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty.clone(), empty.clone(), empty.clone(), empty.clone()

    if total_Q > max_tetrahedra:
        raise ValueError(
            f"Tetrahedra count {total_Q} exceeds max_tetrahedra={max_tetrahedra}. "
            f"Reduce max_edge_length or increase max_tetrahedra."
        )

    prefix = torch.cumsum(counts.to(torch.int64), dim=0) - counts.to(torch.int64)
    prefix = prefix.to(torch.int32)

    # Pass 2: emit
    out_v0 = torch.empty(total_Q, dtype=torch.int32, device=device)
    out_v1 = torch.empty(total_Q, dtype=torch.int32, device=device)
    out_v2 = torch.empty(total_Q, dtype=torch.int32, device=device)
    out_v3 = torch.empty(total_Q, dtype=torch.int32, device=device)
    out_mr = torch.empty(total_Q, dtype=torch.int32, device=device)

    _tet_emit_kernel[grid](
        tri_v0, tri_v1, tri_v2, tri_max_rank,
        adj_ptr, adj_idx, adj_rank,
        prefix, out_v0, out_v1, out_v2, out_v3, out_mr, T,
        BLOCK_T=_BLOCK_T, MAX_DEGREE=max_deg_po2,
    )

    return out_v0, out_v1, out_v2, out_v3, out_mr
