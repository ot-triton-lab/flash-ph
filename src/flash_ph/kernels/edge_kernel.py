"""Triton Rips edge enumeration kernels.

Direct two-pass approach: count edges per block-pair, then emit edges.
"""
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


# NOTE: Block-sparse edge enumeration (rips_edge_enumerate) was removed.
# It required Morton/AABB/CSR infrastructure from flash-tda. The direct
# edge enumeration below is the active code path for flash-ph.



# ---------------------------------------------------------------------------
# Direct edge enumeration
# ---------------------------------------------------------------------------

@triton.jit
def _direct_edge_count_kernel(
    pts_ptr,        # (n, D) original (unsorted) points
    counts_ptr,     # (n_blocks, n_blocks) output counts
    threshold_sq,   # squared max_edge_length
    n,              # number of points (runtime value)
    stride_pn,      # pts.stride(0)
    stride_pd,      # pts.stride(1)
    n_blocks,       # ceildiv(n, B)
    B: tl.constexpr,
    D: tl.constexpr,
):
    """Count qualifying edges per upper-triangle tile (2D grid)."""
    I = tl.program_id(0)
    J = tl.program_id(1)

    # Lower triangle: store 0 and return
    if I > J:
        tl.store(counts_ptr + I * n_blocks + J, 0)
        return

    src_base = I * B
    tgt_base = J * B
    src_offsets = tl.arange(0, B)
    tgt_offsets = tl.arange(0, B)
    src_global = src_base + src_offsets
    tgt_global = tgt_base + tgt_offsets

    # B×B squared distances, dimension by dimension
    # Use range(D) — NOT tl.static_range(D) — to avoid JIT bloat at large D
    dist_sq = tl.zeros([B, B], dtype=tl.float32)
    for d_idx in range(D):
        x_d = tl.load(
            pts_ptr + src_global * stride_pn + d_idx * stride_pd,
            mask=src_global < n, other=0.0,
        ).to(tl.float32)
        y_d = tl.load(
            pts_ptr + tgt_global * stride_pn + d_idx * stride_pd,
            mask=tgt_global < n, other=0.0,
        ).to(tl.float32)
        diff = x_d[:, None] - y_d[None, :]
        dist_sq += diff * diff

    # Valid mask: within threshold AND within n
    valid = (dist_sq < threshold_sq)
    valid = valid & (src_global[:, None] < n)
    valid = valid & (tgt_global[None, :] < n)

    # Diagonal blocks: upper triangle only (src < tgt)
    is_diag = (I == J)
    upper_tri = src_offsets[:, None] < tgt_offsets[None, :]
    valid = valid & (upper_tri | ~is_diag)

    count = tl.sum(valid.to(tl.int32))
    tl.store(counts_ptr + I * n_blocks + J, count)


@triton.jit
def _direct_edge_emit_kernel(
    pts_ptr,        # (n, D) original (unsorted) points
    prefix_ptr,     # (n_blocks, n_blocks) exclusive prefix sums
    out_i_ptr,      # (E,) output source indices
    out_j_ptr,      # (E,) output target indices
    out_dsq_ptr,    # (E,) output squared distances
    threshold_sq,   # squared max_edge_length
    n,              # number of points (runtime value)
    stride_pn,      # pts.stride(0)
    stride_pd,      # pts.stride(1)
    n_blocks,       # ceildiv(n, B)
    B: tl.constexpr,
    D: tl.constexpr,
):
    """Emit qualifying edges with stream compaction (2D grid)."""
    I = tl.program_id(0)
    J = tl.program_id(1)

    if I > J:
        return

    write_base = tl.load(prefix_ptr + I * n_blocks + J)

    src_base = I * B
    tgt_base = J * B
    src_offsets = tl.arange(0, B)
    tgt_offsets = tl.arange(0, B)
    src_global = src_base + src_offsets
    tgt_global = tgt_base + tgt_offsets

    # B×B squared distances (same as count kernel)
    dist_sq = tl.zeros([B, B], dtype=tl.float32)
    for d_idx in range(D):
        x_d = tl.load(
            pts_ptr + src_global * stride_pn + d_idx * stride_pd,
            mask=src_global < n, other=0.0,
        ).to(tl.float32)
        y_d = tl.load(
            pts_ptr + tgt_global * stride_pn + d_idx * stride_pd,
            mask=tgt_global < n, other=0.0,
        ).to(tl.float32)
        diff = x_d[:, None] - y_d[None, :]
        dist_sq += diff * diff

    valid = (dist_sq < threshold_sq)
    valid = valid & (src_global[:, None] < n)
    valid = valid & (tgt_global[None, :] < n)

    is_diag = (I == J)
    upper_tri = src_offsets[:, None] < tgt_offsets[None, :]
    valid = valid & (upper_tri | ~is_diag)

    # Flatten B×B → B² for stream compaction
    flat_valid = tl.reshape(valid, [B * B]).to(tl.int32)
    flat_dist_sq = tl.reshape(dist_sq, [B * B])
    flat_src = tl.reshape(
        tl.broadcast_to(src_global[:, None], [B, B]), [B * B]
    )
    flat_tgt = tl.reshape(
        tl.broadcast_to(tgt_global[None, :], [B, B]), [B * B]
    )

    scan = tl.cumsum(flat_valid)
    bool_mask = flat_valid != 0
    write_idx = write_base + scan - 1

    tl.store(out_i_ptr + write_idx, flat_src, mask=bool_mask)
    tl.store(out_j_ptr + write_idx, flat_tgt, mask=bool_mask)
    tl.store(out_dsq_ptr + write_idx, flat_dist_sq, mask=bool_mask)


def direct_edge_enumerate(
    pts: Tensor,
    n: int,
    threshold_sq: float,
    B: int = 32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Direct Triton edge enumeration without Morton/AABB/CSR overhead.

    Uses a 2D grid of (n_blocks, n_blocks) programs. Each program handles
    a B×B tile of pairwise distances. No spatial sorting or padding needed.

    Parameters
    ----------
    pts : (n, d) float32
        Point cloud coordinates, contiguous. Unsorted (original order).
    n : int
        Number of points.
    threshold_sq : float
        Squared max_edge_length for filtering.
    B : int
        Tile size (default 32).

    Returns
    -------
    out_i : (E,) int32 — source indices in original order, i < j
    out_j : (E,) int32 — target indices in original order, i < j
    out_dsq : (E,) float32 — squared distances
    """
    device = pts.device
    _empty = lambda dt: torch.empty(0, dtype=dt, device=device)

    if n <= 1:
        return _empty(torch.int32), _empty(torch.int32), _empty(torch.float32)

    pts = pts.contiguous()
    _, D = pts.shape
    n_blocks = (n + B - 1) // B

    # Pass 1: count edges per tile (kernel writes every entry, no init needed)
    counts = torch.empty(n_blocks, n_blocks, dtype=torch.int32, device=device)
    grid = (n_blocks, n_blocks)
    _direct_edge_count_kernel[grid](
        pts, counts, threshold_sq, n,
        pts.stride(0), pts.stride(1), n_blocks,
        B=B, D=D,
    )

    total_E = int(counts.sum(dtype=torch.int64).item())
    if total_E == 0:
        return _empty(torch.int32), _empty(torch.int32), _empty(torch.float32)

    assert total_E < 2**31, (
        f"Edge count {total_E} exceeds int32 range; reduce max_edge_length"
    )

    # Exclusive prefix sum over flattened counts (single int64 copy)
    flat_counts = counts.reshape(-1).to(torch.int64)
    flat_prefix = torch.cumsum(flat_counts, dim=0) - flat_counts
    prefix = flat_prefix.reshape(n_blocks, n_blocks).to(torch.int32)

    # Pass 2: emit edges
    out_i = torch.empty(total_E, dtype=torch.int32, device=device)
    out_j = torch.empty(total_E, dtype=torch.int32, device=device)
    out_dsq = torch.empty(total_E, dtype=torch.float32, device=device)

    _direct_edge_emit_kernel[grid](
        pts, prefix, out_i, out_j, out_dsq,
        threshold_sq, n,
        pts.stride(0), pts.stride(1), n_blocks,
        B=B, D=D,
    )

    return out_i, out_j, out_dsq
