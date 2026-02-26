"""Threshold estimation for Rips persistence.

Provides two strategies for choosing ``max_edge_length``:

- ``auto_threshold``: percentile of k-nearest-neighbor distances.
  Produces a threshold that keeps the Rips graph sparse while
  preserving topological features at the chosen scale.

- ``enclosing_radius``: minimum over points of the maximum distance
  to any other point.  Any simplex with diameter > R cannot
  contribute a non-trivial persistence pair (Ripser++ Proposition 4).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def auto_threshold(
    points,
    k: int = 20,
    percentile: float = 95.0,
    subsample: int | None = None,
) -> float:
    """Estimate Rips threshold from k-NN distance distribution.

    Computes the k-th nearest neighbor distance for each point and
    returns the given percentile.  This produces a threshold that
    keeps the Rips graph sparse (each vertex has ~k neighbors at
    the chosen scale) while preserving most topological features.

    Parameters
    ----------
    points : (n, d) array-like
        Point cloud.  Accepts numpy arrays or PyTorch tensors.
    k : int
        Number of nearest neighbors.  Default 20.
    percentile : float
        Percentile of the k-NN distance distribution.  Default 95.
    subsample : int or None
        If set, randomly subsample this many points before computing
        pairwise distances (reduces O(n^2) cost for large n).

    Returns
    -------
    float
        Estimated threshold (Euclidean distance).
    """
    pts = _to_tensor(points)
    n = pts.shape[0]
    if n <= 1:
        return 0.0

    k = max(1, min(k, n - 1))

    if subsample is not None and subsample >= 2 and subsample < n:
        idx = torch.randperm(n, device=pts.device)[:subsample]
        pts = pts[idx]
        n = pts.shape[0]
        k = max(1, min(k, n - 1))

    percentile = max(0.0, min(100.0, percentile))

    # Tiled cdist to avoid O(n^2) memory
    knn_dists = _tiled_knn(pts, k)

    # Return percentile of k-th NN distances
    p = float(torch.quantile(knn_dists, percentile / 100.0).item())
    return p


def enclosing_radius(
    points,
    subsample: int | None = None,
) -> float:
    """Compute enclosing radius: min_x max_y d(x, y).

    Any simplex with diameter > R cannot contribute a non-trivial
    persistence pair (Ripser++ Proposition 4).  Using this as
    ``max_edge_length`` guarantees no topological features are missed.

    Parameters
    ----------
    points : (n, d) array-like
        Point cloud.  Accepts numpy arrays or PyTorch tensors.
    subsample : int or None
        If set, randomly subsample this many points before computing
        pairwise distances (reduces O(n^2) cost for large n).
        The result is an approximation.

    Returns
    -------
    float
        Enclosing radius (Euclidean distance).
    """
    pts = _to_tensor(points)
    n = pts.shape[0]
    if n <= 1:
        return 0.0

    if subsample is not None and subsample >= 2 and subsample < n:
        idx = torch.randperm(n, device=pts.device)[:subsample]
        pts = pts[idx]

    # Tiled max-distance computation
    r = _tiled_enclosing_radius(pts)
    return float(r)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(points) -> Tensor:
    """Convert numpy or tensor input to float32 tensor."""
    if isinstance(points, np.ndarray):
        t = torch.from_numpy(points).float()
        if torch.cuda.is_available():
            t = t.cuda()
        return t
    return points.float()


def _tiled_knn(pts: Tensor, k: int, tile: int = 4096) -> Tensor:
    """Compute k-th nearest neighbor distance per point via tiled cdist.

    Tiles both dimensions so peak memory is O(tile^2) instead of O(n*tile).
    Each query chunk maintains a running top-k heap across target chunks.
    Returns (n,) tensor of k-th NN Euclidean distances.
    """
    n = pts.shape[0]
    device = pts.device
    # Per-point running top-k: store k smallest distances seen so far
    # Initialize with inf so any real distance replaces them
    topk_buf = torch.full((n, k), float('inf'), device=device)

    for i_start in range(0, n, tile):
        i_end = min(i_start + tile, n)
        qi = pts[i_start:i_end]  # (qi_size, d)
        qi_size = i_end - i_start

        for j_start in range(0, n, tile):
            j_end = min(j_start + tile, n)
            tj = pts[j_start:j_end]  # (tj_size, d)

            dist = torch.cdist(qi, tj)  # (qi_size, tj_size)

            # Mask self-distances (only when tiles overlap)
            if i_start < j_end and j_start < i_end:
                overlap_i_lo = max(i_start, j_start)
                overlap_i_hi = min(i_end, j_end)
                for idx in range(overlap_i_lo, overlap_i_hi):
                    dist[idx - i_start, idx - j_start] = float('inf')

            # Take top-k from this tile, then merge with running top-k.
            # topk on (tile, tile) → (tile, k) is a partial sort,
            # then merge (tile, 2k) instead of (tile, k+tile).
            tj_size = j_end - j_start
            k_tile = min(k, tj_size)
            tile_topk, _ = dist.topk(k_tile, dim=1, largest=False)  # (qi_size, k_tile)
            cur_topk = topk_buf[i_start:i_end]  # (qi_size, k)
            combined = torch.cat([cur_topk, tile_topk], dim=1)  # (qi_size, k + k_tile)
            new_topk, _ = combined.topk(k, dim=1, largest=False)
            topk_buf[i_start:i_end] = new_topk

    # k-th NN distance is the largest in each point's top-k
    return topk_buf[:, -1]


def _tiled_enclosing_radius(pts: Tensor, tile: int = 4096) -> float:
    """Compute min_x max_y d(x, y) via doubly-tiled cdist.

    Peak memory is O(tile^2) instead of O(n*tile).
    """
    n = pts.shape[0]
    device = pts.device
    max_dists = torch.zeros(n, device=device)

    for j_start in range(0, n, tile):
        j_end = min(j_start + tile, n)
        chunk = pts[j_start:j_end]
        # Tile the query dimension too for very large n
        for i_start in range(0, n, tile):
            i_end = min(i_start + tile, n)
            dist = torch.cdist(pts[i_start:i_end], chunk)  # (i_tile, j_tile)
            tile_max, _ = dist.max(dim=1)
            max_dists[i_start:i_end] = torch.maximum(
                max_dists[i_start:i_end], tile_max,
            )

    return float(max_dists.min().item())
