"""Public API for Vietoris-Rips persistent homology (H0 + H1 + H2).

Pipeline:
- GPU (CUDA): block-sparse edge enumeration -> Boruvka MST (H0)
- H1 only (max_dim=1): GPU-native reduction (Triton apparent pairs +
  triangle enumeration + Numba residual reduction)
- H1+H2 (max_dim=2): GPU edges -> COO sparse distance matrix ->
  giotto-ph C++ reduction (H1/H2)
"""
from __future__ import annotations


import numpy as np
import torch
from torch import Tensor

from flash_ph.rips import rips_filtration
from flash_ph.boruvka import gpu_boruvka_mst

_VERTEX_PACK_BITS = 16


# ---------------------------------------------------------------------------
# giotto-ph C++ backend
# ---------------------------------------------------------------------------

def _rips_h1h2_giotto(filt, n, max_dim, max_edge_length, device):
    """H1 (+H2) via giotto-ph C++ lockfree reduction engine.

    Builds a COO sparse distance matrix from GPU-computed edges and
    delegates to ``gph.ripser_parallel``.
    """
    from gph import ripser_parallel
    import scipy.sparse

    # COO upper-triangle sparse distance matrix
    row = filt.edge_i.cpu().numpy().astype(np.int64)
    col = filt.edge_j.cpu().numpy().astype(np.int64)
    dist = torch.sqrt(filt.edge_dist_sq).cpu().numpy().astype(np.float32)

    # Zero diagonal (vertex birth = 0)
    diag = np.arange(n, dtype=np.int64)
    row = np.concatenate([row, diag])
    col = np.concatenate([col, diag])
    dist = np.concatenate([dist, np.zeros(n, dtype=np.float32)])

    dm = scipy.sparse.coo_matrix((dist, (row, col)), shape=(n, n))
    result = ripser_parallel(
        dm, maxdim=max_dim, thresh=max_edge_length,
        metric="precomputed", collapse_edges=True, n_threads=-1,
    )

    diagrams = []
    for d in range(1, max_dim + 1):
        dgm_np = result["dgms"][d]
        if len(dgm_np) > 0:
            t = torch.from_numpy(dgm_np).float().to(device)
        else:
            t = torch.empty(0, 2, dtype=torch.float32, device=device)
        diagrams.append(t)
    return diagrams  # [H1] or [H1, H2]


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

    # Step 3+4: H1 (+H2)
    if max_dim == 1 and device.type == 'cuda':
        # GPU-native H1 reduction (no giotto-ph dependency)
        from flash_ph.reduce_h1_gpu import rips_h1_gpu_native
        h1 = rips_h1_gpu_native(filt, n, mst_idx, device)
        return [h0, h1]

    # max_dim == 2 (or CPU fallback): giotto-ph C++ backend
    higher_dims = _rips_h1h2_giotto(
        filt, n, max_dim, max_edge_length, device,
    )
    result = [h0]
    for diag in higher_dims:
        if diag.numel() > 0:
            diag = diag[diag[:, 0].argsort(stable=True)]
        result.append(diag)
    return result
