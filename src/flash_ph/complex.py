"""GUDHI-style RipsComplex wrapper around flash_ph.rips_persistence."""
from __future__ import annotations

import numpy as np
import torch

from flash_ph.persistence import rips_persistence


class RipsComplex:
    """GUDHI-compatible Vietoris-Rips complex backed by flash_ph GPU pipeline.

    Parameters
    ----------
    points : Tensor or ndarray
        (n, d) point cloud coordinates.
    max_edge_length : float
        Maximum edge length for the Rips filtration.

    Examples
    --------
    >>> rips = RipsComplex(points, max_edge_length=2.0)
    >>> rips.compute_persistence(max_dim=1)
    >>> h1 = rips.persistence_intervals_in_dimension(1)
    >>> betti = rips.betti_numbers()
    """

    def __init__(self, points, max_edge_length: float):
        if max_edge_length < 0:
            raise ValueError(
                f"max_edge_length must be non-negative, got {max_edge_length}"
            )
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
            if torch.cuda.is_available():
                points = points.cuda()
        self._points = points
        self._max_edge_length = max_edge_length
        self._diagrams = None
        self._max_dim = None

    def compute_persistence(self, max_dim: int = 1, homology_coeff_field: int = 2,
                             backend: str = 'auto'):
        """Compute persistent homology and store results.

        Parameters
        ----------
        max_dim : int
            Maximum homology dimension (0, 1, or 2). Default 1.
        homology_coeff_field : int
            Must be 2 (Z/2Z). Kept for GUDHI API compatibility.
        backend : str
            H1 reduction backend (``'auto'``, ``'gpu'``, or ``'giotto'``).
            Only used when ``max_dim=1`` on CUDA. Default ``'auto'``.

        Returns None (GUDHI convention).
        """
        if homology_coeff_field != 2:
            raise ValueError("Only Z/2Z coefficients are supported (homology_coeff_field=2)")
        self._max_dim = max_dim
        self._diagrams = rips_persistence(
            self._points, self._max_edge_length, max_dim=max_dim,
            backend=backend,
        )

    def persistence(self, max_dim: int = 1, homology_coeff_field: int = 2):
        """Compute and return GUDHI-format persistence pairs.

        Returns
        -------
        list[tuple[int, tuple[float, float]]]
            Sorted list of ``(dim, (birth, death))`` pairs.
        """
        self.compute_persistence(max_dim, homology_coeff_field)
        result = []
        for dim, dgm in enumerate(self._diagrams):
            if dgm.numel() == 0:
                continue
            arr = dgm.cpu().numpy()
            for i in range(arr.shape[0]):
                result.append((dim, (float(arr[i, 0]), float(arr[i, 1]))))
        result.sort(key=lambda x: (x[0], x[1][0], x[1][1]))
        return result

    def persistence_intervals_in_dimension(self, dim: int):
        """Return persistence intervals for a single dimension.

        Returns
        -------
        ndarray
            (K, 2) float64 array of ``(birth, death)`` pairs.
        """
        self._check_computed()
        if dim < 0 or dim > self._max_dim:
            return np.empty((0, 2), dtype=np.float64)
        dgm = self._diagrams[dim]
        if dgm.numel() == 0:
            return np.empty((0, 2), dtype=np.float64)
        return dgm.cpu().numpy().astype(np.float64)

    def betti_numbers(self):
        """Count essential (infinite-death) features per dimension.

        Returns
        -------
        list[int]
            ``[beta_0, beta_1, ...]``
        """
        self._check_computed()
        return [
            int((d[:, 1] == float("inf")).sum()) if d.numel() > 0 else 0
            for d in self._diagrams
        ]

    def persistent_betti_numbers(self, from_value: float, to_value: float):
        """Count features born <= from_value and dying > to_value, per dimension.

        Returns
        -------
        list[int]
        """
        self._check_computed()
        result = []
        for dgm in self._diagrams:
            if dgm.numel() == 0:
                result.append(0)
                continue
            count = int(((dgm[:, 0] <= from_value) & (dgm[:, 1] > to_value)).sum())
            result.append(count)
        return result

    def num_vertices(self) -> int:
        """Return the number of input points."""
        return self._points.shape[0]

    def _check_computed(self):
        if self._diagrams is None:
            raise RuntimeError(
                "Call compute_persistence() or persistence() first"
            )
