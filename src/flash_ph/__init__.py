"""flash-ph: GPU-accelerated exact Rips persistent homology."""

__version__ = "0.1.0"

from flash_ph.persistence import (
    SimplexPairs, rips_apparent_pairs, rips_persistence, rips_persistence_pairs,
)
from flash_ph.threshold import auto_threshold, enclosing_radius
from flash_ph.complex import RipsComplex

__all__ = [
    "rips_persistence", "rips_persistence_pairs", "rips_apparent_pairs",
    "SimplexPairs", "auto_threshold", "enclosing_radius", "RipsComplex",
]
