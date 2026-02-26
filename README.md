# flash-ph

[![PyPI](https://img.shields.io/pypi/v/flash-ph)](https://pypi.org/project/flash-ph/)
[![Python](https://img.shields.io/pypi/pyversions/flash-ph)](https://pypi.org/project/flash-ph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GPU-accelerated exact Rips persistent homology**

flash-ph computes exact Vietoris-Rips persistent homology (H0, H1, H2) over Z/2Z using PyTorch + Triton kernels on NVIDIA GPUs. The public API includes `rips_persistence`, `auto_threshold`, `enclosing_radius`, and a GUDHI-compatible `RipsComplex` class.

## Install

```bash
# From source (development)
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, PyTorch >= 2.5 with CUDA, Numba.
**Dev extras:** pytest, ripser, giotto-ph.

## Quick Start

### Functional API

```python
from flash_ph import rips_persistence, auto_threshold
import torch

pts = torch.randn(1000, 10, device="cuda")
thresh = auto_threshold(pts, k=20, percentile=95)
diagrams = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
# diagrams = [H0_tensor, H1_tensor]  -- each (K, 2) float32
```

### GUDHI-style API

```python
from flash_ph import RipsComplex

rips = RipsComplex(points=pts, max_edge_length=thresh)
rips.compute_persistence(max_dim=1)

h1 = rips.persistence_intervals_in_dimension(1)   # numpy (K, 2) float64
betti = rips.betti_numbers()                        # [1, 3]
pbetti = rips.persistent_betti_numbers(0.1, 0.5)   # [1, 2]

# Or compute + return in one call (GUDHI's persistence() pattern)
pairs = rips.persistence(max_dim=1)
# [(0, (0.0, inf)), (1, (0.3, 0.7)), ...]
```

## Scope and Guarantees

- **Exact** Vietoris-Rips persistent homology over Z/2Z.
- **Euclidean metric** (squared internally, sqrt at output).
- `max_dim` in {0, 1, 2}. H2 is opt-in (`max_dim=2`) and requires n < 65536.
- **Strict filtration**: only simplices with diameter < `max_edge_length` are included.
- **Output format**: list of `(K, 2)` float32 tensors of `(birth, death)` pairs. Essential features have `death = inf`.
- **Input**: accepts NumPy arrays or PyTorch tensors. NumPy inputs are automatically converted and moved to CUDA if available.
- **GPU (CUDA) required** for speed. CPU fallback is available but not the target use case.

## Choosing `max_edge_length`

Two built-in strategies:

- **`auto_threshold(pts, k=20, percentile=95)`** -- returns the 95th percentile of k-nearest-neighbor distances. Keeps the Rips graph sparse; recommended for d > 3 where distance concentration makes the full complex unnecessary.
- **`enclosing_radius(pts)`** -- returns min_x max_y d(x, y). Guarantees no topological features are missed, but can be expensive for large point clouds.

In high dimensions, pairwise distances concentrate around their mean, so `auto_threshold` with moderate k is usually sufficient to capture all relevant features while keeping memory and runtime manageable.

## Architecture Overview

flash-ph uses a hybrid GPU/CPU pipeline:

- **H0 (connected components):** GPU Boruvka MST on pairwise edge distances. Fully parallel, no CPU reduction needed.
- **H1 (loops):** Triton apparent pair detection kernel (BLOCK_E=128 edges per program) identifies ~95--99% of columns. Remaining columns are reduced by a Numba cohomology reduction on CPU.
- **H2 (voids):** Triton triangle and tetrahedron enumeration via CSR merge-intersection, followed by a GPU CSR builder with restricted ranking and packed int64 face matching. Residual columns fall through to Numba cohomology reduction with clearing cascade from H1.

All Triton kernels pay a one-time JIT compilation cost on first invocation (see Warmup below).

## Benchmarks

flash-ph targets **high-dimensional point clouds (d > 3)** where distance concentration keeps the Rips graph sparse and the GPU pipeline excels. All timings on A100-80GB, computing H0+H1+H2. Parity: exact match with ripser on every configuration.

| Config | Threshold | flash-ph | ripser | giotto-ph | vs ripser | vs giotto-ph |
|--------|-----------|----------|--------|-----------|-----------|--------------|
| n=500 d=10 | 2.8 | 73 ms | 134 ms | 100 ms | 1.8x | 1.4x |
| n=1K d=10 | 2.5 | 68 ms | 218 ms | 153 ms | 3.2x | 2.2x |
| n=2K d=10 | 2.2 | 69 ms | 372 ms | 280 ms | 5.4x | 4.0x |
| n=5K d=10 | 1.8 | 61 ms | 1,259 ms | 535 ms | 20.5x | 8.7x |
| n=10K d=10 | 1.6 | 49 ms | 4,938 ms | 1,409 ms | 101x | 29x |
| n=20K d=10 | 1.4 | 43 ms | 19,161 ms | 3,406 ms | 445x | 79x |
| n=500 d=50 | 8.0 | 21 ms | 24 ms | 134 ms | 1.1x | 6.2x |
| n=1K d=50 | 7.5 | 17 ms | 35 ms | 158 ms | 2.0x | 9.2x |

Ripser: CPU, full O(n^2) distance matrix. Giotto-ph: CPU, 8 threads, sparse radius-neighbor input.

## Warmup

The first call to `rips_persistence` pays Triton and Numba JIT compilation costs (typically 5--15 seconds depending on GPU and kernel configuration). Subsequent calls reuse cached kernels. All benchmark numbers above exclude warmup.

## Reproduce Results

```bash
pip install -e ".[dev]"
python -m flash_ph.bench.bench_persistence
```

## License

MIT
