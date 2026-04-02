# flash-ph: Algorithm Design and Theory

## 1. Background: Vietoris-Rips Persistent Homology

Given a point cloud $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ and a threshold $\varepsilon > 0$, the **Vietoris-Rips complex** $\mathrm{VR}(X, \varepsilon)$ is the simplicial complex where a $k$-simplex $[v_0, \dots, v_k]$ is included iff all pairwise distances $d(v_i, v_j) < \varepsilon$ (strict inequality — this matches ripser's convention and flash-ph's implementation).

**Persistent homology** tracks how homology groups $H_k$ change as $\varepsilon$ increases from 0 to some maximum threshold. Each topological feature (connected component, loop, void) is born at some $\varepsilon_b$ and dies at $\varepsilon_d$, recorded as a bar $(b, d)$ in the **persistence diagram**.

- **H0**: connected components. Born when a vertex appears, dies when two components merge.
- **H1**: loops/tunnels. Born when a cycle forms that doesn't bound a filled region.
- **H2**: voids/cavities. Born when a void is enclosed.

### Computational Complexity

The naive approach builds the full filtration (all simplices ordered by diameter) and applies the **persistence algorithm** — essentially column reduction on the boundary matrix. For $n$ points:

- Number of edges: $\binom{n}{2} = O(n^2)$
- Number of triangles: $\binom{n}{3} = O(n^3)$
- Number of tetrahedra: $\binom{n}{4} = O(n^4)$
- Column reduction: $O(n^3)$ for H1, $O(n^4)$ for H2 in the worst case

This is why persistent homology is expensive. The art is in **never building most of this**.

## 2. Ripser's Key Insights (What We Build On)

Ripser (Bauer, 2021) introduced three optimizations that made Rips persistence practical:

### 2.1 Cohomology Instead of Homology

The persistence algorithm can be run on the **coboundary** matrix (cohomology) instead of the boundary matrix (homology). The algebraic result is identical (persistence diagrams match), but the computational structure differs:

- **Homology H1**: reduces over triangle columns (cofacets of edges). There are $T$ triangles.
- **Cohomology H1**: reduces over edge columns (faces of triangles). There are $E$ edges.

Since $T \gg E$ in typical Rips complexes (e.g., $T = 354K$ vs $E = 33K$ for a 1024-point O(3) dataset), cohomology processes far fewer columns.

**The pivot convention reverses**: in cohomology, the pivot is the **smallest** element (lowest filtration cofacet), not the largest. This is the dual of the standard homology reduction.

### 2.2 Apparent Pairs (~98% of Columns)

An edge $e$ is an **apparent pair** with triangle $t$ if:
1. $t$ is the youngest cofacet of $e$ (the triangle with smallest diameter containing $e$)
2. $e$ is the oldest face of $t$ (the edge with largest diameter in $t$)

If both conditions hold, $e$ and $t$ form a persistence pair without any column reduction — we know immediately that $e$ creates a cycle at birth and $t$ fills it at death.

In practice, **~98% of edges** in a Rips complex are apparent pairs. Only ~2% require actual column reduction. This is why ripser is fast: it avoids almost all matrix algebra.

### 2.3 Edge Collapse

Before column reduction, the complex can be simplified by **edge collapse**: removing edges that provably don't affect persistence. giotto-ph implements this as a preprocessing step that can reduce the edge count by 66-86%.

## 3. flash-ph Pipeline: GPU Acceleration

flash-ph accelerates each stage of the Rips persistence pipeline:

```
                    ┌─────────────────────────────────┐
                    │        Point Cloud (GPU)         │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
         Stage 1    │   Triton Edge Kernel             │  O(n²/B) blocks
                    │   threshold: dist² < ε²          │  sparse output
                    │   → E edges, CSR adjacency       │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
         Stage 2    │   GPU Boruvka MST                │  O(E log n)
                    │   parallel union-find            │  → H0 diagram
                    └──────────────┬──────────────────┘
                                   │
                  ┌────────────────┼───────────────────┐
                  │ max_dim=1      │                    │ max_dim=2
                  ▼                │                    ▼
    ┌─────────────────────┐       │      ┌─────────────────────────┐
    │ Triton Triangle     │       │      │ GPU → COO sparse matrix │
    │ Enumeration         │       │      │ → giotto-ph C++ reduce  │
    │ (CSR merge-intersect)│      │      │ (collapse_edges=True)   │
    └─────────┬───────────┘       │      └────────────┬────────────┘
              │                   │                    │
    ┌─────────▼───────────┐       │                    │
    │ Triton Apparent     │       │                    │
    │ Pairs (~98%)        │       │                    │
    └─────────┬───────────┘       │                    │
              │                   │                    │
    ┌─────────▼───────────┐       │                    │
    │ Numba Residual      │       │                    │
    │ Reduction (~2%)     │       │                    │
    └─────────┬───────────┘       │                    │
              │                   │                    │
              ▼                   │                    ▼
         H1 diagram               │              H1, H2 diagrams
```

### Stage 1: GPU Edge Enumeration

**Problem**: enumerate all edges $(i,j)$ with $d(x_i, x_j) < \varepsilon$.

**Naive approach** (ripser, giotto-ph): compute the full $n \times n$ distance matrix on CPU, then filter. Cost: $O(n^2 d)$ time, $O(n^2)$ memory.

**flash-ph approach**: Triton kernel with block-parallel enumeration.

```
For each block of B=128 source points:
    For each block of B target points (j > i):
        Compute B×B pairwise distances (GPU GEMM-like)
        Mask: keep only dist² < threshold²
        Compact and write surviving edges
```

Key properties:
- **Never materializes the full distance matrix** — processes B×B tiles
- **Threshold applied during enumeration** — output is already sparse
- **GPU parallelism**: $\lceil n/B \rceil \times \lceil n/B \rceil$ blocks execute in parallel
- At 5% density, this outputs 50K edges from 200M possible pairs — only the sparse result is stored

The output includes a **CSR adjacency structure** (row pointers + column indices + edge ranks), which is needed for triangle enumeration and apparent pair detection.

### Stage 2: GPU Boruvka MST (H0)

H0 persistence is equivalent to computing a minimum spanning tree — each MST edge merges two components, creating a persistence bar.

**CPU approach** (Kruskal): sort edges, process one at a time with union-find. Inherently serial: $O(E \log E)$.

**GPU Boruvka** (flash-ph):

```
Repeat until one component remains:
    1. For each component, find the lightest outgoing edge (parallel scan)
    2. Merge components along selected edges (parallel union-find)
    3. Record merging events as H0 bars
```

Each round processes all components in parallel. Boruvka needs $O(\log n)$ rounds, each doing $O(E)$ work in parallel. The result is equivalent to Kruskal but exploits GPU parallelism.

### Stage 3a: Triton Triangle Enumeration

**Problem**: enumerate all triangles in the Rips complex with their maximum edge rank.

**Algorithm**: CSR merge-intersection. For each edge $(i,j)$:

```
For each edge (i,j):
    N(i) = neighbors of i (from CSR adjacency)
    N(j) = neighbors of j (from CSR adjacency)
    Triangles = {(i,j,k) : k ∈ N(i) ∩ N(j)}
```

The intersection $N(i) \cap N(j)$ is computed via merge of sorted adjacency lists — a classic CSR operation.

**Triton implementation**:
- BLOCK_T=64 edges per Triton program
- Each program loads $N(i)$ and $N(j)$ into shared memory
- Merge-intersection via two-pointer scan
- Outputs: triangle vertices $(v_0, v_1, v_2)$ and max edge rank

**Why Triton, not just PyTorch?** The merge-intersection has data-dependent control flow (pointer advances depend on comparison results). PyTorch has no efficient primitive for this. The Triton kernel expresses it naturally with `tl.where` and masked loads.

### Stage 3b: GPU Apparent Pair Pre-Detection

**Problem**: for each edge $e$, determine if it's an apparent pair.

**Algorithm**: for edge $e = (i,j)$ with rank $r_e$:

```
For each cofacet triangle t containing e:
    Let r_t = max edge rank of t
    If r_t is the smallest among all cofacets of e:
        Let e' = face of t with rank r_t
        If e' == e (i.e., e is the oldest face of t):
            e is an apparent pair with t
```

**Implementation**: flash-ph uses vectorized PyTorch operations on the cofacet CSR:
- `scatter_reduce` (min) over cofacet triangle ranks → find youngest cofacet per edge
- `searchsorted` on packed face keys → look up whether the edge is the oldest face
- Vectorized comparison → boolean mask of apparent pairs

This resolves ~98% of edges without any column reduction. The remaining ~2% are residual columns.

> **Note**: flash-tda (the parent project) implements this as a Triton kernel (`apparent_pair_kernel.py`, BLOCK_E=128). flash-ph uses the PyTorch path which is simpler and achieves similar performance since the scatter/searchsorted operations are already GPU-accelerated.

### Stage 3c: Adaptive Numba Residual Reduction

The ~2% of edges that are not apparent pairs require actual cohomology column reduction. This is done on CPU with Numba, with an adaptive time-budget fallback to giotto-ph:

```
For each non-apparent edge e (in reverse filtration order):
    column = coboundary of e
    While column has a pivot that collides with an existing column:
        XOR with the existing column (Z/2Z coefficients)
    If column is non-zero:
        Record pivot as a persistence pair
    Else:
        e is an essential feature (never killed)
```

The XOR operation is implemented via sorted merge (Numba `_sorted_merge_xor`). This is the only serial bottleneck, but it only processes ~2% of columns.

**Adaptive time-budget fallback** (`backend='auto'`): the reduction cost per column is topology-dependent and unpredictable — O(3) manifolds in d=9 can have columns that take 6000ms each while sphere data takes 0.25ms/column. Since no static metric predicts this, flash-ph uses time-budget monitoring:

1. Estimate giotto-ph cost: `E × _GIOTTO_US_PER_EDGE` (calibrated at ~7 μs/edge on A100)
2. Process residual columns in chunks of 100
3. Each Numba call gets a time budget equal to the remaining giotto-ph estimate
4. Within Numba, `time.perf_counter()` is checked every 100 XOR iterations via `objmode`
5. If the budget is exceeded mid-column, abort and fall back to giotto-ph for the full H1

This catches catastrophically expensive columns that no external post-call check can detect.

### Stage 4: giotto-ph C++ Reduction (H2, and H1 Fallback)

giotto-ph's C++ engine is used in two cases:
- **H2 (max_dim=2)**: always, since H2 apparent pair optimization is less effective
- **H1 fallback**: when the adaptive backend (`backend='auto'`) determines that Numba residual reduction is slower than giotto-ph, or when `backend='giotto'` is explicitly requested

The pipeline:
1. GPU-computed sparse edges → COO matrix on CPU
2. `gph.ripser_parallel(sparse_dm, collapse_edges=True)` — C++ lockfree parallel reduction

The key: GPU edge enumeration produces a **pre-thresholded sparse COO matrix** that giotto-ph's edge collapse can reduce very efficiently. Starting from sparse edges (vs a dense point cloud) means fewer simplices to collapse.

## 4. Complexity Analysis

### Per-Stage Costs

| Stage | CPU (ripser) | flash-ph | Improvement |
|-------|-------------|----------|-------------|
| Edge enumeration | $O(n^2 d)$ | $O(n^2 d / B^2)$ parallel | $B^2$ GPU blocks |
| H0 (MST) | $O(E \log E)$ serial | $O(E \log n)$ with $O(E)$ parallelism | GPU parallel |
| Triangle enumeration | implicit | $O(E \cdot \bar{d})$ parallel | GPU parallel |
| Apparent pairs | $O(E)$ serial | $O(E)$ GPU scatter/search | GPU-parallel |
| Residual reduction | $O(E_{res} \cdot \bar{c})$ | same (CPU), adaptive fallback | Only ~2% of $E$ |

Where:
- $B$ = block size for edge enumeration (128)
- $B_E$ = edges per Triton program for apparent pairs (128)
- $\bar{d}$ = average vertex degree
- $\bar{c}$ = average column density in reduction
- $E_{res} \approx 0.02 E$ = residual (non-apparent) edges

### Where flash-ph Wins and Loses

**Wins** (sparse regime, density < 30%):
- GPU edge enumeration dominates: $O(n^2/B^2)$ parallel vs $O(n^2)$ serial
- Apparent pair detection is embarrassingly parallel
- Boruvka MST has $O(E)$ parallelism per round
- Sparse COO handoff means giotto-ph starts from fewer simplices

**Loses** (dense regime, density > 40%):
- Column reduction becomes the bottleneck: $O(E_{res} \cdot \bar{c})$
- $\bar{c}$ grows with density (longer XOR chains)
- The CPU reduction is the same in flash-ph and ripser
- GPU preprocessing overhead (kernel launches, data transfers) adds latency with little benefit

**Crossover**: at ~30% edge density, the GPU preprocessing advantage is fully consumed by the column reduction cost. Above this, CPU methods with optimized BLAS win.

## 5. Correctness Guarantees

### Exact Parity with Ripser

flash-ph produces **identical** persistence diagrams to ripser (verified across 44+ test configurations). This is not approximate — the same bars, same birth/death values, to floating-point precision.

The parity comes from:
1. Same mathematical framework: Vietoris-Rips with $\mathbb{Z}/2\mathbb{Z}$ coefficients
2. Same algorithmic structure: cohomology reduction with apparent pairs
3. Same edge ordering: colex order within each filtration value
4. Careful floating-point handling: `float32` throughout (matching ripser's default)

### Threshold and Essential Bars

When computing with a finite threshold $\varepsilon$, features whose death time exceeds $\varepsilon$ appear as **essential bars** (death = $\infty$). These represent:
- H0: connected components that persist beyond $\varepsilon$ (always exactly 1 for a connected point cloud)
- H1: loops that are never filled by triangles within the threshold
- H2: voids that are never filled by tetrahedra within the threshold

For the Clifford torus $T^2 \subset \mathbb{R}^4$ with $n=500$:
- H1 features die at filtration ~1.23
- H2 feature dies at filtration ~1.28
- At threshold 1.1: H1 essential bars = 2, H2 essential bars = 1 → correct Betti numbers $(\beta_0, \beta_1, \beta_2) = (1, 2, 1)$

**Caveat**: The threshold at which essential bars stabilize to true Betti numbers is sample-dependent — it must exceed the birth time of the last significant feature, which varies with $n$ and the random seed.

## 6. Memory Model

### GPU Memory

The dominant GPU allocations:
- Edge arrays: $3E$ tensors (edge_i, edge_j, edge_dist_sq) — $O(E)$
- CSR adjacency: $O(n + E)$
- Triangle arrays: $4T$ tensors — $O(T)$
- Apparent pair workspace: $O(E)$

Total GPU memory: $O(E + T)$. For $n=20K$ at moderate threshold: ~50MB.

### CPU Memory

- COO sparse matrix (for giotto-ph): $O(E)$
- giotto-ph internal: $O(E + T)$ for reduction
- Numba residual buffers: $O(E_{res} \cdot \bar{c})$

### Why Not Fully GPU?

The column reduction (Numba residual + giotto-ph C++) remains on CPU because:
1. **Data-dependent control flow**: reduction involves chasing pivot chains that are inherently serial
2. **Sparse data structures**: columns are sparse with variable-length entries — GPU parallelism doesn't help when each thread follows a different-length chain
3. **Small workload**: only ~2% of columns for H1 residual; the overhead of GPU kernel launches would exceed the computation

The 98%/2% split (GPU apparent pairs / CPU residual) is the sweet spot: the GPU handles the embarrassingly parallel part, the CPU handles the inherently serial part.

### Scaling Limits (A100-80GB, H0+H1)

flash-ph targets $d \geq 4$ where Alpha complexes are unavailable. The practical ceiling depends critically on **graph density**, which in turn depends on the interplay between the data geometry and the threshold.

#### High-Dimensional Data (Sparse Regime)

In $d \geq 10$, distance concentration keeps the Rips graph naturally sparse at moderate thresholds. This is flash-ph's sweet spot:

| n | d | threshold | edges | time | GPU mem |
|---|---|-----------|-------|------|---------|
| 10,000 | 10 | 1.60 | 26K | 19ms | 19MB |
| 100,000 | 10 | 1.10 | 81K | 114ms | 266MB |
| 500,000 | 10 | 0.90 | 298K | 2.0s | 6.5GB |
| 50,000 | 50 | 6.20 | 31K | 134ms | 75MB |
| 200,000 | 50 | 6.00 | 176K | 1.7s | 1.1GB |

$d = 4$ with sparse thresholds (local structure, not full manifold topology):

| n | threshold | edges | time | GPU mem |
|---|-----------|-------|------|---------|
| 50,000 | 0.35 | 566K | 942ms | 857MB |
| 200,000 | 0.23 | 1.7M | 3.3s | 2.0GB |
| 1,000,000 | 0.11 | 2.3M | 4.9s | 26GB |

#### Low-Dimensional Manifolds (Dense Regime)

For data sampled from a low-dimensional manifold (e.g., the Clifford torus $T^2 \subset \mathbb{R}^4$, a 2D manifold in 4D ambient space), topology-preserving thresholds produce **dense** Rips graphs. This is a fundamental limitation of the Rips complex, not specific to flash-ph.

On the Clifford torus with threshold $\varepsilon = 1.10$ (needed for correct $\beta_1 = 2$ via essential bars):

| n | edges | density | triangles | topology correct? |
|---|-------|---------|-----------|-------------------|
| 500 | 29K | 23% | ~1.5M | Yes ($H_1$ ess = 2) |
| 1,000 | 117K | 23% | ~12M | Yes ($H_1$ ess = 2) |
| 2,000 | ~470K | 23% | ~42M | Triangle limit exceeded |

**Why**: the torus is a 2D manifold, so for a fixed threshold the number of neighbors per point is $O(n \cdot \varepsilon^2 / \text{area})$ — density stays at ~23% regardless of $n$. Edge count grows as $O(n^2)$ and triangle count as $O(n^3)$. All Rips-based methods (ripser, giotto-ph, flash-ph) hit this wall.

**Practical limits by data type:**

| Data type | Max n (correct topology) | Max n (sparse threshold) | Recommended alternative |
|-----------|------------------------|-------------------------|------------------------|
| Random $d \geq 10$ | ~500K | ~500K (same) | — |
| Random $d = 4$ | ~1K (topology-preserving) | ~1M (local features) | — |
| 2D manifold in $\mathbb{R}^4$ | **~1K** | ~1M (no topology) | Flood complex, Alpha ($d \leq 3$) |
| 1D manifold (circle) | ~500 (topology-preserving) | ~1M (local) | Alpha complex |

For large-scale manifold data where global topology matters, use the **Flood complex** (available in flash-tda as `flood_persistence`) which avoids the dense Rips problem by flooding a Delaunay triangulation of landmarks.

#### Hard Constraints

1. **H2 vertex limit**: $n < 65{,}536$ (16-bit packing in int64 keys). Does not apply to H0/H1.
2. **Triangle safety limit**: `max_triangles=10M` by default. Fires when the threshold produces a dense graph. Configurable via parameter.
3. **GPU memory**: dominant allocations are edge arrays ($3E$ tensors), CSR adjacency ($O(n+E)$), and triangle arrays ($4T$ tensors). At $n=1M$ $d=4$ with sparse threshold, this reaches 26GB.
4. **Practical ceiling**: $n \times \text{density}$ determines tractability. At 0.01% density, $n = 1M$ produces 2.3M edges (5 seconds). At 23% density, $n = 2K$ produces 42M triangles (infeasible).

## 7. Numerical Considerations

### Float32 Throughout

flash-ph uses `float32` for all distance computations, matching ripser's default. This means:
- Edge distances are `sqrt(float32)` — precision to ~7 decimal digits
- Filtration ordering is stable: ties broken by colex order on vertex indices
- No accumulated rounding: each distance is computed once and stored

### Colex Ordering

Within a filtration value, simplices are ordered by **colexicographic order** on their vertex indices. For edges: $(i,j) < (i',j')$ iff $j < j'$ or ($j = j'$ and $i < i'$). This is a convention shared with ripser and ensures deterministic output.

### Packed Integer Keys

For fast lookup of simplices (triangle → face ranks, cofacet adjacency), vertex tuples are packed into 64-bit integers:

```
key = v0 + (v1 << 16) + (v2 << 32)     # triangle
key = v0 + (v1 << 16) + (v2 << 32) + (v3 << 48)  # tetrahedron
```

This limits vertex count to $n < 65536$ (16 bits per vertex) but enables $O(\log n)$ lookup via `torch.searchsorted` instead of $O(1)$ Python dict lookup with enormous constant factor.

For tetrahedron colex ranks, naive computation of $\binom{n}{4}$ overflows `int64`. flash-ph uses early division: `v*(v-1)//2 * (v-2)*(v-3)//2 // 6`.

## 8. Related Work and Landscape

### 8.1 Ripser (Bauer, 2021)

The foundational algorithm that flash-ph builds on. Introduced three key optimizations: cohomology instead of homology (process $E$ columns instead of $T$), apparent pairs (~98% resolved without reduction), and implicit coboundary matrix (never materializes the boundary matrix). All subsequent Rips PH software builds on these ideas.

- Bauer, U. "Ripser: efficient computation of Vietoris-Rips persistence barcodes." *JACT*, 2021.

### 8.2 Ripser++ (Zhang et al., SoCG 2020)

The first GPU-accelerated Rips PH software. Parallelizes apparent pair detection on GPU using a 2-layer hashmap data structure. Achieves **30x over ripser**. Non-apparent columns fall back to CPU "submatrix reduction."

**Key difference from flash-ph**: Ripser++ still constructs the full $O(n^2)$ distance matrix on CPU before GPU processing. flash-ph moves edge enumeration itself to GPU (Triton), producing a pre-thresholded sparse edge list. Ripser++ also uses CUDA C++ (harder to modify/extend), while flash-ph uses Triton/PyTorch. Ripser++ is unmaintained (last commit 2021).

- Zhang, S. et al. "GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes." *SoCG*, 2020.

### 8.3 giotto-ph (Burella Schiavo et al., 2021)

Lockfree multicore C++ implementation of ripser + GUDHI edge collapse. Establishes the CPU state-of-the-art, surpassing Ripser++ with 5-10 CPU cores. flash-ph uses giotto-ph as its C++ backend for H1 fallback and H2 computation.

- Burella Perez, J., Hauke, S., Lupo, U., Caorsi, M., and Dassatti, A. "giotto-ph: A Python Library for High-Performance Computation of Persistent Homology of Vietoris-Rips Filtrations." *arXiv:2107.05412*, 2021.

### 8.4 Edge Collapse (Boissonnat & Pritam, SoCG 2020)

Shows that "dominated" edges in a flag complex can be removed using only the 1-skeleton, preserving persistent homology. Reduces edge count by 66-86% as a preprocessing step. Implemented in GUDHI and giotto-ph.

flash-ph exploits this via giotto-ph's `collapse_edges=True`: the GPU-computed sparse COO is the input, so edge collapse starts from fewer edges than a dense point cloud.

- Boissonnat, J.-D. and Pritam, S. "Edge Collapse and Persistence of Flag Complexes." *SoCG*, 2020.

### 8.5 SpecSeq++ (JPDC 2025)

The first fully GPU-parallelized *explicit* boundary matrix reduction. Uses spectral sequence decomposition to partition the boundary matrix into blocks with dynamic load balancing. Reports **62-88x average speedup** (up to 775x peak) over serial reduction.

This is the most promising complement to flash-ph: SpecSeq++ parallelizes the *reduction* stage (the part flash-ph delegates to CPU). Combining flash-ph's GPU sparse complex construction with SpecSeq++-style GPU reduction could eliminate the dense-regime bottleneck entirely.

- Li, Q., Huang, Z., Chen, Y., Hu, D., Dai, Z., Yu, M., and Liu, Z. "SpecSeq++: A high parallel boundary matrix reduction to support real large-scale point clouds." *JPDC*, Vol. 198, 2025.

### 8.6 Sparse Rips Approximations (Sheehy, 2013; Cavanna et al., 2015)

Constructs an $O(n)$-size filtered complex whose persistence diagram $(1+\epsilon)$-approximates the full Rips filtration. Uses greedy permutations and growing/shrinking balls. The constant depends on doubling dimension.

**Relation**: This is an *approximation* with linear complexity, while flash-ph computes *exact* persistence on a thresholded subcomplex. The sparsity structures differ fundamentally — geometric net-based sparsification vs threshold-based pruning.

- Sheehy, D. "Linear-Size Approximations to the Vietoris-Rips Filtration." *DCG*, 2013.
- Cavanna, N., Jahanseir, M., and Sheehy, D. "A Geometric Perspective on Sparse Filtrations." *CCCG*, 2015. arXiv:1506.03797.

### 8.7 Flood Complex (NeurIPS 2025)

Computes PH on millions of points by flooding a Delaunay triangulation of a small landmark subset. GPU-accelerated via PyTorch. Scales to $n > 10^6$ in 3D. flash-tda (the parent project) implements the Flood complex as `flood_persistence`.

- Graf, F., Pellizzoni, P., Uray, M., Huber, S., and Kwitt, R. "The Flood Complex: Large-Scale Persistent Homology on Millions of Points." *NeurIPS*, 2025. arXiv:2509.22432.

### 8.8 Differentiable PH — TopologyLayer, torchph

PyTorch layers that backpropagate through persistence diagrams. Not about speed per se, but about making PH usable in end-to-end gradient-based learning. flash-ph's kernels fall back to differentiable PyTorch when `requires_grad=True`, enabling integration.

- Brüel-Gabrielsson, R. et al. "A Topology Layer for Machine Learning." *AISTATS*, 2020.
- Hofer, C. et al. "torchph: PyTorch extensions for persistent homology."

### 8.9 Summary: Where flash-ph Sits

| Technique | Ripser | Ripser++ | giotto-ph | SpecSeq++ | Sparse Rips | flash-ph |
|-----------|--------|----------|-----------|-----------|-------------|----------|
| Edge enumeration | CPU $O(n^2)$ | CPU $O(n^2)$ | CPU $O(n^2)$ | CPU | CPU $O(n)$ approx | **GPU Triton** |
| H0 (MST) | CPU serial | CPU serial | CPU serial | — | — | **GPU Boruvka** |
| Apparent pairs | CPU serial | **GPU CUDA** | CPU parallel | — | — | **GPU PyTorch** |
| Boundary reduction | CPU serial | CPU serial | CPU lockfree | **GPU parallel** | CPU serial | CPU (Numba/giotto) |
| Edge collapse | No | No | **Yes (C++)** | No | N/A | Yes (via giotto) |
| Exact? | Yes | Yes | Yes | Yes | No (approx) | **Yes** |
| Sparse input? | COO | COO | Dense/COO | Dense | Geometric net | **Thresholded COO/CSR** |

**The main open direction**: combining flash-ph's GPU sparse complex construction (stages 1-3) with SpecSeq++'s GPU boundary matrix reduction (stage 4) would create a fully GPU-native exact Rips PH pipeline with no CPU bottleneck.
