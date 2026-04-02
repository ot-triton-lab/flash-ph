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
