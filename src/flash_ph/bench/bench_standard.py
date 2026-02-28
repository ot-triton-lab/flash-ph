"""Benchmark flash-ph on standard TDA datasets (ripser-benchmark / PH-roadmap).

Datasets (synthetically generated with matching distributions):
  - sphere3:   192 points on S^3 in R^4, no threshold, H0-H2
  - o3_1024:  1024 points on O(3) in R^9, thresh=1.8, H0-H2
  - o3_4096:  4096 points on O(3) in R^9, thresh=1.4, H0-H2
  - random16:   50 points in R^16, no threshold, H0-H2

These are the standard benchmark datasets used by Ripser, giotto-ph, and
Ripser++ papers.  Synthetic generation uses fixed seeds for reproducibility.
Point distributions match the originals (uniform on S^3, Haar on O(3),
Gaussian in R^16) but are not bit-identical to the downloaded files.

Usage:
    python -u -m flash_ph.bench.bench_standard
"""
import gc
import sys
import time

import numpy as np
import torch


NREPS = 3
NWARMUP = 1


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _median(times):
    s = sorted(times)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _generate_sphere3(n=192, seed=42):
    """Sample n points uniformly on S^3 in R^4 via normalized Gaussians."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 4)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _generate_o3(n, seed=42):
    """Sample n points from O(3) via QR of Gaussian matrices, flatten to R^9."""
    rng = np.random.default_rng(seed)
    matrices = rng.standard_normal((n, 3, 3))
    result = np.empty((n, 9), dtype=np.float32)
    for i in range(n):
        q, r = np.linalg.qr(matrices[i])
        # Ensure uniform Haar measure: fix signs so diag(R) > 0
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        q = q * d[np.newaxis, :]
        result[i] = q.ravel()
    return result


def _generate_random16(n=50, seed=42):
    """Sample n points from standard Gaussian in R^16."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 16)).astype(np.float32)


# ---------------------------------------------------------------------------
# Configs: (name, n, ambient_d, thresh_or_None, max_dim, generator_func)
# ---------------------------------------------------------------------------
CONFIGS = [
    # Original ripser-benchmark sizes
    ("sphere3_192",   192,  4, None, 2, lambda: _generate_sphere3(192)),
    ("random16_50",    50, 16, None, 2, lambda: _generate_random16(50)),
    ("o3_1024",      1024,  9, 1.8,  2, lambda: _generate_o3(1024)),
    ("o3_4096",      4096,  9, 1.4,  2, lambda: _generate_o3(4096)),
    # Scaled-up variants (>= 1000 points)
    ("sphere3_1K",   1000,  4, 0.8,  2, lambda: _generate_sphere3(1000, seed=43)),
    ("sphere3_5K",   5000,  4, 0.5,  2, lambda: _generate_sphere3(5000, seed=44)),
    ("random16_1K",  1000, 16, 3.5,  2, lambda: _generate_random16(1000, seed=43)),
    ("random16_2K",  2000, 16, 3.2,  2, lambda: _generate_random16(2000, seed=44)),
]


# ---------------------------------------------------------------------------
# Parity check vs CPU Ripser
# ---------------------------------------------------------------------------

def _compare_diagrams(gpu_dgm, ref_dgm, atol=1e-4):
    """Compare GPU and reference diagrams. Returns (ok, count_str, maxerr)."""
    # Filter to finite bars with positive persistence
    gpu_fin = gpu_dgm[torch.isfinite(gpu_dgm[:, 1])]
    if gpu_fin.numel() > 0:
        gpu_pos = gpu_fin[(gpu_fin[:, 1] - gpu_fin[:, 0]) > 1e-10]
    else:
        gpu_pos = gpu_fin

    ref_fin = ref_dgm[np.isfinite(ref_dgm[:, 1])]
    if ref_fin.shape[0] > 0:
        ref_pos = ref_fin[(ref_fin[:, 1] - ref_fin[:, 0]) > 1e-10]
    else:
        ref_pos = ref_fin

    # Sort by (birth, death)
    if gpu_pos.shape[0] > 0:
        gpu_np = gpu_pos.cpu().numpy()
        idx = np.lexsort((gpu_np[:, 1], gpu_np[:, 0]))
        gpu_sorted = gpu_np[idx]
    else:
        gpu_sorted = gpu_pos.cpu().numpy()

    if ref_pos.shape[0] > 0:
        idx = np.lexsort((ref_pos[:, 1], ref_pos[:, 0]))
        ref_sorted = ref_pos[idx]
    else:
        ref_sorted = ref_pos

    if gpu_sorted.shape != ref_sorted.shape:
        return False, f"{gpu_sorted.shape[0]}/{ref_sorted.shape[0]}", float("inf")

    if gpu_sorted.size > 0:
        maxerr = float(np.max(np.abs(gpu_sorted - ref_sorted)))
    else:
        maxerr = 0.0

    return maxerr < atol, str(gpu_sorted.shape[0]), maxerr


def run_parity():
    """Parity checks against CPU Ripser on all standard datasets."""
    import ripser
    from flash_ph.persistence import rips_persistence

    device = "cuda"

    print("=" * 70)
    print("PARITY: flash-ph GPU vs CPU Ripser (standard datasets)")
    print("=" * 70)
    print(f"{'dataset':>10} {'n':>5} {'d':>3} {'thresh':>6} {'dim':>4} "
          f"{'count':>6} {'maxerr':>10} {'status':>7}")
    print("-" * 60)
    _flush()

    all_ok = True
    for name, n, ambient_d, thresh, max_dim, gen_fn in CONFIGS:
        pts_np = gen_fn()

        # For no-threshold configs, use enclosing radius
        if thresh is None:
            from scipy.spatial.distance import pdist
            effective_thresh = float(pdist(pts_np).max()) + 1e-3
        else:
            effective_thresh = thresh

        pts_cuda = torch.from_numpy(pts_np).to(device)

        dgms = rips_persistence(
            pts_cuda, max_edge_length=effective_thresh,
            max_dim=max_dim,
        )
        result = ripser.ripser(pts_np, maxdim=max_dim, thresh=effective_thresh)

        for dim in range(max_dim + 1):
            ok, count_str, maxerr = _compare_diagrams(
                dgms[dim], result["dgms"][dim]
            )
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            thresh_str = f"{effective_thresh:.2f}" if thresh is not None else "inf"
            print(f"{name:>10} {n:5d} {ambient_d:3d} {thresh_str:>6} "
                  f"H{dim:>2} {count_str:>6} {maxerr:10.2e} {status:>7}")
        _flush()

    print("-" * 60)
    print(f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Performance benchmark: flash-ph vs Ripser vs Giotto-ph on standard datasets."""
    import ripser
    from flash_ph.persistence import rips_persistence

    device = "cuda"

    try:
        from gph import ripser_parallel
        has_giotto = True
    except ImportError:
        has_giotto = False

    try:
        from scipy.spatial.distance import pdist, squareform
        has_scipy = True
    except ImportError:
        has_scipy = False

    ncores = 8

    # Global warm-up: compile Triton kernels for high-d paths
    for wn, wd, wt in [(100, 4, 2.0), (100, 9, 3.0), (50, 16, 5.0)]:
        torch.manual_seed(0)
        _ = rips_persistence(
            torch.randn(wn, wd, device=device),
            max_edge_length=wt, max_dim=2,
        )
        torch.cuda.synchronize()

    # Warm-up Ripser + Giotto once
    _warmup_np = np.random.randn(100, 4).astype(np.float32)
    ripser.ripser(_warmup_np, maxdim=2, thresh=2.0)
    if has_giotto:
        ripser_parallel(_warmup_np, maxdim=2, thresh=2.0, n_threads=ncores)

    print("=" * 80)
    print("PERFORMANCE: standard TDA datasets (flash-ph vs Ripser vs Giotto-ph)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  max_dim=2 (H0+H1+H2)")
    if has_giotto:
        print(f"  Giotto-ph: {ncores} CPU threads")
    print(f"  Repetitions: {NREPS} (median), warm-up: {NWARMUP}")
    print("=" * 80)

    header = f"{'dataset':>10} {'n':>5} {'d':>3} {'thresh':>6} | {'ripser':>8} "
    if has_giotto:
        header += f"{'giotto':>8} "
    header += f"{'flash':>8} | {'vs rip':>7}"
    if has_giotto:
        header += f" {'vs gio':>7}"
    print(header)
    print("-" * len(header))
    _flush()

    for name, n, ambient_d, thresh, max_dim, gen_fn in CONFIGS:
        pts_np = gen_fn()

        # Effective threshold
        if thresh is None and has_scipy:
            effective_thresh = float(pdist(pts_np).max()) + 1e-3
        elif thresh is None:
            # Fallback: brute force max pairwise distance
            pts_t = torch.from_numpy(pts_np)
            dists = torch.cdist(pts_t, pts_t)
            effective_thresh = float(dists.max().item()) + 1e-3
            del pts_t, dists
        else:
            effective_thresh = thresh

        pts_cuda = torch.from_numpy(pts_np).to(device)

        # Per-config warm-up
        for wi in range(NWARMUP):
            gc.collect()
            torch.cuda.empty_cache()
            rips_persistence(
                pts_cuda, max_edge_length=effective_thresh,
                max_dim=max_dim,
            )
            torch.cuda.synchronize()
            ripser.ripser(pts_np, maxdim=max_dim, thresh=effective_thresh)
            if has_giotto:
                if has_scipy:
                    dm = squareform(pdist(pts_np))
                    ripser_parallel(
                        dm, maxdim=max_dim, thresh=effective_thresh,
                        metric="precomputed", collapse_edges=True,
                        n_threads=ncores,
                    )
                else:
                    ripser_parallel(
                        pts_np, maxdim=max_dim, thresh=effective_thresh,
                        n_threads=ncores,
                    )

        # Timed runs
        times_rip = []
        times_gio = []
        times_flash = []

        for rep in range(NREPS):
            # flash-ph GPU
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rips_persistence(
                pts_cuda, max_edge_length=effective_thresh,
                max_dim=max_dim,
            )
            torch.cuda.synchronize()
            times_flash.append(time.perf_counter() - t0)

            # Ripser
            gc.collect()
            t0 = time.perf_counter()
            ripser.ripser(pts_np, maxdim=max_dim, thresh=effective_thresh)
            times_rip.append(time.perf_counter() - t0)

            # Giotto-ph
            if has_giotto:
                gc.collect()
                t0 = time.perf_counter()
                if has_scipy:
                    dm = squareform(pdist(pts_np))
                    ripser_parallel(
                        dm, maxdim=max_dim, thresh=effective_thresh,
                        metric="precomputed", collapse_edges=True,
                        n_threads=ncores,
                    )
                else:
                    ripser_parallel(
                        pts_np, maxdim=max_dim, thresh=effective_thresh,
                        n_threads=ncores,
                    )
                times_gio.append(time.perf_counter() - t0)

        t_flash = _median(times_flash)
        t_rip = _median(times_rip)
        vs_rip = t_rip / t_flash if t_flash > 0 else float("inf")

        thresh_str = f"{effective_thresh:.2f}" if thresh is not None else "inf"
        row = f"{name:>10} {n:5d} {ambient_d:3d} {thresh_str:>6} | {t_rip*1000:7.0f}ms "
        if has_giotto:
            t_gio = _median(times_gio)
            row += f"{t_gio*1000:7.0f}ms "
        row += f"{t_flash*1000:7.0f}ms | {vs_rip:6.1f}x"
        if has_giotto:
            vs_gio = t_gio / t_flash
            row += f" {vs_gio:6.1f}x"
        print(row)
        _flush()

        del pts_cuda

    # Published reference numbers
    print()
    print("Published reference numbers (for comparison):")
    print("-" * 60)
    print("  From giotto-ph paper (Table 3, Xeon Gold, Ripser v1.2):")
    print("    sphere3 H2:   Ripser 0.7s, giotto-ph(8t) 0.4s")
    print("    o3_1024 H3:   Ripser 2.3s, giotto-ph(8t) 0.5s")
    print("    o3_4096 H3:   Ripser 45.6s, giotto-ph(8t) 9.3s")
    print("    random16 H7:  Ripser 3.9s, giotto-ph(8t) 1.0s")
    print("  From Ripser paper (Table 1, 4GHz Core i7):")
    print("    sphere3 H2:   Ripser 1.1s, GUDHI 46.7s")
    print("  Note: our max_dim=2 (H0-H2), published go to H3/H7.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    ok = run_parity()
    if not ok:
        print("PARITY FAILED -- skipping performance benchmark.")
        return

    run_benchmark()


if __name__ == "__main__":
    main()
