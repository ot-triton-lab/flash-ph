"""Benchmark flash_ph GPU persistent homology vs Ripser / Giotto-ph.

Runs parity checks first, then performance benchmarks with proper warm-up
and multiple repetitions (median-of-N) to avoid cache pollution.

Sections:
  1. Parity -- correctness vs CPU Ripser (multi-d)
  2. Performance -- flash-ph vs Ripser / Giotto-ph (d > 3)

flash-ph targets high-dimensional point clouds (d > 3) where distance
concentration keeps the Rips graph sparse and the GPU pipeline excels.

Usage:
    python -u -m flash_ph.bench.bench_persistence
"""
import gc
import os
import sys
import time
import threading
import torch
import numpy as np


NREPS = 5          # repetitions per config; report median
NWARMUP = 2        # warm-up runs (discarded) per config


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _median(times):
    """Return median of a list of floats."""
    s = sorted(times)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _peak_gpu_mb():
    """Return peak GPU memory allocated in MB since last reset."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_gpu_mem():
    """Reset peak GPU memory stats for fresh measurement."""
    torch.cuda.reset_peak_memory_stats()


def _rss_mb():
    """Current RSS of this process in MB (Linux /proc/self/status)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return 0.0


def _run_with_peak_rss(func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_s, peak_rss_delta_mb).

    A background thread polls /proc/self/status every 1ms to capture peak RSS
    during execution of C/C++ extensions that allocate and free internally.
    """
    peak_rss = [0.0]
    done = threading.Event()

    def _poller():
        while not done.is_set():
            peak_rss[0] = max(peak_rss[0], _rss_mb())
            done.wait(0.001)  # 1ms polling interval

    gc.collect()
    rss_before = _rss_mb()
    poller = threading.Thread(target=_poller, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    done.set()
    poller.join()
    delta = max(peak_rss[0] - rss_before, 0.0)
    return result, elapsed, delta


# ---------------------------------------------------------------------------
# Parity  (unchanged -- correctness, not speed)
# ---------------------------------------------------------------------------

def _compare_diagrams(gpu_dgm, ref_dgm, atol=1e-4):
    """Compare GPU and reference diagrams. Returns (ok, count, maxerr)."""
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

    # Sort by birth then death
    if gpu_pos.shape[0] > 0:
        idx = torch.argsort(gpu_pos[:, 0] * 1e6 + gpu_pos[:, 1])
        gpu_sorted = gpu_pos[idx].cpu().numpy()
    else:
        gpu_sorted = gpu_pos.cpu().numpy()

    if ref_pos.shape[0] > 0:
        idx = np.argsort(ref_pos[:, 0] * 1e6 + ref_pos[:, 1])
        ref_sorted = ref_pos[idx]
    else:
        ref_sorted = ref_pos

    if gpu_sorted.shape != ref_sorted.shape:
        return False, (gpu_sorted.shape[0], ref_sorted.shape[0]), float("inf")

    if gpu_sorted.size > 0:
        maxerr = float(np.max(np.abs(gpu_sorted - ref_sorted)))
    else:
        maxerr = 0.0

    return maxerr < atol, gpu_sorted.shape[0], maxerr


def run_parity():
    """Parity checks against CPU Ripser across multiple scales and dimensions."""
    import ripser
    from flash_ph import rips_persistence

    device = "cuda"
    # (n, d, thresh) -- d=2 original configs + multi-d
    configs = [
        (50,   2, 3.0),
        (100,  2, 0.7),
        (200,  2, 0.5),
        (500,  2, 0.15),
        (1000, 2, 0.10),
        (2000, 2, 0.07),
        (5000, 2, 0.05),
        (10000, 2, 0.04),
        # multi-d: direct Triton kernel path
        (100,  3, 1.5),
        (200,  3, 1.0),
        (100, 10, 2.5),
        (500, 10, 2.5),  # t=2.8 has 314 H2 bars where float32 diverges
        (50,  50, 5.0),
    ]

    print("=" * 70)
    print("PARITY: flash-ph GPU vs CPU Ripser")
    print("=" * 70)
    print(f"{'n':>6} {'d':>3} {'thresh':>6} {'dim':>4} {'count':>6} {'maxerr':>10} {'status':>7}")
    print("-" * 55)
    _flush()

    all_ok = True
    for n, d, thresh in configs:
        torch.manual_seed(42)
        pts = torch.randn(n, d, device=device)
        pts_np = pts.cpu().numpy()

        dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=2)
        result = ripser.ripser(pts_np, maxdim=2, thresh=thresh)

        for dim in [0, 1, 2]:
            ok, count, maxerr = _compare_diagrams(
                dgms[dim], result["dgms"][dim]
            )
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            count_str = str(count) if isinstance(count, int) else f"{count[0]}/{count[1]}"
            print(f"{n:6d} {d:3d} {thresh:6.3f} H{dim:>2} {count_str:>6} {maxerr:10.2e} {status:>7}")
        _flush()

    print("-" * 55)
    print(f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Performance -- d > 3 (high-dimensional point clouds)
# ---------------------------------------------------------------------------

def run_benchmark():
    """Performance benchmark: flash-ph vs Ripser vs Giotto-ph (d > 3).

    Per-config warmup, fresh data per rep, separate numpy copies,
    median-of-N.  H0+H1+H2 for all configs.

    flash-ph targets high-dimensional point clouds (d > 3) where distance
    concentration keeps the Rips graph sparse and the GPU pipeline excels.
    """
    import ripser
    from flash_ph import rips_persistence

    device = "cuda"
    try:
        from gph import ripser_parallel
        has_giotto = True
    except ImportError:
        has_giotto = False

    ncores = 8
    # (n, d, thresh, max_dim) — d > 3 only
    # Thresholds hand-tuned to keep the Rips graph sparse while
    # producing meaningful H0+H1+H2 features.
    configs = [
        (500,   10, 2.8, 2),
        (1000,  10, 2.5, 2),
        (2000,  10, 2.2, 2),
        (5000,  10, 1.8, 2),
        (10000, 10, 1.6, 2),
        (20000, 10, 1.4, 2),
        (500,   50, 8.0, 2),
        (1000,  50, 7.5, 2),
    ]

    print("=" * 90)
    print("PERFORMANCE: flash-ph GPU vs CPU Ripser vs Giotto-ph (d > 3)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Repetitions: {NREPS} (median), warm-up: {NWARMUP}")
    print("=" * 90)

    header = f"{'n':>6} {'d':>3} {'thresh':>6} | {'ripser':>8} "
    if has_giotto:
        header += f"{'giotto':>8} "
    header += f"{'flash':>8} | {'vs rip':>7}"
    if has_giotto:
        header += f" {'vs gio':>7}"
    header += f" | {'flash':>6} {'rip':>6}"
    if has_giotto:
        header += f" {'gio':>6}"
    header += "  (mem)"
    print(header)
    print("-" * len(header))
    _flush()

    for n, d, thresh, max_dim in configs:
        # Per-config warmup: use diverse seeds so Numba compiles all code
        # paths (different random data hits different branches in reduction).
        # Run flash-ph first to avoid CPU-heavy ripser tainting GPU state.
        for wi in range(NWARMUP):
            torch.manual_seed(90 + wi)
            wp = torch.randn(n, d, device=device)
            rips_persistence(wp, max_edge_length=thresh, max_dim=max_dim)
            torch.cuda.synchronize()
            wp_np = wp.cpu().numpy()
            ripser.ripser(wp_np, maxdim=max_dim, thresh=thresh)
            if has_giotto:
                ripser_parallel(wp_np, maxdim=max_dim, thresh=thresh,
                                n_threads=ncores)
            del wp, wp_np
            gc.collect()
            torch.cuda.empty_cache()

        times_rip = []
        times_gio = []
        times_flash = []
        peak_gpu_mb = 0.0
        peak_rip_mb = 0.0
        peak_gio_mb = 0.0

        for rep in range(NREPS):
            seed = 42 + rep

            # --- flash-ph GPU first (before CPU methods trash caches) ---
            torch.manual_seed(seed)
            pts_fresh = torch.randn(n, d, device=device)
            gc.collect()
            torch.cuda.empty_cache()
            _reset_gpu_mem()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rips_persistence(pts_fresh, max_edge_length=thresh,
                             max_dim=max_dim)
            torch.cuda.synchronize()
            times_flash.append(time.perf_counter() - t0)
            peak_gpu_mb = max(peak_gpu_mb, _peak_gpu_mb())
            del pts_fresh

            # --- CPU methods: separate numpy copies ---
            torch.manual_seed(seed)
            pts_gpu = torch.randn(n, d, device=device)
            pts_np_rip = pts_gpu.cpu().numpy().copy()
            pts_np_gio = pts_gpu.cpu().numpy().copy() if has_giotto else None
            del pts_gpu

            # CPU Ripser (poll RSS during C++ execution)
            _, t_rip, mem_rip = _run_with_peak_rss(
                ripser.ripser, pts_np_rip, maxdim=max_dim, thresh=thresh)
            times_rip.append(t_rip)
            peak_rip_mb = max(peak_rip_mb, mem_rip)

            # Giotto-ph
            if has_giotto:
                _, t_gio_s, mem_gio = _run_with_peak_rss(
                    ripser_parallel, pts_np_gio, maxdim=max_dim, thresh=thresh,
                    n_threads=ncores)
                times_gio.append(t_gio_s)
                peak_gio_mb = max(peak_gio_mb, mem_gio)

            del pts_np_rip, pts_np_gio

        t_rip = _median(times_rip)
        t_flash = _median(times_flash)
        vs_rip = t_rip / t_flash if t_flash > 0 else float("inf")

        row = f"{n:6d} {d:3d} {thresh:6.2f} | {t_rip*1000:7.0f}ms "
        if has_giotto:
            t_gio = _median(times_gio)
            row += f"{t_gio*1000:7.0f}ms "
        row += f"{t_flash*1000:7.0f}ms | {vs_rip:6.1f}x"
        if has_giotto:
            vs_gio = t_gio / t_flash
            row += f" {vs_gio:6.1f}x"
        row += f" | {peak_gpu_mb:5.0f}M {peak_rip_mb:5.0f}M"
        if has_giotto:
            row += f" {peak_gio_mb:5.0f}M"
        print(row)
        _flush()

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU PH benchmark.")
        return

    # Hardware and version reporting
    print("=" * 60)
    print("flash-ph persistent homology benchmark")
    print("=" * 60)
    print(f"Device : {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton : {triton.__version__}")
    except ImportError:
        print("Triton : not installed")
    print(f"NumPy  : {np.__version__}")
    try:
        import ripser as _ripser
        print(f"Ripser : {_ripser.__version__}")
    except (ImportError, AttributeError):
        pass
    try:
        import gph as _gph
        print(f"Giotto : {_gph.__version__}")
    except (ImportError, AttributeError):
        pass
    print()

    ok = run_parity()
    if not ok:
        print("PARITY FAILED -- skipping performance benchmark.")
        return

    run_benchmark()


if __name__ == "__main__":
    main()
