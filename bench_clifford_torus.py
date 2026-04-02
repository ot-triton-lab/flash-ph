"""Benchmark flash-ph on the Clifford torus (T^2 ⊂ R^4), H0+H1+H2.

The Clifford torus is (cos θ, sin θ, cos φ, sin φ)/√2 — a flat torus
embedded in S^3 ⊂ R^4. Expected topology: H0=Z, H1=Z², H2=Z.

Compares: flash-ph (GPU) vs ripser (CPU) vs giotto-ph (CPU, 8 threads).

Usage:
    conda activate flash-tda
    python -u bench_clifford_torus.py
"""
import gc
import os
import sys
import time
import threading
import numpy as np
import torch


NREPS = 5
NWARMUP = 2


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()


def _median(times):
    s = sorted(times)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_gpu_mem():
    torch.cuda.reset_peak_memory_stats()


def _rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _run_with_peak_rss(func, *args, **kwargs):
    peak_rss = [0.0]
    done = threading.Event()

    def _poller():
        while not done.is_set():
            peak_rss[0] = max(peak_rss[0], _rss_mb())
            done.wait(0.001)

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


def clifford_torus(n, noise=0.02, seed=42):
    """Sample n points from the Clifford torus T^2 ⊂ R^4 with Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(t), np.sin(t), np.cos(s), np.sin(s)]) / np.sqrt(2)
    pts += rng.standard_normal((n, 4)) * noise
    return pts.astype(np.float32)


def _compare_diagrams(gpu_dgm, ref_dgm, atol=1e-4):
    """Compare GPU and reference diagrams. Returns (ok, count, maxerr)."""
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


def run_parity(configs):
    """Parity checks against CPU Ripser on Clifford torus."""
    import ripser
    from flash_ph import rips_persistence

    device = "cuda"

    print("=" * 70)
    print("PARITY: flash-ph GPU vs CPU Ripser (Clifford torus T² ⊂ R⁴)")
    print("=" * 70)
    print(f"{'n':>6} {'thresh':>6} {'dim':>4} {'count':>6} {'maxerr':>10} {'status':>7}")
    print("-" * 50)
    _flush()

    all_ok = True
    for n, thresh, max_dim in configs:
        pts_np = clifford_torus(n)
        pts_gpu = torch.tensor(pts_np, device=device)

        dgms = rips_persistence(pts_gpu, max_edge_length=thresh, max_dim=max_dim)
        result = ripser.ripser(pts_np, maxdim=max_dim, thresh=thresh)

        for dim in range(max_dim + 1):
            ok, count, maxerr = _compare_diagrams(dgms[dim], result["dgms"][dim])
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            count_str = str(count) if isinstance(count, int) else f"{count[0]}/{count[1]}"
            print(f"{n:6d} {thresh:6.3f} H{dim:>2} {count_str:>6} {maxerr:10.2e} {status:>7}")
        _flush()

    print("-" * 50)
    print(f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    return all_ok


def run_benchmark(configs):
    """Performance benchmark: flash-ph vs Ripser vs Giotto-ph on Clifford torus."""
    import ripser
    from flash_ph import rips_persistence

    device = "cuda"
    try:
        from gph import ripser_parallel
        has_giotto = True
    except ImportError:
        has_giotto = False

    ncores = 8

    print("=" * 95)
    print("PERFORMANCE: flash-ph GPU vs Ripser vs Giotto-ph  (Clifford torus T² ⊂ R⁴, H0+H1+H2)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Repetitions: {NREPS} (median), warm-up: {NWARMUP}")
    print("=" * 95)

    header = f"{'n':>6} {'thresh':>6} {'dim':>3} | {'ripser':>8} "
    if has_giotto:
        header += f"{'giotto':>8} "
    header += f"{'flash':>8} | {'vs rip':>7}"
    if has_giotto:
        header += f" {'vs gio':>7}"
    header += f" | {'H1':>4} {'H2':>3} | {'flash':>6} {'rip':>6}"
    if has_giotto:
        header += f" {'gio':>6}"
    header += "  (mem MB)"
    print(header)
    print("-" * len(header))
    _flush()

    for n, thresh, max_dim in configs:
        # Warmup
        for wi in range(NWARMUP):
            pts_np = clifford_torus(n, seed=90 + wi)
            pts_gpu = torch.tensor(pts_np, device=device)
            rips_persistence(pts_gpu, max_edge_length=thresh, max_dim=max_dim)
            torch.cuda.synchronize()
            ripser.ripser(pts_np, maxdim=max_dim, thresh=thresh)
            if has_giotto:
                ripser_parallel(pts_np, maxdim=max_dim, thresh=thresh, n_threads=ncores)
            del pts_gpu
            gc.collect()
            torch.cuda.empty_cache()

        times_rip = []
        times_gio = []
        times_flash = []
        peak_gpu_mb = 0.0
        peak_rip_mb = 0.0
        peak_gio_mb = 0.0
        h1_count = 0
        h2_count = 0

        for rep in range(NREPS):
            pts_np = clifford_torus(n, seed=42 + rep)
            pts_gpu = torch.tensor(pts_np, device=device)

            # --- flash-ph GPU ---
            gc.collect()
            torch.cuda.empty_cache()
            _reset_gpu_mem()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dgms = rips_persistence(pts_gpu, max_edge_length=thresh, max_dim=max_dim)
            torch.cuda.synchronize()
            times_flash.append(time.perf_counter() - t0)
            peak_gpu_mb = max(peak_gpu_mb, _peak_gpu_mb())
            if rep == 0:
                h1_count = dgms[1].shape[0] if len(dgms) > 1 else 0
                h2_count = dgms[2].shape[0] if len(dgms) > 2 else 0
            del dgms, pts_gpu

            # --- CPU Ripser ---
            pts_rip = pts_np.copy()
            _, t_rip, mem_rip = _run_with_peak_rss(
                ripser.ripser, pts_rip, maxdim=max_dim, thresh=thresh)
            times_rip.append(t_rip)
            peak_rip_mb = max(peak_rip_mb, mem_rip)

            # --- Giotto-ph ---
            if has_giotto:
                pts_gio = pts_np.copy()
                _, t_gio, mem_gio = _run_with_peak_rss(
                    ripser_parallel, pts_gio, maxdim=max_dim, thresh=thresh,
                    n_threads=ncores)
                times_gio.append(t_gio)
                peak_gio_mb = max(peak_gio_mb, mem_gio)

        t_rip = _median(times_rip)
        t_flash = _median(times_flash)
        vs_rip = t_rip / t_flash if t_flash > 0 else float("inf")

        dim_str = f"H0-{max_dim}"
        row = f"{n:6d} {thresh:6.2f} {dim_str:>3} | {t_rip*1000:7.0f}ms "
        if has_giotto:
            t_gio = _median(times_gio)
            row += f"{t_gio*1000:7.0f}ms "
        row += f"{t_flash*1000:7.0f}ms | {vs_rip:6.1f}x"
        if has_giotto:
            vs_gio = t_gio / t_flash
            row += f" {vs_gio:6.1f}x"
        row += f" | {h1_count:4d} {h2_count:3d} | {peak_gpu_mb:5.0f}M {peak_rip_mb:5.0f}M"
        if has_giotto:
            row += f" {peak_gio_mb:5.0f}M"
        print(row)
        _flush()

    print()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print("=" * 60)
    print("flash-ph benchmark: Clifford torus T² ⊂ R⁴")
    print("=" * 60)
    print(f"Device : {torch.cuda.get_device_name()}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU mem: {gpu_mem:.0f} GB")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton; print(f"Triton : {triton.__version__}")
    except ImportError:
        print("Triton : not installed")
    print(f"NumPy  : {np.__version__}")
    try:
        import ripser as _r; print(f"Ripser : {_r.__version__}")
    except (ImportError, AttributeError):
        pass
    try:
        import gph as _g; print(f"Giotto : {_g.__version__}")
    except (ImportError, AttributeError):
        pass
    print()

    # Configs: (n, thresh, max_dim)
    # H0+H1+H2 for n<=5K, H0+H1 for n>=10K (tet count explodes in d=4)
    configs = [
        (500,  0.50, 2),
        (1000, 0.40, 2),
        (2000, 0.30, 2),
        (5000, 0.25, 2),
        (10000, 0.16, 1),
        (20000, 0.12, 1),
    ]

    parity_configs = [(n, t, md) for n, t, md in configs[:4]]
    ok = run_parity(parity_configs)
    if not ok:
        print("PARITY FAILED -- skipping performance benchmark.")
        return

    run_benchmark(configs)


if __name__ == "__main__":
    main()
