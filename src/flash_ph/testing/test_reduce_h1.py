"""Tests for GPU-native H1 persistence reduction (reduce_h1_gpu.py).

Parity tests against ripser across various point cloud configurations.
"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _compare_h1_diagrams(gpu_dgm, ref_dgm, atol=1e-4):
    """Compare GPU H1 diagram with reference (ripser). Returns (ok, info)."""
    # Filter to finite positive-persistence bars
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
        return False, f"count mismatch: {gpu_sorted.shape[0]} vs {ref_sorted.shape[0]}"

    if gpu_sorted.size > 0:
        maxerr = float(np.max(np.abs(gpu_sorted - ref_sorted)))
    else:
        maxerr = 0.0

    ok = maxerr < atol
    return ok, f"maxerr={maxerr:.2e}, count={gpu_sorted.shape[0]}"


# ---------------------------------------------------------------------------
# Small cases (n=20-50)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 43, 44])
def test_h1_small_random_d2(seed):
    """Small 2D random point cloud, H1 parity vs ripser."""
    import ripser
    rng = np.random.default_rng(seed)
    pts_np = rng.standard_normal((30, 2)).astype(np.float32)
    thresh = 1.5

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


@pytest.mark.parametrize("seed", [42, 43])
def test_h1_small_random_d4(seed):
    """Small 4D random point cloud, H1 parity vs ripser."""
    import ripser
    rng = np.random.default_rng(seed)
    pts_np = rng.standard_normal((25, 4)).astype(np.float32)
    thresh = 2.0

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


def test_h1_small_d10():
    """Small 10D random, H1 parity."""
    import ripser
    rng = np.random.default_rng(42)
    pts_np = rng.standard_normal((20, 10)).astype(np.float32)
    thresh = 3.0

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


# ---------------------------------------------------------------------------
# Medium cases (n=100-500)
# ---------------------------------------------------------------------------

def test_h1_medium_clifford():
    """Clifford torus n=200, H1 parity."""
    import ripser
    n = 200
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    pts_np = np.column_stack([
        np.cos(theta), np.sin(theta),
        np.cos(phi), np.sin(phi),
    ]).astype(np.float32) / np.sqrt(2)
    thresh = 0.6

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


def test_h1_medium_random_n100():
    """Random n=100 d=3, H1 parity."""
    import ripser
    rng = np.random.default_rng(42)
    pts_np = rng.standard_normal((100, 3)).astype(np.float32)
    thresh = 1.5

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


def test_h1_medium_n200_d9():
    """O(3)-like n=200 d=9, H1 parity."""
    import ripser
    rng = np.random.default_rng(42)
    matrices = rng.standard_normal((200, 3, 3))
    pts_np = np.empty((200, 9), dtype=np.float32)
    for i in range(200):
        q, r = np.linalg.qr(matrices[i])
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        pts_np[i] = (q * d[np.newaxis, :]).ravel()
    thresh = 1.8

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_h1_no_edges():
    """Very small threshold → no edges → empty H1."""
    pts = torch.randn(50, 3, device="cuda")
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=1e-10, max_dim=1)
    assert dgms[1].shape[1] == 2
    assert dgms[1].shape[0] == 0


def test_h1_single_component_no_holes():
    """Tight cluster → single component, no H1 features."""
    import ripser
    rng = np.random.default_rng(42)
    pts_np = rng.standard_normal((10, 2)).astype(np.float32) * 0.01
    thresh = 0.5  # all pairwise distances well within threshold

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"


def test_h1_essential_features():
    """Circle with large threshold → should have essential H1."""
    import ripser
    n = 30
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    pts_np = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)
    thresh = 1.5  # partial connectivity

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed: {info}"

    # Also check essential (infinite) bars match
    gpu_inf = dgms[1][torch.isinf(dgms[1][:, 1])]
    ref_inf = ref["dgms"][1][np.isinf(ref["dgms"][1][:, 1])]
    assert gpu_inf.shape[0] == ref_inf.shape[0], (
        f"Essential H1 count mismatch: {gpu_inf.shape[0]} vs {ref_inf.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Standard benchmark configs
# ---------------------------------------------------------------------------

def test_h1_sphere3_192():
    """sphere3 192 points, no threshold, H1 parity."""
    import ripser
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(42)
    pts_np = rng.standard_normal((192, 4)).astype(np.float32)
    pts_np /= np.linalg.norm(pts_np, axis=1, keepdims=True)
    thresh = float(pdist(pts_np).max()) + 1e-3

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed for sphere3_192: {info}"


def test_h1_o3_1024():
    """O(3) 1024 points, thresh=1.8, H1 parity."""
    import ripser
    rng = np.random.default_rng(42)
    n = 1024
    matrices = rng.standard_normal((n, 3, 3))
    pts_np = np.empty((n, 9), dtype=np.float32)
    for i in range(n):
        q, r = np.linalg.qr(matrices[i])
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        pts_np[i] = (q * d[np.newaxis, :]).ravel()
    thresh = 1.8

    pts = torch.from_numpy(pts_np).cuda()
    from flash_ph.persistence import rips_persistence
    dgms = rips_persistence(pts, max_edge_length=thresh, max_dim=1)
    ref = ripser.ripser(pts_np, maxdim=1, thresh=thresh)

    ok, info = _compare_h1_diagrams(dgms[1], ref["dgms"][1])
    assert ok, f"H1 parity failed for o3_1024: {info}"
