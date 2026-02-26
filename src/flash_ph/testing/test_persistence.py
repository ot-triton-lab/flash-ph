"""End-to-end tests for flash-ph: parity with ripser."""
import pytest
import torch
import numpy as np

from flash_ph import rips_persistence, auto_threshold, enclosing_radius, RipsComplex

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
device = "cuda"


def _compare_diagrams(gpu_dgm, ripser_dgm, atol=1e-4):
    """Compare GPU and ripser diagrams after filtering and sorting."""
    # Filter to finite bars with positive persistence
    gpu_finite = gpu_dgm[torch.isfinite(gpu_dgm[:, 1])]
    if gpu_finite.numel() > 0:
        gpu_pos = gpu_finite[(gpu_finite[:, 1] - gpu_finite[:, 0]) > 1e-10]
    else:
        gpu_pos = gpu_finite

    rip_finite_mask = np.isfinite(ripser_dgm[:, 1])
    rip_finite = ripser_dgm[rip_finite_mask]
    if rip_finite.shape[0] > 0:
        rip_pos = rip_finite[(rip_finite[:, 1] - rip_finite[:, 0]) > 1e-10]
    else:
        rip_pos = rip_finite

    # Sort by birth then death
    if gpu_pos.shape[0] > 0:
        sort_idx = torch.argsort(gpu_pos[:, 0] * 1e6 + gpu_pos[:, 1])
        gpu_sorted = gpu_pos[sort_idx].cpu()
    else:
        gpu_sorted = gpu_pos.cpu()

    if rip_pos.shape[0] > 0:
        sort_idx = np.argsort(rip_pos[:, 0] * 1e6 + rip_pos[:, 1])
        rip_sorted = torch.tensor(rip_pos[sort_idx], dtype=torch.float32)
    else:
        rip_sorted = torch.tensor(rip_pos, dtype=torch.float32)

    assert gpu_sorted.shape == rip_sorted.shape, (
        f"Shape mismatch: GPU {gpu_sorted.shape} vs ripser {rip_sorted.shape}"
    )

    if gpu_sorted.numel() > 0:
        torch.testing.assert_close(gpu_sorted, rip_sorted, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------------------
# H0 + H1 parity tests
# ---------------------------------------------------------------------------

def test_parity_ripser_n50():
    """H0 + H1 parity with ripser, n=50."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n, d = 50, 2
    pts = torch.randn(n, d, device=device)
    threshold = 3.0

    dgms = rips_persistence(pts, max_edge_length=threshold)
    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)

    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


def test_parity_ripser_n100():
    """H0 + H1 parity with ripser, n=100 with tight threshold."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n, d = 100, 2
    pts = torch.randn(n, d, device=device)
    threshold = 0.7

    dgms = rips_persistence(pts, max_edge_length=threshold)
    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)

    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


def test_circle_prominent_h1():
    """Circle point cloud: H1 parity with ripser."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n = 12
    theta = torch.linspace(0, 2 * 3.14159265, n + 1, device=device)[:n]
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    pts += torch.randn_like(pts) * 0.02
    threshold = 1.8

    dgms = rips_persistence(pts, max_edge_length=threshold)
    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)

    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_max_dim_0():
    """max_dim=0: only H0, no H1."""
    torch.manual_seed(42)
    pts = torch.randn(20, 2, device=device)

    dgms = rips_persistence(pts, max_edge_length=2.0, max_dim=0)
    assert len(dgms) == 1, f"Expected 1 diagram, got {len(dgms)}"
    assert dgms[0].shape[1] == 2


def test_empty_and_single():
    """Edge cases: n=0 and n=1."""
    # n=0
    pts0 = torch.empty(0, 3, device=device)
    dgms0 = rips_persistence(pts0, max_edge_length=1.0)
    assert len(dgms0) == 2
    assert dgms0[0].shape == (0, 2)
    assert dgms0[1].shape == (0, 2)

    # n=1
    pts1 = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    dgms1 = rips_persistence(pts1, max_edge_length=1.0)
    assert len(dgms1) == 2
    assert dgms1[0].shape == (1, 2)
    assert dgms1[0][0, 0].item() == 0.0
    assert dgms1[0][0, 1].item() == float('inf')
    assert dgms1[1].shape == (0, 2)


def test_device_consistency():
    """Output device matches input device."""
    pts = torch.randn(20, 2, device=device)
    dgms = rips_persistence(pts, max_edge_length=2.0)
    for dgm in dgms:
        assert dgm.device.type == "cuda", f"Expected cuda, got {dgm.device}"


# ---------------------------------------------------------------------------
# H2 tests
# ---------------------------------------------------------------------------

def test_parity_ripser_h2_n30():
    """H2 parity with ripser on small 3D cloud (n=30, loose threshold)."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n, d = 30, 3
    pts = torch.randn(n, d, device=device)
    threshold = 2.0

    dgms = rips_persistence(pts, max_edge_length=threshold, max_dim=2)
    assert len(dgms) == 3, f"Expected 3 diagrams, got {len(dgms)}"

    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=2, thresh=threshold)

    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)
    _compare_diagrams(dgms[2], result["dgms"][2], atol=1e-4)


def test_parity_ripser_h2_n50_tight():
    """H2 parity with ripser, n=50 tight threshold (sparse graph)."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(123)
    n, d = 50, 3
    pts = torch.randn(n, d, device=device)
    threshold = 1.2

    dgms = rips_persistence(pts, max_edge_length=threshold, max_dim=2)
    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=2, thresh=threshold)

    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)
    _compare_diagrams(dgms[2], result["dgms"][2], atol=1e-4)


def test_sphere_h2():
    """Points on a sphere: should have a prominent H2 feature (void)."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n = 20
    pts = torch.randn(n, 3, device=device)
    pts = pts / pts.norm(dim=1, keepdim=True)
    pts += torch.randn_like(pts) * 0.05
    threshold = 2.0

    dgms = rips_persistence(pts, max_edge_length=threshold, max_dim=2)
    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=2, thresh=threshold)

    _compare_diagrams(dgms[2], result["dgms"][2], atol=1e-4)

    h2 = dgms[2]
    h2_finite = h2[torch.isfinite(h2[:, 1])]
    if h2_finite.numel() > 0:
        max_pers = (h2_finite[:, 1] - h2_finite[:, 0]).max().item()
        assert max_pers > 0.1, (
            f"Expected prominent H2 feature on sphere, max persistence={max_pers}"
        )


def test_max_dim_2_edge_cases():
    """max_dim=2 edge cases: n=0, n=1, and no tetrahedra."""
    pts0 = torch.empty(0, 3, device=device)
    dgms0 = rips_persistence(pts0, max_edge_length=1.0, max_dim=2)
    assert len(dgms0) == 3
    for d in dgms0:
        assert d.shape == (0, 2)

    pts1 = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    dgms1 = rips_persistence(pts1, max_edge_length=1.0, max_dim=2)
    assert len(dgms1) == 3
    assert dgms1[0].shape == (1, 2)
    assert dgms1[1].shape == (0, 2)
    assert dgms1[2].shape == (0, 2)

    torch.manual_seed(42)
    pts_sparse = torch.randn(20, 3, device=device)
    dgms_sparse = rips_persistence(pts_sparse, max_edge_length=0.3, max_dim=2)
    assert len(dgms_sparse) == 3
    assert dgms_sparse[2].ndim == 2 and dgms_sparse[2].shape[1] == 2


# ---------------------------------------------------------------------------
# d > 3 tests
# ---------------------------------------------------------------------------

def test_rips_high_dim_d10():
    """Rips persistence on d=10 point cloud: H0 + H1 parity with ripser."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n, d = 40, 10
    pts = torch.randn(n, d, device=device)
    threshold = 4.0

    dgms = rips_persistence(pts, max_edge_length=threshold)
    assert len(dgms) == 2

    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)
    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


def test_rips_high_dim_d50():
    """Rips persistence on d=50 point cloud: H0 + H1 parity with ripser."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    n, d = 30, 50
    pts = torch.randn(n, d, device=device)
    threshold = 10.0

    dgms = rips_persistence(pts, max_edge_length=threshold)
    assert len(dgms) == 2

    pts_np = pts.cpu().numpy()
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)
    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


def test_h2_colex_tiebreak_n100_d3():
    """Regression: n=100 d=3 H2 requires colex tiebreaking in restricted ranking."""
    ripser_mod = pytest.importorskip("ripser")
    torch.manual_seed(42)
    pts = torch.randn(100, 3, device=device)
    dgms = rips_persistence(pts, max_edge_length=1.5, max_dim=2)
    ref = ripser_mod.ripser(pts.cpu().numpy(), maxdim=2, thresh=1.5)
    _compare_diagrams(dgms[2], ref["dgms"][2], atol=1e-4)


def test_rips_grid_ties_h2():
    """Grid/lattice points create many distance ties — stress-test ordering."""
    ripser_mod = pytest.importorskip("ripser")

    coords = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                coords.append([float(x), float(y), float(z)])
    pts = torch.tensor(coords, dtype=torch.float32, device=device)
    threshold = 2.0

    dgms = rips_persistence(pts, max_edge_length=threshold, max_dim=2)
    assert len(dgms) == 3

    result = ripser_mod.ripser(pts.cpu().numpy(), maxdim=2, thresh=threshold)
    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)
    _compare_diagrams(dgms[2], result["dgms"][2], atol=1e-4)


def test_rips_high_dim_edge_cases():
    """d > 3 edge cases: n=0 and n=1."""
    pts0 = torch.empty(0, 10, device=device)
    dgms0 = rips_persistence(pts0, max_edge_length=1.0)
    assert len(dgms0) == 2
    assert dgms0[0].shape == (0, 2)

    pts1 = torch.randn(1, 10, device=device)
    dgms1 = rips_persistence(pts1, max_edge_length=1.0)
    assert dgms1[0].shape == (1, 2)
    assert dgms1[0][0, 1].item() == float("inf")


# ---------------------------------------------------------------------------
# numpy input acceptance
# ---------------------------------------------------------------------------

def test_numpy_input():
    """rips_persistence accepts numpy arrays (auto-converts to CUDA)."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    pts_np = torch.randn(30, 2).numpy()
    threshold = 2.0

    dgms = rips_persistence(pts_np, max_edge_length=threshold)
    assert len(dgms) == 2
    # Output should be on CUDA since auto-converted
    assert dgms[0].device.type == "cuda"

    # Verify parity
    result = ripser_mod.ripser(pts_np, maxdim=1, thresh=threshold)
    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


# ---------------------------------------------------------------------------
# auto_threshold and enclosing_radius tests
# ---------------------------------------------------------------------------

def test_auto_threshold_basic():
    """auto_threshold returns a positive float for random data."""
    torch.manual_seed(42)
    pts = torch.randn(100, 3, device=device)
    thresh = auto_threshold(pts, k=10, percentile=90)
    assert isinstance(thresh, float)
    assert thresh > 0


def test_auto_threshold_numpy():
    """auto_threshold accepts numpy input."""
    pts_np = np.random.randn(50, 3).astype(np.float32)
    thresh = auto_threshold(pts_np, k=5, percentile=95)
    assert isinstance(thresh, float)
    assert thresh > 0


def test_auto_threshold_small_n():
    """auto_threshold handles n <= k gracefully."""
    pts = torch.randn(3, 2, device=device)
    thresh = auto_threshold(pts, k=20, percentile=95)
    assert thresh > 0


def test_enclosing_radius_basic():
    """enclosing_radius returns a positive float."""
    torch.manual_seed(42)
    pts = torch.randn(50, 3, device=device)
    r = enclosing_radius(pts)
    assert isinstance(r, float)
    assert r > 0


def test_enclosing_radius_with_subsample():
    """enclosing_radius with subsample gives an approximation."""
    torch.manual_seed(42)
    pts = torch.randn(200, 3, device=device)
    r_full = enclosing_radius(pts)
    r_sub = enclosing_radius(pts, subsample=50)
    # Subsample should give a result in the same ballpark
    assert r_sub > 0
    assert abs(r_sub - r_full) / r_full < 0.5  # within 50%


def test_threshold_produces_valid_rips():
    """auto_threshold output can be used with rips_persistence."""
    torch.manual_seed(42)
    pts = torch.randn(50, 3, device=device)
    thresh = auto_threshold(pts, k=10, percentile=90)
    dgms = rips_persistence(pts, max_edge_length=thresh)
    assert len(dgms) == 2
    # Should have some features
    assert dgms[0].shape[0] > 0


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_rips_cpu_fallback():
    """CPU path produces same results as GPU path (H0 + H1)."""
    ripser_mod = pytest.importorskip("ripser")

    torch.manual_seed(42)
    pts = torch.randn(30, 2)  # CPU tensor
    threshold = 2.0

    dgms = rips_persistence(pts, max_edge_length=threshold)
    assert dgms[0].device.type == "cpu"

    result = ripser_mod.ripser(pts.numpy(), maxdim=1, thresh=threshold)
    _compare_diagrams(dgms[0], result["dgms"][0], atol=1e-4)
    _compare_diagrams(dgms[1], result["dgms"][1], atol=1e-4)


# ---------------------------------------------------------------------------
# RipsComplex wrapper tests
# ---------------------------------------------------------------------------


def test_rips_complex_basic():
    """RipsComplex: compute_persistence + persistence_intervals_in_dimension
    returns numpy float64 (K,2) matching flat rips_persistence."""
    torch.manual_seed(42)
    pts = torch.randn(50, 2, device=device)
    threshold = 3.0

    rips = RipsComplex(pts, max_edge_length=threshold)
    rips.compute_persistence(max_dim=1)

    dgms_flat = rips_persistence(pts, max_edge_length=threshold)

    for dim in range(2):
        intervals = rips.persistence_intervals_in_dimension(dim)
        assert isinstance(intervals, np.ndarray)
        assert intervals.dtype == np.float64
        expected = dgms_flat[dim].cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(intervals, expected, atol=1e-6)

    # Out-of-range dim returns empty
    empty = rips.persistence_intervals_in_dimension(5)
    assert empty.shape == (0, 2)
    assert empty.dtype == np.float64


def test_rips_complex_persistence_format():
    """RipsComplex.persistence() returns list[(int, (float, float))], sorted."""
    torch.manual_seed(42)
    pts = torch.randn(30, 2, device=device)
    threshold = 2.5

    rips = RipsComplex(pts, max_edge_length=threshold)
    result = rips.persistence(max_dim=1)

    assert isinstance(result, list)
    assert len(result) > 0
    for entry in result:
        dim, (birth, death) = entry
        assert isinstance(dim, int)
        assert isinstance(birth, float)
        assert isinstance(death, float)

    # Check sorted by (dim, birth, death)
    keys = [(d, b, de) for d, (b, de) in result]
    assert keys == sorted(keys)


def test_rips_complex_betti():
    """Circle: beta_0=1, beta_1=1 with appropriate threshold."""
    torch.manual_seed(42)
    n = 20
    theta = torch.linspace(0, 2 * 3.14159265, n + 1, device=device)[:n]
    pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    # Threshold connects adjacent points but doesn't create triangles,
    # so the H1 loop survives as an essential (inf-death) feature.
    threshold = 0.5

    rips = RipsComplex(pts, max_edge_length=threshold)
    rips.compute_persistence(max_dim=1)
    betti = rips.betti_numbers()

    assert len(betti) == 2
    assert betti[0] == 1  # one connected component
    assert betti[1] == 1  # one loop


def test_rips_complex_persistent_betti():
    """persistent_betti_numbers filters by from_value/to_value correctly."""
    torch.manual_seed(42)
    pts = torch.randn(40, 2, device=device)
    threshold = 3.0

    rips = RipsComplex(pts, max_edge_length=threshold)
    rips.compute_persistence(max_dim=1)

    # All features born at 0 with death > 0 for H0
    pb = rips.persistent_betti_numbers(0.0, 0.0)
    assert isinstance(pb, list)
    assert len(pb) == 2

    # born <= 0 and death > inf: nothing survives
    pb_tight = rips.persistent_betti_numbers(0.0, float("inf"))
    assert pb_tight == [0, 0]


def test_rips_complex_not_computed_raises():
    """RuntimeError before calling compute_persistence."""
    pts = torch.randn(10, 2, device=device)
    rips = RipsComplex(pts, max_edge_length=1.0)

    with pytest.raises(RuntimeError, match="compute_persistence"):
        rips.persistence_intervals_in_dimension(0)

    with pytest.raises(RuntimeError, match="compute_persistence"):
        rips.betti_numbers()

    with pytest.raises(RuntimeError, match="compute_persistence"):
        rips.persistent_betti_numbers(0.0, 1.0)


def test_rips_complex_coeff_field_error():
    """ValueError for homology_coeff_field != 2."""
    pts = torch.randn(10, 2, device=device)
    rips = RipsComplex(pts, max_edge_length=1.0)

    with pytest.raises(ValueError, match="Z/2Z"):
        rips.compute_persistence(homology_coeff_field=3)

    with pytest.raises(ValueError, match="Z/2Z"):
        rips.persistence(homology_coeff_field=7)
