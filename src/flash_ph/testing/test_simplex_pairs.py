"""Compare native simplex identities against small F2 boundary reductions."""
from itertools import combinations
from math import dist, inf

import pytest
import torch

from flash_ph import (
    RipsComplex, rips_apparent_pairs, rips_persistence, rips_persistence_pairs,
)


SQUARE = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.fixture(params=[
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
])
def device(request):
    return torch.device(request.param)


def _facets(simplex):
    return list(combinations(simplex, len(simplex) - 1)) if len(simplex) > 1 else []


def _reference_pairs(points, threshold):
    vertices = [(i,) for i in range(len(points))]
    values = {vertex: 0 for vertex in vertices}
    for i, j in combinations(range(len(points)), 2):
        squared = sum((x - y) ** 2 for x, y in zip(points[i], points[j]))
        if squared < threshold ** 2:
            values[i, j] = squared
    edges = sorted((s for s in values if len(s) == 2), key=lambda s: (values[s], s[::-1]))
    edge_ranks = {edge: rank for rank, edge in enumerate(edges)}
    triangles = []
    for triangle in combinations(range(len(points)), 3):
        faces = _facets(triangle)
        if all(face in edge_ranks for face in faces):
            values[triangle] = max(values[face] for face in faces)
            triangles.append(triangle)
    triangles.sort(key=lambda s: (max(edge_ranks[f] for f in _facets(s)), s[::-1]))
    ordered = sorted(vertices + edges + triangles, key=lambda s: (values[s], len(s)))
    ranks = {simplex: rank for rank, simplex in enumerate(ordered)}

    pairs = [{}, {}]
    apparent = [{}, {}]
    reduced = {}
    earliest_cofacets = {}
    for simplex in ordered:
        dim = len(simplex) - 1
        faces = _facets(simplex)
        for face in faces:
            earliest_cofacets.setdefault(face, simplex)
        column = {ranks[face] for face in faces}
        while column and max(column) in reduced:
            column ^= reduced[max(column)]
        if column:
            pivot = max(column)
            reduced[pivot] = column
            pairs[dim - 1][ordered[pivot]] = simplex
        elif dim < 2:
            pairs[dim][simplex] = None

    for face, cofacet in earliest_cofacets.items():
        if face == max(_facets(cofacet), key=ranks.__getitem__):
            apparent[len(face) - 1][face] = cofacet

    def sort_pairs(tables):
        return [sorted(t.items(), key=lambda row: ranks[row[0]]) for t in tables]

    return sort_pairs(pairs), sort_pairs(apparent)


def _rows(table):
    return [
        (tuple(birth), None if death[0] == -1 else tuple(death))
        for birth, death in zip(table.birth_vertices.tolist(), table.death_vertices.tolist())
    ]


@pytest.mark.parametrize("points, threshold", [
    ([], 0.0),
    ([(0, 0)], 1.0),
    ([(0,), (5,), (2,), (4,), (30,)], 2.1),
    ([(0,), (2,)], 2.0),
    ([(0, 0), (0, 0), (0, 1)], 1.5),
    (SQUARE, 1.1),
    (SQUARE, 1.5),
    ([(0, 0), (0, 2), (2, 1), (2, 2)], 2.3),
    ([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], 1.5),
    ([(0, 0), (0, 2), (2, 0), (2, 2), (10, 0), (10, 1), (12, 1)], 2.3),
], ids=[
    "empty", "singleton", "elder-not-endpoint", "strict-threshold", "coincident",
    "essential-h1", "positive-residual", "zero-residual", "tied-cofacets", "mixed-h1",
])
def test_exact_pair_identities(device, points, threshold):
    cloud = torch.tensor(points, dtype=torch.float32, device=device)
    if not points:
        cloud = cloud.reshape(0, 2)
    expected_pairs, expected_apparent = _reference_pairs(points, threshold)
    for query, expected in [
        (rips_persistence_pairs, expected_pairs),
        (rips_apparent_pairs, expected_apparent),
    ]:
        result = query(cloud, threshold)
        assert len(result) == 2
        for dim, (table, rows) in enumerate(zip(result, expected)):
            assert table.birth_vertices.shape == (len(rows), dim + 1)
            assert table.death_vertices.shape == (len(rows), dim + 2)
            assert table.birth_vertices.dtype == table.death_vertices.dtype == torch.int32
            assert table.birth_vertices.device == table.death_vertices.device == cloud.device
            assert _rows(table) == rows
            for row, (_, death) in zip(table.death_vertices.tolist(), rows):
                if death is None:
                    assert row == [-1] * (dim + 2)


def test_pair_projection_matches_diagram(device):
    if device.type == "cpu":
        pytest.importorskip("gph")
    cloud = torch.tensor(SQUARE, dtype=torch.float32, device=device)
    pairs = rips_persistence_pairs(cloud, 1.5)
    diagrams = rips_persistence(cloud, 1.5, backend="gpu")

    def diameter(simplex):
        return max((dist(SQUARE[i], SQUARE[j]) for i, j in combinations(simplex, 2)),
                   default=0.0)

    for dim, (table, diagram) in enumerate(zip(pairs, diagrams)):
        values = [(diameter(b), inf if d is None else diameter(d)) for b, d in _rows(table)]
        if dim == 1:
            values = [(birth, death) for birth, death in values if birth < death]
        torch.testing.assert_close(
            torch.tensor(sorted(values)), torch.tensor(sorted(diagram.tolist())),
        )


def test_apparent_query_skips_forest_and_reduction(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The apparent query entered the persistence reducer")

    monkeypatch.setattr("flash_ph.persistence.gpu_boruvka_mst", unexpected)
    monkeypatch.setattr("flash_ph.reduce_h1_gpu._prepare_numba_state", unexpected)
    cloud = torch.tensor(SQUARE, dtype=torch.float32)
    assert _rows(rips_apparent_pairs(cloud, 1.5)[1]) == [
        ((1, 2), (0, 1, 2)), ((0, 3), (0, 1, 3)),
    ]


@pytest.mark.parametrize("capacity, message", [
    (0, "apparent-pair pivot pool"),
    (4, "reduction pivot pool"),
])
def test_complete_pair_query_raises_if_pivot_pool_is_full(monkeypatch, capacity, message):
    monkeypatch.setattr("flash_ph._numba_utils.MAX_POOL_ENTRIES", capacity)
    cloud = torch.tensor(SQUARE, dtype=torch.float32)
    with pytest.raises(MemoryError, match=message):
        rips_persistence_pairs(cloud, 1.5)


def test_wrapper_and_unsupported_dimension():
    cloud = torch.tensor(SQUARE, dtype=torch.float32)
    complex_ = RipsComplex(cloud, 1.5)
    for method, query in [
        (complex_.persistence_pairs, rips_persistence_pairs),
        (complex_.apparent_pairs, rips_apparent_pairs),
    ]:
        assert list(map(_rows, method(0))) == list(map(_rows, query(cloud, 1.5, 0)))
        with pytest.raises(NotImplementedError, match="H0 and H1"):
            method(2)


def test_giotto_h2_diagram():
    pytest.importorskip("gph")
    octahedron = torch.cat((torch.eye(3), -torch.eye(3)))
    diagrams = rips_persistence(octahedron, 1.5, max_dim=2)
    torch.testing.assert_close(diagrams[2], torch.tensor([[2 ** 0.5, inf]]))
