"""GPU-parallel Borůvka MST for H0 persistent homology.

Uses scatter_reduce for per-component minimum edge selection.
Works with integer edge_rank (position in sorted edge list) as weights,
ensuring consistency with filtration ordering used in H1.

Component IDs are root vertex indices (comp[root] == root), which
ensures pointer-jumping merges work correctly.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def gpu_boruvka_mst(
    edge_i: Tensor,
    edge_j: Tensor,
    edge_rank: Tensor,
    n: int,
) -> tuple[Tensor, Tensor]:
    """GPU-parallel Borůvka MST using scatter_reduce.

    Parameters
    ----------
    edge_i, edge_j : (E,) int32
        Edge endpoints (i < j), sorted by filtration.
    edge_rank : (E,) int32
        Edge ranks (= position in sorted edge list).
    n : int
        Number of vertices.

    Returns
    -------
    mst_edge_indices : (K,) int64
        Indices into the sorted edge list for MST edges.
    final_comp : (n,) int64
        Final component labels (root vertex indices).
    """
    device = edge_i.device
    E = edge_i.shape[0]
    SENTINEL = torch.iinfo(torch.int64).max

    # comp[v] = root vertex of v's component (initially each vertex is its own root)
    comp = torch.arange(n, dtype=torch.int64, device=device)
    mst_parts: list[Tensor] = []

    max_iters = int(math.ceil(math.log2(max(n, 2)))) + 2

    for _ in range(max_iters):
        # 1. Pointer-jump to flatten component tree
        for _ in range(64):
            comp_new = comp[comp]
            if torch.equal(comp_new, comp):
                break
            comp = comp_new

        # 2. Check convergence
        K = comp.unique().numel()
        if K == 1:
            break

        # 3. Per-component lightest crossing edge via scatter_reduce
        comp_u = comp[edge_i.long()]
        comp_v = comp[edge_j.long()]
        crossing = comp_u != comp_v
        cross_idx = crossing.nonzero(as_tuple=True)[0]

        if cross_idx.numel() == 0:
            break

        # Key = edge_rank (int64)
        key = edge_rank[cross_idx].to(torch.int64)

        # Both endpoints claim lightest for their component
        all_comp = torch.cat([comp_u[cross_idx], comp_v[cross_idx]])
        all_key = torch.cat([key, key])

        # scatter_reduce 'amin' — dense array of size n since comp IDs are vertex indices
        min_key = torch.full((n,), SENTINEL, dtype=torch.int64, device=device)
        min_key.scatter_reduce_(0, all_comp, all_key, reduce="amin",
                                include_self=True)

        # Collect unique selected edge ranks
        valid = min_key < SENTINEL
        selected_ranks = min_key[valid]
        selected = selected_ranks.unique()
        mst_parts.append(selected)

        # 4. Merge: hook higher root → lower root
        # Since comp IDs are root vertex indices (comp[root]==root),
        # comp[hook_from] modifies the correct root entry.
        sel_u = comp[edge_i[selected].long()]
        sel_v = comp[edge_j[selected].long()]
        hook_from = torch.maximum(sel_u, sel_v)
        hook_to = torch.minimum(sel_u, sel_v)
        comp.scatter_reduce_(0, hook_from, hook_to, reduce="amin",
                             include_self=True)

    # Final flatten
    for _ in range(64):
        comp_new = comp[comp]
        if torch.equal(comp_new, comp):
            break
        comp = comp_new

    if mst_parts:
        mst_edge_indices = torch.cat(mst_parts).unique()
    else:
        mst_edge_indices = torch.empty(0, dtype=torch.int64, device=device)

    return mst_edge_indices, comp
