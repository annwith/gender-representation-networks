from __future__ import annotations

import math
import random
from math import comb

import igraph as ig
import networkx as nx
import numpy as np
import pytest

from gender_networks.partitions import (
    agreement,
    ari,
    canonical_labels,
    inverse_purity,
    leiden,
    louvain,
    mean_pairwise_nmi,
    nmi,
    nmi_baseline,
    purity,
    valid_labels,
)


def _two_cliques(size: int = 5) -> ig.Graph:
    graph = ig.Graph.Full(size)
    graph.add_vertices(size)
    graph.add_edges([(size + i, size + j) for i in range(size) for j in range(i + 1, size)])
    graph.add_edge(size - 1, size)
    return graph


def _planted(groups: int, size: int, p_in: float, p_out: float, seed: int) -> ig.Graph:
    graph_nx = nx.planted_partition_graph(groups, size, p_in, p_out, seed=seed)
    return ig.Graph(n=groups * size, edges=list(graph_nx.edges()))


def _nmi_reference(a, b) -> float:
    """NMI with the arithmetic-mean normalization 2 I / (H_a + H_b) used by igraph."""

    _, a = np.unique(np.asarray(a), return_inverse=True)
    _, b = np.unique(np.asarray(b), return_inverse=True)
    n = a.size
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    p = table / n
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return 1.0 if ha + hb == 0 else 2 * mi / (ha + hb)


def _ari_reference(a, b) -> float:
    _, a = np.unique(np.asarray(a), return_inverse=True)
    _, b = np.unique(np.asarray(b), return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    index = sum(comb(int(x), 2) for x in table.ravel())
    rows = sum(comb(int(x), 2) for x in table.sum(1))
    cols = sum(comb(int(x), 2) for x in table.sum(0))
    expected = rows * cols / comb(a.size, 2)
    return (index - expected) / ((rows + cols) / 2 - expected)


def test_leiden_splits_two_cliques_joined_by_an_edge() -> None:
    graph = _two_cliques()
    partition = leiden(graph, runs=5, seed=0)
    assert partition.n_communities == 2
    assert partition.membership.tolist() == [0] * 5 + [1] * 5
    assert partition.sizes == [5, 5]
    assert partition.memberships.shape == (5, 10)
    assert partition.stability == pytest.approx(1.0)
    graph_nx = nx.Graph(graph.get_edgelist())
    expected = nx.community.modularity(graph_nx, [set(range(5)), set(range(5, 10))])
    assert partition.modularity == pytest.approx(expected)


def test_leiden_is_deterministic_given_the_seed_and_restores_random_state() -> None:
    graph = _planted(6, 30, 0.3, 0.02, seed=1)
    random.seed(123)
    before = random.random()
    random.seed(123)
    first = leiden(graph, runs=6, seed=11)
    assert random.random() == before  # caller's random stream untouched
    second = leiden(graph, runs=6, seed=11)
    np.testing.assert_array_equal(first.membership, second.membership)
    np.testing.assert_array_equal(first.memberships, second.memberships)
    assert first.modularity == second.modularity
    assert first.quality == pytest.approx(max(first.qualities))
    assert first.memberships.dtype == np.int32
    assert 0.0 <= first.stability <= 1.0
    # With the modularity objective the planted groups are recovered.
    truth = np.repeat(np.arange(6), 30)
    assert nmi(first.membership, truth) > 0.9


def test_leiden_resolution_and_louvain_cross_check() -> None:
    graph = _planted(4, 25, 0.4, 0.02, seed=2)
    coarse = leiden(graph, runs=3, resolution=0.05, seed=0)
    fine = leiden(graph, runs=3, resolution=5.0, seed=0)
    assert coarse.n_communities < fine.n_communities
    assert fine.resolution == 5.0
    assert fine.quality == pytest.approx(graph.modularity(fine.membership.tolist(), resolution=5))
    base = leiden(graph, runs=3, seed=0)
    multilevel = louvain(graph, seed=0)
    assert multilevel.method == "louvain"
    assert nmi(multilevel.membership, base.membership) > 0.9
    again = louvain(graph, seed=0)
    np.testing.assert_array_equal(multilevel.membership, again.membership)
    assert louvain(_two_cliques(), seed=3).n_communities == 2
    with pytest.raises(ValueError):
        leiden(ig.Graph(n=3, edges=[(0, 1)], directed=True))


def test_canonical_labels_order_by_size_then_first_vertex() -> None:
    labels = canonical_labels([7, 3, 3, 9, 9, 9, 7, 5])
    assert labels.tolist() == [1, 2, 2, 0, 0, 0, 1, 3]
    assert labels.dtype == np.int32


def test_nmi_and_ari_on_known_partitions() -> None:
    a = [0, 0, 0, 1, 1, 1, 2, 2]
    relabelled = ["x", "x", "x", "b", "b", "b", "q", "q"]
    assert nmi(a, relabelled) == pytest.approx(1.0)
    assert ari(a, relabelled) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.integers(0, 4, size=50)
        y = rng.integers(10, 16, size=50)
        assert nmi(x, y) == pytest.approx(_nmi_reference(x, y))
        assert ari(x, y) == pytest.approx(_ari_reference(x, y))
    with pytest.raises(ValueError):
        nmi([0, 1], [0])


def test_purity_and_inverse_purity() -> None:
    pred = [0, 0, 0, 1, 1, 2]
    true = ["a", "a", "a", "a", "b", "b"]
    assert purity(pred, true) == pytest.approx(5 / 6)
    assert inverse_purity(pred, true) == pytest.approx(4 / 6)
    singletons = list(range(6))
    assert purity(singletons, true) == 1.0
    assert inverse_purity(singletons, true) == pytest.approx(2 / 6)


def test_nmi_baseline_is_near_zero_for_random_labels() -> None:
    rng = np.random.default_rng(1)
    pred = rng.integers(0, 5, size=3000)
    true = rng.integers(0, 5, size=3000)
    baseline = nmi_baseline(pred, true, permutations=10, seed=0)
    assert baseline < 0.01
    assert abs(nmi(pred, true) - baseline) < 0.01
    assert nmi_baseline(pred, true, permutations=10, seed=0) == baseline
    # Many small communities inflate NMI even against unrelated labels; the baseline shows it.
    fine = rng.integers(0, 600, size=3000)
    assert nmi_baseline(fine, true, permutations=5, seed=0) > 0.1


def test_mean_pairwise_nmi() -> None:
    runs = [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]]
    expected = np.mean([1.0, _nmi_reference(runs[0], runs[2]), _nmi_reference(runs[1], runs[2])])
    assert mean_pairwise_nmi(runs) == pytest.approx(expected)
    assert math.isnan(mean_pairwise_nmi(runs[:1]))


def test_agreement_rows_skip_missing_labels() -> None:
    pred = np.array([0, 0, 0, 1, 1, 1])
    labels = {
        "theme": np.array(["f", "f", "f", "e", "e", "e"]),
        "sense": np.array(["a", "", "a", "b", None, "b"], dtype=object),
        "score": np.array([1.0, np.nan, 1.0, 2.0, 2.0, np.nan]),
    }
    rows = {row["label"]: row for row in agreement(pred, labels, permutations=5, seed=0)}
    assert rows["theme"]["nmi"] == pytest.approx(1.0)
    assert rows["theme"]["ari"] == pytest.approx(1.0)
    assert rows["theme"]["purity"] == rows["theme"]["inverse_purity"] == 1.0
    assert rows["theme"]["nmi_minus_baseline"] == pytest.approx(1.0 - rows["theme"]["nmi_baseline"])
    assert rows["theme"]["n_classes"] == 2 and rows["theme"]["n"] == 6
    assert rows["sense"]["n"] == 4 and rows["sense"]["n_classes"] == 2
    assert rows["score"]["n"] == 4
    assert valid_labels(labels["sense"]).tolist() == [True, False, True, True, False, True]
    with pytest.raises(ValueError):
        agreement(pred, {"bad": np.zeros(3)})
