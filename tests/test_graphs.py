from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pytest

from gender_networks.graphs import (
    directed_edges,
    directed_graph,
    directed_metrics,
    double_sweep_lower_bound,
    edge_hash,
    graph_metrics,
    mutual_graph,
    symmetrized_edges,
    to_igraph,
    union_graph,
)
from gender_networks.network_analysis import build_union_knn_graph


def _random_knn(n: int, k: int, seed: int) -> np.ndarray:
    """Neighbour rows of a random point cloud (distinct ids, never the row itself)."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    dist = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1, kind="stable")[:, :k]


def _nx_from_igraph(g: ig.Graph) -> nx.Graph:
    graph = nx.DiGraph() if g.is_directed() else nx.Graph()
    graph.add_nodes_from(range(g.vcount()))
    graph.add_edges_from(g.get_edgelist())
    return graph


def _nx_avg_clustering(graph: nx.Graph) -> float:
    """Mean local clustering over the vertices of degree >= 2, where C_i is defined."""

    defined = [v for v, d in graph.degree() if d >= 2]
    return float(np.mean([nx.clustering(graph, v) for v in defined]))


def _load_export_networks():
    path = Path(__file__).parents[1] / "scripts" / "partial_delivery" / "export_networks.py"
    spec = importlib.util.spec_from_file_location("export_networks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _edge_set(g: ig.Graph) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in g.get_edgelist()}


def _reference_union(neighbors: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(i, int(j)), max(i, int(j)))
        for i, row in enumerate(neighbors)
        for j in row
        if i != int(j)
    }


def _reference_mutual(neighbors: np.ndarray) -> set[tuple[int, int]]:
    chosen = {(i, int(j)) for i, row in enumerate(neighbors) for j in row if i != int(j)}
    return {(min(i, j), max(i, j)) for i, j in chosen if (j, i) in chosen}


@pytest.mark.parametrize(
    "graph_nx",
    [
        nx.gnp_random_graph(60, 0.06, seed=1),
        nx.gnp_random_graph(80, 0.03, seed=2),  # several components
        nx.watts_strogatz_graph(70, 4, 0.2, seed=3),
        nx.barabasi_albert_graph(90, 2, seed=4),
    ],
)
def test_undirected_metrics_match_networkx(graph_nx: nx.Graph) -> None:
    n = graph_nx.number_of_nodes()
    g = to_igraph(np.array(list(graph_nx.edges()), dtype=np.int64), n, directed=False)
    metrics = graph_metrics(g, distances="exact")

    assert metrics["n_vertices"] == n
    assert metrics["n_edges"] == graph_nx.number_of_edges()
    assert metrics["density"] == pytest.approx(nx.density(graph_nx))
    assert metrics["mean_degree"] == pytest.approx(2 * graph_nx.number_of_edges() / n)
    assert metrics["transitivity"] == pytest.approx(nx.transitivity(graph_nx))
    assert metrics["avg_local_clustering"] == pytest.approx(_nx_avg_clustering(graph_nx))

    components = sorted((len(c) for c in nx.connected_components(graph_nx)), reverse=True)
    assert metrics["n_components"] == len(components)
    assert metrics["largest_component"] == components[0]
    assert metrics["largest_fraction"] == pytest.approx(components[0] / n)
    expected_sizes: dict[int, int] = {}
    for size in components:
        expected_sizes[size] = expected_sizes.get(size, 0) + 1
    assert metrics["component_sizes"] == dict(sorted(expected_sizes.items()))

    lcc = graph_nx.subgraph(max(nx.connected_components(graph_nx), key=len))
    assert metrics["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))
    assert metrics["diameter"] == nx.diameter(lcc)
    assert metrics["distance_method"] == "exact"
    lengths = dict(nx.all_pairs_shortest_path_length(lcc))
    hist: dict[int, int] = {}
    for source, targets in lengths.items():
        for target, d in targets.items():
            if source < target:
                hist[d] = hist.get(d, 0) + 1
    assert metrics["distance_hist"] == dict(sorted(hist.items()))

    degree_counts = nx.degree_histogram(graph_nx)
    assert metrics["degree_hist"]["undirected"] == {
        d: c for d, c in enumerate(degree_counts) if c
    }


def test_knn_union_graph_metrics_match_networkx() -> None:
    neighbors = _random_knn(120, 4, seed=7)
    g = union_graph(neighbors, 120)
    reference = _nx_from_igraph(g)
    metrics = graph_metrics(g)
    assert metrics["transitivity"] == pytest.approx(nx.transitivity(reference))
    assert metrics["avg_local_clustering"] == pytest.approx(_nx_avg_clustering(reference))
    lcc = reference.subgraph(max(nx.connected_components(reference), key=len))
    assert metrics["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))
    assert metrics["diameter"] == nx.diameter(lcc)


def test_avg_local_clustering_excludes_degree_below_two() -> None:
    # Triangle 0-1-2 plus the pendant vertex 3 on 2 and the isolated vertex 4:
    # C = 1, 1, 1/3 and undefined for 3 and 4, so the mean is 7/9 (not 7/15 with zeros).
    g = to_igraph(np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64), 5, directed=False)
    assert graph_metrics(g, distances="none")["avg_local_clustering"] == pytest.approx(7 / 9)
    leaves = to_igraph(np.array([[0, 1]], dtype=np.int64), 2, directed=False)
    assert math.isnan(graph_metrics(leaves, distances="none")["avg_local_clustering"])


def test_directed_clustering_is_undefined_below_two_neighbours() -> None:
    export = _load_export_networks()
    # Triangle 0 -> 1 -> 2 -> 0, plus 3 <-> 0 (one reciprocal neighbour) and 4 -> 1.
    edges = [(0, 1), (1, 2), (2, 0), (3, 0), (0, 3), (4, 1)]
    g = ig.Graph(n=5, edges=edges, directed=True)
    local, _ = export.directed_clustering(g)
    assert np.isnan(local[3]) and np.isnan(local[4])
    reference = nx.clustering(_nx_from_igraph(g))
    defined = [0, 1, 2]
    assert local[defined] == pytest.approx([reference[i] for i in defined])
    assert reference[3] == reference[4] == 0
    mean = export.measure_directed(g)["avg_local_clustering"]
    assert mean == pytest.approx(np.mean([reference[i] for i in defined]))


def test_sampled_distances_close_to_exact_on_small_world() -> None:
    random.seed(0)  # igraph's generators draw from Python's random module
    graph = ig.Graph.Watts_Strogatz(1, 3000, 3, 0.05)
    graph.simplify()
    exact = graph_metrics(graph, distances="exact")
    sampled = graph_metrics(graph, distances="sampled", sample_sources=300, seed=5)
    assert sampled["distance_method"] == "sampled"
    assert sampled["n_distance_sources"] == 300
    assert sampled["mean_distance"] == pytest.approx(exact["mean_distance"], rel=0.03)
    assert exact["diameter"] - 2 <= sampled["diameter"] <= exact["diameter"]
    # Same seed, same estimate.
    again = graph_metrics(graph, distances="sampled", sample_sources=300, seed=5)
    assert again["mean_distance"] == sampled["mean_distance"]


def test_sampled_with_enough_sources_is_exact_and_none_skips_distances() -> None:
    random.seed(1)
    graph = ig.Graph.Watts_Strogatz(1, 50, 2, 0.1)
    exact = graph_metrics(graph, distances="exact")
    fallback = graph_metrics(graph, distances="sampled", sample_sources=50)
    assert fallback["distance_method"] == "exact"
    assert fallback["mean_distance"] == exact["mean_distance"]
    skipped = graph_metrics(graph, distances="none")
    assert skipped["distance_method"] == "none"
    assert math.isnan(skipped["mean_distance"]) and skipped["distance_hist"] == {}


def test_double_sweep_is_a_lower_bound() -> None:
    graph = ig.Graph.Lattice([30], circular=False)  # a path: diameter 29
    assert double_sweep_lower_bound(graph, 12) == 29
    ring = ig.Graph.Ring(40)
    assert double_sweep_lower_bound(ring, 0) == 20


def test_union_and_mutual_edge_sets() -> None:
    neighbors = _random_knn(50, 5, seed=11)
    assert _edge_set(union_graph(neighbors, 50)) == _reference_union(neighbors)
    assert _edge_set(mutual_graph(neighbors, 50)) == _reference_mutual(neighbors)
    assert _edge_set(mutual_graph(neighbors, 50)) <= _edge_set(union_graph(neighbors, 50))
    assert union_graph(neighbors, 50).is_simple()
    directed = directed_graph(neighbors, 50)
    assert directed.is_directed() and directed.ecount() == 50 * 5
    assert set(directed.outdegree()) == {5}


def test_union_graph_matches_frozen_pilot() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, 6))
    pilot = build_union_knn_graph(vectors, 3)
    ours = union_graph(pilot.neighbors, 40)
    assert _edge_set(ours) == {(min(a, b), max(a, b)) for a, b in pilot.graph.edges()}


def test_csr_neighbors_self_loops_and_repeats_are_dropped() -> None:
    indptr = np.array([0, 3, 4, 4, 6])
    idx = np.array([1, 1, 0, 1, 3, 2])  # row 0 repeats 1 and lists itself; row 3 lists itself
    edges = directed_edges((indptr, idx), 4)
    assert edges.tolist() == [[0, 1], [3, 2]]
    assert symmetrized_edges((indptr, idx), 4, "union").tolist() == [[0, 1], [2, 3]]
    assert symmetrized_edges((indptr, idx), 4, "mutual").tolist() == []
    dense = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    csr = (np.array([0, 2, 4, 6, 8]), dense.reshape(-1))
    assert _edge_set(union_graph(csr, 4)) == _edge_set(union_graph(dense, 4))
    with pytest.raises(ValueError):
        directed_edges(np.array([[1], [5]]), 2)


def test_edge_hash_depends_only_on_the_edge_set() -> None:
    a = np.array([[1, 2], [0, 2], [0, 1]])
    b = np.array([[2, 1], [2, 0], [1, 0]])  # same sets, different order
    for sym in ("directed", "union", "mutual"):
        ea, eb = symmetrized_edges(a, 3, sym), symmetrized_edges(b, 3, sym)
        assert edge_hash(ea, 3, sym == "directed") == edge_hash(eb, 3, sym == "directed")
    union = symmetrized_edges(a, 3, "union")
    assert edge_hash(union, 3, directed=False) != edge_hash(union, 3, directed=True)
    assert edge_hash(union, 3, directed=False) != edge_hash(union, 4, directed=False)


def test_reciprocity_and_hubs_on_hand_built_digraph() -> None:
    neighbors = np.array([[1, 2], [0, 2], [3, 0], [2, 1]])
    metrics = directed_metrics(neighbors, 4)
    # Reciprocated: 0->1, 0->2, 1->0, 2->3, 2->0, 3->2; not: 1->2, 3->1.
    assert metrics["reciprocity"] == pytest.approx(0.75)
    graph = directed_graph(neighbors, 4)
    assert metrics["reciprocity"] == pytest.approx(graph.reciprocity())
    assert metrics["reciprocity"] == pytest.approx(nx.reciprocity(_nx_from_igraph(graph)))
    assert metrics["hubs"][:4] == [(2, 3), (0, 2), (1, 2), (3, 1)]
    assert metrics["in_degree_hist"] == {1: 1, 2: 2, 3: 1}
    assert metrics["in_degree_max"] == 3
    assert metrics["in_degree_mean"] == pytest.approx(2.0)
    assert metrics["in_degree_zero_fraction"] == 0.0
    assert metrics["out_degree_min"] == metrics["out_degree_max"] == 2


def test_directed_metrics_statistics() -> None:
    neighbors = _random_knn(200, 6, seed=2)
    metrics = directed_metrics(neighbors, 200, top_hubs=5)
    indeg = np.bincount(neighbors.reshape(-1), minlength=200)
    assert metrics["in_degree_std"] == pytest.approx(indeg.std())
    centered = indeg - indeg.mean()
    skew = np.mean(centered**3) / np.mean(centered**2) ** 1.5
    assert metrics["in_degree_skewness"] == pytest.approx(skew)
    assert metrics["in_degree_zero_fraction"] == pytest.approx(np.mean(indeg == 0))
    assert [v for v, _ in metrics["hubs"]] == list(np.lexsort((np.arange(200), -indeg))[:5])
    assert len(metrics["hubs"]) == 5
    reference = _nx_from_igraph(directed_graph(neighbors, 200))
    assert metrics["reciprocity"] == pytest.approx(nx.reciprocity(reference))


def test_graph_metrics_on_digraph_reports_in_and_out_degrees() -> None:
    neighbors = _random_knn(30, 3, seed=9)
    g = directed_graph(neighbors, 30)
    metrics = graph_metrics(g, distances="none")
    assert metrics["mean_degree"] == pytest.approx(3.0)
    assert metrics["degree_hist"]["out"] == {3: 30}
    assert sum(metrics["degree_hist"]["in"].values()) == 30
    assert math.isnan(metrics["transitivity"])
    weak = nx.number_weakly_connected_components(_nx_from_igraph(g))
    assert metrics["n_components"] == weak
    with pytest.raises(ValueError):
        graph_metrics(g, distances="exact")
