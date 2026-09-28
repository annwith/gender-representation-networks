"""Deterministic k-NN graphs and comparisons for token-occurrence networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphResult:
    """A k-NN graph and its directed neighbor choices before symmetrization."""

    graph: nx.Graph
    neighbors: np.ndarray
    similarities: np.ndarray


def build_union_knn_graph(vectors: np.ndarray, k: int) -> GraphResult:
    """Build an undirected union-symmetrized cosine k-NN graph.

    For every source occurrence, the ``k`` highest cosine similarities excluding
    itself are retained.  Equal scores are resolved by the lower *position* in
    the sequence, making the lexical-embedding ties reproducible.  An undirected
    edge is present when either endpoint selected the other as a neighbor.
    """

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must have shape [n_occurrences, hidden_size]")
    n = values.shape[0]
    if n < 2:
        raise ValueError("at least two token occurrences are required")
    if not 1 <= k < n:
        raise ValueError("k must be at least 1 and smaller than the number of occurrences")

    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("cosine similarity is undefined for zero vectors")
    cosine = (values / norms[:, None]) @ (values / norms[:, None]).T
    # Numerical multiplication can produce values microscopically outside [-1, 1].
    cosine = np.clip(cosine, -1.0, 1.0)
    positions = np.arange(n)
    neighbors = np.empty((n, k), dtype=np.int64)
    selected_similarities = np.empty((n, k), dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for source in range(n):
        candidates = positions[positions != source]
        # lexsort uses the last key as primary: similarity descending, then position ascending.
        order = np.lexsort((candidates, -cosine[source, candidates]))[:k]
        chosen = candidates[order]
        neighbors[source] = chosen
        selected_similarities[source] = cosine[source, chosen]
        for target, similarity in zip(chosen, selected_similarities[source], strict=True):
            if not graph.has_edge(source, int(target)):
                graph.add_edge(source, int(target), similarity=float(similarity))
    return GraphResult(graph=graph, neighbors=neighbors, similarities=selected_similarities)


def graph_metrics(graph: nx.Graph) -> dict[str, int | float]:
    """Calculate the requested global metrics, using the largest component for paths."""

    n = graph.number_of_nodes()
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_nodes = components[0] if components else set()
    largest = graph.subgraph(largest_nodes)
    if largest.number_of_nodes() <= 1:
        average_distance = 0.0
        diameter = 0
    else:
        average_distance = float(nx.average_shortest_path_length(largest))
        diameter = int(nx.diameter(largest))
    return {
        "vertices": n,
        "edges": graph.number_of_edges(),
        "average_degree": float(sum(dict(graph.degree()).values()) / n) if n else 0.0,
        "density": float(nx.density(graph)),
        "average_clustering": float(nx.average_clustering(graph)),
        "global_clustering": float(nx.transitivity(graph)),
        "connected_components": len(components),
        "largest_component_size": len(largest_nodes),
        "average_distance_largest_component": average_distance,
        "diameter_largest_component": diameter,
    }


def detect_communities(graph: nx.Graph) -> tuple[list[int], dict[str, Any]]:
    """Detect deterministic greedy-modularity communities and return node labels."""

    communities = list(nx.community.greedy_modularity_communities(graph))
    communities.sort(key=lambda group: (-len(group), min(group)))
    labels = [-1] * graph.number_of_nodes()
    for community_id, group in enumerate(communities):
        for node in group:
            labels[node] = community_id
    modularity = float(nx.community.modularity(graph, communities)) if communities else 0.0
    return labels, {
        "community_count": len(communities),
        "community_sizes": ";".join(str(len(group)) for group in communities),
        "modularity": modularity,
    }


def jaccard_per_occurrence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return Jaccard similarity between every pair of same-position neighbor sets."""

    if left.shape != right.shape:
        raise ValueError("neighbor matrices must have identical shapes")
    scores = np.empty(left.shape[0], dtype=np.float64)
    for position, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        first, second = set(left_row.tolist()), set(right_row.tolist())
        scores[position] = len(first & second) / len(first | second)
    return scores


def same_token_neighbor_fraction(neighbors: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """Fraction of each occurrence's chosen neighbors with the same token ID."""

    ids = np.asarray(token_ids)
    return (ids[neighbors] == ids[:, None]).mean(axis=1)
