"""Graphs built from k-NN neighbour sets and their structural metrics (plan decision 10).

The k-NN relation is directed: every vertex chooses its neighbours, so the ``knn`` stage output
is a digraph ``i -> j`` for ``j in N_i`` (used for in-degree, hubs and reciprocity). Clustering,
distances, components and communities use the undirected union graph (``{i, j}`` when either
endpoint chose the other); the mutual graph (both chose each other) is a robustness variant.

Metrics are computed with igraph, whose all-pairs BFS runs in C: exact distances on a
15 000-vertex graph take seconds, where the NetworkX pilot needed 20-25 minutes. NetworkX is
kept as the reference in the tests.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Literal

import igraph as ig
import numpy as np

from gender_networks.neighborhood import Neighbors, neighbor_rows

DistanceMode = Literal["exact", "sampled", "none"]
DISTANCE_MODES = ("exact", "sampled", "none")


def directed_edges(neighbors: Neighbors, n: int) -> np.ndarray:
    """Sorted distinct ``[E, 2]`` edges ``i -> j`` for ``j in N_i``, without self loops.

    Args:
        neighbors: Neighbour sets, dense ``[N, k]`` or CSR ``(indptr, idx)``.
        n: Number of vertices; may exceed the number of rows.

    Returns:
        int64 array ``[E, 2]`` sorted by ``(i, j)``, each edge once.

    Raises:
        ValueError: If there are more rows than vertices or an id is outside ``[0, n)``.
    """

    rows, ids, n_rows = neighbor_rows(neighbors)
    if n_rows > n:
        raise ValueError(f"{n_rows} neighbour rows for a graph with {n} vertices")
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise ValueError("neighbour ids must lie in [0, n)")
    keep = rows != ids
    keys = np.unique(rows[keep] * n + ids[keep])
    return np.column_stack((keys // n, keys % n))


def undirected_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Distinct ``{i, j}`` pairs (``i < j``) of a directed edge list: the union symmetrization.

    Args:
        edges: int64 array ``[E, 2]`` of directed edges.
        n: Number of vertices.

    Returns:
        int64 array ``[M, 2]`` sorted by ``(i, j)``, with ``{i, j}`` present when either
        direction is.
    """

    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    keys = np.unique(lo * n + hi)
    return np.column_stack((keys // n, keys % n))


def mutual_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Pairs ``{i, j}`` (``i < j``) present in both directions of a directed edge list.

    ``edges`` must be distinct and sorted (as returned by :func:`directed_edges`).

    Args:
        edges: int64 array ``[E, 2]`` of directed edges.
        n: Number of vertices.

    Returns:
        int64 array ``[M, 2]`` with one row per reciprocated pair, in the order of ``edges``.
    """

    keys = edges[:, 0] * n + edges[:, 1]
    reverse = edges[:, 1] * n + edges[:, 0]
    both = np.isin(reverse, keys) & (edges[:, 0] < edges[:, 1])
    return edges[both]


def symmetrized_edges(neighbors: Neighbors, n: int, sym: str) -> np.ndarray:
    """Edge list of the ``directed``, ``union`` or ``mutual`` graph of a neighbour structure.

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.
        sym: ``directed``, ``union`` or ``mutual``.

    Returns:
        int64 array ``[E, 2]`` in canonical sorted form.

    Raises:
        ValueError: If ``sym`` is not one of the three names.
    """

    return symmetrize(directed_edges(neighbors, n), n, sym)


def symmetrize(edges: np.ndarray, n: int, sym: str) -> np.ndarray:
    """``directed``, ``union`` or ``mutual`` edge list from :func:`directed_edges` output.

    Lets a caller derive the three graphs of one neighbour structure from a single pass.

    Args:
        edges: int64 array ``[E, 2]`` from :func:`directed_edges`.
        n: Number of vertices.
        sym: ``directed`` (returned unchanged), ``union`` or ``mutual``.

    Returns:
        int64 array ``[E', 2]``.

    Raises:
        ValueError: If ``sym`` is not one of the three names.
    """

    if sym == "directed":
        return edges
    if sym == "union":
        return undirected_edges(edges, n)
    if sym == "mutual":
        return mutual_edges(edges, n)
    raise ValueError(f"Unknown symmetrization '{sym}' (directed, union or mutual)")


def edge_hash(edges: np.ndarray, n: int, directed: bool) -> str:
    """Fingerprint of an edge set, used to skip graphs identical to one already measured.

    ``edges`` must be in the canonical sorted form produced by the functions above.

    Args:
        edges: int64 array ``[E, 2]``.
        n: Number of vertices.
        directed: Whether the edges are directed; part of the hash.

    Returns:
        SHA-1 hex digest of the direction, ``n`` and the edge bytes.
    """

    digest = hashlib.sha1(f"{'directed' if directed else 'undirected'}|{n}|".encode())
    digest.update(np.ascontiguousarray(edges, dtype=np.int64).tobytes())
    return digest.hexdigest()


def to_igraph(edges: np.ndarray, n: int, directed: bool) -> ig.Graph:
    """igraph graph from an edge list; ``simplify`` guards against loops and multi-edges.

    Args:
        edges: Edge array ``[E, 2]``.
        n: Number of vertices (isolated ones included).
        directed: Build a digraph.

    Returns:
        Simple igraph graph with ``n`` vertices.
    """

    pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    graph = ig.Graph(n=n, edges=pairs, directed=directed)
    graph.simplify(multiple=True, loops=True)
    return graph


def directed_graph(neighbors: Neighbors, n: int) -> ig.Graph:
    """Digraph with an edge ``i -> j`` for every ``j in N_i`` (out-degree ``|N_i|``).

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.

    Returns:
        Directed igraph graph.
    """

    return to_igraph(directed_edges(neighbors, n), n, directed=True)


def union_graph(neighbors: Neighbors, n: int) -> ig.Graph:
    """Undirected simple graph with ``{i, j}`` when ``j in N_i`` or ``i in N_j``.

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.

    Returns:
        Undirected igraph graph.
    """

    return to_igraph(symmetrized_edges(neighbors, n, "union"), n, directed=False)


def mutual_graph(neighbors: Neighbors, n: int) -> ig.Graph:
    """Undirected simple graph with ``{i, j}`` only when ``j in N_i`` and ``i in N_j``.

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.

    Returns:
        Undirected igraph graph.
    """

    return to_igraph(symmetrized_edges(neighbors, n, "mutual"), n, directed=False)


def histogram(values: np.ndarray | list[int]) -> dict[int, int]:
    """``{value: count}`` of an integer sequence, sorted by value.

    Args:
        values: Integers (degrees, component sizes).

    Returns:
        Dictionary with Python ``int`` keys and counts.
    """

    uniq, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(uniq, counts, strict=True)}


def distance_histogram(hist: ig.Histogram) -> dict[int, int]:
    """``{distance: number of pairs}`` of a ``path_length_hist`` (finite distances only).

    Args:
        hist: Histogram returned by ``Graph.path_length_hist``.

    Returns:
        Distance to the number of vertex pairs at that distance; empty bins are left out.
    """

    return {int(low): int(count) for low, _high, count in hist.bins() if count}


def _exact_distances(lcc: ig.Graph) -> dict[str, Any]:
    """Mean distance, diameter and histogram over all unordered pairs (one all-pairs BFS).

    Args:
        lcc: Connected undirected graph (the largest component).

    Returns:
        ``mean_distance``, ``diameter``, ``distance_hist`` (unordered pairs per distance),
        ``distance_method`` (``exact``) and ``n_distance_sources`` (every vertex).
    """

    hist = distance_histogram(lcc.path_length_hist(directed=False))
    pairs = sum(hist.values())
    mean = sum(d * c for d, c in hist.items()) / pairs if pairs else 0.0
    return {
        "mean_distance": float(mean),
        "diameter": max(hist) if hist else 0,
        "distance_hist": hist,
        "distance_method": "exact",
        "n_distance_sources": lcc.vcount(),
    }


def _bfs_layers(graph: ig.Graph, source: int) -> tuple[np.ndarray, list[int]]:
    """Layer sizes of a BFS from ``source`` (index d holds the vertices at distance d).

    Args:
        graph: The graph searched.
        source: Start vertex.

    Returns:
        ``(sizes, vids)``: the number of vertices at each distance (index 0 is the source
        itself) and the vertices in visiting order, so the last one is a farthest vertex.
    """

    vids, layers, _parents = graph.bfs(source)
    return np.diff(np.asarray(layers, dtype=np.int64)), vids


def double_sweep_lower_bound(graph: ig.Graph, start: int, max_rounds: int = 10) -> int:
    """Diameter lower bound by repeated double sweep from ``start``.

    The eccentricity of the farthest vertex found by a BFS is a lower bound on the diameter;
    restarting from that vertex while the bound grows is cheap and usually tight on k-NN graphs.

    Args:
        graph: Connected undirected graph.
        start: First BFS source.
        max_rounds: Most BFS runs. Defaults to 10.

    Returns:
        The largest eccentricity seen, a lower bound on the diameter (at least 0).
    """

    best = -1
    current = start
    for _ in range(max_rounds):
        sizes, vids = _bfs_layers(graph, current)
        eccentricity = sizes.size - 1
        if eccentricity <= best:
            break
        best = eccentricity
        current = int(vids[-1])
    return max(best, 0)


def _sampled_distances(lcc: ig.Graph, sources: int, seed: int) -> dict[str, Any]:
    """Distance estimates from BFS trees rooted at ``sources`` random vertices.

    The mean over ordered (source, target) pairs is an unbiased estimate of the mean distance;
    the diameter is a lower bound (largest sampled eccentricity or double sweep). The histogram
    counts ordered pairs from the sampled sources.

    Args:
        lcc: Connected undirected graph (the largest component).
        sources: Number of BFS sources, drawn without replacement; at most ``lcc.vcount()``.
        seed: Seed of the numpy generator that draws the sources.

    Returns:
        The same keys as :func:`_exact_distances`, with ``distance_method`` ``sampled`` and
        ``n_distance_sources`` equal to ``sources``.
    """

    n = lcc.vcount()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=sources, replace=False)
    totals = np.zeros(1, dtype=np.int64)
    eccentricity = 0
    for source in chosen.tolist():
        sizes, _ = _bfs_layers(lcc, source)
        if sizes.size > totals.size:
            totals = np.pad(totals, (0, sizes.size - totals.size))
        totals[: sizes.size] += sizes
        eccentricity = max(eccentricity, sizes.size - 1)
    totals[0] = 0  # distance 0 is the source itself
    reached = int(totals.sum())
    distances = np.arange(totals.size)
    mean = float((distances * totals).sum() / reached) if reached else 0.0
    diameter = max(eccentricity, double_sweep_lower_bound(lcc, int(chosen[0])))
    return {
        "mean_distance": mean,
        "diameter": int(diameter),
        "distance_hist": {int(d): int(c) for d, c in enumerate(totals.tolist()) if c},
        "distance_method": "sampled",
        "n_distance_sources": int(sources),
    }


def graph_metrics(
    g: ig.Graph,
    distances: DistanceMode = "exact",
    sample_sources: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Structural metrics required by the course, with distances on the largest component.

    Clustering and distances are defined here only for undirected graphs; for a digraph they
    are NaN (the union graph carries them) and components are weak components. With
    ``distances='sampled'`` and at least as many sources as vertices the exact computation is
    used, since it costs the same.

    Args:
        g: Simple igraph graph, directed or not.
        distances: ``exact``, ``sampled`` or ``none``.
        sample_sources: BFS sources of the sampled mode. Defaults to 500.
        seed: Seed of the source draw.

    Returns:
        Dictionary with ``n_vertices``, ``n_edges``, ``mean_degree`` (``2m/n`` undirected,
        ``m/n`` directed), ``density`` (``2m/[n(n-1)]`` or ``m/[n(n-1)]``), ``transitivity``,
        ``avg_local_clustering`` (mean over the vertices of degree at least 2),
        ``degree_hist``, ``n_components``, ``largest_component``, ``largest_fraction``,
        ``component_sizes`` and the distance keys of :func:`_exact_distances`.

    Raises:
        ValueError: If the mode is unknown, ``sample_sources`` is below 1 in sampled mode,
            or distances are asked for a digraph.
    """

    if distances not in DISTANCE_MODES:
        raise ValueError(f"distances must be one of {DISTANCE_MODES}")
    if distances == "sampled" and sample_sources < 1:
        raise ValueError("sample_sources must be positive")
    directed = g.is_directed()
    if directed and distances != "none":
        raise ValueError("distances are computed on undirected graphs only")
    n, m = g.vcount(), g.ecount()
    pairs = n * (n - 1)
    out: dict[str, Any] = {
        "n_vertices": n,
        "n_edges": m,
        "mean_degree": (m if directed else 2 * m) / n if n else math.nan,
        "density": (m if directed else 2 * m) / pairs if pairs else math.nan,
    }
    if directed:
        out["transitivity"] = math.nan
        out["avg_local_clustering"] = math.nan
        out["degree_hist"] = {
            "in": histogram(g.indegree()),
            "out": histogram(g.outdegree()),
        }
    else:
        transitivity = g.transitivity_undirected()
        # igraph returns NaN without connected triples; NetworkX (the reference) returns 0.
        out["transitivity"] = 0.0 if math.isnan(transitivity) else float(transitivity)
        # C_i is undefined for degree < 2, so those vertices are left out of the mean
        # (NetworkX's average_clustering would count them as 0).
        out["avg_local_clustering"] = (
            float(g.transitivity_avglocal_undirected(mode="nan")) if n else math.nan
        )
        out["degree_hist"] = {"undirected": histogram(g.degree())}

    components = g.connected_components(mode="weak")
    sizes = components.sizes()
    largest = max(sizes) if sizes else 0
    out["n_components"] = len(sizes)
    out["largest_component"] = largest
    out["largest_fraction"] = largest / n if n else math.nan
    out["component_sizes"] = histogram(sizes)

    if distances == "none" or n == 0:
        out.update(
            mean_distance=math.nan,
            diameter=math.nan,
            distance_hist={},
            distance_method="none",
            n_distance_sources=0,
        )
        return out
    lcc = components.giant()
    if lcc.vcount() <= 1:
        out.update(
            mean_distance=0.0,
            diameter=0,
            distance_hist={},
            distance_method="exact",
            n_distance_sources=lcc.vcount(),
        )
    elif distances == "exact" or sample_sources >= lcc.vcount():
        out.update(_exact_distances(lcc))
    else:
        out.update(_sampled_distances(lcc, sample_sources, seed))
    return out


def _skewness(values: np.ndarray) -> float:
    """Population (Fisher-Pearson) skewness; NaN for a constant sequence.

    Args:
        values: float array.

    Returns:
        ``m3 / m2 ** 1.5`` with the central moments taken over the whole sequence.
    """

    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    if m2 == 0:
        return math.nan
    return float(np.mean(centered**3) / m2**1.5)


def directed_metrics(neighbors: Neighbors, n: int, top_hubs: int = 20) -> dict[str, Any]:
    """In-degree (hubness) statistics and reciprocity of the directed k-NN graph.

    A hub is a vertex chosen as neighbour by many others; in high-dimensional k-NN graphs the
    in-degree distribution becomes right-skewed, so skewness, the zero-in-degree fraction and
    the top hubs summarize it. Reciprocity is the fraction of edges ``i -> j`` whose reverse
    ``j -> i`` also exists.

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.
        top_hubs: Number of hubs listed. Defaults to 20.

    Returns:
        See :func:`directed_metrics_from_edges`.
    """

    return directed_metrics_from_edges(directed_edges(neighbors, n), n, top_hubs)


def directed_metrics_from_edges(edges: np.ndarray, n: int, top_hubs: int = 20) -> dict[str, Any]:
    """:func:`directed_metrics` on distinct loop-free edges from :func:`directed_edges`.

    Args:
        edges: int64 array ``[E, 2]`` of directed edges.
        n: Number of vertices.
        top_hubs: Number of hubs listed. Defaults to 20.

    Returns:
        Dictionary with the vertex and edge counts, ``reciprocity``, the mean, standard
        deviation (population), maximum, skewness and zero fraction of the in-degree, the
        mean, minimum and maximum out-degree, ``hubs`` (``(vertex, in_degree)`` by decreasing
        in-degree, ties by smaller vertex) and the in- and out-degree histograms.
    """

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    indeg = np.bincount(edges[:, 1], minlength=n)
    outdeg = np.bincount(edges[:, 0], minlength=n)
    keys = edges[:, 0] * n + edges[:, 1]
    reverse = edges[:, 1] * n + edges[:, 0]
    order = np.lexsort((np.arange(n), -indeg))[:top_hubs]
    return {
        "n_vertices": n,
        "n_edges": int(edges.shape[0]),
        "reciprocity": float(np.isin(reverse, keys).mean()) if keys.size else math.nan,
        "in_degree_mean": float(indeg.mean()) if n else math.nan,
        "in_degree_std": float(indeg.std()) if n else math.nan,
        "in_degree_max": int(indeg.max()) if n else 0,
        "in_degree_skewness": _skewness(indeg.astype(np.float64)) if n else math.nan,
        "in_degree_zero_fraction": float(np.mean(indeg == 0)) if n else math.nan,
        "out_degree_mean": float(outdeg.mean()) if n else math.nan,
        "out_degree_min": int(outdeg.min()) if n else 0,
        "out_degree_max": int(outdeg.max()) if n else 0,
        "hubs": [(int(v), int(indeg[v])) for v in order],
        "in_degree_hist": histogram(indeg),
        "out_degree_hist": histogram(outdeg),
    }


# --------------------------------------------------------------------------------------------
# comparing graphs on the same vertices, components, directed clustering


def edge_keys(g: ig.Graph, ignore_direction: bool = False) -> np.ndarray:
    """Sorted distinct ``i * n + j`` keys of the edges (``i < j`` when undirected, or when
    ``ignore_direction``).

    Args:
        g: igraph graph.
        ignore_direction: Treat the arcs of a digraph as unordered pairs.

    Returns:
        int64 array of distinct keys, sorted.
    """

    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    if ignore_direction or not g.is_directed():
        e = np.sort(e, axis=1)
    return np.unique(e[:, 0] * g.vcount() + e[:, 1])


def key_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index of two arrays of distinct keys (edge sets from :func:`edge_keys`).

    Args:
        a: Distinct keys of the first edge set.
        b: Distinct keys of the second edge set.

    Returns:
        ``|a & b| / |a | b|``.

    Raises:
        ZeroDivisionError: If both arrays are empty.
    """

    common = np.intersect1d(a, b, assume_unique=True).size
    return common / (a.size + b.size - common)


def same_label_share(g: ig.Graph, labels: np.ndarray) -> float:
    """Share of the edges (arcs) whose two ends have the same label (token id, theme...).

    Args:
        g: igraph graph.
        labels: Label of every vertex, ``[n]``.

    Returns:
        Fraction of edges with equal end labels; NaN for a graph without edges.
    """

    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels)
    return float(np.mean(labels[e[:, 0]] == labels[e[:, 1]]))


def component_rank(clustering: ig.VertexClustering) -> np.ndarray:
    """Component index of every vertex, renumbered so that 0 is the largest (stable on ties).

    Args:
        clustering: Components from ``Graph.connected_components``.

    Returns:
        int64 array ``[n]`` with the rank of each vertex's component by decreasing size.
    """

    order = np.argsort([-s for s in clustering.sizes()], kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[np.asarray(clustering.membership)]


def directed_clustering(g: ig.Graph) -> tuple[np.ndarray, float]:
    """Local and global clustering of a digraph (Fagiolo 2007), on ``S = A + A^T``.

    ``t_i = (S^3)_ii / 2`` counts the directed triangles through ``i`` (every arc pattern of the
    triangle counts) and ``d_i^tot (d_i^tot - 1) - 2 d_i^<->`` the triangles that could exist
    given the in-, out- and reciprocal degrees of ``i``. The local coefficient is their ratio,
    undefined (NaN) when the denominator is 0, which happens exactly when ``i`` has fewer than
    two distinct neighbours (``networkx.clustering`` gives 0 there). The global one, the
    directed transitivity, is the ratio of their sums. On a symmetric graph both reduce to the
    undirected coefficients.

    Args:
        g: Simple directed igraph graph (no loops, no repeated arcs).

    Returns:
        ``(local, transitivity)``: float array ``[n]`` and the global coefficient (NaN when
        no vertex has a possible triangle).

    Notes:
        The triangle count is a pure Python loop over pairs of neighbours of every vertex,
        so the cost grows with the sum of squared total degrees.
    """

    n = g.vcount()
    weights: list[dict[int, int]] = [{} for _ in range(n)]
    for a, b in g.get_edgelist():
        weights[a][b] = weights[a].get(b, 0) + 1
        weights[b][a] = weights[b].get(a, 0) + 1
    triangles = np.zeros(n)
    total = np.zeros(n)
    reciprocal = np.zeros(n)
    for i, row in enumerate(weights):
        items = list(row.items())
        total[i] = sum(row.values())
        reciprocal[i] = sum(1 for w in row.values() if w == 2)
        closed = 0
        for j, wj in items:
            row_j = weights[j]
            for h, wh in items:
                wjh = row_j.get(h)
                if wjh:
                    closed += wj * wh * wjh
        triangles[i] = closed / 2
    possible = total * (total - 1) - 2 * reciprocal
    local = np.divide(triangles, possible, out=np.full(n, np.nan), where=possible > 0)
    return local, float(triangles.sum() / possible.sum())
