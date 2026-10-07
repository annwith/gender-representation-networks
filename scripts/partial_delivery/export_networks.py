"""GraphML networks, figures and table of the partial delivery (Entrega Parcial, MO438).

Exports the four networks the rest of the project uses — the occurrence k-NN graphs of the
representations ``lex``, ``L01``, ``L18`` and ``L36`` with k = 10, random tie-breaking, seed 0 —
as GraphML, with the occurrence metadata and the per-vertex measures as vertex attributes. The
edges are built with the same functions as the ``metrics`` stage, from the ``knn`` stage
neighbour sets of the main run, and their fingerprint is checked against
``metrics/graph_metrics.csv``. ``--sym`` picks the version:

- ``union`` (default): the undirected graph with ``{i, j}`` when either endpoint chose the other;
- ``directed``: the k-NN relation itself, an arc ``i -> j`` for every ``j`` in ``N_i``.

Every number and figure is then computed from the GraphML files *read back from disk* (with
igraph), so the delivered files are exactly what was measured:

1. vertices, edges, mean degree;
2. global clustering: transitivity (ratio of closed triples) and mean local clustering;
3. local clustering of every vertex and its distribution;
4. mean distance and 5. diameter;
6. density; 7. degree distribution; 8. components and their size distribution.

In the union graph, distances are exact over all pairs of the largest component. In the
directed graph, ``d(i, j)`` is the length of the shortest directed path and is infinite when
``j`` cannot be reached from ``i``; the script reports the fraction of reachable ordered pairs,
the mean and maximum over the reachable pairs, the same two inside the largest strongly
connected component and the global efficiency (mean of ``1/d``, with ``1/inf = 0``). Directed
clustering follows Fagiolo (2007) and is checked against ``networkx.clustering``.

Usage::

    .venv/bin/python scripts/partial_delivery/export_networks.py [--sym union|directed]
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from gender_networks import plots
from gender_networks.artifacts import neighbors_name
from gender_networks.graphs import edge_hash, graph_metrics, histogram, symmetrized_edges
from gender_networks.metrics import community_file, graph_id, select_neighbors

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "outputs" / "experiment" / "main"
DEFAULT_GRAPHML = REPO / "data" / "graphml"
DEFAULT_OUT = {
    "union": REPO / "report" / "entrega-parcial",
    "directed": REPO / "report" / "entrega-parcial-direcionada",
}

REPS = ["lex", "L01", "L18", "L36"]
K = 10
TIE = "rand"
SEED = 0
RESOLUTION = 1.0
MODEL = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
REP_DESCRIPTIONS = {
    "lex": "embedding de entrada do token (rede lexical: ocorrências do mesmo token coincidem)",
    "L01": "saída do bloco 1",
    "L18": "saída do bloco 18",
    "L36": "saída do bloco 36 (último)",
}
SYM_DESCRIPTIONS = {
    "union": "união: {i, j} quando j está entre os k vizinhos de i ou i entre os de j",
    "directed": "direcionada: arco i -> j quando j está entre os k vizinhos de i",
}

# Occurrence columns kept as vertex attributes (the rest stays in occurrences.csv).
VERTEX_COLUMNS = {
    "occurrence_id": int,
    "token_id": int,
    "token_text": str,
    "has_leading_space": bool,
    "token_category": str,
    "is_function_word": bool,
    "word": str,
    "stratum": str,
    "theme": str,
    "target_word": str,
    "sense_theme": str,
    "pageid": int,
    "revid": int,
    "title": str,
    "paragraph_id": str,
    "pos_in_sequence": int,
    "local_context": str,
}

# Characters XML 1.0 forbids (control characters other than tab, newline, carriage return).
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")


def xml_safe(text: str) -> str:
    return _XML_INVALID.sub("�", text)


def vertex_table(run: Path) -> pd.DataFrame:
    occ = pd.read_csv(run / "sample" / "occurrences.csv", keep_default_na=False)
    if not np.array_equal(occ["occurrence_id"].to_numpy(), np.arange(len(occ))):
        raise ValueError("occurrence_id must be the row index (vertex id of the k-NN graphs)")
    table = pd.DataFrame(index=occ.index)
    for column, kind in VERTEX_COLUMNS.items():
        values = occ[column]
        if kind is bool:
            table[column] = values.astype(str).str.lower().eq("true")
        elif kind is int:
            table[column] = values.astype(np.int64)
        else:
            table[column] = values.astype(str).map(xml_safe)
    return table


def reference_row(run: Path, gid: str) -> Mapping[str, Any]:
    table = pd.read_csv(run / "metrics" / "graph_metrics.csv")
    rows = table[table["graph_id"] == gid]
    if len(rows) != 1:
        raise ValueError(f"{gid} is not (once) in graph_metrics.csv")
    return rows.iloc[0].to_dict()


def build_graph(run: Path, rep: str, vertices: pd.DataFrame, sym: str) -> tuple[nx.Graph, str]:
    """k-NN graph of ``rep`` with vertex attributes; returns it and its edge hash."""

    n = len(vertices)
    directed = sym == "directed"
    with np.load(run / "knn" / neighbors_name(rep, K)) as data:
        neighbors = select_neighbors(data, TIE, SEED)
    arcs = symmetrized_edges(neighbors, n, "directed")
    edges = symmetrized_edges(neighbors, n, sym)
    fingerprint = edge_hash(edges, n, directed=directed)
    gid = graph_id(rep, K, sym, TIE, SEED)
    expected = reference_row(run, gid)["edge_hash"]
    if fingerprint != expected:
        raise ValueError(
            f"{gid}: edges differ from the metrics stage ({fingerprint} != {expected})"
        )

    # Communities are always found on the union graph (Leiden optimizes undirected modularity).
    union_gid = graph_id(rep, K, "union", TIE, SEED)
    with np.load(run / "metrics" / "communities" / community_file(union_gid, RESOLUTION)) as data:
        community = np.asarray(data["membership"], dtype=np.int64)
    in_degree = np.bincount(arcs[:, 1], minlength=n)

    g = (nx.DiGraph if directed else nx.Graph)(
        name=f"{rep}_k{K}_{sym}",
        representation=rep,
        representation_description=REP_DESCRIPTIONS[rep],
        k=K,
        symmetrization=SYM_DESCRIPTIONS[sym],
        similarity="cosseno em float64",
        tie_breaking=f"aleatório, semente {SEED}",
        community="Leiden (modularidade, resolução 1,0) na rede de união, melhor de 10 execuções",
        model=f"{MODEL}@{MODEL_REVISION}",
        corpus="Wikipédia em português (revisões em data/corpus/manifest.csv)",
        edge_hash=fingerprint,
    )
    records = vertices.to_dict(orient="records")
    for i, attrs in enumerate(records):
        attrs["label"] = attrs["token_text"]
        if not directed:
            attrs["knn_in_degree"] = int(in_degree[i])
        attrs["community"] = int(community[i])
        g.add_node(i, **attrs)
    g.add_edges_from(map(tuple, edges.tolist()))
    return g, fingerprint


def _size_rank(clustering: ig.VertexClustering) -> np.ndarray:
    """Component index of every vertex, renumbered so that 0 is the largest (stable on ties)."""

    order = np.argsort([-s for s in clustering.sizes()], kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[np.asarray(clustering.membership)]


def directed_clustering(g: ig.Graph) -> tuple[np.ndarray, float]:
    """Local and global clustering of a digraph (Fagiolo 2007), on ``S = A + A^T``.

    ``t_i = (S^3)_ii / 2`` counts the directed triangles through ``i`` (every arc pattern of the
    triangle counts) and ``d_i^tot (d_i^tot - 1) - 2 d_i^<->`` the triangles that could exist
    given the in-, out- and reciprocal degrees of ``i``. The local coefficient is their ratio
    (0 when the denominator is 0, as in ``networkx.clustering``) and the global one, the
    directed transitivity, is the ratio of their sums. On a symmetric graph both reduce to the
    undirected coefficients.
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
    local = np.divide(triangles, possible, out=np.zeros(n), where=possible > 0)
    return local, float(triangles.sum() / possible.sum())


def add_vertex_measures(g: nx.Graph) -> None:
    """Degrees, local clustering and components of every vertex (computed with igraph)."""

    directed = g.is_directed()
    ig_graph = ig.Graph(n=g.number_of_nodes(), edges=list(g.edges()), directed=directed)
    if directed:
        clustering, _ = directed_clustering(ig_graph)
        weak = _size_rank(ig_graph.connected_components(mode="weak"))
        strong = _size_rank(ig_graph.connected_components(mode="strong"))
        for i, (din, dout) in enumerate(
            zip(ig_graph.indegree(), ig_graph.outdegree(), strict=True)
        ):
            g.nodes[i]["in_degree"] = int(din)
            g.nodes[i]["out_degree"] = int(dout)
            g.nodes[i]["local_clustering"] = float(clustering[i])
            g.nodes[i]["weak_component"] = int(weak[i])
            g.nodes[i]["strong_component"] = int(strong[i])
        return
    clustering = ig_graph.transitivity_local_undirected(mode="zero")
    component = _size_rank(ig_graph.connected_components())
    for i, (deg, cc) in enumerate(zip(ig_graph.degree(), clustering, strict=True)):
        g.nodes[i]["degree"] = int(deg)
        g.nodes[i]["local_clustering"] = float(cc)
        g.nodes[i]["component"] = int(component[i])


def _distance_summary(hist: ig.Histogram, prefix: str) -> dict[str, Any]:
    """Mean, maximum and histogram of the finite distances of a ``path_length_hist``."""

    bins = {int(low): int(count) for low, _high, count in hist.bins() if count}
    reached = sum(bins.values())
    return {
        f"{prefix}distance_hist": bins,
        f"{prefix}reachable_pairs": reached,
        f"{prefix}unreachable_pairs": int(hist.unconnected),
        f"{prefix}mean_distance": sum(d * c for d, c in bins.items()) / reached,
        f"{prefix}diameter": max(bins),
    }


def measure_union(g: ig.Graph) -> dict[str, Any]:
    result = graph_metrics(g, distances="exact")
    result["local_clustering"] = np.asarray(
        g.transitivity_local_undirected(mode="zero"), dtype=float
    )
    result["degrees"] = np.asarray(g.degree(), dtype=np.int64)
    return result


def measure_directed(g: ig.Graph) -> dict[str, Any]:
    n, m = g.vcount(), g.ecount()
    pairs = n * (n - 1)
    local, transitivity = directed_clustering(g)
    weak = g.connected_components(mode="weak")
    strong = g.connected_components(mode="strong")
    in_degree = np.asarray(g.indegree(), dtype=np.int64)
    result: dict[str, Any] = {
        "n_vertices": n,
        "n_edges": m,
        "mean_degree": m / n,
        "density": m / pairs,
        "reciprocity": g.reciprocity(ignore_loops=True, mode="default"),
        "in_degrees": in_degree,
        "out_degrees": np.asarray(g.outdegree(), dtype=np.int64),
        "in_degree_max": int(in_degree.max()),
        "in_degree_zero_fraction": float(np.mean(in_degree == 0)),
        "transitivity": transitivity,
        "avg_local_clustering": float(local.mean()),
        "local_clustering": local,
        "n_components": len(weak),
        "largest_component": max(weak.sizes()),
        "n_strong_components": len(strong),
        "largest_strong": max(strong.sizes()),
        "strong_sizes": histogram(strong.sizes()),
    }
    result.update(_distance_summary(g.path_length_hist(directed=True), ""))
    if result["reachable_pairs"] + result["unreachable_pairs"] != pairs:
        raise ValueError("path_length_hist does not cover every ordered pair")
    result["reachable_fraction"] = result["reachable_pairs"] / pairs
    # Global efficiency (Latora & Marchiori 2001): mean of 1/d over all ordered pairs, 1/inf = 0.
    result["efficiency"] = sum(c / d for d, c in result["distance_hist"].items()) / pairs
    result.update(_distance_summary(strong.giant().path_length_hist(directed=True), "scc_"))
    return result


def measure(path: Path) -> tuple[dict[str, Any], ig.Graph]:
    """Every metric of the delivery, from the GraphML file on disk."""

    g = ig.Graph.Read_GraphML(str(path))
    g.simplify()
    return (measure_directed if g.is_directed() else measure_union)(g), g


REFERENCE_KEYS = {
    "union": (
        "n_vertices",
        "n_edges",
        "mean_degree",
        "density",
        "transitivity",
        "avg_local_clustering",
        "n_components",
        "largest_component",
        "mean_distance",
        "diameter",
    ),
    "directed": (
        "n_vertices",
        "n_edges",
        "mean_degree",
        "density",
        "reciprocity",
        "in_degree_max",
        "in_degree_zero_fraction",
        "n_components",
        "largest_component",
    ),
}


def check_against_reference(
    rep: str, sym: str, result: Mapping[str, Any], ref: Mapping[str, Any]
) -> None:
    for key in REFERENCE_KEYS[sym]:
        if not math.isclose(float(result[key]), float(ref[key]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{rep}: {key} = {result[key]} but graph_metrics.csv has {ref[key]}")


def check_clustering_with_networkx(rep: str, path: Path, local: np.ndarray) -> None:
    """Directed local clustering against the reference implementation of NetworkX."""

    reference = nx.clustering(nx.read_graphml(path, node_type=int))
    expected = np.array([reference[i] for i in range(local.size)])
    if not np.allclose(local, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{rep}: directed clustering differs from networkx.clustering")


# --------------------------------------------------------------------------------------------
# figures (half text width, so that two fit side by side; legend above the plot)

HALF = 3.1


def _half_figure(
    results: Mapping[str, Mapping[str, Any]], title: str, grid: str
) -> tuple[Figure, Axes, dict[str, plots.Style]]:
    styles = plots.rep_styles(list(results))
    fig, axes = plots.figure(1, 1, height=2.4, width=HALF)
    ax = axes[0, 0]
    plots.prepare(ax, title, grid=grid)
    plots.legend_top(fig, [plots.line_handle(styles[r]) for r in results], ncol=len(results))
    return fig, ax, styles


def _sparse_markers(ax: Axes, x: np.ndarray, y: np.ndarray, style: plots.Style) -> None:
    every = max(1, x.size // 8)
    ax.plot(
        x[::every],
        y[::every],
        linestyle="none",
        marker=style.marker,
        color=style.color,
        markersize=3.8,
        markeredgecolor=plots.SURFACE,
        markeredgewidth=0.5,
    )


def figure_degrees(
    results: Mapping[str, Mapping[str, Any]],
    key: str = "degrees",
    title: str = "Distribuição de graus",
    xlabel: str = "grau d",
) -> Figure:
    """Complementary cumulative degree distribution, log-log (the tail stays readable)."""

    fig, ax, styles = _half_figure(results, title, "both")
    for rep, result in results.items():
        degrees, counts = np.unique(result[key], return_counts=True)
        x, y = plots.ccdf(degrees, counts)
        ax.step(x, y, where="post", color=styles[rep].color, linewidth=1.3)
        _sparse_markers(ax, x, y, styles[rep])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(K ≥ d)")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t >= ax.get_xlim()[0]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    return fig


def figure_clustering(
    results: Mapping[str, Mapping[str, Any]],
    xlabel: str = "coeficiente de clusterização local $C_i$",
) -> Figure:
    fig, ax, styles = _half_figure(results, "Clusterização local dos vértices", "y")
    bins = np.linspace(0, 1, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    for rep, result in results.items():
        counts, _ = np.histogram(result["local_clustering"], bins=bins)
        share = counts / counts.sum()
        style = styles[rep]
        ax.step(bins, np.append(share, share[-1]), where="post", color=style.color, linewidth=1.2)
        _sparse_markers(ax, centers[2:], share[2:], style)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fração dos vértices")
    plots.decimal_axis(ax, "x")
    ax.yaxis.set_major_formatter(plots.percent_formatter())
    return fig


def figure_distances(results: Mapping[str, Mapping[str, Any]]) -> Figure:
    fig, ax, styles = _half_figure(results, "Distâncias na maior componente", "y")
    for rep, result in results.items():
        hist = result["distance_hist"]
        d = np.array(sorted(hist))
        c = np.array([hist[x] for x in d], dtype=float)
        plots.plot_series(ax, d, c / c.sum(), styles[rep], linewidth=1.2, markersize=3.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("distância (número de arestas)")
    ax.set_ylabel("fração dos pares")
    ax.yaxis.set_major_formatter(plots.percent_formatter())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def figure_directed_distances(results: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Finite directed distances as a share of *all* ordered pairs, and the unreachable rest."""

    styles = plots.rep_styles(list(results))
    fig, axes = plots.figure(1, 2, height=2.3, width_ratios=[2.2, 1])
    ax, bar_ax = axes[0]
    plots.prepare(ax, "Distância d(i, j) entre pares ordenados com caminho", grid="y")
    for rep, result in results.items():
        hist = result["distance_hist"]
        n = result["n_vertices"]
        d = np.array(sorted(hist))
        c = np.array([hist[x] for x in d], dtype=float)
        style = styles[rep]
        ax.plot(d, c / (n * (n - 1)), color=style.color, linewidth=1.2)
        _sparse_markers(ax, d, c / (n * (n - 1)), style)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, None)
    ax.set_xlabel("distância (número de arcos)")
    ax.set_ylabel("fração dos pares ordenados")
    ax.yaxis.set_major_formatter(plots.percent_formatter())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plots.prepare(bar_ax, "Pares sem caminho (d = ∞)", grid="x")
    reps = list(results)
    shares = [1 - results[r]["reachable_fraction"] for r in reps]
    ys = np.arange(len(reps))[::-1]
    for y, rep, share in zip(ys, reps, shares, strict=True):
        bar_ax.barh(y, share, height=0.62, color=styles[rep].color, linewidth=0)
        bar_ax.text(
            share + 0.02,
            y,
            f"{plots.fmt_dec(100 * share, 1)}%",
            va="center",
            ha="left",
            fontsize=7.5,
            color=plots.INK_SECONDARY,
        )
    bar_ax.set_yticks(ys, reps)
    bar_ax.set_xlim(0, 1.15)
    bar_ax.set_xticks([0, 0.5, 1])
    bar_ax.xaxis.set_major_formatter(plots.percent_formatter())
    bar_ax.spines["left"].set_visible(False)
    plots.legend_top(fig, [plots.line_handle(styles[r]) for r in reps], ncol=len(reps))
    return fig


def figure_components(
    results: Mapping[str, Mapping[str, Any]],
    sizes_key: str = "component_sizes",
    giant_key: str = "largest_component",
    noun: tuple[str, str] = ("componente", "componentes"),
    noun_in_title: bool = True,
) -> Figure:
    styles = plots.rep_styles(list(results))
    fig, axes = plots.figure(1, len(results), height=1.9, sharex=True, sharey=True)
    ymax = 1
    for ax, (rep, result) in zip(axes[0], results.items(), strict=True):
        sizes = result[sizes_key]
        total = sum(sizes.values())
        ymax = max(ymax, max(sizes.values()))
        title = f"{rep}: {plots.fmt_int(total)}"
        if noun_in_title:
            title += f" {noun[0] if total == 1 else noun[1]}"
        plots.prepare(ax, title, grid="both")
        x = np.array(sorted(sizes), dtype=float)
        y = np.array([sizes[int(s)] for s in x], dtype=float)
        style = styles[rep]
        ax.vlines(x, 0.7, y, color=style.color, linewidth=1.0)
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=style.marker,
            color=style.color,
            markersize=4.2,
            markeredgecolor=plots.SURFACE,
            markeredgewidth=0.5,
            clip_on=False,
        )
        giant = result[giant_key]
        ax.annotate(
            plots.fmt_int(giant),
            (giant, 1),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=plots.INK_SECONDARY,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("tamanho (vértices)")
    axes[0, 0].set_ylabel(f"nº de {noun[1]}")
    axes[0, 0].set_ylim(0.7, 40 if ymax < 40 else 2 * ymax)
    small = min(min(r[sizes_key]) for r in results.values())
    axes[0, 0].set_xlim(0.7 if small < 7 else 7, 25_000)
    ticks = [1, 100, 10_000] if small < 7 else [10, 100, 1_000, 10_000]
    axes[0, 0].set_xticks(ticks)
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    return fig


def write_figures(results: Mapping[str, Mapping[str, Any]], sym: str, figures: Path) -> None:
    plots.apply_style()
    if sym == "union":
        plots.save(figure_degrees(results), figures / "graus.pdf")
        plots.save(figure_clustering(results), figures / "clusterizacao_local.pdf")
        plots.save(figure_distances(results), figures / "distancias.pdf")
        plots.save(figure_components(results), figures / "componentes.pdf")
        return
    plots.save(
        figure_degrees(
            results, "in_degrees", "Distribuição dos graus de entrada", "grau de entrada d"
        ),
        figures / "graus.pdf",
    )
    plots.save(
        figure_clustering(results, "clusterização local dirigida $C_i^{\\rightarrow}$"),
        figures / "clusterizacao_local.pdf",
    )
    plots.save(figure_directed_distances(results), figures / "distancias.pdf")
    plots.save(
        figure_components(
            results,
            "strong_sizes",
            "largest_strong",
            ("componente forte", "componentes fortes"),
            noun_in_title=False,
        ),
        figures / "componentes.pdf",
    )


# --------------------------------------------------------------------------------------------
# tables


def metrics_table(results: Mapping[str, Mapping[str, Any]], sym: str) -> pd.DataFrame:
    rows = []
    for rep, r in results.items():
        row = {
            "rede": rep,
            "vertices": r["n_vertices"],
            "arestas": r["n_edges"],
            "grau_medio": r["mean_degree"],
            "densidade": r["density"],
            "transitividade": r["transitivity"],
            "clusterizacao_local_media": r["avg_local_clustering"],
            "clusterizacao_local_mediana": float(np.median(r["local_clustering"])),
            "componentes": r["n_components"],
            "maior_componente": r["largest_component"],
            "fracao_maior_componente": r["largest_component"] / r["n_vertices"],
        }
        if sym == "union":
            row |= {
                "grau_min": int(r["degrees"].min()),
                "grau_max": int(r["degrees"].max()),
                "distancia_media": r["mean_distance"],
                "diametro": r["diameter"],
            }
        else:
            n = r["n_vertices"]
            row |= {
                "grau_entrada_max": r["in_degree_max"],
                "fracao_grau_entrada_zero": r["in_degree_zero_fraction"],
                "reciprocidade": r["reciprocity"],
                "componentes_fortes": r["n_strong_components"],
                "maior_componente_forte": r["largest_strong"],
                "fracao_maior_componente_forte": r["largest_strong"] / n,
                "fracao_pares_alcancaveis": r["reachable_fraction"],
                "distancia_media_alcancaveis": r["mean_distance"],
                "maior_distancia_finita": r["diameter"],
                "distancia_media_cfc": r["scc_mean_distance"],
                "diametro_cfc": r["scc_diameter"],
                "eficiencia_global": r["efficiency"],
                "distancia_harmonica": 1 / r["efficiency"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _count_share(count: int, share: float) -> str:
    return f"{plots.fmt_int(count)} ({plots.fmt_dec(100 * share, 1)}\\%)"


def _dec(digits: int) -> Any:
    return lambda row, key: plots.fmt_dec(row[key], digits)


def _int(row: Mapping[str, Any], key: str) -> str:
    return plots.fmt_int(row[key])


def _pct(row: Mapping[str, Any], key: str) -> str:
    return f"{plots.fmt_dec(100 * row[key], 1)}\\%"


LATEX_ROWS = {
    "union": [
        ("Vértices", "vertices", _int),
        ("Arestas", "arestas", _int),
        ("Grau médio", "grau_medio", _dec(2)),
        ("Grau mínimo / máximo", "grau_min", lambda r, _: f"{r['grau_min']} / {r['grau_max']}"),
        (
            "Densidade ($\\times 10^{-3}$)",
            "densidade",
            lambda r, k: plots.fmt_dec(r[k] * 1e3, 3),
        ),
        ("Transitividade (clusterização global)", "transitividade", _dec(3)),
        ("Clusterização local média", "clusterizacao_local_media", _dec(3)),
        ("Componentes", "componentes", _int),
        (
            "Maior componente (vértices)",
            "maior_componente",
            lambda r, k: _count_share(r[k], r["fracao_maior_componente"]),
        ),
        ("Distância média$^\\ast$", "distancia_media", _dec(2)),
        ("Diâmetro$^\\ast$", "diametro", _int),
    ],
    "directed": [
        ("Vértices", "vertices", _int),
        ("Arcos", "arestas", _int),
        ("Grau médio de entrada = de saída", "grau_medio", _dec(0)),
        ("Grau de entrada máximo", "grau_entrada_max", _int),
        ("Vértices com grau de entrada 0", "fracao_grau_entrada_zero", _pct),
        ("Reciprocidade (arcos com arco inverso)", "reciprocidade", _pct),
        (
            "Densidade ($\\times 10^{-4}$)",
            "densidade",
            lambda r, k: plots.fmt_dec(r[k] * 1e4, 3),
        ),
        ("Transitividade dirigida $T^{\\rightarrow}$", "transitividade", _dec(3)),
        (
            "Clusterização local média $\\bar C^{\\rightarrow}$",
            "clusterizacao_local_media",
            _dec(3),
        ),
        ("Componentes fracas", "componentes", _int),
        ("Componentes fortes", "componentes_fortes", _int),
        (
            "Maior componente forte (vértices)",
            "maior_componente_forte",
            lambda r, k: _count_share(r[k], r["fracao_maior_componente_forte"]),
        ),
        ("Pares ordenados com caminho", "fracao_pares_alcancaveis", _pct),
        ("(a) Distância média entre pares com caminho", "distancia_media_alcancaveis", _dec(2)),
        ("(a) Maior distância finita", "maior_distancia_finita", _int),
        ("(b) Distância média na maior comp.\\ forte", "distancia_media_cfc", _dec(2)),
        ("(b) Diâmetro da maior comp.\\ forte", "diametro_cfc", _int),
        ("(c) Eficiência global $E$", "eficiencia_global", _dec(3)),
        ("(c) Distância harmônica $1/E$", "distancia_harmonica", _dec(2)),
    ],
}


def latex_table(table: pd.DataFrame, sym: str) -> str:
    """Rows = metrics, columns = networks (fits the half page of the delivery)."""

    reps = table["rede"].tolist()
    lines = [
        "\\begin{tabular}{l" + "r" * len(reps) + "}",
        "\\toprule",
        "Métrica & " + " & ".join(reps) + " \\\\",
        "\\midrule",
    ]
    for name, key, cell in LATEX_ROWS[sym]:
        cells = [cell(row, key) for _, row in table.iterrows()]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sym", choices=sorted(DEFAULT_OUT), default="union")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    sym = args.sym
    out = args.out or DEFAULT_OUT[sym]

    vertices = vertex_table(args.run)
    args.graphml.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for rep in REPS:
        g, fingerprint = build_graph(args.run, rep, vertices, sym)
        add_vertex_measures(g)
        path = args.graphml / f"{rep}_k{K}_{sym}.graphml"
        nx.write_graphml(g, path, encoding="utf-8", prettyprint=True)

        result, back = measure(path)
        ref = reference_row(args.run, graph_id(rep, K, sym, TIE, SEED))
        check_against_reference(rep, sym, result, ref)
        stored = np.asarray(back.vs["local_clustering"], dtype=float)
        if not np.allclose(stored, result["local_clustering"], equal_nan=True):
            raise ValueError(f"{rep}: stored local clustering differs from the recomputed one")
        if sym == "directed":
            check_clustering_with_networkx(rep, path, result["local_clustering"])
        results[rep] = result
        size = path.stat().st_size / 2**20
        print(
            f"{path.relative_to(REPO)}: {result['n_vertices']} vértices, "
            f"{result['n_edges']} arestas, {size:.1f} MiB, arestas = metrics ({fingerprint[:10]})"
        )

    write_figures(results, sym, out / "figuras")
    table = metrics_table(results, sym)
    tables = out / "tabelas"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "metricas.csv", index=False)
    (tables / "metricas.tex").write_text(latex_table(table, sym), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.set_index("rede").T)


if __name__ == "__main__":
    main()
