"""GraphML networks, figures and table of the partial delivery (Entrega Parcial, MO438).

Exports the four networks the rest of the project uses — the occurrence k-NN graphs of the
representations ``lex``, ``L01``, ``L18`` and ``L36`` with k = 10, union symmetrization, random
tie-breaking, seed 0 — as GraphML, with the occurrence metadata and the per-vertex measures as
vertex attributes. The edges are built with the same functions as the ``metrics`` stage, from the
``knn`` stage neighbour sets of the main run, and their fingerprint is checked against
``metrics/graph_metrics.csv``.

Every number and figure is then computed from the GraphML files *read back from disk* (with
igraph), so the delivered files are exactly what was measured:

1. vertices, edges, mean degree;
2. global clustering: transitivity (ratio of closed triples) and mean local clustering;
3. local clustering of every vertex and its distribution;
4. mean distance and 5. diameter (exact, over all pairs of the largest component);
6. density; 7. degree distribution; 8. components and their size distribution.

Usage::

    .venv/bin/python scripts/partial_delivery/export_networks.py [--run outputs/experiment/main]
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
from gender_networks.graphs import edge_hash, graph_metrics, symmetrized_edges
from gender_networks.metrics import community_file, graph_id, select_neighbors

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "outputs" / "experiment" / "main"
DEFAULT_GRAPHML = REPO / "data" / "graphml"
DEFAULT_OUT = REPO / "report" / "entrega-parcial"

REPS = ["lex", "L01", "L18", "L36"]
K = 10
SYM = "union"
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


def build_graph(run: Path, rep: str, vertices: pd.DataFrame) -> tuple[nx.Graph, str]:
    """Union k-NN graph of ``rep`` with vertex attributes; returns it and its edge hash."""

    n = len(vertices)
    with np.load(run / "knn" / neighbors_name(rep, K)) as data:
        neighbors = select_neighbors(data, TIE, SEED)
    directed = symmetrized_edges(neighbors, n, "directed")
    edges = symmetrized_edges(neighbors, n, SYM)
    fingerprint = edge_hash(edges, n, directed=False)
    gid = graph_id(rep, K, SYM, TIE, SEED)
    expected = reference_row(run, gid)["edge_hash"]
    if fingerprint != expected:
        raise ValueError(
            f"{gid}: edges differ from the metrics stage ({fingerprint} != {expected})"
        )

    with np.load(run / "metrics" / "communities" / community_file(gid, RESOLUTION)) as data:
        community = np.asarray(data["membership"], dtype=np.int64)
    in_degree = np.bincount(directed[:, 1], minlength=n)

    g = nx.Graph(
        name=f"{rep}_k{K}_{SYM}",
        representation=rep,
        representation_description=REP_DESCRIPTIONS[rep],
        k=K,
        symmetrization="união: {i, j} quando j está entre os k vizinhos de i ou i entre os de j",
        similarity="cosseno em float64",
        tie_breaking=f"aleatório, semente {SEED}",
        community="Leiden (modularidade, resolução 1,0), melhor de 10 execuções",
        model=f"{MODEL}@{MODEL_REVISION}",
        corpus="Wikipédia em português (revisões em data/corpus/manifest.csv)",
        edge_hash=fingerprint,
    )
    records = vertices.to_dict(orient="records")
    for i, attrs in enumerate(records):
        attrs["label"] = attrs["token_text"]
        attrs["knn_in_degree"] = int(in_degree[i])
        attrs["community"] = int(community[i])
        g.add_node(i, **attrs)
    g.add_edges_from(map(tuple, edges.tolist()))
    return g, fingerprint


def add_vertex_measures(g: nx.Graph) -> None:
    """Degree, local clustering and component of every vertex (computed with igraph)."""

    ig_graph = ig.Graph(n=g.number_of_nodes(), edges=list(g.edges()))
    clustering = ig_graph.transitivity_local_undirected(mode="zero")
    components = ig_graph.connected_components()
    # Component 0 is the largest, then by decreasing size.
    order = np.argsort([-s for s in components.sizes()], kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    for i, (deg, cc, comp) in enumerate(
        zip(ig_graph.degree(), clustering, components.membership, strict=True)
    ):
        g.nodes[i]["degree"] = int(deg)
        g.nodes[i]["local_clustering"] = float(cc)
        g.nodes[i]["component"] = int(rank[comp])


def measure(path: Path) -> tuple[dict[str, Any], ig.Graph]:
    """Every metric of the delivery, from the GraphML file on disk."""

    g = ig.Graph.Read_GraphML(str(path))
    g.simplify()
    result = graph_metrics(g, distances="exact")
    result["local_clustering"] = np.asarray(
        g.transitivity_local_undirected(mode="zero"), dtype=float
    )
    result["degrees"] = np.asarray(g.degree(), dtype=np.int64)
    return result, g


def check_against_reference(rep: str, result: Mapping[str, Any], ref: Mapping[str, Any]) -> None:
    for key in (
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
    ):
        if not math.isclose(float(result[key]), float(ref[key]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{rep}: {key} = {result[key]} but graph_metrics.csv has {ref[key]}")


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


def figure_degrees(results: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Complementary cumulative degree distribution, log-log (the tail stays readable)."""

    fig, ax, styles = _half_figure(results, "Distribuição de graus", "both")
    for rep, result in results.items():
        degrees, counts = np.unique(result["degrees"], return_counts=True)
        x, y = plots.ccdf(degrees, counts)
        ax.step(x, y, where="post", color=styles[rep].color, linewidth=1.3)
        _sparse_markers(ax, x, y, styles[rep])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grau d")
    ax.set_ylabel("P(K ≥ d)")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    return fig


def figure_clustering(results: Mapping[str, Mapping[str, Any]]) -> Figure:
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
    ax.set_xlabel("coeficiente de clusterização local $C_i$")
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


def figure_components(results: Mapping[str, Mapping[str, Any]]) -> Figure:
    styles = plots.rep_styles(list(results))
    fig, axes = plots.figure(1, len(results), height=1.9, sharex=True, sharey=True)
    for ax, (rep, result) in zip(axes[0], results.items(), strict=True):
        sizes = result["component_sizes"]
        total = sum(sizes.values())
        plots.prepare(ax, f"{rep}: {total} componente{'s' if total > 1 else ''}", grid="both")
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
        giant = result["largest_component"]
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
    axes[0, 0].set_ylabel("nº de componentes")
    axes[0, 0].set_ylim(0.7, 40)
    axes[0, 0].set_xlim(7, 25_000)
    axes[0, 0].set_xticks([10, 100, 1_000, 10_000])
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: plots.fmt_int(v)))
    return fig


# --------------------------------------------------------------------------------------------
# table


def metrics_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for rep, r in results.items():
        cc = r["local_clustering"]
        rows.append(
            {
                "rede": rep,
                "vertices": r["n_vertices"],
                "arestas": r["n_edges"],
                "grau_medio": r["mean_degree"],
                "grau_min": int(r["degrees"].min()),
                "grau_max": int(r["degrees"].max()),
                "densidade": r["density"],
                "transitividade": r["transitivity"],
                "clusterizacao_local_media": r["avg_local_clustering"],
                "clusterizacao_local_mediana": float(np.median(cc)),
                "componentes": r["n_components"],
                "maior_componente": r["largest_component"],
                "fracao_maior_componente": r["largest_fraction"],
                "distancia_media": r["mean_distance"],
                "diametro": r["diameter"],
            }
        )
    return pd.DataFrame(rows)


def latex_table(table: pd.DataFrame) -> str:
    """Rows = metrics, columns = networks (fits the half page of the delivery)."""

    fmt = {
        "vertices": ("Vértices", plots.fmt_int),
        "arestas": ("Arestas", plots.fmt_int),
        "grau_medio": ("Grau médio", lambda v: plots.fmt_dec(v, 2)),
        "grau_min": ("Grau mínimo / máximo", None),
        "densidade": ("Densidade ($\\times 10^{-3}$)", lambda v: plots.fmt_dec(v * 1e3, 3)),
        "transitividade": ("Transitividade (clusterização global)", lambda v: plots.fmt_dec(v, 3)),
        "clusterizacao_local_media": ("Clusterização local média", lambda v: plots.fmt_dec(v, 3)),
        "componentes": ("Componentes", plots.fmt_int),
        "maior_componente": ("Maior componente (vértices)", None),
        "distancia_media": ("Distância média$^\\ast$", lambda v: plots.fmt_dec(v, 2)),
        "diametro": ("Diâmetro$^\\ast$", plots.fmt_int),
    }
    reps = table["rede"].tolist()
    lines = [
        "\\begin{tabular}{l" + "r" * len(reps) + "}",
        "\\toprule",
        "Métrica & " + " & ".join(reps) + " \\\\",
        "\\midrule",
    ]
    for key, (name, f) in fmt.items():
        cells = []
        for _, row in table.iterrows():
            if key == "grau_min":
                cells.append(f"{row['grau_min']} / {row['grau_max']}")
            elif key == "maior_componente":
                pct = plots.fmt_dec(100 * row["fracao_maior_componente"], 1)
                cells.append(f"{plots.fmt_int(row['maior_componente'])} ({pct}\\%)")
            else:
                cells.append(f(row[key]))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    vertices = vertex_table(args.run)
    args.graphml.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for rep in REPS:
        g, fingerprint = build_graph(args.run, rep, vertices)
        add_vertex_measures(g)
        path = args.graphml / f"{rep}_k{K}_{SYM}.graphml"
        nx.write_graphml(g, path, encoding="utf-8", prettyprint=True)

        result, back = measure(path)
        ref = reference_row(args.run, graph_id(rep, K, SYM, TIE, SEED))
        check_against_reference(rep, result, ref)
        stored = np.asarray(back.vs["local_clustering"], dtype=float)
        if not np.allclose(stored, result["local_clustering"], equal_nan=True):
            raise ValueError(f"{rep}: stored local clustering differs from the recomputed one")
        results[rep] = result
        size = path.stat().st_size / 2**20
        print(
            f"{path.relative_to(REPO)}: {result['n_vertices']} vértices, "
            f"{result['n_edges']} arestas, {size:.1f} MiB, arestas = metrics ({fingerprint[:10]})"
        )

    plots.apply_style()
    figures = args.out / "figuras"
    plots.save(figure_degrees(results), figures / "graus.pdf")
    plots.save(figure_clustering(results), figures / "clusterizacao_local.pdf")
    plots.save(figure_distances(results), figures / "distancias.pdf")
    plots.save(figure_components(results), figures / "componentes.pdf")

    table = metrics_table(results)
    tables = args.out / "tabelas"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "metricas.csv", index=False)
    (tables / "metricas.tex").write_text(latex_table(table), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.set_index("rede").T)


if __name__ == "__main__":
    main()
