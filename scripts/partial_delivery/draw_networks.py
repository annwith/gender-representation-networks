"""Drawings of the partial-delivery networks (Entrega Parcial, MO438, item 9).

Reads the four union GraphML files delivered in ``data/graphml`` (``lex``, ``L01``, ``L18`` and
``L36``, k = 10) and draws each with its own UMAP layout (igraph's ``layout_umap``, unweighted,
seeded through Python's ``random``), vertices colored by the article theme. It writes

- ``figuras/redes.pdf`` in both report folders: a 2x2 static figure. The view is the disc
  around the median position that holds 99% of the vertices; the rest are small groups pushed
  to the periphery, and the caption says how many are left out;
- ``docs/redes/``: the interactive page (``index.html``, copied from ``redes.html`` next to this
  script) and its data files, one with the vertex attributes and one per network with the
  positions and edges, so the page shows the same drawing as the PDF.

The directed GraphML of each network, with the arc directions dropped, must give exactly the
union edges: the same drawing therefore serves the directed report, with the arcs undirected.

Layouts take about two minutes per network and are cached in the run folder
(``layout/umap_<rep>.npz``), keyed by the graph's edge hash; ``--recompute`` ignores the cache.

Usage::

    .venv/bin/python scripts/partial_delivery/draw_networks.py [--recompute]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import igraph as ig
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from gender_networks import plots

REPO = Path(__file__).resolve().parents[2]
GRAPHML = REPO / "data" / "graphml"
LAYOUTS = REPO / "outputs" / "experiment" / "main" / "layout"
REPORTS = [REPO / "report" / "entrega-parcial", REPO / "report" / "entrega-parcial-direcionada"]
PAGE = REPO / "docs" / "redes"
TEMPLATE = Path(__file__).with_name("redes.html")

REPS = ["lex", "L01", "L18", "L36"]
K = 10
LAYOUT_SEED = 438
ORDER_SEED = 0
VIEW_QUANTILE = 0.99  # fraction of the vertices inside the static view
REP_TITLES = {
    "lex": "lex: embedding de entrada",
    "L01": "L01: saída do bloco 1",
    "L18": "L18: saída do bloco 18",
    "L36": "L36: saída do bloco 36",
}
THEMES = list(plots.THEME_LABELS)

# Vertex attributes shown by the page, all equal across the four networks.
PAGE_ATTRIBUTES = [
    "token_text",
    "word",
    "theme",
    "stratum",
    "token_category",
    "target_word",
    "title",
    "local_context",
]


def read(rep: str, sym: str) -> ig.Graph:
    g = ig.Graph.Read_GraphML(str(GRAPHML / f"{rep}_k{K}_{sym}.graphml"))
    g.simplify()
    return g


def edge_keys(g: ig.Graph) -> np.ndarray:
    """Sorted distinct ``lo * n + hi`` keys of the edges, ignoring their direction."""

    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    lo, hi = e.min(axis=1), e.max(axis=1)
    return np.unique(lo * g.vcount() + hi)


def check_directed_matches_union(rep: str, union: ig.Graph) -> None:
    directed = read(rep, "directed")
    if directed.vs["occurrence_id"] != union.vs["occurrence_id"]:
        raise ValueError(f"{rep}: directed and union files list the vertices in different orders")
    if not np.array_equal(edge_keys(directed), edge_keys(union)):
        raise ValueError(f"{rep}: the directed arcs, undirected, differ from the union edges")


def layout(rep: str, g: ig.Graph, recompute: bool) -> np.ndarray:
    """UMAP positions, from the cache when it was computed for the same edges."""

    path = LAYOUTS / f"umap_{rep}.npz"
    if path.exists() and not recompute:
        cached = np.load(path)
        if str(cached["edge_hash"]) == g["edge_hash"]:
            return cached["xy"]
    random.seed(LAYOUT_SEED)
    start = time.time()
    xy = np.asarray(g.layout_umap(min_dist=0.1, epochs=500).coords, dtype=np.float64)
    print(f"{rep}: layout UMAP em {time.time() - start:.0f} s")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, xy=xy, edge_hash=g["edge_hash"])
    return xy


def normalize(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions centered on the median and scaled so the view disc has radius 1.

    Returns the positions and the boolean mask of the vertices outside that disc.
    """

    centered = xy - np.median(xy, axis=0)
    radius = np.linalg.norm(centered, axis=1)
    scaled = centered / np.quantile(radius, VIEW_QUANTILE)
    return scaled, np.linalg.norm(scaled, axis=1) > 1


def theme_colors(g: ig.Graph) -> np.ndarray:
    styles = plots.categorical_styles(THEMES, plots.THEME_LABELS)
    return np.array([styles[t].color for t in g.vs["theme"]])


def figure_networks(graphs: dict[str, ig.Graph], positions: dict[str, np.ndarray]) -> Figure:
    fig, axes = plots.figure(2, 2, height=6.75)
    fig.set_dpi(300)  # the edges and vertices are rasterized at this resolution
    order = np.random.default_rng(ORDER_SEED).permutation(next(iter(graphs.values())).vcount())
    for ax, rep in zip(axes.flat, REPS, strict=True):
        g, (xy, _outside) = graphs[rep], normalize(positions[rep])
        edges = np.asarray(g.get_edgelist())
        ax.add_collection(
            LineCollection(
                xy[edges], colors=plots.AXIS, linewidths=0.06, alpha=0.18, rasterized=True
            )
        )
        colors = theme_colors(g)
        ax.scatter(
            xy[order, 0],
            xy[order, 1],
            s=1.6,
            c=colors[order],
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(REP_TITLES[rep], color=plots.INK, loc="left")
    styles = plots.categorical_styles(THEMES, plots.THEME_LABELS)
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=5,
            color=styles[t].color,
            label=styles[t].label,
        )
        for t in THEMES
    ]
    plots.legend_top(fig, handles, ncol=4)
    return fig


def compact(values: list[str]) -> tuple[list[str], list[int]]:
    """Distinct values (in first-seen order) and the index of each value among them."""

    table: dict[str, int] = {}
    codes = [table.setdefault(v, len(table)) for v in values]
    return list(table), codes


def write_page(graphs: dict[str, ig.Graph], positions: dict[str, np.ndarray]) -> None:
    data = PAGE / "dados"
    data.mkdir(parents=True, exist_ok=True)
    first = graphs[REPS[0]]
    vertices: dict[str, object] = {"n": first.vcount()}
    for attr in PAGE_ATTRIBUTES:
        values = ["" if v is None else str(v) for v in first.vs[attr]]
        if attr in ("theme", "stratum", "token_category", "title", "target_word"):
            vertices[attr] = dict(zip(("values", "codes"), compact(values), strict=True))
        else:
            vertices[attr] = values
    (data / "vertices.json").write_text(
        json.dumps(vertices, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    for rep in REPS:
        g = graphs[rep]
        xy, outside = normalize(positions[rep])
        network = {
            "rep": rep,
            "title": REP_TITLES[rep],
            "n_edges": g.ecount(),
            "n_outside": int(outside.sum()),
            "x": np.round(xy[:, 0] * 1000).astype(int).tolist(),
            "y": np.round(xy[:, 1] * 1000).astype(int).tolist(),
            "edges": np.asarray(g.get_edgelist(), dtype=int).ravel().tolist(),
            "community": [int(c) for c in g.vs["community"]],
            "degree": g.degree(),
        }
        (data / f"{rep}.json").write_text(
            json.dumps(network, separators=(",", ":")), encoding="utf-8"
        )
    shutil.copyfile(TEMPLATE, PAGE / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true", help="ignore the cached layouts")
    args = parser.parse_args()

    graphs: dict[str, ig.Graph] = {}
    positions: dict[str, np.ndarray] = {}
    for rep in REPS:
        g = read(rep, "union")
        if graphs and g.vs["occurrence_id"] != graphs[REPS[0]].vs["occurrence_id"]:
            raise ValueError(f"{rep}: the vertices differ from those of {REPS[0]}")
        check_directed_matches_union(rep, g)
        graphs[rep] = g
        positions[rep] = layout(rep, g, args.recompute)
        print(f"{rep}: {g.vcount()} vértices, {g.ecount()} arestas (= arcos direcionados)")

    plots.apply_style()
    for report in REPORTS:
        plots.save(figure_networks(graphs, positions), report / "figuras" / "redes.pdf")
    write_page(graphs, positions)
    print(f"figuras/redes.pdf nos dois relatórios; página em {PAGE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
