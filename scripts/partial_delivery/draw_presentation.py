"""One network per figure, and the arc numbers, for the partial-delivery talk (MO438).

The talk (``report/apresentacao-parcial``) shows one network per slide, larger than the 2 x 2
panels of ``draw_networks.py``. This script reuses that script's reading and colors and its
cached UMAP layouts (``outputs/experiment/main/layout``); it never recomputes a layout, so each
drawing is the one of the visualization delivery and of the interactive page, seen closer. It
writes, in ``report/apresentacao-parcial``:

- ``figuras/rede_<rep>_tema.pdf``: the network, vertices colored by the article theme. The view
  is a square around the central 96% of the vertices along each axis (``VIEW_TAIL``), with a
  10% margin: closer than the 99% disc of ``draw_networks.py``, so the dense center is
  legible. The number of vertices left out is printed (2% to 4% of each network);
- ``figuras/rede_<rep>_palavra.pdf``: the same drawing, colored by the word position;
- ``figuras/graus_pk.pdf``: the in-degree distribution ``P(K = d)`` of the directed networks,
  log-log, one point per degree, beside the CCDF ``graus.pdf`` of ``export_networks.py``. The
  vertices of in-degree 0 do not fit the log axis and are left out;
- ``tabelas/arcos.csv``: what the arcs of the directed networks join, and the nominal
  assortativity of theme and word position over the arcs. It is the directed counterpart of
  ``entrega-parcial-visualizacao/tabelas/arestas.csv``, which counts the union edges.

The drawings show the union edges, which are the arcs with their direction dropped
(``draw_networks.check_directed_matches_union``).

Usage::

    .venv/bin/python scripts/partial_delivery/draw_presentation.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draw_networks as dn  # noqa: E402
import export_networks as en  # noqa: E402

from gender_networks import labels, plots  # noqa: E402
from gender_networks.graphs import same_label_share  # noqa: E402

REPORT = dn.REPO / "report" / "apresentacao-parcial"
VIEW_TAIL = 0.02  # fraction of the vertices cut at each end of each axis
WORD_LEGEND = ["continua no token seguinte", "termina no token (ou pontuação)"]


def cached_layout(rep: str, g: ig.Graph) -> np.ndarray:
    """The UMAP positions of ``draw_networks.py``, only when they belong to these edges."""

    path = dn.LAYOUTS / f"umap_{rep}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run draw_networks.py first")
    cached = np.load(path)
    if str(cached["edge_hash"]) != g["edge_hash"]:
        raise ValueError(f"{path}: the layout belongs to other edges; run draw_networks.py")
    return cached["xy"]


def view(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Center and half side of the square view, and how many vertices fall outside it."""

    low, high = np.quantile(xy, [VIEW_TAIL, 1 - VIEW_TAIL], axis=0)
    center, half = (low + high) / 2, np.max(high - low) / 2 * 1.1
    outside = np.any(np.abs(xy - center) > half, axis=1)
    return center, half, int(outside.sum())


def figure_network(
    g: ig.Graph, xy: np.ndarray, colors: np.ndarray, legend: Sequence[tuple[str, str]]
) -> Figure:
    """One network, in the square ``view``, with the legend on the right."""

    fig, axes = plots.figure(1, 1, height=3.2, width=4.6)
    fig.set_dpi(300)  # the edges and vertices are rasterized at this resolution
    ax = axes[0, 0]
    center, half, _ = view(xy)
    edges = np.asarray(g.get_edgelist())
    ax.add_collection(
        LineCollection(xy[edges], colors=plots.AXIS, linewidths=0.08, alpha=0.14, rasterized=True)
    )
    order = np.random.default_rng(dn.ORDER_SEED).permutation(colors.size)
    ax.scatter(xy[order, 0], xy[order, 1], s=3.0, c=colors[order], linewidths=0, rasterized=True)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    handles = [
        Line2D([], [], linestyle="", marker="o", markersize=6, color=color, label=name)
        for name, color in legend
    ]
    fig.legend(handles=handles, loc="outside right upper", frameon=False, handletextpad=0.4)
    return fig


def figure_in_degrees(in_degrees: dict[str, np.ndarray]) -> Figure:
    """``P(K = d)`` of the in-degree, log-log, in the size and style of ``graus.pdf``."""

    fig, ax, styles = en._half_figure(in_degrees, "Distribuição dos graus de entrada", "both")
    for rep, degrees in in_degrees.items():
        d, counts = np.unique(degrees, return_counts=True)
        keep = d > 0
        ax.plot(
            d[keep],
            counts[keep] / degrees.size,
            linestyle="none",
            marker=styles[rep].marker,
            color=styles[rep].color,
            markersize=2.6,
            markeredgewidth=0,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grau de entrada d")
    ax.set_ylabel("P(K = d)")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: labels.fmt_int(v)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    return fig


def arc_composition(g: ig.Graph, continues: np.ndarray) -> dict[str, float]:
    """``draw_networks.edge_composition`` over the arcs of a directed network."""

    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    token = np.asarray(g.vs["token_id"], dtype=np.int64)
    _, theme = np.unique(np.asarray(g.vs["theme"]), return_inverse=True)
    position = continues.astype(np.int64)
    return {
        "arcos": g.ecount(),
        "mesmo_token": same_label_share(g, token),
        "mesmo_tema": same_label_share(g, theme),
        "assortatividade_tema": g.assortativity_nominal(theme.tolist(), directed=True),
        "continua_termina": float(np.mean(position[e[:, 0]] != position[e[:, 1]])),
        "assortatividade_palavra": g.assortativity_nominal(position.tolist(), directed=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    plots.apply_style()
    figures = REPORT / "figuras"
    rows = {}
    in_degrees = {}
    for rep in dn.REPS:
        union = dn.read(rep, "union")
        dn.check_directed_matches_union(rep, union)
        xy = cached_layout(rep, union)
        continues = dn.word_continues(union)
        word_colors, word_legend = dn.word_colors(continues)
        word_legend = [
            (name, color) for name, (_, color) in zip(WORD_LEGEND, word_legend, strict=True)
        ]
        for name, (colors, legend) in {
            "tema": dn.theme_colors(union),
            "palavra": (word_colors, word_legend),
        }.items():
            fig = figure_network(union, xy, colors, legend)
            plots.save(fig, figures / f"rede_{rep}_{name}.pdf")
        directed = dn.read(rep, "directed")
        if directed.vs["occurrence_id"] != union.vs["occurrence_id"]:
            raise ValueError(f"{rep}: the directed file lists the vertices in another order")
        rows[rep] = arc_composition(directed, continues)
        in_degrees[rep] = np.asarray(directed.indegree())
        print(f"{rep}: {view(xy)[2]} vértices fora do recorte")
    plots.save(figure_in_degrees(in_degrees), figures / "graus_pk.pdf")
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("rede")
    tables = REPORT / "tabelas"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "arcos.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.T)
    print(f"figuras e tabela em {REPORT.relative_to(dn.REPO)}")


if __name__ == "__main__":
    main()
