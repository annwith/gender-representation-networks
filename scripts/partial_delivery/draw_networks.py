"""Drawings of the partial-delivery networks and their numbers (Entrega Parcial, MO438, item 9).

Reads the four union GraphML files delivered in ``data/graphml`` (``lex``, ``L01``, ``L18`` and
``L36``, k = 10) and draws each with its own UMAP layout (igraph's ``layout_umap``, unweighted,
seeded through Python's ``random``). The visualization is delivered as a document of its own,
``report/entrega-parcial-visualizacao``, and this script writes its material:

- ``figuras/redes.pdf``: the four networks, vertices colored by the article theme;
- ``figuras/redes_palavra.pdf``: the same drawings, vertices colored by whether the word goes on
  in the next token (``pos_in_word < word_n_tokens - 1``: the start or the middle of a word of
  several tokens) or not (the last or only token of a word, or punctuation, which belongs to no
  word);
- ``tabelas/arestas.{csv,tex}``: what the edges join (occurrences of the same token, of the same
  theme, of the two word positions) and the nominal assortativity of theme and word position;
- ``tabelas/blocos.{csv,tex}``: Leiden communities at a low resolution (modularity, resolution
  0.05, best of 10 seeded runs, as in the ``metrics`` stage), which keeps only the coarsest
  division of each network, and the block that holds the most positions where the word goes on;
  the number of communities at resolution 1 (the ``community`` attribute) is there to compare.

The view of both figures is the disc around the median position that holds 99% of the
vertices; the rest are small groups pushed to the periphery, and the caption says how many are
left out. The script also writes ``docs/redes/``: the interactive page (``index.html``, copied
from ``redes.html`` next to this script) and its data files, one with the vertex attributes and
one per network with the positions and edges, so the page shows the same drawing as the PDF.

The directed GraphML of each network, with the arc directions dropped, must give exactly the
union edges: the same drawings therefore serve the directed version, with the arcs undirected.

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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from gender_networks import plots
from gender_networks.partitions import leiden

REPO = Path(__file__).resolve().parents[2]
GRAPHML = REPO / "data" / "graphml"
LAYOUTS = REPO / "outputs" / "experiment" / "main" / "layout"
REPORT = REPO / "report" / "entrega-parcial-visualizacao"
PAGE = REPO / "docs" / "redes"
TEMPLATE = Path(__file__).with_name("redes.html")

REPS = ["lex", "L01", "L18", "L36"]
K = 10
LAYOUT_SEED = 438
ORDER_SEED = 0
VIEW_QUANTILE = 0.99  # fraction of the vertices inside the static view
# Coarse blocks: the best of BLOCK_RUNS Leiden runs at BLOCK_RESOLUTION; a community is a block
# when it holds at least BLOCK_MIN_SHARE of the vertices.
BLOCK_RESOLUTION = 0.05
BLOCK_RUNS = 10
BLOCK_SEED = 0
BLOCK_MIN_SHARE = 0.01
REP_TITLES = {
    "lex": "lex: embedding de entrada",
    "L01": "L01: saída do bloco 1",
    "L18": "L18: saída do bloco 18",
    "L36": "L36: saída do bloco 36",
}
THEMES = list(plots.THEME_LABELS)
# Word position of a token: the class that matters highlighted, the rest in the muted gray.
WORD_POSITIONS = {
    "continues": ("a palavra continua no token seguinte", plots.CATEGORICAL[0]),
    "ends": ("a palavra termina no token (ou é pontuação)", plots.MUTED),
}

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


def word_continues(g: ig.Graph) -> np.ndarray:
    """Whether the word goes on in the next token, i.e. the token is not the last of its word.

    Punctuation belongs to no word (``word_n_tokens = 0``, ``pos_in_word = -1``), so it never
    goes on.
    """

    if "pos_in_word" not in g.vs.attributes():
        raise ValueError("the GraphML has no pos_in_word: export it again with export_networks.py")
    pos = np.asarray(g.vs["pos_in_word"], dtype=np.int64)
    size = np.asarray(g.vs["word_n_tokens"], dtype=np.int64)
    return pos < size - 1


def theme_colors(g: ig.Graph) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Color of every vertex by theme, and the legend entries (label, color)."""

    styles = plots.categorical_styles(THEMES, plots.THEME_LABELS)
    colors = np.array([styles[t].color for t in g.vs["theme"]])
    return colors, [(styles[t].label, styles[t].color) for t in THEMES]


def word_colors(continues: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Color of every vertex by word position, and the legend entries (label, color)."""

    (on_label, on_color), (off_label, off_color) = WORD_POSITIONS.values()
    return np.where(continues, on_color, off_color), [(on_label, on_color), (off_label, off_color)]


def figure_networks(
    graphs: dict[str, ig.Graph],
    positions: dict[str, np.ndarray],
    colors: np.ndarray,
    legend: Sequence[tuple[str, str]],
) -> Figure:
    fig, axes = plots.figure(2, 2, height=6.75)
    fig.set_dpi(300)  # the edges and vertices are rasterized at this resolution
    order = np.random.default_rng(ORDER_SEED).permutation(colors.size)
    for ax, rep in zip(axes.flat, REPS, strict=True):
        g, (xy, _outside) = graphs[rep], normalize(positions[rep])
        edges = np.asarray(g.get_edgelist())
        ax.add_collection(
            LineCollection(
                xy[edges], colors=plots.AXIS, linewidths=0.06, alpha=0.18, rasterized=True
            )
        )
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
    handles = [
        Line2D([], [], linestyle="", marker="o", markersize=5, color=color, label=name)
        for name, color in legend
    ]
    plots.legend_top(fig, handles, ncol=min(len(handles), 4))
    return fig


# --------------------------------------------------------------------------------------------
# numbers of the visualization


def edge_composition(g: ig.Graph, continues: np.ndarray) -> dict[str, float]:
    """What the edges join, and the nominal assortativity of theme and word position."""

    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    a, b = e[:, 0], e[:, 1]
    token = np.asarray(g.vs["token_id"], dtype=np.int64)
    _, theme = np.unique(np.asarray(g.vs["theme"]), return_inverse=True)
    position = continues.astype(np.int64)
    return {
        "mesmo_token": float(np.mean(token[a] == token[b])),
        "mesmo_tema": float(np.mean(theme[a] == theme[b])),
        "assortatividade_tema": g.assortativity_nominal(theme.tolist(), directed=False),
        "continua_termina": float(np.mean(position[a] != position[b])),
        "assortatividade_palavra": g.assortativity_nominal(position.tolist(), directed=False),
    }


def coarse_blocks(g: ig.Graph, continues: np.ndarray) -> dict[str, Any]:
    """Leiden blocks at a low resolution, and the block B with the most word-continuing positions.

    Communities come labelled by decreasing size; the blocks are the communities with at least
    ``BLOCK_MIN_SHARE`` of the vertices.
    """

    partition = leiden(g, runs=BLOCK_RUNS, resolution=BLOCK_RESOLUTION, seed=BLOCK_SEED)
    membership = partition.membership
    sizes = np.asarray(partition.sizes)
    blocks = np.flatnonzero(sizes >= BLOCK_MIN_SHARE * g.vcount())
    if not blocks.size:
        raise ValueError("no community holds enough vertices to be a block")
    continuing = np.bincount(membership[continues], minlength=sizes.size)
    inside = membership == blocks[np.argmax(continuing[blocks])]
    e = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    return {
        "comunidades_resolucao_1": len(set(g.vs["community"])),
        "comunidades": partition.n_communities,
        "blocos": int(blocks.size),
        "fracao_vertices_blocos": float(sizes[blocks].sum() / g.vcount()),
        "maior_comunidade": int(sizes[0]),
        "segunda_comunidade": int(sizes[1]) if sizes.size > 1 else 0,
        "bloco_b_vertices": int(inside.sum()),
        "bloco_b_continua": float(continues[inside].mean()),
        "bloco_b_cobertura": float(inside[continues].mean()),
        "bloco_b_arestas_saem": float(np.mean(inside[e[:, 0]] != inside[e[:, 1]])),
        "modularidade": partition.modularity,
        "estabilidade": partition.stability,
    }


def _pct(value: float) -> str:
    return f"{plots.fmt_dec(100 * value, 1)}\\%"


def _dec(value: float) -> str:
    return plots.fmt_dec(value, 3)


Row = tuple[str, Callable[[pd.Series], str]]

EDGE_ROWS: list[Row] = [
    ("Arestas entre ocorrências do mesmo token", lambda r: _pct(r["mesmo_token"])),
    ("Arestas entre ocorrências do mesmo tema", lambda r: _pct(r["mesmo_tema"])),
    ("Assortatividade por tema", lambda r: _dec(r["assortatividade_tema"])),
    ("Arestas entre \\emph{continua} e \\emph{termina}", lambda r: _pct(r["continua_termina"])),
    ("Assortatividade por posição na palavra", lambda r: _dec(r["assortatividade_palavra"])),
]
BLOCK_ROWS: list[Row] = [
    ("Comunidades com $\\gamma = 1$", lambda r: plots.fmt_int(r["comunidades_resolucao_1"])),
    ("Comunidades com $\\gamma = 0{,}05$", lambda r: plots.fmt_int(r["comunidades"])),
    ("Blocos (comunidades com ao menos 1\\% dos vértices)", lambda r: plots.fmt_int(r["blocos"])),
    ("\\quad vértices nos blocos", lambda r: _pct(r["fracao_vertices_blocos"])),
    (
        "\\quad as duas maiores comunidades (vértices)",
        lambda r: (
            f"{plots.fmt_int(r['maior_comunidade'])} / {plots.fmt_int(r['segunda_comunidade'])}"
        ),
    ),
    ("Bloco $B$ (vértices)", lambda r: plots.fmt_int(r["bloco_b_vertices"])),
    ("\\quad fração de $B$ que é \\emph{continua}", lambda r: _pct(r["bloco_b_continua"])),
    (
        "\\quad fração dos \\emph{continua} que está em $B$",
        lambda r: _pct(r["bloco_b_cobertura"]),
    ),
    ("\\quad arestas que saem de $B$", lambda r: _pct(r["bloco_b_arestas_saem"])),
]


def latex_table(table: pd.DataFrame, rows: Sequence[Row]) -> str:
    """Rows = measures, columns = networks."""

    reps = table.index.tolist()
    lines = [
        "\\begin{tabular}{l" + "r" * len(reps) + "}",
        "\\toprule",
        "Medida & " + " & ".join(reps) + " \\\\",
        "\\midrule",
    ]
    for name, cell in rows:
        lines.append(f"{name} & " + " & ".join(cell(table.loc[rep]) for rep in reps) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write_tables(graphs: dict[str, ig.Graph], continues: np.ndarray) -> None:
    tables = REPORT / "tabelas"
    tables.mkdir(parents=True, exist_ok=True)
    measures = {"arestas": (edge_composition, EDGE_ROWS), "blocos": (coarse_blocks, BLOCK_ROWS)}
    for name, (measure, rows) in measures.items():
        table = pd.DataFrame.from_dict(
            {rep: measure(g, continues) for rep, g in graphs.items()}, orient="index"
        )
        table.rename_axis("rede").to_csv(tables / f"{name}.csv")
        (tables / f"{name}.tex").write_text(latex_table(table, rows), encoding="utf-8")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(table.T)


# --------------------------------------------------------------------------------------------
# interactive page


def compact(values: list[str]) -> tuple[list[str], list[int]]:
    """Distinct values (in first-seen order) and the index of each value among them."""

    table: dict[str, int] = {}
    codes = [table.setdefault(v, len(table)) for v in values]
    return list(table), codes


def write_page(
    graphs: dict[str, ig.Graph], positions: dict[str, np.ndarray], continues: np.ndarray
) -> None:
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
    positions_in_word = ["continues" if c else "ends" for c in continues]
    vertices["word_position"] = dict(
        zip(("values", "codes"), compact(positions_in_word), strict=True)
    )
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
        n_outside = int(normalize(positions[rep])[1].sum())
        print(
            f"{rep}: {g.vcount()} vértices, {g.ecount()} arestas (= arcos direcionados), "
            f"{n_outside} fora do recorte"
        )
    continues = word_continues(graphs[REPS[0]])

    plots.apply_style()
    figures = REPORT / "figuras"
    by_theme = theme_colors(graphs[REPS[0]])
    plots.save(figure_networks(graphs, positions, *by_theme), figures / "redes.pdf")
    by_word = word_colors(continues)
    plots.save(figure_networks(graphs, positions, *by_word), figures / "redes_palavra.pdf")
    write_tables(graphs, continues)
    write_page(graphs, positions, continues)
    print(f"figuras e tabelas em {REPORT.relative_to(REPO)}; página em {PAGE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
