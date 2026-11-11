"""Drawings of subgraphs with fixed positions across representations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.labels import (
    ordered,
    theme_label,
)
from gender_networks.report.context import Context, Skip
from gender_networks.representations import parse

SUBGRAPH_WINDOW = 14


SUBGRAPH_TOP_TYPES = 7


EGO_WORD = "banco"


LAYOUT_SEED = 438


def _layout(n: int, edges: np.ndarray) -> np.ndarray:
    """Spring layout of the union of every panel's edges (deterministic seed)."""

    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, edges.tolist()))
    positions = nx.spring_layout(graph, seed=LAYOUT_SEED, iterations=150)
    return np.array([positions[i] for i in range(n)], dtype=float)


def _ego_panels(
    ctx: Context, focal: np.ndarray, reps: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[plots.GraphPanel]]:
    """Vertex set (focal + out-neighbours in every rep), shared positions and one panel per rep."""

    sets = {rep: ctx.nbr(rep)["rand"][0][focal] for rep in reps}
    vertices = np.unique(np.concatenate([focal, *[s.ravel() for s in sets.values()]]))
    local = {int(v): i for i, v in enumerate(vertices)}
    panels = []
    all_edges = []
    for rep in reps:
        src = np.repeat(focal, sets[rep].shape[1])
        dst = sets[rep].ravel()
        edges = np.array(
            [[local[int(a)], local[int(b)]] for a, b in zip(src, dst, strict=True)], dtype=np.int64
        ).reshape(-1, 2)
        active = np.zeros(vertices.size, dtype=bool)
        active[edges.ravel()] = True
        panels.append(plots.GraphPanel(rep, edges, active))
        all_edges.append(edges)
    positions = _layout(vertices.size, np.concatenate(all_edges))
    return vertices, positions, panels


def _drawing_reps(ctx: Context) -> list[str]:
    shown = [r for r in ctx.main_reps if ctx.shows(r)]
    drawable = [r for r in shown if parse(r).is_lexical or parse(r).is_block]
    reps = drawable or shown or ctx.main_reps
    return reps[:4]


def figure_subgrafo_camadas(ctx: Context) -> Any:
    occ = ctx.occurrences
    reps = _drawing_reps(ctx)
    core = occ[occ["stratum"] == "core"]
    if core.empty:
        raise Skip("a amostra não tem núcleo")
    sizes = core.groupby("paragraph_id", sort=False).size()
    candidates = sizes[sizes >= SUBGRAPH_WINDOW]
    paragraph = candidates.index[0] if len(candidates) else sizes.idxmax()
    focal = core.index[core["paragraph_id"] == paragraph].to_numpy()[:SUBGRAPH_WINDOW]
    vertices, positions, panels = _ego_panels(ctx, focal, reps)
    token_ids = occ["token_id"].to_numpy(dtype=np.int64)[vertices]
    counts = pd.Series(token_ids).value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    top = [int(t) for t in counts.index[:SUBGRAPH_TOP_TYPES] if counts[t] > 1]
    key_of = {t: i for i, t in enumerate(top)}
    keys = np.array([key_of.get(int(t), -1) for t in token_ids])
    texts = occ.drop_duplicates("token_id").set_index("token_id")["token_text"]
    base = plots.categorical_styles([str(t) for t in top])
    styles = [
        plots.Style(base[str(t)].color, base[str(t)].marker, label=_visible(texts.get(t, "")))
        for t in top
    ]
    is_focal = np.isin(vertices, focal)
    title = occ.loc[focal[0], "title"] if "title" in occ else paragraph
    fig = plots.fixed_layout_figure(
        positions,
        panels,
        is_focal,
        styles,
        keys,
        keys,
        legend_title=f"Tipos de token mais frequentes no subgrafo (parágrafo de «{title}»)",
    )
    return fig


def _visible(text: Any) -> str:
    return str(text).replace(" ", "␣") or "∅"


def figure_p4_ego_banco(ctx: Context) -> Any:
    occ = ctx.occurrences
    reps = _drawing_reps(ctx)
    marked = occ[occ["target_word"] != ""]
    targets = [w for w in ctx.settings.sample.targets if (marked["target_word"] == w).any()]
    if not targets:
        targets = sorted(set(marked["target_word"]))
    if not targets:
        raise Skip("a amostra não tem ocorrências de palavras-alvo")
    word = (
        EGO_WORD
        if EGO_WORD in targets
        else max(targets, key=lambda w: (int((marked["target_word"] == w).sum()), w))
    )
    focal = marked.index[marked["target_word"] == word].to_numpy()
    vertices, positions, panels = _ego_panels(ctx, focal, reps)
    themes_all = list(ctx.settings.corpus.themes)
    sense = occ["sense_theme"].where(occ["sense_theme"] != "", occ["theme"]).to_numpy()
    present = ctx.theme_order(sense[focal])
    base = plots.categorical_styles(ordered(themes_all + present, themes_all))
    styles = [plots.Style(base[t].color, base[t].marker, label=theme_label(t)) for t in present]
    key_of = {t: i for i, t in enumerate(present)}
    keys = np.array([key_of.get(t, -1) for t in sense[vertices]])
    is_focal = np.isin(vertices, focal)
    title = f"Tema de sentido das ocorrências de «{word}»"
    if word != EGO_WORD:
        title += f" («{EGO_WORD}» não está na amostra)"
    return plots.fixed_layout_figure(
        positions, panels, is_focal, styles, keys, None, legend_title=title
    )
