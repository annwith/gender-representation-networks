"""Global metrics of the networks (Section 8, overview)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.artifacts import (
    graph_id,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import (
    DASH,
    Tabular,
    finite_or_none,
    header_row,
    integer,
    num,
    pct,
    pm,
    sci,
    tex_text,
)
from gender_networks.representations import is_lexical

GLOBAL_ROWS = [
    # (label, column, kind, digits, source sym)
    ("$|V|$", "n_vertices", "int", 0, "union"),
    ("$|E|$", "n_edges", "int", 0, "union"),
    (r"Grau médio $\langle k\rangle$", "mean_degree", "num", 2, "union"),
    ("Densidade", "density", "sci", 2, "union"),
    ("Reciprocidade (direcionada)", "reciprocity", "num", 3, "directed"),
    ("Clusterização global", "transitivity", "num", 3, "union"),
    ("Clusterização local média", "avg_local_clustering", "num", 3, "union"),
    ("Componentes", "n_components", "int", 0, "union"),
    ("Maior componente (fração)", "largest_fraction", "pct", 1, "union"),
    ("Distância média", "mean_distance", "num", 2, "union"),
    ("Diâmetro", "diameter", "int", 0, "union"),
    ("Grau de entrada máximo", "in_degree_max", "int", 0, "directed"),
    ("Assimetria do grau de entrada", "in_degree_skewness", "num", 2, "directed"),
    ("Modularidade (Leiden)", "modularity", "num", 3, "union"),
    ("Comunidades", "n_communities", "int", 0, "union"),
]


def _metric_cell(part: pd.DataFrame, column: str, kind: str, digits: int) -> str:
    """Cell of the global metrics table: the mean over the rows, with the deviation when they
    differ.

    Args:
        part: Rows of ``graph_metrics.csv`` for one network (one per tie seed).
        column: Metric column.
        kind: ``int``, ``num``, ``pct`` or ``sci``.
        digits: Decimal places.

    Returns:
        The LaTeX text; ``--`` when the column is absent or has no value. The mean and the
        sample standard deviation run over the rows that have a value.
    """

    if column not in part:
        return DASH
    values = pd.to_numeric(part[column], errors="coerce").dropna()
    if values.empty:
        return DASH
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else math.nan
    if kind == "sci":
        return sci(mean, digits)
    if kind == "pct":
        text = pct(mean, digits)
        spread = finite_or_none(sd) and sd > 0
        return text + (r"\,$\pm$\," + num(sd * 100, digits) if spread else "")
    if kind == "int":
        if values.size > 1 and sd > 0:
            return pm(mean, sd, 1, "num") if mean < 1000 else pm(mean, sd, 0, "int")
        return integer(mean)
    return pm(mean, sd, digits)


def table_metricas_globais(ctx: Context) -> Tabular:
    """Table ``metricas_globais``: the course metrics of the main networks and of the vocabulary
    network.

    Each shown representation is a column: the union and directed graphs at ``k_main`` with
    random ties (every seed for ``lex``, seed 0 elsewhere). The vocabulary column uses the
    ``k`` that has communities.

    Args:
        ctx: Context of the run.

    Returns:
        The table, one row per metric of :data:`GLOBAL_ROWS`.

    Raises:
        Skip: If ``graph_metrics.csv`` is missing or lacks the main networks.
    """

    metrics = ctx.metrics_table("graph_metrics")
    k = ctx.k
    columns: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = []
    for rep in ctx.shown_reps:
        main = metrics[(metrics["rep"] == rep) & (metrics["k"] == k) & (metrics["tie"] == "rand")]
        if not is_lexical(rep):
            main = main[main["seed"] == 0]
        union, directed = main[main["sym"] == "union"], main[main["sym"] == "directed"]
        if union.empty and directed.empty:
            continue
        columns.append((r"\texttt{" + tex_text(rep) + "}", union, directed, ""))
    sampled_note = False
    try:
        vk = ctx.vocab_k
    except Skip:
        vk = None
    if vk is not None:
        main = metrics[
            (metrics["rep"] == "vocab")
            & (metrics["k"] == vk)
            & (metrics["tie"] == "rand")
            & (metrics["seed"] == 0)
        ]
        union, directed = main[main["sym"] == "union"], main[main["sym"] == "directed"]
        if not union.empty:
            method = str(union["distance_method"].iat[0]) if "distance_method" in union else ""
            mark = "$^{*}$" if method == "sampled" else ""
            sampled_note = bool(mark)
            columns.append((f"vocab. ($k={vk}$){mark}", union, directed, method))
    if not columns:
        raise Skip(f"graph_metrics.csv não tem as redes principais com k = {k}")
    rows = []
    for name, column, kind, digits, sym in GLOBAL_ROWS:
        cells = [name]
        for _, union, directed, method in columns:
            part = union if sym == "union" else directed
            cell = _metric_cell(part, column, kind, digits)
            if method == "sampled" and column == "diameter" and cell != DASH:
                cell = r"$\geq$\," + cell
            cells.append(cell)
        rows.append(cells)
    lex_seeds = 0
    if any(c[0].startswith(r"\texttt{lex}") for c in columns):
        lex_seeds = int(columns[0][1]["seed"].nunique())
    notes = [
        f"Redes com $k={k}$, versão por união, desempate por sorteio "
        f"(média $\\pm$ desvio entre as {lex_seeds} sementes na rede lexical, semente 0 nas "
        "demais); reciprocidade e grau de entrada vêm da versão direcionada."
    ]
    if sampled_note:
        notes.append(
            "$^{*}$ Distâncias estimadas por busca em largura a partir de fontes "
            "sorteadas; o diâmetro é um limite inferior."
        )
    header = [header_row(["Métrica", *[c[0] for c in columns]])]
    return Tabular("l" + "r" * len(columns), header, rows, size=r"\footnotesize", notes=notes)


def figure_graus_camadas(ctx: Context) -> Any:
    """Figure ``graus_camadas``: in-degree CCDF of the directed graphs (random tie seed 0).

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure, one line per shown representation.

    Raises:
        Skip: If ``degree_hist.csv`` is missing or lacks those graphs.
    """

    hist = ctx.metrics_table("degree_hist")
    hist = hist[hist["kind"] == "in"]
    series = {}
    for rep in ctx.shown_reps:
        part = hist[hist["graph_id"] == graph_id(rep, ctx.k, "directed", "rand", 0)]
        if not part.empty:
            series[rep] = (
                part["degree"].to_numpy(dtype=float),
                part["count"].to_numpy(dtype=float),
            )
    if not series:
        raise Skip(f"degree_hist.csv não tem os grafos direcionados com k = {ctx.k}")
    styles = plots.rep_styles(list(series))
    return plots.degree_ccdf_figure(
        series, ctx.k, styles, "Grau de entrada nas redes direcionadas (semente 0)"
    )
