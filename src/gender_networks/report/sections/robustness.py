"""Robustness of the main results (Section 9)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.artifacts import (
    graph_id,
)
from gender_networks.labels import (
    ordered,
)
from gender_networks.neighborhood import jaccard
from gender_networks.report.context import Context, Skip, band_order
from gender_networks.report.latex import DASH, Tabular, header_row, num, sci, tex_text
from gender_networks.representations import LEX, parse


def _graph_row(metrics: pd.DataFrame, rep: str, k: int, sym: str, tie: str) -> pd.Series | None:
    """Row of ``graph_metrics.csv`` for one graph with tie seed 0.

    Args:
        metrics: The metrics table.
        rep: Representation name.
        k: Number of neighbours.
        sym: Symmetrization.
        tie: Tie treatment.

    Returns:
        The row, or ``None`` when the graph was not measured.
    """

    hit = metrics[metrics["graph_id"] == graph_id(rep, k, sym, tie, 0)]
    return None if hit.empty else hit.iloc[0]


def table_robustez(ctx: Context) -> Tabular:
    """Table ``robustez``: summary measures of each robustness variation next to the main
    configuration.

    Per shown contextual layer: mean local clustering, mean distance, Leiden modularity and
    the mean Jaccard with the lexical network of the same ``k`` and tie treatment. Rows: the
    main configuration, the other ``k`` values, the mutual graph, the position rule,
    variant (b), the larger tolerance, the final-norm and centered variants, the other
    Leiden resolutions and Louvain.

    Args:
        ctx: Context of the run.

    Returns:
        The table; a variation without any value is left out.

    Raises:
        Skip: If ``graph_metrics.csv`` is missing or lacks the contextual networks.
    """

    metrics = ctx.metrics_table("graph_metrics")
    net = ctx.settings.networks
    k = ctx.k
    neighbors = pd.DataFrame()
    path = ctx.paths.analysis_table("robustness_neighbors")
    if path.exists():
        neighbors = ctx.table(path)
    index = pd.DataFrame()
    index_path = ctx.paths.communities_index
    if index_path.exists():
        index = ctx.table(index_path)
    reps = [
        r
        for r in ctx.shown_contextual_reps
        if _graph_row(metrics, r, k, "union", "rand") is not None
    ]
    if not reps:
        raise Skip(f"graph_metrics.csv não tem as redes contextuais com k = {k}")

    def j_vs_lex(rep: str, kk: int, tie: str) -> Any:
        """Mean Jaccard with the lexical sets, from ``robustness_neighbors.csv``.

        Args:
            rep: Representation name.
            kk: Number of neighbours.
            tie: Tie rule (``pos``, ``rand`` or ``all``).

        Returns:
            The value at ``networks.eps``, or NaN when the table has no such row.
        """

        if neighbors.empty:
            return np.nan
        hit = neighbors[
            (neighbors["rep"] == rep)
            & (neighbors["k"] == kk)
            & (neighbors["tie"] == tie)
            & np.isclose(pd.to_numeric(neighbors["eps"]), net.eps)
        ]
        return hit["J_vs_lex_mean"].iat[0] if len(hit) else np.nan

    def centered_name(rep: str) -> str:
        """Name of the centered variant of a representation.

        Args:
            rep: Representation name.

        Returns:
            The name with the trailing ``c``.
        """

        return parse(rep).with_centering().name

    def graph_cells(rep: str | None, kk: int, sym: str, tie: str, j: Any) -> list[str]:
        """The four cells of one layer in one variation.

        Args:
            rep: Representation whose graph is read; ``None`` gives four dashes.
            kk: Number of neighbours.
            sym: Symmetrization.
            tie: Tie treatment.
            j: Mean Jaccard with the lexical network, already looked up.

        Returns:
            Clustering, mean distance, modularity and Jaccard, formatted.
        """

        if rep is None:
            return [DASH] * 4
        row = _graph_row(metrics, rep, kk, sym, tie)
        if row is None:
            return [DASH] * 4
        return [
            num(row.get("avg_local_clustering"), 3),
            num(row.get("mean_distance"), 2),
            num(row.get("modularity"), 3),
            num(j, 3),
        ]

    variations: list[tuple[str, Callable[[str], list[str]]]] = [
        (
            f"Principal ($k={k}$, união, sorteio)",
            lambda r: graph_cells(r, k, "union", "rand", j_vs_lex(r, k, "rand")),
        ),
    ]
    for kk in net.k_values:
        if kk != k:
            variations.append(
                (
                    f"$k={kk}$",
                    lambda r, kk=kk: graph_cells(r, kk, "union", "rand", j_vs_lex(r, kk, "rand")),
                )
            )
    variations += [
        ("Versão mútua", lambda r: graph_cells(r, k, "mutual", "rand", np.nan)),
        (
            "Desempate por posição",
            lambda r: graph_cells(r, k, "union", "pos", j_vs_lex(r, k, "pos")),
        ),
        (
            "(b) todos os empatados",
            lambda r: graph_cells(r, k, "union", "all", j_vs_lex(r, k, "all")),
        ),
        (
            f"$\\varepsilon$ = {sci(net.eps_sensitivity)}",
            lambda r: graph_cells(r, k, "union", "rand_eps", np.nan),
        ),
    ]
    robust = [r for r in net.robust_representations]
    for variant in robust:
        rep = parse(variant)
        base = rep.raw().name if rep.kind == "final_norm" else None
        variations.append(
            (
                f"\\texttt{{{tex_text(variant)}}} (RMSNorm final)",
                lambda r, v=variant, b=base: graph_cells(
                    v if r == b else None, k, "union", "rand", j_vs_lex(v, k, "rand")
                ),
            )
        )
    if net.centered:
        variations.append(
            (
                "Cosseno centrado",
                lambda r: graph_cells(
                    centered_name(r) if parse(r).is_block else None,
                    k,
                    "union",
                    "rand",
                    j_vs_lex(centered_name(r), k, "rand"),
                ),
            )
        )
    rows: list[Any] = []
    for name, build in variations:
        cells = [name]
        for rep in reps:
            cells += build(rep)
        if all(c == DASH for c in cells[1:]):
            continue
        rows.append(cells)
    if not index.empty:
        community_rows = []
        for resolution in ctx.settings.analysis.resolution_sweep:
            if math.isclose(resolution, ctx.settings.analysis.resolution):
                continue
            community_rows.append((f"Leiden $\\gamma={num(resolution, 1)}$", "leiden", resolution))
        community_rows.append(("Louvain (checagem)", "louvain", ctx.settings.analysis.resolution))
        for name, method, resolution in community_rows:
            cells = [name]
            found = False
            for rep in reps:
                hit = index[
                    (index["graph_id"] == graph_id(rep, k, "union", "rand", 0))
                    & (index["method"] == method)
                    & np.isclose(pd.to_numeric(index["resolution"]), resolution)
                ]
                q = hit["modularity"].iat[0] if len(hit) else np.nan
                found |= len(hit) > 0
                cells += [DASH, DASH, num(q, 3), DASH]
            if found:
                rows.append(cells)
    header = [
        header_row([""] + [rf"\multicolumn{{4}}{{c}}{{\texttt{{{tex_text(r)}}}}}" for r in reps]),
        "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(reps))),
        header_row(["Variação"] + [r"$\bar C$", r"$\bar\ell$", "$Q$", "$J$"] * len(reps)),
    ]
    notes = [
        r"$\bar C$: clusterização local média; $\bar\ell$: distância média; $Q$: modularidade "
        r"do Leiden (só nos grafos com comunidades); $J$: Jaccard médio com a rede \texttt{lex} "
        "de mesmo $k$ e mesmo tratamento de empates. Cada linha muda uma dimensão em relação à "
        r"principal; nas linhas \texttt{L36n} e centrada, cada coluna usa a variante da camada."
    ]
    return Tabular(
        "l" + "rrrr" * len(reps), header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt"
    )


def figure_robustez_empates(ctx: Context) -> Any:
    """Figure ``robustez_empates``: mean ``J(lex, l)`` per frequency band under each tie
    treatment, and the type-composition Jaccard.

    The position rule and variant (b) are computed here from the neighbour files; the random
    rule, variant (c) and ``J^w`` come from the vertex measures.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure, one panel per shown contextual layer.

    Raises:
        Skip: If the neighbour files are missing.
    """

    occ = ctx.occurrences
    lex = ctx.nbr(LEX)
    reps = [r for r in ctx.shown_contextual_reps if ctx.paths.nbr(r, ctx.k).exists()]
    if not reps:
        raise Skip("faltam os vizinhos das camadas contextuais")
    measures = None
    try:
        measures = ctx.vertex_measures()
    except Skip:
        pass
    bands = ordered([b for b in occ["band_sample"] if b], band_order(ctx.settings))
    band = occ["band_sample"].to_numpy()
    variants = {
        "a": "(a) sorteio",
        "pos": "posição",
        "b": "(b) todos os empatados",
        "c": "(c) tipos distintos",
        "w": "Jʷ (composição de tipos)",
    }
    rows = []
    for rep in reps:
        data = ctx.nbr(rep)
        per_vertex = {
            "pos": jaccard(lex["pos"], data["pos"]),
            "b": jaccard(
                (lex["all_indptr"], lex["all_idx"]), (data["all_indptr"], data["all_idx"])
            ),
        }
        if measures is not None:
            for key, prefix in (("a", "J"), ("c", "Jc"), ("w", "Jw")):
                column = f"{prefix}_lex_{rep}"
                if column in measures:
                    per_vertex[key] = measures[column].to_numpy(dtype=float)
        for key, values in per_vertex.items():
            for b in bands:
                mask = band == b
                rows.append(
                    {
                        "rep": rep,
                        "variant": key,
                        "band": b,
                        "value": float(np.nanmean(values[mask])) if mask.any() else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    keys = [k for k in variants if (frame["variant"] == k).any()]
    return plots.tie_variants_figure(frame, reps, bands, keys, variants)
