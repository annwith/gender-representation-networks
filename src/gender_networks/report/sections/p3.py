"""P3: communities and context (Section 8)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.labels import (
    LABEL_LABELS,
    label,
    ordered,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import Tabular, header_row, integer, num, tex_text


def _p3_core(ctx: Context) -> pd.DataFrame:
    table = ctx.analysis("p3_agreement")
    core = table[table["subset"].astype(str) == "core"]
    if core.empty:
        raise Skip("p3_agreement.csv não tem linhas do núcleo")
    return core


def figure_p3_nmi_camadas(ctx: Context) -> Any:
    core = _p3_core(ctx)
    reps = [r for r in ctx.settings.networks.representations if (core["rep"] == r).any()]
    from gender_networks.analysis import CORE_LABELS

    labels = ordered(list(core["label"]), CORE_LABELS)
    stability = core.groupby("rep")["stability"].first().astype(float).to_dict()
    return plots.nmi_layers_figure(core, reps, labels, stability)


def table_p3_concordancia(ctx: Context) -> Tabular:
    core = _p3_core(ctx)
    from gender_networks.analysis import CORE_LABELS

    reps = [r for r in ctx.shown_reps if (core["rep"] == r).any()]
    rows: list[Any] = []
    for i, name in enumerate(ordered(list(core["label"]), CORE_LABELS)):
        if i:
            rows.append(r"\midrule")
        first = True
        for rep in reps:
            hit = core[(core["label"] == name) & (core["rep"] == rep)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            rows.append(
                [
                    label(LABEL_LABELS, name) if first else "",
                    integer(row.get("n_classes")) if first else "",
                    r"\texttt{" + tex_text(rep) + "}",
                    integer(row.get("n_communities")),
                    num(row.get("nmi"), 3),
                    num(row.get("nmi_baseline"), 3),
                    num(row.get("nmi_minus_baseline"), 3),
                    num(row.get("ari"), 3),
                    num(row.get("purity"), 3),
                ]
            )
            first = False
    header = [
        header_row(
            ["Rótulo", "Classes", "Repr.", "Comun.", "NMI", "Base", r"NMI$-$base", "ARI", "Pureza"]
        )
    ]
    notes = [
        "Comunidades do Leiden (modularidade, "
        f"$\\gamma={num(ctx.settings.analysis.resolution, 1)}$)"
        f" na rede por união com $k={ctx.k}$, semente 0; linha de base: NMI médio com o rótulo "
        f"permutado ({ctx.settings.analysis.permutations} permutações)."
    ]
    return Tabular("lrlrrrrrr", header, rows, size=r"\footnotesize", notes=notes, colsep="4pt")
