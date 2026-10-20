"""P2: in which layers the change happens (Section 8)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.artifacts import (
    parse_bool,
)
from gender_networks.plots import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    label,
    ordered,
)
from gender_networks.report.context import Context, Skip, band_order
from gender_networks.report.latex import Tabular, bold, header_row, integer, num


def _transitions(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = ctx.analysis("p2_transitions")
    groups = ctx.analysis("p2_transitions_groups")
    for frame in (overall, groups):
        frame["value"] = frame["value"].astype(str)
        frame["consecutive"] = parse_bool(frame["consecutive"])
    return overall, groups


def figure_p2_transicoes(ctx: Context) -> Any:
    overall, groups = _transitions(ctx)
    transitions = list(overall.loc[overall["consecutive"], "transition"])
    if not transitions:
        raise Skip("p2_transitions.csv não tem transições consecutivas")

    def block(group: str, values: Sequence[str], labels: Callable[[str], str]) -> pd.DataFrame:
        rows = {}
        for value in values:
            part = groups[(groups["group"] == group) & (groups["value"] == value)]
            part = part[part["consecutive"]].set_index("transition")
            rows[labels(value)] = {
                t: part["share_largest_change"].get(t, np.nan) for t in transitions
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    everything = overall[overall["consecutive"]].set_index("transition")
    todos = pd.DataFrame(
        {t: [everything["share_largest_change"].get(t)] for t in transitions}, index=["todos"]
    )
    categories = ordered(
        list(groups.loc[groups["group"] == "token_category", "value"]), CATEGORY_ORDER
    )
    bands = ordered(
        list(groups.loc[groups["group"] == "band_sample", "value"]), band_order(ctx.settings)
    )
    by_category = pd.concat(
        [todos, block("token_category", categories, lambda v: label(CATEGORY_LABELS, v))]
    )
    by_band = pd.concat(
        [todos, block("band_sample", bands, lambda v: f"f_t {plots.band_label(v)}")]
    )
    return plots.transition_share_figure(
        [("Por categoria de token", by_category), ("Por faixa de frequência", by_band)],
        transitions,
    )


def table_p2_transicoes(ctx: Context) -> Tabular:
    overall, groups = _transitions(ctx)
    transitions = list(dict.fromkeys(overall["transition"]))
    if not transitions:
        raise Skip("p2_transitions.csv está vazio")
    categories = ordered(
        list(groups.loc[groups["group"] == "token_category", "value"]), CATEGORY_ORDER
    )

    def cells(frame: pd.DataFrame) -> list[str]:
        frame = frame.set_index("transition")
        out = []
        for t in transitions:
            out += [
                num(frame["excess_change_mean"].get(t), 2),
                num(frame["delta_D_mean"].get(t), 2),
            ]
        return out

    rows: list[Any] = [[bold("Todos"), integer(overall["n"].iat[0]), *cells(overall)]]
    rows.append(r"\midrule")
    for category in categories:
        part = groups[(groups["group"] == "token_category") & (groups["value"] == category)]
        rows.append([label(CATEGORY_LABELS, category), integer(part["n"].iat[0]), *cells(part)])
    names = [t.replace("->", r"$\to$") for t in transitions]
    header = [
        header_row(["", ""] + [rf"\multicolumn{{2}}{{c}}{{\texttt{{{n}}}}}" for n in names]),
        "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(names))),
        header_row(["Categoria", "$n$"] + [r"$M\!-\!M^{\mathrm{r}}$", r"$\Delta D$"] * len(names)),
    ]
    notes = [
        r"$M\!-\!M^{\mathrm{r}}$: mudança média $1-J$ acima do piso de ruído (o piso só existe "
        r"nas transições que partem de \texttt{lex}; nas outras é $1-J$). $\Delta D$: variação "
        "média da dominância lexical."
    ]
    return Tabular(
        "lr" + "rr" * len(names), header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt"
    )


def figure_p2_curvas_camadas(ctx: Context) -> Any:
    layers = ctx.analysis("p2_layers")
    layers["token_category"] = layers["token_category"].astype(str)
    categories = ordered([c for c in layers["token_category"] if c != "all"], CATEGORY_ORDER)
    return plots.layer_curves_figure(layers, categories)
