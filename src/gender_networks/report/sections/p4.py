"""P4: separation of the occurrences of a token (Section 8)."""

from __future__ import annotations

from typing import Any

import numpy as np

from gender_networks import plots
from gender_networks.labels import (
    GROUP_ORDER,
    ordered,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import Tabular, header_row, integer, num, tex_text
from gender_networks.representations import LEX

P4_MEASURES = [
    ("self_similarity_mean", "Autossimilaridade (cosseno médio)"),
    ("theme_gap_mean", "Cosseno intra − entre temas"),
    ("effective_communities_mean", "Comunidades efetivas (exp H)"),
    ("nmi_community_theme_mean", "NMI comunidade × tema"),
]


def figure_p4_separacao(ctx: Context) -> Any:
    groups = ctx.analysis("p4_groups")
    reps = [r for r in ctx.settings.networks.representations if (groups["rep"] == r).any()]
    keys = ordered([g for g in groups["group"].astype(str) if g in GROUP_ORDER], GROUP_ORDER)
    if not keys:
        raise Skip("p4_groups.csv não tem alvos, controles, multitema nem funcionais")
    measures = [(c, t) for c, t in P4_MEASURES if c in groups]
    return plots.separation_figure(groups, reps, measures, keys)


def table_p4_alvos(ctx: Context) -> Tabular:
    types = ctx.analysis("p4_types")
    types["group"] = types["group"].astype(str)
    types["word"] = types["word"].fillna("").astype(str)
    chosen = types[types["group"].isin(["target", "control"])]
    if chosen.empty:
        raise Skip("p4_types.csv não tem palavras-alvo nem controles")
    deepest = [r for r in ctx.contextual_reps if (chosen["rep"] == r).any()]
    if not deepest:
        raise Skip("p4_types.csv não tem camadas contextuais")
    last = deepest[-1]
    senses = None
    sense_path = ctx.paths.analysis_table("p4_senses")
    if sense_path.exists():
        frame = ctx.table(sense_path)
        if not frame.empty:
            senses = frame[frame["rep"] == last].set_index("word")["sense_gap"]
    rows: list[Any] = []
    words = [
        w
        for w in [*ctx.settings.sample.targets, *ctx.settings.sample.controls]
        if (chosen["word"] == w).any()
    ]
    words += sorted(set(chosen["word"]) - set(words))
    previous_group = None
    for word in words:
        part = chosen[chosen["word"] == word].set_index("rep")
        group = part["group"].iat[0]
        if previous_group is not None and group != previous_group:
            rows.append(r"\midrule")
        previous_group = group
        ref = part.loc[LEX] if LEX in part.index else None
        cur = part.loc[last] if last in part.index else None

        def pair(column: str, digits: int = 2, ref=ref, cur=cur) -> list[str]:
            return [
                num(ref[column] if ref is not None else np.nan, digits),
                num(cur[column] if cur is not None else np.nan, digits),
            ]

        any_row = part.iloc[0]
        cells = [
            tex_text(word),
            "alvo" if group == "target" else "controle",
            integer(any_row["f_sample"]),
            integer(any_row["n_themes"]),
        ]
        cells += pair("self_similarity") + pair("theme_gap", 3)
        cells += pair("effective_communities", 1) + pair("nmi_community_theme")
        if senses is not None:
            cells.append(num(senses.get(word, np.nan), 3))
        rows.append(cells)
    lx, lc = r"\texttt{lex}", r"\texttt{" + tex_text(last) + "}"
    names = [
        "Autossimilaridade",
        "Intra $-$ entre temas",
        "Comunidades efetivas",
        r"NMI com.$\times$tema",
    ]
    top = ["", "", "", ""] + [rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in names]
    sub = ["Palavra", "Papel", "$f_t$", "Temas"] + [lx, lc] * len(names)
    if senses is not None:
        top.append("Sentido")
        sub.append(lc)
    header = [
        header_row(top),
        "".join(rf"\cmidrule(lr){{{5 + 2 * i}-{6 + 2 * i}}}" for i in range(len(names))),
        header_row(sub),
    ]
    notes = [
        "Autossimilaridade: cosseno médio entre as ocorrências do tipo; intra $-$ entre temas: "
        "cosseno médio entre ocorrências do mesmo tema menos o entre temas diferentes; "
        "comunidades efetivas: $\\exp H$ das comunidades ocupadas pelas ocorrências."
        + (" Sentido: o mesmo contraste pelo sentido anotado à mão." if senses is not None else "")
    ]
    spec = "llrr" + "rr" * len(names) + ("r" if senses is not None else "")
    return Tabular(spec, header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt")
