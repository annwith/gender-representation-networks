"""P1: how much and how the organization changes (Section 8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.plots import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    STRATUM_LABELS,
    STRATUM_ORDER,
    label,
    ordered,
)
from gender_networks.report.context import Context, Skip, band_order
from gender_networks.report.latex import (
    DASH,
    Tabular,
    bold,
    header_row,
    integer,
    num,
    tex_text,
    tex_token,
)

CATEGORY_ABBR = {
    "whole_word": "PI",
    "word_start": "IP",
    "continuation": "CO",
    "punctuation": "PU",
    "number": "NU",
}


def figure_p1_jaccard_faixas(ctx: Context) -> Any:
    frame = ctx.vertex_measures()
    reps = [r for r in ctx.contextual_reps if f"J_lex_{r}" in frame]
    if not reps:
        raise Skip("vertex_measures.csv não tem J(lex, ℓ)")
    bands = ordered([b for b in frame["band_sample"] if b], band_order(ctx.settings))
    values = {
        rep: {
            b: frame.loc[frame["band_sample"] == b, f"J_lex_{rep}"].to_numpy(dtype=float)
            for b in bands
        }
        for rep in reps
    }
    floors = {}
    if "NF_lex" in frame:
        floors = {
            b: float(np.nanmean(frame.loc[frame["band_sample"] == b, "NF_lex"])) for b in bands
        }
    return plots.jaccard_bands_figure(values, floors, bands)


def figure_p1_dominancia(ctx: Context) -> Any:
    frame = ctx.vertex_measures()
    frame = frame[pd.to_numeric(frame["f_sample"], errors="coerce") >= 3]
    reps = [r for r in ctx.settings.networks.representations if f"D_{r}" in frame]
    if frame.empty or not reps:
        raise Skip("sem vértices com f_t ≥ 3 ou sem colunas D")
    rows = []
    bands = ordered([b for b in frame["band_sample"] if b], band_order(ctx.settings))
    categories = ordered(list(frame["token_category"]), CATEGORY_ORDER)
    for grouping, groups in (("band_sample", bands), ("token_category", categories)):
        for group in groups:
            part = frame[frame[grouping] == group]
            for rep in reps:
                for measure in ("D", "Dn"):
                    rows.append(
                        {
                            "measure": measure,
                            "grouping": grouping,
                            "group": group,
                            "rep": rep,
                            "value": float(np.nanmean(part[f"{measure}_{rep}"]))
                            if len(part)
                            else np.nan,
                        }
                    )
    panels = [
        ("band_sample", "faixa de frequência", bands),
        ("token_category", "categoria de token", categories),
    ]
    return plots.dominance_figure(pd.DataFrame(rows), reps, panels)


def table_p1_resumo(ctx: Context) -> Tabular:
    frame = ctx.vertex_measures()
    reps = [r for r in ctx.contextual_reps if f"J_lex_{r}" in frame and f"D_{r}" in frame]
    if not reps:
        raise Skip("vertex_measures.csv não tem as medidas por camada")
    groups: list[tuple[str, pd.Series]] = [(bold("Todos"), pd.Series(True, index=frame.index))]
    blocks = [
        ("Estrato", "stratum", ordered(list(frame["stratum"]), STRATUM_ORDER), STRATUM_LABELS),
        (
            "Categoria",
            "token_category",
            ordered(list(frame["token_category"]), CATEGORY_ORDER),
            CATEGORY_LABELS,
        ),
    ]
    rows: list[Any] = []

    def cells(mask: pd.Series) -> list[str]:
        out = []
        part = frame[mask]
        for rep in reps:
            j = part[f"J_lex_{rep}"].to_numpy(dtype=float)
            floor = part["NF_lex"].to_numpy(dtype=float) if "NF_lex" in part else np.ones_like(j)
            out += [
                num(np.nanmean(j) if j.size else np.nan, 2),
                num(np.nanmean(floor - j) if j.size else np.nan, 2),
                num(np.nanmean(part[f"D_{rep}"]) if len(part) else np.nan, 2),
                num(
                    np.nanmean(part[f"Dn_{rep}"]) if f"Dn_{rep}" in part and len(part) else np.nan,
                    2,
                ),
            ]
        return out

    for name, mask in groups:
        rows.append([name, integer(mask.sum()), *cells(mask)])
    width = 2 + 4 * len(reps)
    for title, column, values, labels in blocks:
        rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{{title}}}}} \\")
        for value in values:
            mask = frame[column] == value
            rows.append([r"\quad " + label(labels, value), integer(mask.sum()), *cells(mask)])
    header = [
        header_row(
            ["", ""] + [rf"\multicolumn{{4}}{{c}}{{\texttt{{{tex_text(r)}}}}}" for r in reps]
        ),
        "".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(reps))),
        header_row(
            ["Grupo", "$n$"] + [r"$J$", r"$M\!-\!M^{\mathrm{r}}$", "$D$", r"$\tilde D$"] * len(reps)
        ),
    ]
    notes = [
        r"$J=J_i(\mathrm{lex},\ell)$; $M\!-\!M^{\mathrm{r}}$: mudança acima do piso de ruído "
        r"(piso de $J$ entre sementes lexicais menos $J$); $D$: dominância lexical; "
        r"$\tilde D$: $D$ sobre o teto $\min(f_t-1,k)/k$ (só $f_t\geq2$). Médias por vértice."
    ]
    return Tabular(
        "lr" + "rrrr" * len(reps), header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt"
    )


def table_p1_hubs(ctx: Context) -> Tabular:
    hubs = ctx.analysis("p1_hubs")
    occ = ctx.occurrences
    reps = [r for r in ctx.settings.networks.representations if (hubs["rep"] == r).any()]
    top = int(pd.to_numeric(hubs["rank"]).max())
    blocks = [reps[i : i + 2] for i in range(0, len(reps), 2)]
    rows: list[Any] = []
    pos = occ["pos_bucket"].to_numpy()
    for b, block in enumerate(blocks):
        if b:
            rows.append(r"\midrule")
            rows.append(
                header_row(
                    [""] + [rf"\multicolumn{{4}}{{c}}{{\texttt{{{tex_text(r)}}}}}" for r in block]
                )
            )
            rows.append(header_row([""] + ["Token", "Cat.", "Pos.", "Grau"] * len(block)))
            rows.append(r"\midrule")
        for rank in range(1, top + 1):
            cells = [str(rank)]
            for rep in block:
                hit = hubs[(hubs["rep"] == rep) & (pd.to_numeric(hubs["rank"]) == rank)]
                if hit.empty:
                    cells += [DASH] * 4
                    continue
                row = hit.iloc[0]
                vertex = int(row["vertex"])
                cells += [
                    tex_token(row.get("token_text", "")),
                    CATEGORY_ABBR.get(str(row.get("token_category", "")), DASH),
                    plots.band_label(pos[vertex]) if vertex < len(pos) else DASH,
                    integer(row["in_degree"]),
                ]
            cells += [""] * (4 * (2 - len(block)))
            rows.append(cells)
    first = blocks[0]
    header = [
        header_row(
            [""]
            + [rf"\multicolumn{{4}}{{c}}{{\texttt{{{tex_text(r)}}}}}" for r in first]
            + [""] * (4 * (2 - len(first)))
        ),
        "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(first))),
        header_row(["\\#"] + ["Token", "Cat.", "Pos.", "Grau"] * 2),
    ]
    notes = [
        f"Grafos direcionados com $k={ctx.k}$, semente 0; grau = grau de entrada. Categorias: "
        + "; ".join(f"{abbr} {CATEGORY_LABELS[c]}" for c, abbr in CATEGORY_ABBR.items())
        + ". Pos.: faixa de posição na sequência."
    ]
    return Tabular("r" + "lllr" * 2, header, rows, size=r"\footnotesize", notes=notes, colsep="4pt")
