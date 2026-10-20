"""Report items of the ``sample`` stage (Section 4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.plots import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    STRATUM_LABELS,
    STRATUM_ORDER,
    label,
    ordered,
    theme_label,
)
from gender_networks.report.context import Context, Skip, band_order
from gender_networks.report.latex import Tabular, bold, header_row, integer, tex_text

THEME_ABBR = {
    "fisica": "Fís.",
    "biologia": "Biol.",
    "economia": "Econ.",
    "politica": "Pol.",
    "computacao": "Comp.",
    "musica": "Mús.",
    "esporte": "Esp.",
    "geografia": "Geo.",
}


def _themes_of_sample(ctx: Context) -> list[str]:
    return ctx.theme_order(ctx.occurrences["theme"])


def table_amostra_composicao(ctx: Context) -> Tabular:
    occ = ctx.occurrences
    themes = _themes_of_sample(ctx)
    abbreviate = len(themes) > 4
    strata = ordered(list(occ["stratum"]), STRATUM_ORDER)
    rows: list[Any] = []

    def row(name: str, part: pd.DataFrame) -> list[str]:
        cells = [name]
        cells += [integer((part["theme"] == t).sum()) for t in themes]
        cells += [
            integer(len(part)),
            integer(part["paragraph_id"].nunique()),
            integer(part["token_id"].nunique()),
        ]
        return cells

    for stratum in strata:
        rows.append(row(label(STRATUM_LABELS, stratum), occ[occ["stratum"] == stratum]))
    rows.append(r"\midrule")
    rows.append(row(bold("Total"), occ))
    names = [THEME_ABBR.get(t, theme_label(t)) if abbreviate else theme_label(t) for t in themes]
    header = [
        header_row(
            [
                "",
                rf"\multicolumn{{{len(themes)}}}{{c}}{{Vértices por tema}}",
                r"\multicolumn{3}{c}{Total}",
            ]
        ),
        rf"\cmidrule(lr){{2-{1 + len(themes)}}}\cmidrule(l){{{2 + len(themes)}-{4 + len(themes)}}}",
        header_row(["Estrato", *names, "Vértices", "Parágrafos", "Tipos"]),
    ]
    notes = None
    if abbreviate:
        notes = [
            "Temas: " + "; ".join(f"{THEME_ABBR.get(t, t)} {theme_label(t)}" for t in themes) + "."
        ]
    return Tabular(
        "l" + "r" * (len(themes) + 3),
        header,
        rows,
        size=r"\footnotesize",
        notes=notes,
        colsep="4pt",
    )


def figure_amostra_composicao(ctx: Context) -> Any:
    occ = ctx.occurrences
    panels = []
    strata = ordered(list(occ["stratum"]), STRATUM_ORDER)
    counts = occ["stratum"].value_counts()
    panels.append(
        (
            "Estrato",
            [label(STRATUM_LABELS, s) for s in strata],
            [int(counts.get(s, 0)) for s in strata],
        )
    )
    themes = _themes_of_sample(ctx)
    counts = occ["theme"].value_counts()
    panels.append(
        ("Tema", [theme_label(t) for t in themes], [int(counts.get(t, 0)) for t in themes])
    )
    bands = ordered(list(occ["band_sample"]), band_order(ctx.settings))
    counts = occ["band_sample"].value_counts()
    panels.append(
        (
            "Faixa de frequência na amostra (f_t)",
            [plots.band_label(b) for b in bands],
            [int(counts.get(b, 0)) for b in bands],
        )
    )
    categories = ordered(list(occ["token_category"]), CATEGORY_ORDER)
    counts = occ["token_category"].value_counts()
    panels.append(
        (
            "Categoria de token",
            [label(CATEGORY_LABELS, c) for c in categories],
            [int(counts.get(c, 0)) for c in categories],
        )
    )
    return plots.sample_composition_figure(panels)


def table_amostra_faixas(ctx: Context) -> Tabular:
    occ = ctx.occurrences
    categories = ordered(list(occ["token_category"]), CATEGORY_ORDER)
    rows: list[Any] = []
    width = len(categories) + 2
    for column, title, order in (
        ("band_sample", "Faixa de frequência na amostra", band_order(ctx.settings)),
        ("band_corpus", "Faixa de frequência no corpus", band_order(ctx.settings, True)),
    ):
        if column not in occ or (occ[column] == "").all():
            continue
        if rows:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{{title}}}}} \\")
        for band in ordered([b for b in occ[column] if b], order):
            part = occ[occ[column] == band]
            cells = [r"\quad " + plots.band_label(band)]
            cells += [integer((part["token_category"] == c).sum()) for c in categories]
            cells.append(integer(len(part)))
            rows.append(cells)
    rows.append(r"\midrule")
    total = [bold("Total")]
    total += [integer((occ["token_category"] == c).sum()) for c in categories]
    total.append(integer(len(occ)))
    rows.append(total)
    header = [header_row(["Faixa", *[label(CATEGORY_LABELS, c) for c in categories], "Total"])]
    return Tabular("l" + "r" * (len(categories) + 1), header, rows, size=r"\footnotesize")


def table_amostra_cotas(ctx: Context) -> Tabular:
    sample = ctx.need_manifest("sample", "sample")
    rows: list[Any] = []
    for role, key in (("alvo", "targets"), ("controle", "controls")):
        report = sample.get(key) or {}
        if not report:
            continue
        if rows:
            rows.append(r"\midrule")
        for word, info in report.items():
            themes = info.get("themes") or {}
            first = True
            for theme in ctx.theme_order(themes):
                entry = themes[theme]
                cells = [
                    tex_text(word) if first else "",
                    role if first else "",
                    theme_label(theme) + ("" if entry.get("in_sample", True) else "$^{*}$"),
                    integer(entry.get("requested")),
                    integer(entry.get("core")),
                    integer(entry.get("drawn")),
                    integer(entry.get("obtained")),
                    ("sim" if info.get("kept") else bold("não")) if first else "",
                ]
                rows.append(cells)
                first = False
    if not rows:
        raise Skip("o manifesto da amostra não tem cotas de alvos ou controles")
    notes = []
    dropped = (sample.get("dropped_targets") or []) + (sample.get("dropped_controls") or [])
    if dropped:
        parts = []
        for entry in dropped:
            reached = entry.get("themes_reaching_min") or []
            parts.append(
                f"{tex_text(entry.get('word', ''))} ({len(reached)} tema(s) com o mínimo; "
                f"{integer(entry.get('removed_occurrences', 0))} ocorrências retiradas)"
            )
        notes.append("Descartadas por falta de ocorrências: " + "; ".join(parts) + ".")
    rejected = sample.get("not_single_token") or {}
    if rejected:
        notes.append(
            "Fora por não serem um token só: " + ", ".join(tex_text(w) for w in rejected) + "."
        )
    if any("$^{*}$" in str(cell) for row in rows if isinstance(row, list) for cell in row):
        notes.append("$^{*}$ tema fora dos temas da amostra.")
    notes.append(
        f"Mínimo por tema de sentido: {ctx.settings.sample.min_per_sense} ocorrências em pelo "
        "menos dois temas."
    )
    header = [
        header_row(
            [
                "Palavra",
                "Papel",
                "Tema de sentido",
                "Cota",
                "Núcleo",
                "Sorteadas",
                "Obtidas",
                "Mantida",
            ]
        )
    ]
    return Tabular("lllrrrrl", header, rows, size=r"\footnotesize", notes=notes)
