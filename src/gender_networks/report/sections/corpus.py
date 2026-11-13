"""Report items of the ``corpus`` stage (Section 4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.labels import (
    REASON_LABELS,
    REASON_ORDER,
    label,
    ordered,
    theme_label,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import Tabular, bold, header_row, integer


def table_corpus_temas(ctx: Context) -> Tabular:
    """Table ``corpus_temas``: kept articles, paragraphs and words per theme, and the dropped
    articles by reason.

    Args:
        ctx: Context of the run.

    Returns:
        The table, with one row per theme and a total row; the reason columns exist only
        for the reasons found in ``articles.jsonl``.

    Raises:
        Skip: If the corpus files are missing.
    """

    articles = ctx.articles
    stats = ctx.paragraph_stats
    reason = articles["dropped_reason"].where(articles["dropped_reason"].notna(), None)
    kept = articles[reason.isna()]
    dropped = articles[reason.notna()]
    themes = ctx.theme_order(set(articles["theme"].dropna()) | set(stats["theme"]))
    reasons = ordered(list(dropped["dropped_reason"]), REASON_ORDER) if len(dropped) else []
    rows: list[Any] = []
    totals = {"art": 0, "par": 0, "words": 0, **dict.fromkeys(reasons, 0)}
    for theme in themes:
        n_art = int((kept["theme"] == theme).sum())
        part = stats[stats["theme"] == theme]
        by_reason = dropped[dropped["theme"] == theme]["dropped_reason"].value_counts()
        cells = [
            theme_label(theme),
            integer(n_art),
            integer(len(part)),
            integer(part["words"].sum()),
        ]
        cells += [integer(by_reason.get(r, 0)) for r in reasons]
        rows.append(cells)
        totals["art"] += n_art
        totals["par"] += len(part)
        totals["words"] += int(part["words"].sum())
        for r in reasons:
            totals[r] += int(by_reason.get(r, 0))
    rows.append(r"\midrule")
    rows.append(
        [bold("Total"), integer(totals["art"]), integer(totals["par"]), integer(totals["words"])]
        + [integer(totals[r]) for r in reasons]
    )
    header = []
    if reasons:
        header.append(
            header_row(
                [
                    "",
                    "",
                    "",
                    "",
                    rf"\multicolumn{{{len(reasons)}}}{{c}}"
                    r"{Descartados por motivo}",
                ]
            )
        )
        header.append(rf"\cmidrule(l){{5-{4 + len(reasons)}}}")
    header.append(
        header_row(
            ["Tema", "Artigos", "Parágrafos", "Palavras"]
            + [label(REASON_LABELS, r) for r in reasons]
        )
    )
    return Tabular("l" + "r" * (3 + len(reasons)), header, rows, size=r"\footnotesize")


def figure_corpus_paragrafos(ctx: Context) -> Any:
    """Figure ``corpus_paragrafos``: paragraphs per theme and the paragraph length histogram.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure.

    Raises:
        Skip: If ``paragraphs.jsonl`` is missing or empty.
    """

    stats = ctx.paragraph_stats
    counts = stats["theme"].value_counts()
    return plots.corpus_paragraphs_figure(counts, stats["words"].to_numpy())


def figure_corpus_descartes(ctx: Context) -> Any:
    """Figure ``corpus_descartes``: dropped candidate pages per theme and reason.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure.

    Raises:
        Skip: If ``articles.jsonl`` is missing or no page was dropped.
    """

    articles = ctx.articles
    dropped = articles[articles["dropped_reason"].notna()]
    if dropped.empty:
        raise Skip("nenhuma página descartada")
    themes = ctx.theme_order(articles["theme"].dropna())
    table = pd.crosstab(dropped["theme"], dropped["dropped_reason"]).reindex(themes, fill_value=0)
    return plots.corpus_discards_figure(table)
