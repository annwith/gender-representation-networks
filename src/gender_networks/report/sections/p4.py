"""P4: separation of the occurrences of a token (Section 8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.labels import (
    GROUP_ORDER,
    SENSE_LABELS,
    fmt_dec,
    label,
    ordered,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import Tabular, header_row, integer, num, tex_text
from gender_networks.representations import LEX

P4_MEASURES = [
    ("self_similarity_mean", "Autossimilaridade (cosseno médio)"),
    ("theme_gap_mean", "Cosseno intra − entre temas"),
    ("effective_communities_mean", "Comunidades efetivas (exp H)"),
    ("nmi_community_theme_excess_mean", "NMI comunidade × tema − base"),
]

# p4_senses subsets shown in the table, with their column headings.
SENSE_SUBSETS = {"all": "todas", "left": "esq."}
SENSE_COLUMNS = ["sense_gap", "nmi_community_sense", "nmi_community_sense_baseline"]
SENSE_NOTE = (
    " Sentido: o mesmo contraste pelo sentido anotado (Seção~\\ref{sec:dados-sentidos}), em "
    "todas as ocorrências anotadas e só nas que o contexto à esquerda já decide."
)
# Lexical values shown in the note instead of columns: every occurrence of a type is the same
# vector there, so they do not vary between words.
LEX_NOTE_MEASURES = [
    ("self_similarity", "autossimilaridade", 2),
    ("theme_gap", "intra $-$ entre", 3),
    ("effective_communities", "comunidades efetivas", 1),
    ("nmi_community_theme", "NMI", 2),
]


def _span(values: Any, digits: int) -> str:
    """``0,00`` when every value rounds the same, ``de 0,00 a 0,05`` otherwise.

    Args:
        values: Numeric series.
        digits: Decimal places.

    Returns:
        The pt-BR text of the range.
    """

    low, high = num(values.min(), digits), num(values.max(), digits)
    return low if low == high else f"de {low} a {high}"


def figure_p4_separacao(ctx: Context) -> Any:
    """Figure ``p4_separacao``: mean same-type separation measures per layer and group.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure, one panel per measure of :data:`P4_MEASURES`.

    Raises:
        Skip: If ``p4_groups.csv`` is missing or has none of the four groups.
    """

    groups = ctx.analysis("p4_groups")
    reps = [r for r in ctx.settings.networks.representations if (groups["rep"] == r).any()]
    keys = ordered([g for g in groups["group"].astype(str) if g in GROUP_ORDER], GROUP_ORDER)
    if not keys:
        raise Skip("p4_groups.csv não tem alvos, controles, multitema nem funcionais")
    groups = _with_nmi_excess(ctx, groups)
    measures = [(c, t) for c, t in P4_MEASURES if c in groups]
    return plots.separation_figure(groups, reps, measures, keys)


def _with_nmi_excess(ctx: Context, groups: pd.DataFrame) -> pd.DataFrame:
    """``groups`` plus ``nmi_community_theme_excess_mean``, the group mean of NMI − baseline.

    The baseline (NMI expected by chance) differs a lot between groups, so the raw NMI does
    not compare them; the panel shows what is left above it.

    Args:
        ctx: Context of the run.
        groups: Rows of ``p4_groups.csv``.

    Returns:
        ``groups`` unchanged when ``p4_types.csv`` is missing or has no baseline column.
    """

    try:
        types = ctx.analysis("p4_types")
    except Skip:
        return groups
    if "nmi_community_theme_baseline" not in types:
        return groups
    excess = (
        types.assign(
            excess=pd.to_numeric(types["nmi_community_theme"], errors="coerce")
            - pd.to_numeric(types["nmi_community_theme_baseline"], errors="coerce")
        )
        .groupby(["rep", "group"], as_index=False)["excess"]
        .mean()
        .rename(columns={"excess": "nmi_community_theme_excess_mean"})
    )
    return groups.merge(excess, on=["rep", "group"], how="left")


def table_p4_alvos(ctx: Context) -> Tabular:
    """Table ``p4_alvos``: one row per target and control word with its separation measures at
    the deepest contextual layer.

    Columns: sample frequency, number of themes, self-similarity,
    effective communities, the within-minus-between cosine and the community NMI by theme
    and, when ``p4_senses.csv`` exists, by annotated sense in each subset. Each group (targets,
    controls) ends with a ``média`` row. The notes give the mean NMI baselines and the lexical
    values.

    Args:
        ctx: Context of the run.

    Returns:
        The table.

    Raises:
        Skip: If ``p4_types.csv`` is missing or has no target, control or contextual row.
    """

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
    senses: dict[str, pd.DataFrame] = {}
    sense_path = ctx.paths.analysis_table("p4_senses")
    if sense_path.exists():
        frame = ctx.table(sense_path)
        if not frame.empty:
            frame = frame[frame["rep"] == last]
            subsets = frame["subset"] if "subset" in frame else "all"
            for subset in SENSE_SUBSETS:
                part = frame[subsets == subset]
                if not part.empty:
                    senses[subset] = part.set_index("word").reindex(columns=SENSE_COLUMNS)
    rows: list[Any] = []
    words = [
        w
        for w in [*ctx.settings.sample.targets, *ctx.settings.sample.controls]
        if (chosen["word"] == w).any()
    ]
    words += sorted(set(chosen["word"]) - set(words))
    current = chosen[chosen["rep"] == last].set_index("word")
    # Numeric cells of each word, by group, with their formatters (the role is text).
    formats: list[Any] = [integer, integer, num, lambda v: num(v, 1), lambda v: num(v, 3), num]
    formats += [lambda v: num(v, 3), num] * len(senses)
    by_group: dict[str, list[tuple[str, list[Any]]]] = {}
    for word in words:
        group = chosen.loc[chosen["word"] == word, "group"].iat[0]
        cur = current.loc[word] if word in current.index else None

        def value(column: str, cur=cur) -> Any:
            """A measure of the current word at the deepest layer.

            Args:
                column: Column of ``p4_types.csv``.
                cur: Row of the word, or ``None``; bound when the function is defined.

            Returns:
                The value, or NaN when the word or the column is absent.
            """

            return cur[column] if cur is not None and column in cur.index else np.nan

        def sense(subset: str, column: str, word=word) -> Any:
            """A sense measure of the current word in one subset.

            Args:
                subset: ``all`` or ``left``.
                column: Column of ``p4_senses.csv``.
                word: The word; bound when the function is defined.

            Returns:
                The value, or NaN when the word has no row in that subset.
            """

            table = senses[subset]
            return table.at[word, column] if word in table.index else np.nan

        any_row = chosen[chosen["word"] == word].iloc[0]
        raw = [any_row["f_sample"], any_row["n_themes"]]
        raw += [value("self_similarity"), value("effective_communities")]
        raw += [value("theme_gap"), value("nmi_community_theme")]
        for subset in senses:
            raw += [sense(subset, "sense_gap"), sense(subset, "nmi_community_sense")]
        by_group.setdefault(group, []).append((word, raw))
    for i, (group, members) in enumerate(by_group.items()):
        role = "alvo" if group == "target" else "controle"
        if i:
            rows.append(r"\midrule")
        for word, raw in members:
            rows.append([tex_text(word), role, *(f(v) for f, v in zip(formats, raw, strict=True))])
        rows.append(_mean_row(role, [raw for _, raw in members], formats))
    gap = r"\begin{tabular}[b]{@{}c@{}}Intra $-$\\entre\end{tabular}"
    blocks = ["Tema", *(f"Sentido ({SENSE_SUBSETS[subset]})" for subset in senses)]
    single = ["Palavra", "Papel", "$f_t$", "Temas"]
    single += [
        r"\begin{tabular}[b]{@{}c@{}}Autossi-\\milaridade\end{tabular}",
        r"\begin{tabular}[b]{@{}c@{}}Comunidades\\efetivas\end{tabular}",
    ]
    top = [""] * len(single) + [rf"\multicolumn{{2}}{{c}}{{{b}}}" for b in blocks]
    sub = single + [gap, "NMI"] * len(blocks)
    rules = [(len(single) + 1 + 2 * i, len(single) + 2 + 2 * i) for i in range(len(blocks))]
    header = [
        header_row(top),
        "".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in rules),
        header_row(sub),
    ]
    notes = [
        f"Valores na rede \\texttt{{{tex_text(last)}}}. Autossimilaridade: cosseno médio entre as "
        "ocorrências do tipo; comunidades efetivas: $\\exp H$ das comunidades ocupadas pelas "
        "ocorrências; intra $-$ entre: cosseno médio entre ocorrências do mesmo tema (sentido) "
        "menos o entre temas (sentidos) diferentes; NMI: entre a comunidade das ocorrências e o "
        "seu tema (sentido)." + (SENSE_NOTE if senses else "")
    ]
    baselines = _baseline_note(current, senses, words, chosen)
    if baselines:
        notes.append(baselines)
    lexical = chosen[chosen["rep"] == LEX]
    if not lexical.empty:
        values = _listing(
            [
                f"{name} {_span(lexical[column].astype(float), digits)}"
                for column, name, digits in LEX_NOTE_MEASURES
            ]
        )
        notes.append(
            "Na camada \\texttt{lex}, em que as ocorrências de um tipo são o mesmo vetor, as "
            f"palavras da tabela têm {values}."
        )
    spec = "ll" + "r" * (len(single) - 2) + "rr" * len(blocks)
    return Tabular(spec, header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt")


def table_p4_sentidos(ctx: Context) -> Tabular:
    """Table ``p4_sentidos``: the annotated senses of each target word, with their counts.

    One row per target: the number of annotated occurrences and every sense label, the most
    frequent first, with its number of occurrences and of those whose left context decides it.

    Args:
        ctx: Context of the run.

    Returns:
        The table.

    Raises:
        Skip: If ``p4_sense_communities.csv`` is missing or empty.
    """

    table = ctx.analysis("p4_sense_communities")
    if table.empty:
        raise Skip("p4_sense_communities.csv está vazia")
    # Every representation lists the same annotated occurrences; one is enough.
    table = table[table["rep"].astype(str) == str(table["rep"].iat[0])]
    table = table.assign(word=table["word"].astype(str), sense=table["sense"].astype(str))
    left = table["left_determinable"].astype(str).str.lower() == "true"
    table = table.assign(left=left)
    present = set(table["word"])
    words = [w for w in ctx.settings.sample.targets if w in present]
    words += sorted(present - set(words))
    rows: list[Any] = []
    for word in words:
        part = table[table["word"] == word]
        counts = part.groupby("sense")["left"].agg(["size", "sum"])
        listing = [
            f"{tex_text(label(SENSE_LABELS, sense))}~({integer(counts.at[sense, 'size'])};~"
            f"{integer(counts.at[sense, 'sum'])})"
            for sense in _by_count(part["sense"])
        ]
        rows.append(
            [
                tex_text(word),
                integer(len(part)),
                integer(int(part["left"].sum())),
                integer(len(counts)),
                ", ".join(listing),
            ]
        )
    header = [header_row(["Palavra", "Ocorr.", "Esq.", "Sentidos", "Sentidos anotados"])]
    notes = [
        "Ocorr.: ocorrências anotadas; Esq.: quantas delas o contexto à esquerda já decide. "
        "Cada sentido traz os mesmos dois números (ocorrências; decididas à esquerda) e eles "
        "estão em ordem de frequência."
    ]
    return Tabular(
        "lrrr>{\\raggedright\\arraybackslash}X",
        header,
        rows,
        size=r"\footnotesize",
        notes=notes,
        width=r"\linewidth",
        colsep="4pt",
    )


def _mean_row(role: str, values: list[list[Any]], formats: list[Any]) -> list[str]:
    """The ``média`` row of one group of ``p4_alvos``: the mean of each numeric column.

    Counts (frequency, themes) get one decimal place; the other columns keep the
    format of the word rows. A column without values in the group shows ``--``.

    Args:
        role: ``alvo`` or ``controle``.
        values: Numeric cells of each word of the group, in column order.
        formats: Formatter of each column of the word rows.

    Returns:
        The cells of the row.
    """

    frame = pd.DataFrame(values, dtype=float)
    cells = [r"\emph{média}", role]
    for column, fmt in zip(frame.columns, formats, strict=True):
        mean = frame[column].mean() if frame[column].notna().any() else np.nan
        cells.append(num(mean, 1) if fmt is integer else fmt(mean))
    return cells


def _baseline_note(
    current: pd.DataFrame, senses: dict[str, pd.DataFrame], words: list[str], chosen: pd.DataFrame
) -> str:
    """Mean NMI baselines over the target words, or ``""`` when the tables lack them.

    Args:
        current: Rows of ``p4_types.csv`` at the deepest layer, indexed by word.
        senses: Subset to its rows of ``p4_senses.csv``, indexed by word.
        words: Words of the table.
        chosen: Target and control rows of ``p4_types.csv``.

    Returns:
        A sentence for the table notes.
    """

    targets = [w for w in words if (chosen.loc[chosen["word"] == w, "group"] == "target").any()]
    parts: list[str] = []
    if "nmi_community_theme_baseline" in current:
        theme = current.reindex(targets)["nmi_community_theme_baseline"].astype(float)
        if theme.notna().any():
            parts.append(f"{num(theme.mean())} pelo tema")
    for subset, table in senses.items():
        sense = table.reindex(targets)["nmi_community_sense_baseline"].astype(float)
        if sense.notna().any():
            parts.append(f"{num(sense.mean())} pelo sentido ({SENSE_SUBSETS[subset]})")
    if not parts:
        return ""
    return (
        "O NMI esperado ao acaso (média com os rótulos permutados dentro do tipo) é, em média "
        f"nos alvos, {_listing(parts)}; ele cresce com o número de rótulos."
    )


def _by_count(values: pd.Series) -> list[Any]:
    """Distinct values, the most frequent first (ties in sorted order).

    Args:
        values: Series of labels.

    Returns:
        The distinct values.
    """

    counts = values.value_counts().sort_index()
    return list(counts.sort_values(ascending=False, kind="stable").index)


def _listing(names: list[str]) -> str:
    """Join names as a Portuguese list (``a, b e c``).

    Args:
        names: At least one name.

    Returns:
        The joined text.
    """

    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]


def figure_p4_comunidades_sentidos(ctx: Context) -> Any:
    """Annotated senses of each target split by the communities their occurrences fall in.

    Representations where every word sits in a single community (``lex``, whose occurrences of a
    type are one vector, and ``L01`` in the main run) are left out and named in the legend.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure; each panel title carries the NMI between community and sense
        and its baseline, when ``p4_senses.csv`` has them.

    Raises:
        Skip: If ``p4_sense_communities.csv`` is missing or lacks the shown representations.
    """

    table = ctx.analysis("p4_sense_communities")
    for column in ("rep", "word", "sense"):
        table[column] = table[column].astype(str)
    candidates = [r for r in ctx.main_reps if ctx.shows(r) and (table["rep"] == r).any()]
    if not candidates:
        raise Skip("p4_sense_communities.csv não tem as representações mostradas")
    table = table[table["rep"].isin(candidates)]
    split = table.groupby(["rep", "word"])["community"].nunique().groupby(level="rep").max()
    reps = [r for r in candidates if split[r] > 1] or candidates
    omitted = [r for r in candidates if r not in reps]
    present = set(table["word"])
    words = [w for w in ctx.settings.sample.targets if w in present]
    words += sorted(present - set(words))
    nmi: dict[tuple[str, str], tuple[float, float]] = {}
    sense_path = ctx.paths.analysis_table("p4_senses")
    if sense_path.exists():
        senses = ctx.table(sense_path)
        if not senses.empty:
            for row in senses[senses["subset"].astype(str) == "all"].itertuples():
                nmi[(str(row.rep), str(row.word))] = (
                    row.nmi_community_sense,
                    row.nmi_community_sense_baseline,
                )
    panels = []
    for word in words:
        part = table[table["word"] == word]
        order = _by_count(part[part["rep"] == reps[0]]["sense"])
        counts, titles = [], []
        for rep in reps:
            occ = part[part["rep"] == rep]
            crosstab = pd.crosstab(occ["sense"], occ["community"])
            counts.append(
                crosstab.reindex(index=order, columns=_by_count(occ["community"]), fill_value=0)
                .to_numpy()
            )
            value, baseline = nmi.get((rep, word), (np.nan, np.nan))
            titles.append(
                f"{rep}: {fmt_dec(value)} ({fmt_dec(baseline)})"
                if np.isfinite(value) and np.isfinite(baseline)
                else rep
            )
        panels.append(
            plots.SensePanel(word, [label(SENSE_LABELS, s) for s in order], counts, titles)
        )
    note = ""
    if omitted:
        plural = len(omitted) > 1
        note = (
            f"{_listing(omitted)} não aparece{'m' if plural else ''}: "
            f"nela{'s' if plural else ''}, todas as ocorrências de cada palavra ficam numa só "
            "comunidade."
        )
    return plots.sense_communities_figure(panels, note)
