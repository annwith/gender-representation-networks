"""The lexical type network over the vocabulary (Section 8)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.artifacts import (
    graph_id,
)
from gender_networks.labels import (
    SCRIPT_LABELS,
    SCRIPT_ORDER,
    label,
)
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import (
    DASH,
    Tabular,
    header_row,
    integer,
    num,
    pct,
    tex_text,
    tex_token,
)


def _fold_classes(classes: np.ndarray, keep: int = 5) -> tuple[np.ndarray, list[str]]:
    """Script classes with the ``keep`` largest kept and the rest folded into ``other``.

    Args:
        classes: Script class of every vertex.
        keep: Number of classes kept.

    Returns:
        ``(folded, order)``: the class of every vertex after folding, and the kept classes
        in display order, with ``other`` last when something was folded.
    """

    counts = pd.Series(classes).value_counts()
    top = [c for c in counts.index[:keep]]
    order = [c for c in SCRIPT_ORDER if c in top] + sorted(c for c in top if c not in SCRIPT_ORDER)
    folded = np.where(np.isin(classes, order), classes, "other")
    if (folded == "other").any():
        order.append("other")
    return folded, order


def figure_vocab_graus(ctx: Context) -> Any:
    """Figure ``vocab_graus``: in-degree CCDF of the vocabulary network by script class.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure, from the random tie-break sets of the vocabulary graph.

    Raises:
        Skip: If the vocabulary neighbour file or ``vocab_types.csv`` is missing.
    """

    data = ctx.npz(ctx.paths.vocab_nbr(ctx.vocab_k))
    vocab_ids = data["vocab_ids"].astype(np.int64)
    sets = data["rand"][0] if "rand" in data else data["pos"]
    in_degree = np.bincount(sets.ravel().astype(np.int64), minlength=vocab_ids.size)
    classes = ctx.vocab_types["script_class"].astype(str).to_numpy()[vocab_ids]
    folded, order = _fold_classes(classes)
    series = {}
    for cls in order:
        values = in_degree[folded == cls]
        degrees, counts = np.unique(values, return_counts=True)
        series[cls] = (degrees, counts)
    styles = plots.categorical_styles(SCRIPT_ORDER, SCRIPT_LABELS)
    styles = {
        c: styles.get(c, plots.Style(plots.MUTED, "o", label=label(SCRIPT_LABELS, c)))
        for c in order
    }
    if "other" in styles:
        styles["other"] = plots.Style(plots.MUTED, "o", label=SCRIPT_LABELS["other"])
    return plots.degree_ccdf_figure(
        series,
        ctx.vocab_k,
        styles,
        f"Rede do vocabulário ($k$ = {ctx.vocab_k}): grau de entrada por classe de escrita",
        height=2.8,
    )


def _vocab_partition(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """Leiden partition of the vocabulary graph and the token id of every vertex.

    Args:
        ctx: Context of the run.

    Returns:
        ``(membership, token ids)``, both int64 ``[V']``.

    Raises:
        Skip: If the community index or the partition is missing.
    """

    index = ctx.nonempty(ctx.paths.communities_index)
    gid = graph_id("vocab", ctx.vocab_k, "union", "rand", 0)
    hit = index[
        (index["graph_id"] == gid)
        & (index["method"] == "leiden")
        & np.isclose(pd.to_numeric(index["resolution"]), ctx.settings.analysis.resolution)
    ]
    if hit.empty:
        raise Skip(f"sem partição do Leiden para {gid}")
    data = ctx.npz(ctx.paths.metrics_dir / str(hit["file"].iat[0]))
    membership = data["membership"].astype(np.int64)
    if "vocab_ids" in data:
        ids = data["vocab_ids"].astype(np.int64)
    else:
        ids = ctx.npz(ctx.paths.vocab_nbr(ctx.vocab_k))["vocab_ids"].astype(np.int64)
    return membership, ids


def figure_vocab_comunidades(ctx: Context) -> Any:
    """Figure ``vocab_comunidades``: script-class composition of the 15 largest communities.

    Args:
        ctx: Context of the run.

    Returns:
        The matplotlib figure.

    Raises:
        Skip: If the partition or ``vocab_types.csv`` is missing.
    """

    membership, ids = _vocab_partition(ctx)
    classes = ctx.vocab_types["script_class"].astype(str).to_numpy()[ids]
    folded, order = _fold_classes(classes)
    sizes = np.bincount(membership)
    top = np.argsort(-sizes, kind="stable")[:15]
    shares = pd.DataFrame(
        {cls: [float(np.mean(folded[membership == c] == cls)) for c in top] for cls in order},
        index=[str(i + 1) for i in range(top.size)],
    )
    return plots.community_composition_figure(shares, [int(sizes[c]) for c in top], order)


def table_vocab_comunidades(ctx: Context) -> Tabular:
    """Table ``vocab_comunidades``: the ten largest vocabulary communities, with their main
    script class, share of tokens in the corpus and example tokens.

    Args:
        ctx: Context of the run.

    Returns:
        The table; the notes give the agreement with the script class and with presence in
        the corpus, and the Leiden stability.

    Raises:
        Skip: If ``vocab_communities.csv`` is missing or empty.
    """

    communities = ctx.analysis("vocab_communities")
    communities = communities.sort_values("size", ascending=False, kind="stable").head(10)
    total = None
    try:
        membership, _ = _vocab_partition(ctx)
        total = membership.size
    except Skip:
        pass
    rows = []
    for rank, (_, row) in enumerate(communities.iterrows(), start=1):
        examples = json.loads(row.get("examples") or "[]")[:6]
        rows.append(
            [
                str(rank),
                integer(row["size"]),
                pct(row["size"] / total, 1) if total else DASH,
                label(SCRIPT_LABELS, row.get("main_script_class", "")),
                pct(row.get("main_script_share"), 0),
                pct(row.get("fraction_in_corpus"), 1),
                r"\enspace ".join(tex_token(e) for e in examples) or DASH,
            ]
        )
    notes = []
    summary_path = ctx.paths.analysis_table("vocab_summary")
    if summary_path.exists():
        summary = ctx.table(summary_path)
        for name, text in (
            ("script_class", "classe de escrita"),
            ("in_corpus", "presença no corpus"),
        ):
            hit = summary[summary["label"] == name] if not summary.empty else summary
            if len(hit):
                r = hit.iloc[0]
                notes.append(
                    f"Comunidades $\\times$ {text}: NMI {num(r.get('nmi'), 3)} (linha de base "
                    f"{num(r.get('nmi_baseline'), 3)}), ARI {num(r.get('ari'), 3)}."
                )
        if len(summary) and "stability" in summary:
            notes.append(
                f"Estabilidade do Leiden: {num(summary['stability'].iat[0], 3)}; "
                f"{integer(summary['n_communities'].iat[0])} comunidades."
            )
    header = [
        header_row(
            [
                "\\#",
                "Tamanho",
                "Fração",
                "Classe principal",
                "(\\%)",
                "No corpus",
                "Exemplos (mais frequentes no corpus)",
            ]
        )
    ]
    return Tabular(
        "rrrlrrX",
        header,
        rows,
        size=r"\footnotesize",
        notes=notes or None,
        width=r"\linewidth",
        colsep="4pt",
    )


def table_vocab_vizinhos_alvos(ctx: Context) -> Tabular:
    """Table ``vocab_vizinhos_alvos``: vocabulary neighbours of each target and control word.

    Args:
        ctx: Context of the run.

    Returns:
        The table, with up to ten neighbours per word and how many of them occur in the
        corpus.

    Raises:
        Skip: If ``vocab_target_neighbors.csv`` is missing or has no word.
    """

    table = ctx.analysis("vocab_target_neighbors")
    rows: list[Any] = []
    for role, name in (("target", "alvo"), ("control", "controle")):
        part = [
            [
                tex_text(row["word"]),
                name,
                r"\enspace ".join(tex_token(t) for t in json.loads(row["neighbors"])[:10]),
                integer(row["neighbors_in_corpus"]),
            ]
            for _, row in table[table["role"] == role].iterrows()
        ]
        if part and rows:
            rows.append(r"\midrule")
        rows += part
    if not rows:
        raise Skip("nenhuma palavra-alvo ou de controle tem linha na rede do vocabulário")
    header = [
        header_row(
            [
                "Palavra",
                "Papel",
                f"Vizinhos ($k={ctx.vocab_k}$, desempate por posição)",
                "No corpus",
            ]
        )
    ]
    notes = [
        r"\textvisiblespace{} marca o espaço inicial do token; [U+\ldots] são caracteres "
        "que a fonte do relatório não tem (escritas não latinas ou pedaços de bytes)."
    ]
    return Tabular(
        "llXr", header, rows, size=r"\footnotesize", notes=notes, width=r"\linewidth", colsep="4pt"
    )
