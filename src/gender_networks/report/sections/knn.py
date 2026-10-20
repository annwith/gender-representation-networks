"""Report items of the ``knn`` stage (Section 6)."""

from __future__ import annotations

from typing import Any

from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import (
    DASH,
    Tabular,
    _finite,
    duration,
    header_row,
    integer,
    num,
    pct,
    sci,
    tex_text,
)
from gender_networks.representations import sort_names


def table_knn_empates(ctx: Context) -> Tabular:
    manifest = ctx.need_manifest("knn", "knn")
    reps = manifest.get("representations") or {}
    if not reps:
        raise Skip("o manifesto do knn não tem representações")
    n = _finite(manifest.get("n_occurrences"))
    rows: list[Any] = []
    for rep in sort_names(reps):
        info = reps[rep]
        per_k = info.get("k") or {}
        first = True
        for k in sorted(per_k, key=int):
            stats = per_k[k]
            tied = stats.get("tied_rows")
            rows.append(
                [
                    r"\texttt{" + tex_text(rep) + "}" if first else "",
                    str(k),
                    integer(tied),
                    pct(tied / n, 1) if n and _finite(tied) is not None else DASH,
                    integer(stats.get("max_tie_block")),
                    num(stats.get("all_mean_degree"), 2),
                    integer((info.get("candidates") or {}).get("fallback_rows")) if first else "",
                    duration((info.get("timings_s") or {}).get("total")) if first else "",
                ]
            )
            first = False
    vocab = manifest.get("vocab") or {}
    if vocab.get("k"):
        rows.append(r"\midrule")
        size = _finite(vocab.get("rows"))
        first = True
        for k in sorted(vocab["k"], key=int):
            stats = vocab["k"][k]
            tied = stats.get("tied_rows")
            rows.append(
                [
                    "vocab." if first else "",
                    str(k),
                    integer(tied),
                    pct(tied / size, 1) if size and _finite(tied) is not None else DASH,
                    integer(stats.get("max_tie_block")),
                    num(stats.get("all_mean_degree"), 2),
                    integer((vocab.get("candidates") or {}).get("fallback_rows")) if first else "",
                    duration(vocab.get("total_s")) if first else "",
                ]
            )
            first = False
    header = [
        header_row(
            [
                "Repr.",
                "$k$",
                r"\multicolumn{2}{c}{Linhas com empate}",
                "Maior bloco",
                r"$\overline{|F\cup B|}$",
                "Recalculadas",
                "Tempo",
            ]
        )
    ]
    notes = [
        f"Empate na fronteira: $|B|>k-|F|$ com $\\varepsilon$ = {sci(manifest.get('eps'))}. "
        r"$\overline{|F\cup B|}$: tamanho médio dos conjuntos da variante (b). Recalculadas: "
        "linhas cuja lista de candidatos foi refeita por inteiro (por representação); tempo "
        "de candidatos e vizinhos de todas as variantes."
    ]
    return Tabular("lrrrrrrr", header, rows, size=r"\footnotesize", notes=notes)
