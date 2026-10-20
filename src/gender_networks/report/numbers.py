"""``numeros.tex``: the macros with the numbers cited in the text of the report."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gender_networks.artifacts import (
    STAGE_DIRS,
    ensure_dir,
)
from gender_networks.report.context import STAGE, Context, Skip
from gender_networks.report.latex import HEADER, finite_or_none, integer, pct, sci, tex_text

# Macros of numeros.tex (the \providecommand list of relatorio.tex).
MACROS = [
    "nArtigosCandidatos",
    "nArtigos",
    "nParagrafos",
    "nPalavrasCorpus",
    "nTokensCorpus",
    "nTiposCorpus",
    "nVertices",
    "nTipos",
    "nSequencias",
    "nTokensProcessados",
    "nParagrafosNucleo",
    "nVerticesNucleo",
    "nVerticesAlvo",
    "nVerticesControle",
    "nVerticesMultitema",
    "nAlvosMantidos",
    "nGruposPrefixo",
    "nTiposVocab",
    "nTiposVocabCorpus",
    "difMaxPrefixo",
    "concordanciaPredicao",
    "dataExecucao",
    "revisaoGit",
]


# Stage manifests in pipeline order: (label, directory name inside the run, or None = corpus).
STAGE_MANIFESTS = [("corpus", None)] + [
    (stage, directory) for stage, directory in STAGE_DIRS.items() if stage != STAGE
]


def collect_numbers(ctx: Context) -> dict[str, str]:
    """Value of every macro that the artifacts on disk can give (formatted pt-BR text)."""

    values: dict[str, str] = {}
    corpus = ctx.manifest(None)
    if corpus:
        for macro, key in (
            ("nArtigosCandidatos", "articles_considered"),
            ("nArtigos", "articles_kept"),
            ("nParagrafos", "paragraphs"),
            ("nPalavrasCorpus", "words"),
        ):
            if corpus.get(key) is not None:
                values[macro] = integer(corpus[key])
    sample = ctx.manifest("sample")
    if sample:
        for macro, key in (
            ("nTokensCorpus", "corpus_tokens"),
            ("nVertices", "n_vertices"),
            ("nTipos", "n_types"),
            ("nSequencias", "n_sequences"),
            ("nTokensProcessados", "tokens_to_process"),
            ("nParagrafosNucleo", "core_paragraphs"),
            ("nGruposPrefixo", "prefix_groups"),
        ):
            if sample.get(key) is not None:
                values[macro] = integer(sample[key])
        strata = sample.get("by_stratum") or {}
        for macro, key in (
            ("nVerticesNucleo", "core"),
            ("nVerticesAlvo", "target"),
            ("nVerticesControle", "control"),
            ("nVerticesMultitema", "multitheme"),
        ):
            values[macro] = integer(strata.get(key, 0))
        if sample.get("kept_targets") is not None:
            values["nAlvosMantidos"] = integer(len(sample["kept_targets"]))
        vocab = sample.get("vocab") or {}
        if vocab.get("in_corpus") is not None:
            values["nTiposCorpus"] = integer(vocab["in_corpus"])
    extract = ctx.manifest("reps")
    if extract:
        if extract.get("prefix_groups") is not None:
            values["nGruposPrefixo"] = integer(extract["prefix_groups"])
        diffs = extract.get("prefix_max_abs_diff")
        if isinstance(diffs, Mapping) and diffs:
            numbers = [finite_or_none(v) for v in diffs.values()]
            numbers = [v for v in numbers if v is not None]
            if numbers:
                values["difMaxPrefixo"] = sci(max(numbers), 1)
        elif finite_or_none(diffs) is not None:
            values["difMaxPrefixo"] = sci(diffs, 1)
    knn_manifest = ctx.manifest("knn")
    vocab_ids = None
    try:
        vocab_ids = ctx.npz(ctx.paths.vocab_nbr(ctx.vocab_k))["vocab_ids"].astype(np.int64)
    except Skip:
        pass
    if knn_manifest and (knn_manifest.get("vocab") or {}).get("rows") is not None:
        values["nTiposVocab"] = integer(knn_manifest["vocab"]["rows"])
    elif vocab_ids is not None:
        values["nTiposVocab"] = integer(vocab_ids.size)
    if vocab_ids is not None and ctx.paths.vocab_types.exists():
        f_corpus = ctx.vocab_types["f_corpus"].to_numpy(dtype=np.int64)
        values["nTiposVocabCorpus"] = integer(int(np.sum(f_corpus[vocab_ids] > 0)))
    lens = ctx.paths.lens_file("summary.json")
    if lens.exists():
        agreement = (ctx.json(lens).get("pred_next_agreement") or {}).get("rate")
        if finite_or_none(agreement) is not None:
            values["concordanciaPredicao"] = pct(agreement, 1)
    latest = latest_manifest(ctx)
    if latest is not None:
        stamp = finite_or_none(latest.get("started_unix"))
        if stamp is not None:
            finished = stamp + (finite_or_none(latest.get("elapsed_s")) or 0.0)
            values["dataExecucao"] = datetime.fromtimestamp(finished).strftime("%d/%m/%Y")
        gits = {m.get("git") for m in stage_manifests(ctx).values() if m.get("git")}
        if latest.get("git"):
            text = tex_text(latest["git"])
            if len(gits) > 1:
                text += " (etapas em revisões diferentes; ver Tabela~\\ref{tab:versoes})"
            values["revisaoGit"] = text
    return values


def stage_manifests(ctx: Context) -> dict[str, dict[str, Any]]:
    out = {}
    for stage, directory in STAGE_MANIFESTS:
        data = ctx.manifest(directory)
        if data is not None:
            out[stage] = data
    return out


def latest_manifest(ctx: Context) -> dict[str, Any] | None:
    manifests = [m for m in stage_manifests(ctx).values() if finite_or_none(m.get("started_unix"))]
    return max(manifests, key=lambda m: float(m["started_unix"]), default=None)


def write_numbers(ctx: Context, path: Path) -> dict[str, str]:
    values = collect_numbers(ctx)
    lines = [
        HEADER,
        "% Macros dos números citados no texto; relatorio.tex usa \\providecommand para as que",
        "% faltarem (impressas como [pendente]).",
    ]
    for macro in MACROS:
        if macro in values:
            lines.append(f"\\newcommand{{\\{macro}}}{{{values[macro]}}}")
        else:
            lines.append(f"% \\{macro}: sem dados nesta execução")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values
