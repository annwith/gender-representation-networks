"""Stage ``report``: figures, tables and numbers of the technical report (plan stage 6).

Reads whatever the earlier stages left on disk and writes, under ``paths.report_dir``:

- ``figuras/<name>.pdf``: vector figures in the single style of :mod:`gender_networks.plots`;
- ``tabelas/<name>.tex``: only the ``tabular`` environment (booktabs, Portuguese headers);
  caption and label live in the report (``\\tabelaopcional``);
- ``tabelas/numeros.tex``: ``\\newcommand`` macros with the numbers cited in the text. The file is
  ``\\input`` *before* the ``\\providecommand`` fallbacks of ``relatorio.tex``, so a macro that
  is not written here prints *[pendente]*.

The names and the content of each item follow the captions in ``report/relatorio/secoes``
(see :data:`FIGURES` and :data:`TABLES`). Every item is optional: when its inputs are missing it
is skipped (and a stale file of the same name from an older run is removed, so the report never
mixes runs); the reason is logged and recorded in ``<run>/report/_manifest.json``. An item that
fails with an error does not stop the others; the stage raises at the end, listing them.

Numbers are plain pt-BR text (``15.104``, ``0,83``), never ``\\num``: the report does not load
siunitx. Characters that pdflatex cannot typeset with T1 fonts (CJK, emoji, replacement chars of
byte-fragment tokens) are written as ``[U+XXXX]``.

Freshness: the stage always regenerates everything. It only reads small tables and a few
neighbour files (seconds at the size of the main run), and checking every input of every item
would cost more code than it saves. When ``latexmk`` exists and ``report_dir`` holds
``relatorio.tex`` (the real report, not the mini rehearsal), the document is compiled at the end;
a compilation failure is logged and recorded but does not fail the stage.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gender_networks import plots
from gender_networks.artifacts import RunPaths, ensure_dir, read_json, read_jsonl, write_manifest
from gender_networks.neighborhood import jaccard
from gender_networks.plots import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    GROUP_ORDER,
    LABEL_LABELS,
    REASON_LABELS,
    REASON_ORDER,
    SCRIPT_LABELS,
    SCRIPT_ORDER,
    STRATUM_LABELS,
    STRATUM_ORDER,
    fmt_dec,
    fmt_int,
    label,
    ordered,
    theme_label,
)
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)

LEXICAL = "lex"
STAGE = "report"
# Every file the report expects (\figuraopcional / \tabelaopcional in report/relatorio/secoes).
FIGURES = [
    "corpus_paragrafos",
    "corpus_descartes",
    "amostra_composicao",
    "extracao_camadas",
    "extracao_dimensoes",
    "graus_camadas",
    "subgrafo_camadas",
    "p1_jaccard_faixas",
    "p1_dominancia",
    "p2_transicoes",
    "p2_curvas_camadas",
    "p3_nmi_camadas",
    "p4_separacao",
    "p4_ego_banco",
    "robustez_empates",
    "vocab_graus",
    "vocab_comunidades",
    "lente_ranking",
]
TABLES = [
    "corpus_temas",
    "amostra_composicao",
    "amostra_cotas",
    "amostra_faixas",
    "extracao_diagnosticos",
    "extracao_verificacao",
    "knn_empates",
    "metricas_globais",
    "p1_resumo",
    "p1_hubs",
    "p2_transicoes",
    "p3_concordancia",
    "p4_alvos",
    "robustez",
    "vocab_comunidades",
    "vocab_vizinhos_alvos",
    "versoes",
]
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
STAGE_MANIFESTS = [
    ("corpus", None),
    ("sample", "sample"),
    ("extract", "reps"),
    ("knn", "knn"),
    ("metrics", "metrics"),
    ("analyze", "analysis"),
    ("lens", "lens"),
]
SUBGRAPH_WINDOW = 14
SUBGRAPH_TOP_TYPES = 7
EGO_WORD = "banco"
LAYOUT_SEED = 438
HEADER = "% Gerado pela etapa `report` (gender_networks.report). Não editar à mão.\n"
CATEGORY_ABBR = {
    "whole_word": "PI",
    "word_start": "IP",
    "continuation": "CO",
    "punctuation": "PU",
    "number": "NU",
}
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


class Skip(Exception):
    """An item cannot be produced from the artifacts on disk (the message says why)."""


# --------------------------------------------------------------------------------------------
# LaTeX text and numbers

# Code points pdflatex typesets with inputenc (utf8) + T1 + lmodern, probed one by one.
_TEX_RANGES = [
    (0xA0, 0x125), (0x128, 0x137), (0x139, 0x13E), (0x141, 0x148), (0x14A, 0x165),
    (0x168, 0x17E), (0x192, 0x192), (0x1C4, 0x1D4), (0x1E2, 0x1E3), (0x1E6, 0x1EB),
    (0x1F0, 0x1F0), (0x1F4, 0x1F5), (0x218, 0x21B), (0x232, 0x233), (0x237, 0x237),
    (0x2010, 0x2016), (0x2018, 0x201A), (0x201C, 0x201E), (0x2020, 0x2022), (0x2026, 0x2026),
    (0x2030, 0x2031), (0x2039, 0x203B), (0x203D, 0x203D), (0x20AC, 0x20AC), (0x2122, 0x2122),
]  # fmt: skip
_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "%": r"\%",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    '"': r"\textquotedbl{}",
    "`": r"\textasciigrave{}",
    "\u00ad": "-{}",
}
_LIGATURES = set("-,'!?")  # "--", ",,", "''", "!`" and "?`" would become other glyphs


def _typesettable(char: str) -> bool:
    code = ord(char)
    if 0x20 <= code < 0x7F:
        return True
    return any(low <= code <= high for low, high in _TEX_RANGES)


def tex_text(text: Any, visible_space: bool = False) -> str:
    """Escape arbitrary text for LaTeX; unsupported characters become ``[U+XXXX]``."""

    out: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            shown = " ".join(f"U+{ord(c):04X}" for c in pending[:2])
            more = r"\ldots{}" if len(pending) > 2 else ""
            out.append(f"[{shown}{more}]")
            pending.clear()

    for char in str(text):
        if char in _TEX_SPECIAL:
            flush()
            out.append(_TEX_SPECIAL[char])
        elif char == " ":
            flush()
            out.append(r"\textvisiblespace{}" if visible_space else " ")
        elif char in "\n\r\t":
            flush()
            out.append(
                {
                    "\n": r"\textbackslash{}n",
                    "\r": r"\textbackslash{}r",
                    "\t": r"\textbackslash{}t",
                }[char]
            )
        elif _typesettable(char):
            flush()
            out.append(char + "{}" if char in _LIGATURES else char)
        else:
            pending.append(char)
    flush()
    return "".join(out)


def tex_token(text: Any) -> str:
    """A token as it appears in the text: typewriter, leading/inner spaces made visible."""

    value = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    return r"\texttt{" + (tex_text(value, visible_space=True) or r"\textit{(vazio)}") + "}"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _signed(text: str) -> str:
    return "$-$" + text[1:] if text.startswith("-") else text


DASH = "--"


def num(value: Any, digits: int = 2) -> str:
    """Decimal with comma (``0,83``); ``--`` when missing."""

    number = _finite(value)
    if number is None:
        return DASH
    text = fmt_dec(number, digits)
    if re.fullmatch(r"-0(,0*)?", text):
        text = text[1:]
    return _signed(text)


def integer(value: Any) -> str:
    number = _finite(value)
    return DASH if number is None else _signed(fmt_int(number))


def pct(value: Any, digits: int = 1) -> str:
    number = _finite(value)
    return DASH if number is None else num(number * 100, digits) + r"\,\%"


def sci(value: Any, digits: int = 1) -> str:
    """Scientific notation in math mode for small or large values (``$2{,}4\\times10^{-6}$``)."""

    number = _finite(value)
    if number is None:
        return DASH
    if number == 0:
        return "0"
    exponent = math.floor(math.log10(abs(number)))
    if -3 <= exponent <= 3:
        return num(number, digits - exponent if exponent < 0 else digits)
    mantissa = number / 10**exponent
    text = fmt_dec(mantissa, digits).replace(",", "{,}")
    return f"${text}\\times10^{{{exponent}}}$"


def pm(mean: Any, sd: Any, digits: int = 2, kind: str = "num") -> str:
    """``mean ± sd`` (the sd only when finite and positive)."""

    fmt = integer if kind == "int" else (lambda v: num(v, digits))
    text = fmt(mean)
    sd_value = _finite(sd)
    if sd_value is not None and sd_value > 0 and text != DASH:
        shown = num(sd_value, digits if kind != "int" else 1)
        text += r"\,$\pm$\," + shown
    return text


def duration(seconds: Any) -> str:
    value = _finite(seconds)
    if value is None:
        return DASH
    if value < 60:
        return num(value, 1) + " s"
    minutes, secs = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def bold(text: str) -> str:
    return r"\textbf{" + text + "}"


@dataclass
class Tabular:
    """A booktabs ``tabular`` (or ``tabularx``) fragment built row by row."""

    spec: str
    header: list[str]
    rows: list[Any]
    size: str = ""  # e.g. \footnotesize
    notes: list[str] | None = None
    width: str | None = None  # tabularx width (e.g. \linewidth)
    colsep: str | None = None

    def render(self) -> str:
        env = "tabularx" if self.width else "tabular"
        head = [HEADER]
        if self.size:
            head.append(self.size + "\n")
        if self.colsep:
            head.append(f"\\setlength{{\\tabcolsep}}{{{self.colsep}}}\n")
        width = f"{{{self.width}}}" if self.width else ""
        lines = [f"\\begin{{{env}}}{width}{{{self.spec}}}", r"\toprule"]
        lines += self.header
        lines.append(r"\midrule")
        for row in self.rows:
            if isinstance(row, str):
                lines.append(row)
            else:
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(f"\\end{{{env}}}")
        body = "\n".join(lines) + "\n"
        if self.notes:
            notes = " ".join(self.notes)
            body += "\n\\par\\smallskip{\\footnotesize\\raggedright " + notes + "\\par}\n"
        return "".join(head) + body


def header_row(cells: Sequence[str]) -> str:
    return " & ".join(cells) + r" \\"


# --------------------------------------------------------------------------------------------
# reading


def read_table(path: Path) -> pd.DataFrame:
    """CSV with only empty cells as missing (tokens like ``NA`` stay text); empty -> empty."""

    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def band_order(settings: Settings, open_top: bool = False) -> list[str]:
    bands = settings.sample.bands
    out = []
    for number, (low, high) in enumerate(bands):
        if open_top and number == len(bands) - 1:
            out.append(f"{low}+")
        else:
            out.append(f"{low}" if low == high else f"{low}-{high}")
    return out


class Context:
    """Lazy, cached access to the artifacts of one run."""

    def __init__(self, settings: Settings, paths: RunPaths) -> None:
        self.settings = settings
        self.paths = paths
        self.k = settings.networks.k_main
        self._tables: dict[Path, pd.DataFrame] = {}
        self._npz: dict[Path, dict[str, np.ndarray]] = {}
        self.read: set[Path] = set()

    # generic readers ----------------------------------------------------------------------

    def table(self, path: Path) -> pd.DataFrame:
        if path not in self._tables:
            if not path.exists():
                raise Skip(f"falta {path}")
            self._tables[path] = read_table(path)
            self.read.add(path)
        return self._tables[path]

    def nonempty(self, path: Path) -> pd.DataFrame:
        frame = self.table(path)
        if frame.empty:
            raise Skip(f"{path} está vazio")
        return frame

    def analysis(self, name: str) -> pd.DataFrame:
        return self.nonempty(self.paths.analysis_dir / name)

    def metrics_table(self, name: str) -> pd.DataFrame:
        return self.nonempty(self.paths.metrics_dir / name)

    def npz(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._npz:
            if not path.exists():
                raise Skip(f"falta {path}")
            with np.load(path) as data:
                self._npz[path] = {key: data[key] for key in data.files}
            self.read.add(path)
        return self._npz[path]

    def json(self, path: Path) -> Any:
        if not path.exists():
            raise Skip(f"falta {path}")
        self.read.add(path)
        return read_json(path)

    def manifest(self, directory: str | None) -> dict[str, Any] | None:
        path = self.paths.corpus_manifest if directory is None else self.paths.manifest(directory)
        if not path.exists():
            return None
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        self.read.add(path)
        return data

    def need_manifest(self, directory: str | None, stage: str) -> dict[str, Any]:
        data = self.manifest(directory)
        if data is None:
            raise Skip(f"falta o _manifest.json da etapa {stage}")
        return data

    # run artifacts ------------------------------------------------------------------------

    @cached_property
    def occurrences(self) -> pd.DataFrame:
        frame = self.table(self.paths.occurrences)
        if frame.empty:
            raise Skip("occurrences.csv está vazio")
        for column in (
            "token_text",
            "word",
            "target_word",
            "sense_theme",
            "theme",
            "stratum",
            "token_category",
            "band_sample",
            "band_corpus",
            "pos_bucket",
            "paragraph_id",
        ):
            if column in frame:
                frame[column] = frame[column].fillna("").astype(str)
        return frame

    @cached_property
    def vocab_types(self) -> pd.DataFrame:
        frame = self.table(self.paths.vocab_types)
        if frame.empty:
            raise Skip("vocab_types.csv está vazio")
        frame = frame.sort_values("token_id").reset_index(drop=True)
        frame["token_repr"] = frame["token_repr"].fillna("").astype(str)
        return frame

    def vocab_text(self, token_id: int) -> str:
        return decode_repr(self.vocab_types["token_repr"].iat[int(token_id)])

    @cached_property
    def main_reps(self) -> list[str]:
        reps = [
            rep
            for rep in self.settings.networks.representations
            if self.paths.nbr(rep, self.k).exists()
        ]
        if not reps:
            raise Skip(f"nenhum arquivo de vizinhos com k = {self.k} em {self.paths.knn_dir}")
        return reps

    @property
    def contextual_reps(self) -> list[str]:
        return [rep for rep in self.settings.networks.representations if rep != LEXICAL]

    def nbr(self, rep: str, k: int | None = None) -> dict[str, np.ndarray]:
        return self.npz(self.paths.nbr(rep, self.k if k is None else k))

    @cached_property
    def vocab_k(self) -> int:
        from gender_networks.metrics import vocab_community_k

        value = vocab_community_k(self.settings)
        if value is None:
            raise Skip("networks.vocab.k_values está vazio")
        return value

    @cached_property
    def paragraph_stats(self) -> pd.DataFrame:
        if not self.paths.paragraphs.exists():
            raise Skip(f"falta {self.paths.paragraphs}")
        self.read.add(self.paths.paragraphs)
        rows = [
            (record.get("theme", ""), len(str(record.get("text", "")).split()))
            for record in read_jsonl(self.paths.paragraphs)
        ]
        if not rows:
            raise Skip("paragraphs.jsonl está vazio")
        return pd.DataFrame(rows, columns=["theme", "words"])

    @cached_property
    def articles(self) -> pd.DataFrame:
        if not self.paths.articles.exists():
            raise Skip(f"falta {self.paths.articles}")
        self.read.add(self.paths.articles)
        frame = pd.DataFrame(list(read_jsonl(self.paths.articles)))
        if frame.empty:
            raise Skip("articles.jsonl está vazio")
        return frame

    def theme_order(self, themes: Iterable[str]) -> list[str]:
        return ordered(list(themes), list(self.settings.corpus.themes))


def decode_repr(text: str) -> str:
    """``"' banco'"`` (the ``repr`` stored in vocab_types.csv) -> ``" banco"``."""

    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return value if isinstance(value, str) else str(value)


_REPR = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def split_reprs(text: str) -> list[str]:
    """Tokens of a space-joined list of ``repr`` strings (analysis ``examples`` columns)."""

    return [decode_repr(match) for match in _REPR.findall(str(text))]


# --------------------------------------------------------------------------------------------
# numbers (numeros.tex)


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
            numbers = [_finite(v) for v in diffs.values()]
            numbers = [v for v in numbers if v is not None]
            if numbers:
                values["difMaxPrefixo"] = sci(max(numbers), 1)
        elif _finite(diffs) is not None:
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
    lens = ctx.paths.lens_dir / "summary.json"
    if lens.exists():
        agreement = (ctx.json(lens).get("pred_next_agreement") or {}).get("rate")
        if _finite(agreement) is not None:
            values["concordanciaPredicao"] = pct(agreement, 1)
    latest = latest_manifest(ctx)
    if latest is not None:
        stamp = _finite(latest.get("started_unix"))
        if stamp is not None:
            finished = stamp + (_finite(latest.get("elapsed_s")) or 0.0)
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
    manifests = [m for m in stage_manifests(ctx).values() if _finite(m.get("started_unix"))]
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


# --------------------------------------------------------------------------------------------
# corpus


def table_corpus_temas(ctx: Context) -> Tabular:
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
    stats = ctx.paragraph_stats
    counts = stats["theme"].value_counts()
    return plots.corpus_paragraphs_figure(counts, stats["words"].to_numpy())


def figure_corpus_descartes(ctx: Context) -> Any:
    articles = ctx.articles
    dropped = articles[articles["dropped_reason"].notna()]
    if dropped.empty:
        raise Skip("nenhuma página descartada")
    themes = ctx.theme_order(articles["theme"].dropna())
    table = pd.crosstab(dropped["theme"], dropped["dropped_reason"]).reindex(themes, fill_value=0)
    return plots.corpus_discards_figure(table)


# --------------------------------------------------------------------------------------------
# sample


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


# --------------------------------------------------------------------------------------------
# extraction


def _diagnostics(ctx: Context) -> dict[str, Any]:
    return ctx.json(ctx.paths.diagnostics)


def _rep_order(names: Iterable[str]) -> list[str]:
    order = [LEXICAL, "L01", "L18", "L36", "L36n"]
    names = list(names)
    known = [n for n in order if n in names]
    blocks = sorted((n for n in names if n not in known), key=lambda n: (len(n), n))
    return known + blocks


def table_extracao_diagnosticos(ctx: Context) -> Tabular:
    reps = _diagnostics(ctx).get("representations") or {}
    if not reps:
        raise Skip("diagnostics.json não tem representações")
    rows = []
    factor = None
    n_top = 0
    for rep in _rep_order(reps):
        info = reps[rep]
        q = info.get("norm_quantiles") or {}
        top = info.get("top_dims") or []
        n_top = max(n_top, len(top))
        factor = info.get("outlier_factor", factor)
        rows.append(
            [
                r"\texttt{" + tex_text(rep) + "}",
                num(q.get("5"), 1),
                num(q.get("50"), 1),
                num(q.get("95"), 1),
                num(q.get("100"), 1),
                num(info.get("mean_pair_cosine"), 3),
                integer(top[0]["dim"]) if top else DASH,
                num(info.get("top_dim_mean_abs_over_median_dim"), 1),
                pct(info.get("top_dims_mean_share_sq_norm"), 1),
                integer(info.get("n_norm_above_factor_median")),
            ]
        )
    header = [
        header_row(
            [
                "",
                r"\multicolumn{4}{c}{Norma (quantis)}",
                "",
                r"\multicolumn{3}{c}"
                r"{Dimensões dominantes}",
                "",
            ]
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-9}",
        header_row(
            [
                "Repr.",
                "5\\%",
                "50\\%",
                "95\\%",
                "máx.",
                "Cos. médio",
                "Maior",
                "$|x|$/mediana",
                f"norma² ({n_top} dim.)",
                f"$>{num(factor, 0)}\\times$med." if factor else "Atípicas",
            ]
        ),
    ]
    notes = [
        "Cos. médio: cosseno médio entre pares aleatórios de vértices (anisotropia). Maior: "
        "índice da dimensão de maior $|x|$ médio; $|x|$/mediana: esse valor sobre a mediana das "
        "dimensões; norma²: fração média da norma ao quadrado nas dimensões de maior $|x|$; "
        "a última coluna conta vértices com norma acima desse múltiplo da mediana."
    ]
    return Tabular("l" + "r" * 9, header, rows, size=r"\footnotesize", notes=notes, colsep="4pt")


def figure_extracao_camadas(ctx: Context) -> Any:
    diagnostics = _diagnostics(ctx)
    curve = pd.DataFrame(diagnostics.get("layer_curve") or [])
    if curve.empty:
        raise Skip("diagnostics.json não tem a curva por camada (capture_all_layers desligado?)")
    lexical = (diagnostics.get("representations") or {}).get(LEXICAL)
    marked = {name: layer for name, layer in ctx.settings.model.layers.items()}
    return plots.extraction_layers_figure(curve, lexical, marked)


def figure_extracao_dimensoes(ctx: Context) -> Any:
    reps = _diagnostics(ctx).get("representations") or {}
    if not reps:
        raise Skip("diagnostics.json não tem representações")
    return plots.extraction_dimensions_figure({rep: reps[rep] for rep in _rep_order(reps)})


def _yes(flag: Any) -> str:
    return "ok" if flag else bold("falhou")


def table_extracao_verificacao(ctx: Context) -> Tabular:
    report = ctx.json(ctx.paths.verify_report)
    checks = report.get("checks") or {}
    tol = report.get("tolerance") or {}
    tol_text = f"rtol {sci(tol.get('rtol'))}, atol {sci(tol.get('atol'))}" if tol else DASH
    rows: list[Any] = []
    if "hidden_states_count" in checks:
        c = checks["hidden_states_count"]
        found = ", ".join(str(v) for v in c.get("found", []))
        rows.append(
            [
                "\\texttt{len(hidden\\_states)} $=$ blocos $+1$",
                _yes(c.get("passed")),
                f"{found} (esperado {c.get('expected')})",
                "igualdade",
            ]
        )
    if "embedding_input" in checks:
        c = checks["embedding_input"]
        rows.append(
            [
                "\\texttt{hidden\\_states[0]} $=$ \\texttt{embed\\_tokens(ids)}",
                _yes(c.get("passed")),
                sci(c.get("max_abs_diff")),
                "bit a bit",
            ]
        )
    if "hooks_match_hidden_states" in checks:
        c = checks["hooks_match_hidden_states"]
        for layer, info in (c.get("layers") or {}).items():
            rows.append(
                [
                    f"\\textit{{hook}} em \\texttt{{layers[{int(layer) - 1}]}} $=$ "
                    f"\\texttt{{hidden\\_states[{layer}]}}",
                    _yes(info.get("equal")),
                    sci(info.get("max_abs_diff")),
                    "bit a bit",
                ]
            )
    if "final_norm" in checks:
        c = checks["final_norm"]
        rows.append(
            [
                "bloco final bruto $\\neq$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("raw_block_differs_from_hidden")),
                sci(c.get("raw_block_vs_hidden_max_abs_diff")),
                "fora da tolerância",
            ]
        )
        rows.append(
            [
                "\\texttt{norm(bloco final)} $\\approx$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("norm_of_raw_block_close")),
                sci(c.get("norm_of_raw_block_max_abs_diff")),
                tol_text,
            ]
        )
        rows.append(
            [
                "\\textit{hook} em \\texttt{norm} $=$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("norm_hook_equal")),
                sci(c.get("norm_hook_max_abs_diff")),
                "bit a bit",
            ]
        )
    if "determinism" in checks:
        c = checks["determinism"]
        status = {
            "identical": "idênticas",
            "tiny": "quase idênticas",
            "different": "diferentes",
        }.get(c.get("status"), str(c.get("status")))
        rows.append(
            [
                f"duas execuções ({status})",
                _yes(c.get("passed")),
                sci(c.get("max_abs_diff")),
                tol_text + " (só aviso)",
            ]
        )
    if not rows:
        raise Skip("verify.json não tem checagens")
    model = report.get("model") or {}
    notes = [
        f"{integer(report.get('n_sequences'))} sequências, {integer(report.get('tokens'))} tokens, "
        f"{integer(report.get('n_layers'))} blocos, "
        f"\\texttt{{{tex_text(report.get('dtype', ''))}}}; "
        f"resultado geral: {_yes(report.get('passed'))}."
    ]
    if model.get("revision"):
        notes.append(
            f"Modelo \\texttt{{{tex_text(model.get('name_or_path', ''))}}}, revisão "
            f"\\texttt{{{tex_text(str(model['revision'])[:12])}}}."
        )
    header = [header_row(["Checagem", "Resultado", "Maior dif. absoluta", "Critério"])]
    return Tabular(
        r">{\raggedright\arraybackslash}X l l >{\raggedright\arraybackslash}p{0.25\linewidth}",
        header,
        rows,
        size=r"\footnotesize",
        notes=notes,
        width=r"\linewidth",
    )


# --------------------------------------------------------------------------------------------
# knn


def table_knn_empates(ctx: Context) -> Tabular:
    manifest = ctx.need_manifest("knn", "knn")
    reps = manifest.get("representations") or {}
    if not reps:
        raise Skip("o manifesto do knn não tem representações")
    n = _finite(manifest.get("n_occurrences"))
    rows: list[Any] = []
    for rep in _rep_order(reps):
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


# --------------------------------------------------------------------------------------------
# metrics


GLOBAL_ROWS = [
    # (label, column, kind, digits, source sym)
    ("$|V|$", "n_vertices", "int", 0, "union"),
    ("$|E|$", "n_edges", "int", 0, "union"),
    (r"Grau médio $\langle k\rangle$", "mean_degree", "num", 2, "union"),
    ("Densidade", "density", "sci", 2, "union"),
    ("Reciprocidade (direcionada)", "reciprocity", "num", 3, "directed"),
    ("Clusterização global", "transitivity", "num", 3, "union"),
    ("Clusterização local média", "avg_local_clustering", "num", 3, "union"),
    ("Componentes", "n_components", "int", 0, "union"),
    ("Maior componente (fração)", "largest_fraction", "pct", 1, "union"),
    ("Distância média", "mean_distance", "num", 2, "union"),
    ("Diâmetro", "diameter", "int", 0, "union"),
    ("Grau de entrada máximo", "in_degree_max", "int", 0, "directed"),
    ("Assimetria do grau de entrada", "in_degree_skewness", "num", 2, "directed"),
    ("Modularidade (Leiden)", "modularity", "num", 3, "union"),
    ("Comunidades", "n_communities", "int", 0, "union"),
]


def _metric_cell(part: pd.DataFrame, column: str, kind: str, digits: int) -> str:
    if column not in part:
        return DASH
    values = pd.to_numeric(part[column], errors="coerce").dropna()
    if values.empty:
        return DASH
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else math.nan
    if kind == "sci":
        return sci(mean, digits)
    if kind == "pct":
        text = pct(mean, digits)
        return text + (r"\,$\pm$\," + num(sd * 100, digits) if _finite(sd) and sd > 0 else "")
    if kind == "int":
        if values.size > 1 and sd > 0:
            return pm(mean, sd, 1, "num") if mean < 1000 else pm(mean, sd, 0, "int")
        return integer(mean)
    return pm(mean, sd, digits)


def table_metricas_globais(ctx: Context) -> Tabular:
    metrics = ctx.metrics_table("graph_metrics.csv")
    k = ctx.k
    columns: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = []
    for rep in ctx.settings.networks.representations:
        main = metrics[(metrics["rep"] == rep) & (metrics["k"] == k) & (metrics["tie"] == "rand")]
        if rep != LEXICAL:
            main = main[main["seed"] == 0]
        union, directed = main[main["sym"] == "union"], main[main["sym"] == "directed"]
        if union.empty and directed.empty:
            continue
        columns.append((r"\texttt{" + tex_text(rep) + "}", union, directed, ""))
    sampled_note = False
    try:
        vk = ctx.vocab_k
    except Skip:
        vk = None
    if vk is not None:
        main = metrics[
            (metrics["rep"] == "vocab")
            & (metrics["k"] == vk)
            & (metrics["tie"] == "rand")
            & (metrics["seed"] == 0)
        ]
        union, directed = main[main["sym"] == "union"], main[main["sym"] == "directed"]
        if not union.empty:
            method = str(union["distance_method"].iat[0]) if "distance_method" in union else ""
            mark = "$^{*}$" if method == "sampled" else ""
            sampled_note = bool(mark)
            columns.append((f"vocab. ($k={vk}$){mark}", union, directed, method))
    if not columns:
        raise Skip(f"graph_metrics.csv não tem as redes principais com k = {k}")
    rows = []
    for name, column, kind, digits, sym in GLOBAL_ROWS:
        cells = [name]
        for _, union, directed, method in columns:
            part = union if sym == "union" else directed
            cell = _metric_cell(part, column, kind, digits)
            if method == "sampled" and column == "diameter" and cell != DASH:
                cell = r"$\geq$\," + cell
            cells.append(cell)
        rows.append(cells)
    lex_seeds = 0
    if any(c[0].startswith(r"\texttt{lex}") for c in columns):
        lex_seeds = int(columns[0][1]["seed"].nunique())
    notes = [
        f"Redes com $k={k}$, versão por união, desempate por sorteio "
        f"(média $\\pm$ desvio entre as {lex_seeds} sementes na rede lexical, semente 0 nas "
        "demais); reciprocidade e grau de entrada vêm da versão direcionada."
    ]
    if sampled_note:
        notes.append(
            "$^{*}$ Distâncias estimadas por busca em largura a partir de fontes "
            "sorteadas; o diâmetro é um limite inferior."
        )
    header = [header_row(["Métrica", *[c[0] for c in columns]])]
    return Tabular("l" + "r" * len(columns), header, rows, size=r"\footnotesize", notes=notes)


def figure_graus_camadas(ctx: Context) -> Any:
    hist = ctx.metrics_table("degree_hist.csv")
    hist = hist[hist["kind"] == "in"]
    series = {}
    for rep in ctx.settings.networks.representations:
        part = hist[hist["graph_id"] == f"{rep}|k{ctx.k}|directed|rand|s0"]
        if not part.empty:
            series[rep] = (
                part["degree"].to_numpy(dtype=float),
                part["count"].to_numpy(dtype=float),
            )
    if not series:
        raise Skip(f"degree_hist.csv não tem os grafos direcionados com k = {ctx.k}")
    styles = plots.rep_styles(list(series))
    return plots.degree_ccdf_figure(
        series, ctx.k, styles, "Grau de entrada nas redes direcionadas (semente 0)"
    )


# --------------------------------------------------------------------------------------------
# fixed-layout drawings


def _layout(n: int, edges: np.ndarray) -> np.ndarray:
    """Spring layout of the union of every panel's edges (deterministic seed)."""

    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, edges.tolist()))
    positions = nx.spring_layout(graph, seed=LAYOUT_SEED, iterations=150)
    return np.array([positions[i] for i in range(n)], dtype=float)


def _ego_panels(
    ctx: Context, focal: np.ndarray, reps: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[plots.GraphPanel]]:
    """Vertex set (focal + out-neighbours in every rep), shared positions and one panel per rep."""

    sets = {rep: ctx.nbr(rep)["rand"][0][focal] for rep in reps}
    vertices = np.unique(np.concatenate([focal, *[s.ravel() for s in sets.values()]]))
    local = {int(v): i for i, v in enumerate(vertices)}
    panels = []
    all_edges = []
    for rep in reps:
        src = np.repeat(focal, sets[rep].shape[1])
        dst = sets[rep].ravel()
        edges = np.array(
            [[local[int(a)], local[int(b)]] for a, b in zip(src, dst, strict=True)], dtype=np.int64
        ).reshape(-1, 2)
        active = np.zeros(vertices.size, dtype=bool)
        active[edges.ravel()] = True
        panels.append(plots.GraphPanel(rep, edges, active))
        all_edges.append(edges)
    positions = _layout(vertices.size, np.concatenate(all_edges))
    return vertices, positions, panels


def _drawing_reps(ctx: Context) -> list[str]:
    reps = [r for r in ctx.main_reps if r in (LEXICAL, "L01", "L18", "L36")] or ctx.main_reps
    return reps[:4]


def figure_subgrafo_camadas(ctx: Context) -> Any:
    occ = ctx.occurrences
    reps = _drawing_reps(ctx)
    core = occ[occ["stratum"] == "core"]
    if core.empty:
        raise Skip("a amostra não tem núcleo")
    sizes = core.groupby("paragraph_id", sort=False).size()
    candidates = sizes[sizes >= SUBGRAPH_WINDOW]
    paragraph = candidates.index[0] if len(candidates) else sizes.idxmax()
    focal = core.index[core["paragraph_id"] == paragraph].to_numpy()[:SUBGRAPH_WINDOW]
    vertices, positions, panels = _ego_panels(ctx, focal, reps)
    token_ids = occ["token_id"].to_numpy(dtype=np.int64)[vertices]
    counts = pd.Series(token_ids).value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    top = [int(t) for t in counts.index[:SUBGRAPH_TOP_TYPES] if counts[t] > 1]
    key_of = {t: i for i, t in enumerate(top)}
    keys = np.array([key_of.get(int(t), -1) for t in token_ids])
    texts = occ.drop_duplicates("token_id").set_index("token_id")["token_text"]
    base = plots.categorical_styles([str(t) for t in top])
    styles = [
        plots.Style(base[str(t)].color, base[str(t)].marker, label=_visible(texts.get(t, "")))
        for t in top
    ]
    is_focal = np.isin(vertices, focal)
    title = occ.loc[focal[0], "title"] if "title" in occ else paragraph
    fig = plots.fixed_layout_figure(
        positions,
        panels,
        is_focal,
        styles,
        keys,
        keys,
        legend_title=f"Tipos de token mais frequentes no subgrafo (parágrafo de «{title}»)",
    )
    return fig


def _visible(text: Any) -> str:
    return str(text).replace(" ", "␣") or "∅"


def figure_p4_ego_banco(ctx: Context) -> Any:
    occ = ctx.occurrences
    reps = _drawing_reps(ctx)
    marked = occ[occ["target_word"] != ""]
    targets = [w for w in ctx.settings.sample.targets if (marked["target_word"] == w).any()]
    if not targets:
        targets = sorted(set(marked["target_word"]))
    if not targets:
        raise Skip("a amostra não tem ocorrências de palavras-alvo")
    word = (
        EGO_WORD
        if EGO_WORD in targets
        else max(targets, key=lambda w: (int((marked["target_word"] == w).sum()), w))
    )
    focal = marked.index[marked["target_word"] == word].to_numpy()
    vertices, positions, panels = _ego_panels(ctx, focal, reps)
    themes_all = list(ctx.settings.corpus.themes)
    sense = occ["sense_theme"].where(occ["sense_theme"] != "", occ["theme"]).to_numpy()
    present = ctx.theme_order(sense[focal])
    base = plots.categorical_styles(ordered(themes_all + present, themes_all))
    styles = [plots.Style(base[t].color, base[t].marker, label=theme_label(t)) for t in present]
    key_of = {t: i for i, t in enumerate(present)}
    keys = np.array([key_of.get(t, -1) for t in sense[vertices]])
    is_focal = np.isin(vertices, focal)
    title = f"Tema de sentido das ocorrências de «{word}»"
    if word != EGO_WORD:
        title += f" («{EGO_WORD}» não está na amostra)"
    return plots.fixed_layout_figure(
        positions, panels, is_focal, styles, keys, None, legend_title=title
    )


# --------------------------------------------------------------------------------------------
# P1


def _measures(ctx: Context) -> pd.DataFrame:
    measures = ctx.analysis("vertex_measures.csv")
    occ = ctx.occurrences
    if len(measures) != len(occ):
        raise Skip("vertex_measures.csv e occurrences.csv têm números de linhas diferentes")
    frame = measures.copy()
    for column in ("band_sample", "token_category", "stratum", "f_sample", "pos_bucket"):
        frame[column] = occ[column].to_numpy()
    return frame


def figure_p1_jaccard_faixas(ctx: Context) -> Any:
    frame = _measures(ctx)
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
    frame = _measures(ctx)
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
    frame = _measures(ctx)
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
    hubs = ctx.analysis("p1_hubs.csv")
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


# --------------------------------------------------------------------------------------------
# P2


def _transitions(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = ctx.analysis("p2_transitions.csv")
    groups = ctx.analysis("p2_transitions_groups.csv")
    for frame in (overall, groups):
        frame["value"] = frame["value"].astype(str)
        frame["consecutive"] = _bool(frame["consecutive"])
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
    layers = ctx.analysis("p2_layers.csv")
    layers["token_category"] = layers["token_category"].astype(str)
    categories = ordered([c for c in layers["token_category"] if c != "all"], CATEGORY_ORDER)
    return plots.layer_curves_figure(layers, categories)


# --------------------------------------------------------------------------------------------
# P3


def _p3_core(ctx: Context) -> pd.DataFrame:
    table = ctx.analysis("p3_agreement.csv")
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

    reps = [r for r in ctx.settings.networks.representations if (core["rep"] == r).any()]
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


# --------------------------------------------------------------------------------------------
# P4


P4_MEASURES = [
    ("self_similarity_mean", "Autossimilaridade (cosseno médio)"),
    ("theme_gap_mean", "Cosseno intra − entre temas"),
    ("effective_communities_mean", "Comunidades efetivas (exp H)"),
    ("nmi_community_theme_mean", "NMI comunidade × tema"),
]


def figure_p4_separacao(ctx: Context) -> Any:
    groups = ctx.analysis("p4_groups.csv")
    reps = [r for r in ctx.settings.networks.representations if (groups["rep"] == r).any()]
    keys = ordered([g for g in groups["group"].astype(str) if g in GROUP_ORDER], GROUP_ORDER)
    if not keys:
        raise Skip("p4_groups.csv não tem alvos, controles, multitema nem funcionais")
    measures = [(c, t) for c, t in P4_MEASURES if c in groups]
    return plots.separation_figure(groups, reps, measures, keys)


def table_p4_alvos(ctx: Context) -> Tabular:
    types = ctx.analysis("p4_types.csv")
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
    sense_path = ctx.paths.analysis_dir / "p4_senses.csv"
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
        ref = part.loc[LEXICAL] if LEXICAL in part.index else None
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


# --------------------------------------------------------------------------------------------
# robustness


def _graph_row(metrics: pd.DataFrame, rep: str, k: int, sym: str, tie: str) -> pd.Series | None:
    hit = metrics[metrics["graph_id"] == f"{rep}|k{k}|{sym}|{tie}|s0"]
    return None if hit.empty else hit.iloc[0]


def table_robustez(ctx: Context) -> Tabular:
    metrics = ctx.metrics_table("graph_metrics.csv")
    net = ctx.settings.networks
    k = ctx.k
    neighbors = pd.DataFrame()
    path = ctx.paths.analysis_dir / "robustness_neighbors.csv"
    if path.exists():
        neighbors = ctx.table(path)
    index = pd.DataFrame()
    index_path = ctx.paths.metrics_dir / "communities" / "index.csv"
    if index_path.exists():
        index = ctx.table(index_path)
    reps = [
        r for r in ctx.contextual_reps if _graph_row(metrics, r, k, "union", "rand") is not None
    ]
    if not reps:
        raise Skip(f"graph_metrics.csv não tem as redes contextuais com k = {k}")

    def j_vs_lex(rep: str, kk: int, tie: str) -> Any:
        if neighbors.empty:
            return np.nan
        hit = neighbors[
            (neighbors["rep"] == rep)
            & (neighbors["k"] == kk)
            & (neighbors["tie"] == tie)
            & np.isclose(pd.to_numeric(neighbors["eps"]), net.eps)
        ]
        return hit["J_vs_lex_mean"].iat[0] if len(hit) else np.nan

    def graph_cells(rep: str | None, kk: int, sym: str, tie: str, j: Any) -> list[str]:
        if rep is None:
            return [DASH] * 4
        row = _graph_row(metrics, rep, kk, sym, tie)
        if row is None:
            return [DASH] * 4
        return [
            num(row.get("avg_local_clustering"), 3),
            num(row.get("mean_distance"), 2),
            num(row.get("modularity"), 3),
            num(j, 3),
        ]

    variations: list[tuple[str, Callable[[str], list[str]]]] = [
        (
            f"Principal ($k={k}$, união, sorteio)",
            lambda r: graph_cells(r, k, "union", "rand", j_vs_lex(r, k, "rand")),
        ),
    ]
    for kk in net.k_values:
        if kk != k:
            variations.append(
                (
                    f"$k={kk}$",
                    lambda r, kk=kk: graph_cells(r, kk, "union", "rand", j_vs_lex(r, kk, "rand")),
                )
            )
    variations += [
        ("Versão mútua", lambda r: graph_cells(r, k, "mutual", "rand", np.nan)),
        (
            "Desempate por posição",
            lambda r: graph_cells(r, k, "union", "pos", j_vs_lex(r, k, "pos")),
        ),
        (
            "(b) todos os empatados",
            lambda r: graph_cells(r, k, "union", "all", j_vs_lex(r, k, "all")),
        ),
        (
            f"$\\varepsilon$ = {sci(net.eps_sensitivity)}",
            lambda r: graph_cells(r, k, "union", "rand_eps", np.nan),
        ),
    ]
    robust = [r for r in net.robust_representations]
    for variant in robust:
        base = variant[:-1] if variant.endswith("n") else None
        variations.append(
            (
                f"\\texttt{{{tex_text(variant)}}} (RMSNorm final)",
                lambda r, v=variant, b=base: graph_cells(
                    v if r == b else None, k, "union", "rand", j_vs_lex(v, k, "rand")
                ),
            )
        )
    if net.centered:
        variations.append(
            (
                "Cosseno centrado",
                lambda r: graph_cells(
                    f"{r}c" if r in ctx.settings.model.layers else None,
                    k,
                    "union",
                    "rand",
                    j_vs_lex(f"{r}c", k, "rand"),
                ),
            )
        )
    rows: list[Any] = []
    for name, build in variations:
        cells = [name]
        for rep in reps:
            cells += build(rep)
        if all(c == DASH for c in cells[1:]):
            continue
        rows.append(cells)
    if not index.empty:
        community_rows = []
        for resolution in ctx.settings.analysis.resolution_sweep:
            if math.isclose(resolution, ctx.settings.analysis.resolution):
                continue
            community_rows.append((f"Leiden $\\gamma={num(resolution, 1)}$", "leiden", resolution))
        community_rows.append(("Louvain (checagem)", "louvain", ctx.settings.analysis.resolution))
        for name, method, resolution in community_rows:
            cells = [name]
            found = False
            for rep in reps:
                hit = index[
                    (index["graph_id"] == f"{rep}|k{k}|union|rand|s0")
                    & (index["method"] == method)
                    & np.isclose(pd.to_numeric(index["resolution"]), resolution)
                ]
                q = hit["modularity"].iat[0] if len(hit) else np.nan
                found |= len(hit) > 0
                cells += [DASH, DASH, num(q, 3), DASH]
            if found:
                rows.append(cells)
    header = [
        header_row([""] + [rf"\multicolumn{{4}}{{c}}{{\texttt{{{tex_text(r)}}}}}" for r in reps]),
        "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(reps))),
        header_row(["Variação"] + [r"$\bar C$", r"$\bar\ell$", "$Q$", "$J$"] * len(reps)),
    ]
    notes = [
        r"$\bar C$: clusterização local média; $\bar\ell$: distância média; $Q$: modularidade "
        r"do Leiden (só nos grafos com comunidades); $J$: Jaccard médio com a rede \texttt{lex} "
        "de mesmo $k$ e mesmo tratamento de empates. Cada linha muda uma dimensão em relação à "
        r"principal; nas linhas \texttt{L36n} e centrada, cada coluna usa a variante da camada."
    ]
    return Tabular(
        "l" + "rrrr" * len(reps), header, rows, size=r"\footnotesize", notes=notes, colsep="3.5pt"
    )


def figure_robustez_empates(ctx: Context) -> Any:
    occ = ctx.occurrences
    lex = ctx.nbr(LEXICAL)
    reps = [r for r in ctx.contextual_reps if ctx.paths.nbr(r, ctx.k).exists()]
    if not reps:
        raise Skip("faltam os vizinhos das camadas contextuais")
    measures = None
    try:
        measures = _measures(ctx)
    except Skip:
        pass
    bands = ordered([b for b in occ["band_sample"] if b], band_order(ctx.settings))
    band = occ["band_sample"].to_numpy()
    variants = {
        "a": "(a) sorteio",
        "pos": "posição",
        "b": "(b) todos os empatados",
        "c": "(c) tipos distintos",
        "w": "Jʷ (composição de tipos)",
    }
    rows = []
    for rep in reps:
        data = ctx.nbr(rep)
        per_vertex = {
            "pos": jaccard(lex["pos"], data["pos"]),
            "b": jaccard(
                (lex["all_indptr"], lex["all_idx"]), (data["all_indptr"], data["all_idx"])
            ),
        }
        if measures is not None:
            for key, prefix in (("a", "J"), ("c", "Jc"), ("w", "Jw")):
                column = f"{prefix}_lex_{rep}"
                if column in measures:
                    per_vertex[key] = measures[column].to_numpy(dtype=float)
        for key, values in per_vertex.items():
            for b in bands:
                mask = band == b
                rows.append(
                    {
                        "rep": rep,
                        "variant": key,
                        "band": b,
                        "value": float(np.nanmean(values[mask])) if mask.any() else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    keys = [k for k in variants if (frame["variant"] == k).any()]
    return plots.tie_variants_figure(frame, reps, bands, keys, variants)


# --------------------------------------------------------------------------------------------
# vocabulary network


def _fold_classes(classes: np.ndarray, keep: int = 5) -> tuple[np.ndarray, list[str]]:
    """Script classes with the ``keep`` largest kept and the rest folded into ``other``."""

    counts = pd.Series(classes).value_counts()
    top = [c for c in counts.index[:keep]]
    order = [c for c in SCRIPT_ORDER if c in top] + sorted(c for c in top if c not in SCRIPT_ORDER)
    folded = np.where(np.isin(classes, order), classes, "other")
    if (folded == "other").any():
        order.append("other")
    return folded, order


def figure_vocab_graus(ctx: Context) -> Any:
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
    index = ctx.metrics_table("communities/index.csv")
    gid = f"vocab|k{ctx.vocab_k}|union|rand|s0"
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
    communities = ctx.analysis("vocab_communities.csv")
    communities = communities.sort_values("size", ascending=False, kind="stable").head(10)
    total = None
    try:
        membership, _ = _vocab_partition(ctx)
        total = membership.size
    except Skip:
        pass
    rows = []
    for rank, (_, row) in enumerate(communities.iterrows(), start=1):
        examples = split_reprs(row.get("examples", ""))[:6]
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
    summary_path = ctx.paths.analysis_dir / "vocab_summary.csv"
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
    data = ctx.npz(ctx.paths.vocab_nbr(ctx.vocab_k))
    vocab_ids = data["vocab_ids"].astype(np.int64)
    row_of = {int(t): i for i, t in enumerate(vocab_ids)}
    occ = ctx.occurrences
    marked = occ[occ["target_word"] != ""]
    word_ids = marked.groupby("target_word")["token_id"].first().to_dict()
    f_corpus = ctx.vocab_types["f_corpus"].to_numpy(dtype=np.int64)
    rows: list[Any] = []
    sample = ctx.settings.sample
    for role, words in (("alvo", sample.targets), ("controle", sample.controls)):
        part = []
        for word in words:
            token_id = word_ids.get(word)
            if token_id is None or int(token_id) not in row_of:
                continue
            neighbors = vocab_ids[data["pos"][row_of[int(token_id)]]][:10]
            part.append(
                [
                    tex_text(word),
                    role,
                    r"\enspace ".join(tex_token(ctx.vocab_text(t)) for t in neighbors),
                    integer(int(np.sum(f_corpus[neighbors] > 0))),
                ]
            )
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


# --------------------------------------------------------------------------------------------
# lens


def figure_lente_ranking(ctx: Context) -> Any:
    curve_path = ctx.paths.lens_dir / "layer_curve.csv"
    if curve_path.exists() and not ctx.table(curve_path).empty:
        curve = ctx.table(curve_path).copy()
        curve = curve[curve["group_by"].isin(["all", "token_category"])]
        curve = curve.assign(
            x=curve["layer"].astype(float),
            xlabel=curve["rep"].astype(str),
            group=curve["group"].astype(str),
        )
        discrete = False
    else:
        summary = ctx.json(ctx.paths.lens_dir / "summary.json")
        ranks = summary.get("ranks") or {}
        if not ranks:
            raise Skip("lens/summary.json não tem postos")
        rows = []
        for x, rep in enumerate(summary.get("representations") or list(ranks)):
            entry = ranks.get(rep) or {}
            groups = {"all": entry.get("all") or {}, **(entry.get("by_token_category") or {})}
            for group, stats in groups.items():
                rows.append(
                    {
                        "x": float(x),
                        "xlabel": rep,
                        "group": group,
                        "own_rank_median": stats.get("own_rank_median"),
                        "next_rank_median": stats.get("next_rank_median"),
                    }
                )
        curve = pd.DataFrame(rows)
        discrete = True
    categories = ordered([g for g in curve["group"] if g != "all"], CATEGORY_ORDER)
    return plots.lens_ranks_figure(curve, categories, discrete)


# --------------------------------------------------------------------------------------------
# versions


def _when(manifest: Mapping[str, Any]) -> str:
    stamp = _finite(manifest.get("started_unix"))
    if stamp is None:
        return DASH
    finished = stamp + (_finite(manifest.get("elapsed_s")) or 0.0)
    return datetime.fromtimestamp(finished).strftime("%d/%m/%Y %H:%M")


def table_versoes(ctx: Context) -> Tabular:
    manifests = stage_manifests(ctx)
    if not manifests:
        raise Skip("nenhum _manifest.json de etapa")
    rows: list[Any] = []
    for stage, manifest in manifests.items():
        rows.append(
            [
                r"\texttt{" + stage + "}",
                _when(manifest),
                duration(manifest.get("elapsed_s")),
                r"\texttt{" + tex_text(manifest.get("git") or DASH) + "}",
                r"\texttt{" + tex_text(Path(str(manifest.get("config_source") or DASH)).name) + "}",
            ]
        )
    versions: dict[str, list[str]] = {}
    for manifest in manifests.values():
        for name, version in (manifest.get("versions") or {}).items():
            versions.setdefault(name, [])
            if str(version) not in versions[name]:
                versions[name].append(str(version))
    width = 5
    rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{Bibliotecas e modelo}}}} \\")
    for name, values in versions.items():
        rows.append(
            [tex_text(name), r"\multicolumn{4}{l}{\texttt{" + tex_text(" / ".join(values)) + "}}"]
        )
    model = ctx.settings.model
    revision = model.revision
    verify = ctx.paths.verify_report
    devices: dict[str, Any] = {}
    if verify.exists():
        report = ctx.json(verify)
        revision = (report.get("model") or {}).get("revision") or revision
        devices = report.get("devices") or {}
    rows.append(
        [
            "modelo",
            rf"\multicolumn{{4}}{{l}}{{\texttt{{{tex_text(model.name_or_path)}}} "
            rf"(\texttt{{{tex_text(model.dtype)}}})}}",
        ]
    )
    rows.append(["revisão", rf"\multicolumn{{4}}{{l}}{{\texttt{{{tex_text(revision)}}}}}"])
    if devices.get("cuda_device"):
        rows.append(["GPU", rf"\multicolumn{{4}}{{l}}{{{tex_text(devices['cuda_device'])}}}"])
    header = [
        header_row(["Etapa", "Concluída em", "Tempo", "Revisão \\texttt{git}", "Configuração"])
    ]
    return Tabular("lllll", header, rows, size=r"\footnotesize")


# --------------------------------------------------------------------------------------------
# stage


@dataclass(frozen=True)
class Item:
    kind: str  # figure | table
    name: str
    build: Callable[[Context], Any]


ITEMS = [
    Item("table", "corpus_temas", table_corpus_temas),
    Item("figure", "corpus_paragrafos", figure_corpus_paragrafos),
    Item("figure", "corpus_descartes", figure_corpus_descartes),
    Item("table", "amostra_composicao", table_amostra_composicao),
    Item("figure", "amostra_composicao", figure_amostra_composicao),
    Item("table", "amostra_faixas", table_amostra_faixas),
    Item("table", "amostra_cotas", table_amostra_cotas),
    Item("table", "extracao_diagnosticos", table_extracao_diagnosticos),
    Item("table", "extracao_verificacao", table_extracao_verificacao),
    Item("figure", "extracao_camadas", figure_extracao_camadas),
    Item("figure", "extracao_dimensoes", figure_extracao_dimensoes),
    Item("table", "knn_empates", table_knn_empates),
    Item("table", "metricas_globais", table_metricas_globais),
    Item("figure", "graus_camadas", figure_graus_camadas),
    Item("figure", "subgrafo_camadas", figure_subgrafo_camadas),
    Item("figure", "p1_jaccard_faixas", figure_p1_jaccard_faixas),
    Item("figure", "p1_dominancia", figure_p1_dominancia),
    Item("table", "p1_resumo", table_p1_resumo),
    Item("table", "p1_hubs", table_p1_hubs),
    Item("figure", "p2_transicoes", figure_p2_transicoes),
    Item("table", "p2_transicoes", table_p2_transicoes),
    Item("figure", "p2_curvas_camadas", figure_p2_curvas_camadas),
    Item("figure", "p3_nmi_camadas", figure_p3_nmi_camadas),
    Item("table", "p3_concordancia", table_p3_concordancia),
    Item("figure", "p4_separacao", figure_p4_separacao),
    Item("table", "p4_alvos", table_p4_alvos),
    Item("figure", "p4_ego_banco", figure_p4_ego_banco),
    Item("table", "robustez", table_robustez),
    Item("figure", "robustez_empates", figure_robustez_empates),
    Item("figure", "vocab_graus", figure_vocab_graus),
    Item("figure", "vocab_comunidades", figure_vocab_comunidades),
    Item("table", "vocab_comunidades", table_vocab_comunidades),
    Item("table", "vocab_vizinhos_alvos", table_vocab_vizinhos_alvos),
    Item("figure", "lente_ranking", figure_lente_ranking),
    Item("table", "versoes", table_versoes),
]


def output_path(paths: RunPaths, item: Item) -> Path:
    if item.kind == "figure":
        return paths.figures_dir / f"{item.name}.pdf"
    return paths.tables_dir / f"{item.name}.tex"


def produce(ctx: Context, item: Item) -> Path:
    """Build one item and write it; raises :class:`Skip` when its inputs are missing."""

    path = output_path(ctx.paths, item)
    result = item.build(ctx)
    ensure_dir(path.parent)
    if item.kind == "figure":
        plots.save(result, path)
    else:
        path.write_text(result.render(), encoding="utf-8")
    return path


def compile_report(report_dir: Path) -> dict[str, Any]:
    """``latexmk -pdf relatorio.tex`` when both exist; never raises."""

    latexmk = shutil.which("latexmk")
    if latexmk is None or not (report_dir / "relatorio.tex").exists():
        return {"compiled": False, "reason": "latexmk ou relatorio.tex ausente"}
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "relatorio.tex"],
            cwd=report_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"compiled": False, "reason": str(error)}
    ok = result.returncode == 0
    if not ok:
        LOGGER.warning(
            "latexmk falhou (código %d); veja %s", result.returncode, report_dir / "relatorio.log"
        )
    return {
        "compiled": ok,
        "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 1),
    }


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    """Write every figure and table whose inputs exist, plus numeros.tex (always regenerated)."""

    del force  # the stage is cheap and always regenerates everything
    started = time.time()
    plots.apply_style()
    ctx = Context(settings, paths)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.tables_dir)
    status: dict[str, dict[str, str]] = {"figure": {}, "table": {}}
    failed: list[str] = []
    for item in ITEMS:
        key = f"{item.kind} {item.name}"
        try:
            produce(ctx, item)
        except Skip as reason:
            status[item.kind][item.name] = f"pulada: {reason}"
            stale = output_path(paths, item)
            if stale.exists():
                stale.unlink()
                LOGGER.info("%s antiga removida (%s)", stale, reason)
            LOGGER.info("Pulada %s: %s", key, reason)
        except Exception as error:  # noqa: BLE001 - one broken item must not stop the others
            LOGGER.exception("Falhou %s", key)
            status[item.kind][item.name] = f"erro: {type(error).__name__}: {error}"
            failed.append(key)
            output_path(paths, item).unlink(missing_ok=True)
        else:
            status[item.kind][item.name] = "gerada"
    numbers = write_numbers(ctx, paths.tables_dir / "numeros.tex")
    compiled = compile_report(paths.report_dir)
    generated = sum(v == "gerada" for kind in status.values() for v in kind.values())
    LOGGER.info(
        "Relatório: %d de %d itens gerados, %d macros em numeros.tex (%s)",
        generated,
        len(ITEMS),
        len(numbers),
        paths.report_dir,
    )
    manifest_inputs = sorted(
        {path for path in ctx.read if path.name == "_manifest.json"}
        | {p for p in (paths.manifest(d) for _, d in STAGE_MANIFESTS if d) if p.exists()}
    )
    write_manifest(
        paths.stage_dir(STAGE),
        STAGE,
        settings,
        started,
        extra={
            "report_dir": str(paths.report_dir),
            "figures": status["figure"],
            "tables": status["table"],
            "macros": sorted(numbers),
            "missing_macros": [m for m in MACROS if m not in numbers],
            "latex": compiled,
            "failed": failed,
            "files_read": sorted(str(p) for p in ctx.read),
        },
        root=paths.root,
        inputs=manifest_inputs,
    )
    if failed:
        raise RuntimeError(f"Itens do relatório com erro: {', '.join(failed)}")
