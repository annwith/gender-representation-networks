"""Artifact paths, formats and small I/O helpers shared by every stage.

This module is the contract between stages: each stage reads only what an earlier stage wrote
through the paths below, so any stage can be rerun on its own.

Shared across runs (the corpus is downloaded once):

- ``raw_dir/``: cached Wikipedia API responses (gzip JSON, keyed by request hash).
- ``corpus_dir/articles.jsonl``: one record per candidate article
  (``pageid, revid, title, theme, themes_reached, source_category, depth, origin, is_biography,
  dropped_reason``; ``dropped_reason`` is null for kept articles).
- ``corpus_dir/paragraphs.jsonl``: one record per kept paragraph
  (``paragraph_id, pageid, revid, title, theme, source_category, paragraph_idx, text,
  sentences``), where ``sentences`` is a list of ``[start, end)`` character spans.
- ``corpus_dir/manifest.csv``: versioned list of articles (pageid, revid, title, theme,
  source_category, kept, dropped_reason) so the same revisions can be downloaded again.

Per run (``runs_dir/<name>/``):

- ``sample/occurrences.csv``: one row per vertex, ``occurrence_id`` equal to the row index in
  every representation file (see :data:`OCCURRENCE_COLUMNS`).
- ``sample/sequences.jsonl``: ``sequence_id, paragraph_id, input_ids`` (prefix token at
  position 0, truncated after the last selected position).
- ``sample/vocab_types.csv``: one row per vocabulary id (see :data:`VOCAB_COLUMNS`).
- ``sample/senses.csv``: template for manual sense annotation of target occurrences.
- ``reps/<rep>.safetensors``: tensor ``x`` with shape ``[N, d]`` (bf16) for each contextual
  representation (``L01``, ``L18``, ``L36``, ``L36n``) in ``occurrence_id`` order.
- ``reps/embeddings.safetensors``: tensor ``weight`` with the input embedding rows of every
  tokenizer id ``[V, d]`` (bf16). Lexical occurrence vectors are never materialized.
- ``reps/all_layers.npy``: uint16 view of bf16 block outputs, shape ``[L, N, d]``
  (layer ``l`` at index ``l - 1``).
- ``knn/``: candidate lists and neighbor sets (``.npz``).
- ``metrics/``, ``analysis/``: CSV/JSON tables.
- every stage directory holds a ``_manifest.json`` (see :func:`write_manifest`).

Generated report material goes to ``report_dir/figuras/`` and ``report_dir/tabelas/``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gender_networks.settings import Settings

OCCURRENCE_COLUMNS = [
    "occurrence_id",
    "stratum",  # core | target | control | multitheme
    "theme",
    "source_category",
    "pageid",
    "revid",
    "title",
    "paragraph_id",
    "paragraph_idx",
    "sentence_id",
    "sequence_id",
    "pos_in_sequence",  # >= 1; position 0 is the prefix token
    "pos_bucket",  # 1-4 | 5-16 | 17-64 | 65+
    "prefix_group",  # -1 when the token prefix is unique
    "token_id",
    "token_text",  # exact text slice of the token in the paragraph
    "char_start",
    "char_end",
    "has_leading_space",
    "prev_token_id",
    "next_token_id",  # -1 when the token is the last one of the paragraph
    "word",
    "word_idx",
    "word_n_tokens",
    "pos_in_word",
    "token_category",  # whole_word | word_start | continuation | punctuation | number
    "is_function_word",
    "f_sample",
    "band_sample",
    "f_corpus",
    "band_corpus",
    "target_word",  # target/control word when the id occurs as that whole word, any stratum
    "sense_theme",  # paragraph theme of the rows with a target_word
    "local_context",  # about ±8 tokens, the token marked with «…»
]

VOCAB_COLUMNS = [
    "token_id",
    "token_repr",  # repr() of the decoded token text
    "script_class",
    "is_special",
    "f_corpus",
    "f_sample",
    "in_corpus",
]

POS_BUCKETS = [(1, 4, "1-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, None, "65+")]


def candidates_name(rep: str) -> str:
    return f"cand_{rep}.npz"


def neighbors_name(rep: str, k: int, sensitivity: bool = False) -> str:
    """``nbr_{rep}_k{k}.npz``; ``_eps`` files hold the eps-sensitivity neighbour sets."""

    return f"nbr_{rep}_k{k}{'_eps' if sensitivity else ''}.npz"


def vocab_neighbors_name(k: int) -> str:
    return f"vocab_k{k}.npz"


def position_bucket(position: int) -> str:
    """Label of the position bucket for a sequence position (1-based after the prefix)."""

    for low, high, label in POS_BUCKETS:
        if position >= low and (high is None or position <= high):
            return label
    raise ValueError(f"Position {position} is not a vertex position (must be >= 1)")


@dataclass(frozen=True)
class RunPaths:
    """All artifact locations of one run."""

    root: Path
    raw_dir: Path
    corpus_dir: Path
    run_dir: Path
    report_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings, root: str | Path = ".") -> RunPaths:
        base = Path(root)
        return cls(
            root=base,
            raw_dir=base / settings.paths.raw_dir,
            corpus_dir=base / settings.paths.corpus_dir,
            run_dir=base / settings.paths.runs_dir / settings.name,
            report_dir=base / settings.paths.report_dir,
        )

    # corpus (shared)
    @property
    def corpus_manifest(self) -> Path:
        return self.corpus_dir / "_manifest.json"

    def manifest(self, stage: str) -> Path:
        """``_manifest.json`` of a per-run stage directory (sample, reps, knn, metrics, ...)."""

        return self.stage_dir(stage) / "_manifest.json"

    @property
    def articles(self) -> Path:
        return self.corpus_dir / "articles.jsonl"

    @property
    def paragraphs(self) -> Path:
        return self.corpus_dir / "paragraphs.jsonl"

    @property
    def corpus_manifest_csv(self) -> Path:
        return self.corpus_dir / "manifest.csv"

    # per-run stage directories
    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage

    @property
    def sample_dir(self) -> Path:
        return self.stage_dir("sample")

    @property
    def occurrences(self) -> Path:
        return self.sample_dir / "occurrences.csv"

    @property
    def sequences(self) -> Path:
        return self.sample_dir / "sequences.jsonl"

    @property
    def vocab_types(self) -> Path:
        return self.sample_dir / "vocab_types.csv"

    @property
    def senses(self) -> Path:
        return self.sample_dir / "senses.csv"

    @property
    def reps_dir(self) -> Path:
        return self.stage_dir("reps")

    def rep(self, name: str) -> Path:
        return self.reps_dir / f"{name}.safetensors"

    @property
    def embeddings(self) -> Path:
        return self.reps_dir / "embeddings.safetensors"

    @property
    def all_layers(self) -> Path:
        return self.reps_dir / "all_layers.npy"

    @property
    def pred_next(self) -> Path:
        return self.reps_dir / "pred_next.npy"

    @property
    def diagnostics(self) -> Path:
        return self.reps_dir / "diagnostics.json"

    @property
    def verify_report(self) -> Path:
        return self.reps_dir / "verify.json"

    @property
    def knn_dir(self) -> Path:
        return self.stage_dir("knn")

    def cand(self, rep: str) -> Path:
        return self.knn_dir / candidates_name(rep)

    def nbr(self, rep: str, k: int, sensitivity: bool = False) -> Path:
        return self.knn_dir / neighbors_name(rep, k, sensitivity)

    @property
    def vocab_cand(self) -> Path:
        return self.knn_dir / "vocab_cand.npz"

    def vocab_nbr(self, k: int) -> Path:
        return self.knn_dir / vocab_neighbors_name(k)

    @property
    def metrics_dir(self) -> Path:
        return self.stage_dir("metrics")

    @property
    def analysis_dir(self) -> Path:
        return self.stage_dir("analysis")

    @property
    def lens_dir(self) -> Path:
        return self.stage_dir("lens")

    # generated report material
    @property
    def figures_dir(self) -> Path:
        return self.report_dir / "figuras"

    @property
    def tables_dir(self) -> Path:
        return self.report_dir / "tabelas"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_gz(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)


def _git_describe(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "torch", "transformers", "igraph", "networkx", "pandas"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def file_fingerprint(paths: Iterable[Path]) -> dict[str, list[int] | None]:
    """Size and modification time of each input (None when missing), keyed by path."""

    out: dict[str, list[int] | None] = {}
    for path in paths:
        try:
            stat = Path(path).stat()
        except FileNotFoundError:
            out[str(path)] = None
            continue
        out[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return out


def _string_keys(value: Any) -> Any:
    """Copy with every mapping key as a string (``max_memory`` mixes ``0`` and ``"cpu"``)."""

    if isinstance(value, dict):
        return {str(key): _string_keys(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_string_keys(item) for item in value]
    return value


def settings_digest(settings: Settings) -> str:
    """Hash of the run configuration (the source file path does not count)."""

    payload = _string_keys(settings.to_dict())
    payload.pop("source", None)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_is_fresh(
    stage_dir: Path, settings: Settings, inputs: Iterable[Path], outputs: Iterable[Path]
) -> bool:
    """True when the stage's outputs exist and were built from the same inputs and settings.

    Stages skip only when this holds, so rerunning an upstream stage (for example ``sample
    --force``) makes every downstream stage recompute instead of reusing stale artifacts.
    """

    manifest = stage_dir / "_manifest.json"
    if not manifest.exists() or not all(Path(path).exists() for path in outputs):
        return False
    try:
        recorded = read_json(manifest)
    except (OSError, json.JSONDecodeError):
        return False
    return recorded.get("settings_digest") == settings_digest(settings) and recorded.get(
        "inputs"
    ) == file_fingerprint(inputs)


def write_manifest(
    stage_dir: Path,
    stage: str,
    settings: Settings,
    started: float,
    extra: dict[str, Any] | None = None,
    root: Path = Path("."),
    inputs: Iterable[Path] | None = None,
) -> Path:
    """Record what produced a stage's artifacts: config, inputs, versions, timings and facts.

    ``inputs`` are the upstream files the stage read; their fingerprint lets
    :func:`stage_is_fresh` notice when an upstream stage was rerun.
    """

    path = stage_dir / "_manifest.json"
    write_json(
        path,
        {
            "stage": stage,
            "run": settings.name,
            "config_source": settings.source,
            "settings_digest": settings_digest(settings),
            "inputs": file_fingerprint(inputs or []),
            "settings": settings.to_dict(),
            "git": _git_describe(root),
            "versions": library_versions(),
            "started_unix": started,
            "elapsed_s": round(time.time() - started, 3),
            **(extra or {}),
        },
    )
    return path
