"""Artifact paths, formats and small I/O helpers shared by every stage.

This module is the contract between stages: each stage reads only what an earlier stage wrote
through the paths below, so any stage can be rerun on its own.

Shared across runs (the corpus is downloaded once):

- ``raw_dir/``: cached Wikipedia API responses (gzip JSON, keyed by request hash).
- ``corpus_dir/articles.jsonl``: one record per candidate article
  (``pageid, revid, title, theme, themes_reached, source_category, depth, origin, is_biography,
  dropped_reason``; ``dropped_reason`` is null for kept articles).
- ``corpus_dir/paragraphs.jsonl``: one record per kept paragraph
  (``paragraph_id, pageid, revid, title, theme, source_category, origin, paragraph_idx, text,
  sentences``), where ``sentences`` is a list of ``[start, end)`` character spans and
  ``origin`` is ``root`` or the extra category whose walk reached the article.
- ``corpus_dir/manifest.csv``: versioned list of articles (pageid, revid, title, theme,
  source_category, origin, kept, dropped_reason) so the same revisions can be downloaded again.
- ``data/annotations/senses.csv``: sense labels of target occurrences, keyed by
  ``(paragraph_id, char_start)`` so they hold for every run that draws the same paragraph
  (see :data:`SENSE_ANNOTATION_COLUMNS`).

Per run (``runs_dir/<name>/``):

- ``sample/occurrences.csv``: one row per vertex, ``occurrence_id`` equal to the row index in
  every representation file (see :data:`OCCURRENCE_COLUMNS`).
- ``sample/sequences.jsonl``: ``sequence_id, paragraph_id, input_ids`` (prefix token at
  position 0, truncated after the last selected position).
- ``sample/vocab_types.csv``: one row per vocabulary id (see :data:`VOCAB_COLUMNS`).
- ``sample/senses.csv``: template of target occurrences to annotate; the labels themselves go
  to ``data/annotations/senses.csv``.
- ``reps/<rep>.safetensors``: tensor ``x`` with shape ``[N, d]`` (bf16) for each contextual
  representation (``L01``, ``L18``, ``L36``, ``L36n``) in ``occurrence_id`` order.
- ``reps/embeddings.safetensors``: tensor ``weight`` with the input embedding rows of every
  tokenizer id ``[V, d]`` (bf16). Lexical occurrence vectors are never materialized.
- ``reps/all_layers.npy``: uint16 view of bf16 block outputs, shape ``[L, N, d]``
  (layer ``l`` at index ``l - 1``).
- ``knn/``: candidate lists and neighbor sets (``.npz``).
- ``metrics/``: the tables of :data:`METRICS_TABLES` and ``communities/`` (one partition per
  :func:`community_name`, listed in ``communities/index.csv``).
- ``analysis/``: the tables of :data:`ANALYSIS_TABLES`.
- ``lens/``: the files of :data:`LENS_FILES` and ``topk_<rep>.npy``.
- every stage directory holds a ``_manifest.json`` (see :func:`write_manifest`).

Graphs are named by :func:`graph_id`. File names are built here and nowhere else: the
``RunPaths`` methods refuse names that are not listed, so a typo fails instead of making a
report item silently "pending".

Generated report material goes to ``report_dir/figuras/`` and ``report_dir/tabelas/``.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import logging
import platform
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gender_networks.settings import Settings

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)

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
    "token_repr",  # repr() of the decoded token text (see token_text)
    "script_class",
    "is_special",
    "f_corpus",
    "f_sample",
    "in_corpus",
]

SENSE_ANNOTATION_COLUMNS = [
    "paragraph_id",
    "char_start",  # with paragraph_id, the occurrence (as in occurrences.csv)
    "target_word",
    "title",  # title and local_context only help the reader
    "local_context",
    "sense",  # blank = not annotated
    "left_determinable",  # true when the paragraph up to the word decides the sense
    "annotator",
]

# Position buckets as (low, high, label), both ends inclusive; ``None`` leaves the last one open.
# Positions count tokens in the sequence, and position 0 (the prefix token) is in no bucket.
POS_BUCKETS = [(1, 4, "1-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, None, "65+")]

# Name of the file each stage writes last; its presence marks the stage as complete.
MANIFEST = "_manifest.json"
# Stage (CLI name) -> its directory inside the run, in pipeline order (corpus is shared).
STAGE_DIRS = {
    "sample": "sample",
    "extract": "reps",
    "knn": "knn",
    "metrics": "metrics",
    "analyze": "analysis",
    "lens": "lens",
    "report": "report",
}
# Declared outputs of the metrics, analyze and lens stages. ``RunPaths`` refuses any other name,
# and the stages use these tuples as their list of expected outputs.
METRICS_TABLES = ("graph_metrics", "degree_hist", "components", "distance_hist", "hubs")
ANALYSIS_TABLES = (
    "vertex_measures",
    "p1_summary",
    "p1_global",
    "p1_hubs",
    "p2_transitions",
    "p2_transitions_groups",
    "p2_layers",
    "p3_agreement",
    "p3_layer_nmi",
    "p4_types",
    "p4_groups",
    "p4_senses",
    "p4_sense_communities",
    "robustness_neighbors",
    "robustness_graphs",
    "core_metrics",
    "vocab_summary",
    "vocab_communities",
    "vocab_target_neighbors",
)
LENS_FILES = ("ranks.csv", "topk_jaccard.csv", "summary.json", "layer_curve.csv")


def graph_id(rep: str, k: int, sym: str, tie: str, seed: int) -> str:
    """Identifier ``rep|k{k}|{sym}|{tie}|s{seed}`` of a graph specification.

    Args:
        rep: Representation name (``lex``, ``L18``...) or ``vocab``.
        k: Number of neighbours.
        sym: Symmetrization: ``directed``, ``union`` or ``mutual``.
        tie: Tie treatment code, as written by the ``metrics`` stage.
        seed: Tie-break seed index.

    Returns:
        The identifier used as the key of every metrics and analysis table.
    """

    return f"{rep}|k{k}|{sym}|{tie}|s{seed}"


def file_safe(gid: str) -> str:
    """File-name version of a graph id (``|`` becomes ``__``).

    Args:
        gid: Graph id from :func:`graph_id`.

    Returns:
        The id with every ``|`` replaced by ``__``.
    """

    return gid.replace("|", "__")


def community_name(gid: str, resolution: float, method: str = "leiden") -> str:
    """Name of a partition file inside ``metrics/communities``.

    Args:
        gid: Graph id from :func:`graph_id`.
        resolution: Modularity resolution; written as a float (``1`` becomes ``res1.0``).
        method: ``leiden`` (no suffix) or another method name, appended as ``__{method}``.

    Returns:
        ``{file_safe(gid)}__res{resolution}[__{method}].npz``.
    """

    suffix = "" if method == "leiden" else f"__{method}"
    return f"{file_safe(gid)}__res{float(resolution)}{suffix}.npz"


def _listed(name: str, allowed: tuple[str, ...], where: str) -> str:
    """``name`` when it is one of the files a stage declares; an error otherwise.

    Args:
        name: Table or file name asked for.
        allowed: The names the stage declares (:data:`METRICS_TABLES`...).
        where: Stage name for the error message.

    Returns:
        ``name`` unchanged.

    Raises:
        ValueError: If ``name`` is not in ``allowed``, so that a typo fails instead of
            pointing at a file that never exists.
    """

    if name not in allowed:
        raise ValueError(f"'{name}' is not one of the {where} files: {', '.join(allowed)}")
    return name


def candidates_name(rep: str) -> str:
    """File name of the candidate lists of a representation: ``cand_{rep}.npz``.

    Args:
        rep: Representation name.

    Returns:
        The file name, without directory.
    """

    return f"cand_{rep}.npz"


def neighbors_name(rep: str, k: int, sensitivity: bool = False) -> str:
    """``nbr_{rep}_k{k}.npz``; ``_eps`` files hold the eps-sensitivity neighbour sets.

    Args:
        rep: Representation name.
        k: Number of neighbours.
        sensitivity: True for the sets selected with ``networks.eps_sensitivity``.

    Returns:
        The file name, without directory.
    """

    return f"nbr_{rep}_k{k}{'_eps' if sensitivity else ''}.npz"


def vocab_neighbors_name(k: int) -> str:
    """File name of the vocabulary network neighbours for one ``k``: ``vocab_k{k}.npz``.

    Args:
        k: Number of neighbours.

    Returns:
        The file name, without directory.
    """

    return f"vocab_k{k}.npz"


def position_bucket(position: int) -> str:
    """Label of the position bucket for a sequence position (1-based after the prefix).

    Args:
        position: Token position in the sequence; 0 is the prefix token, so vertices start
            at 1.

    Returns:
        One of ``1-4``, ``5-16``, ``17-64``, ``65+`` (see :data:`POS_BUCKETS`).

    Raises:
        ValueError: If ``position`` is smaller than 1.
    """

    for low, high, label in POS_BUCKETS:
        if position >= low and (high is None or position <= high):
            return label
    raise ValueError(f"Position {position} is not a vertex position (must be >= 1)")


@dataclass(frozen=True)
class RunPaths:
    """All artifact locations of one run.

    Stages never build a path themselves: they ask this object, so a file has one name
    everywhere. The object only computes paths; nothing is created or checked.

    Attributes:
        root: Repository root every other path was resolved against.
        raw_dir: Cache of raw Wikipedia API responses (shared by every run).
        corpus_dir: Corpus files (shared by every run).
        run_dir: ``runs_dir/<name>``, holding one directory per stage of this run.
        report_dir: Where the generated report material goes (``figuras/``, ``tabelas/``).
    """

    root: Path
    raw_dir: Path
    corpus_dir: Path
    run_dir: Path
    report_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings, root: str | Path = ".") -> RunPaths:
        """Paths of a run, from the ``paths`` section and the run name.

        Args:
            settings: Run configuration.
            root: Directory the configured relative paths are joined to. Defaults to the current
                directory.

        Returns:
            The :class:`RunPaths` of the run.
        """

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
        """``_manifest.json`` of the corpus stage, inside ``corpus_dir``."""

        return self.corpus_dir / MANIFEST

    @property
    def sense_annotations(self) -> Path:
        """Versioned sense labels of target occurrences, shared by every run."""

        return self.root / "data" / "annotations" / "senses.csv"

    def manifest(self, directory: str) -> Path:
        """``_manifest.json`` of a per-run stage directory (sample, reps, knn, metrics, ...).

        Args:
            directory: Directory name inside the run (a value of :data:`STAGE_DIRS`).

        Returns:
            Path of the manifest.
        """

        return self.stage_dir(directory) / MANIFEST

    def stage_manifest(self, stage: str) -> Path:
        """``_manifest.json`` of a stage given by its CLI name (``extract`` lives in ``reps``).

        Args:
            stage: CLI stage name, a key of :data:`STAGE_DIRS` (``corpus`` is not one: use
                :attr:`corpus_manifest`).

        Returns:
            Path of the manifest.

        Raises:
            KeyError: If ``stage`` is not a per-run stage.
        """

        return self.manifest(STAGE_DIRS[stage])

    @property
    def articles(self) -> Path:
        """``articles.jsonl``: one record per candidate article, kept or dropped."""

        return self.corpus_dir / "articles.jsonl"

    @property
    def paragraphs(self) -> Path:
        """``paragraphs.jsonl``: one record per kept paragraph, with its sentence spans."""

        return self.corpus_dir / "paragraphs.jsonl"

    @property
    def corpus_manifest_csv(self) -> Path:
        """``manifest.csv``: the versioned list of fetched articles with their ``revid``."""

        return self.corpus_dir / "manifest.csv"

    # per-run stage directories
    def stage_dir(self, stage: str) -> Path:
        """Directory of a stage inside the run.

        Args:
            stage: Directory name (``sample``, ``reps``, ``knn``...), not the CLI stage name; the
                value is joined as given, without checking it against :data:`STAGE_DIRS`.

        Returns:
            ``run_dir / stage``.
        """

        return self.run_dir / stage

    @property
    def sample_dir(self) -> Path:
        """Directory of the ``sample`` stage."""

        return self.stage_dir("sample")

    @property
    def occurrences(self) -> Path:
        """``sample/occurrences.csv``: one row per vertex (see :data:`OCCURRENCE_COLUMNS`)."""

        return self.sample_dir / "occurrences.csv"

    @property
    def sequences(self) -> Path:
        """``sample/sequences.jsonl``: the token ids of every sequence the model runs."""

        return self.sample_dir / "sequences.jsonl"

    @property
    def vocab_types(self) -> Path:
        """``sample/vocab_types.csv``: one row per vocabulary id (see :data:`VOCAB_COLUMNS`)."""

        return self.sample_dir / "vocab_types.csv"

    @property
    def senses(self) -> Path:
        """``sample/senses.csv``: template of the target occurrences to annotate."""

        return self.sample_dir / "senses.csv"

    @property
    def reps_dir(self) -> Path:
        """Directory of the ``extract`` stage (``reps``)."""

        return self.stage_dir("reps")

    def rep(self, name: str) -> Path:
        """File of a stored contextual representation.

        Args:
            name: Name of a stored representation (``L01``, ``L36n``); centered names have no
                file, and the name is not checked.

        Returns:
            ``reps/{name}.safetensors``, holding tensor ``x`` of shape ``[N, d]`` in bfloat16.
        """

        return self.reps_dir / f"{name}.safetensors"

    @property
    def embeddings(self) -> Path:
        """``reps/embeddings.safetensors``: tensor ``weight``, the embedding rows ``[V, d]``."""

        return self.reps_dir / "embeddings.safetensors"

    @property
    def all_layers(self) -> Path:
        """``reps/all_layers.npy``: bfloat16 bits as uint16, ``[L, N, d]`` (layer l at l - 1)."""

        return self.reps_dir / "all_layers.npy"

    @property
    def pred_next(self) -> Path:
        """``reps/pred_next.npy``: predicted next token id of every occurrence."""

        return self.reps_dir / "pred_next.npy"

    @property
    def diagnostics(self) -> Path:
        """``reps/diagnostics.json``: norm, anisotropy and dominant-dimension diagnostics."""

        return self.reps_dir / "diagnostics.json"

    @property
    def verify_report(self) -> Path:
        """``reps/verify.json``: result of ``extract --verify`` on the real model."""

        return self.reps_dir / "verify.json"

    @property
    def knn_dir(self) -> Path:
        """Directory of the ``knn`` stage."""

        return self.stage_dir("knn")

    def cand(self, rep: str) -> Path:
        """Candidate lists of a representation.

        Args:
            rep: Representation name, centered ones included.

        Returns:
            ``knn/cand_{rep}.npz``.
        """

        return self.knn_dir / candidates_name(rep)

    def nbr(self, rep: str, k: int, sensitivity: bool = False) -> Path:
        """Neighbour sets of a representation for one ``k``.

        Args:
            rep: Representation name.
            k: Number of neighbours.
            sensitivity: True for the sets selected with ``networks.eps_sensitivity``.

        Returns:
            ``knn/nbr_{rep}_k{k}.npz`` or, with ``sensitivity``, ``knn/nbr_{rep}_k{k}_eps.npz``.
        """

        return self.knn_dir / neighbors_name(rep, k, sensitivity)

    @property
    def vocab_cand(self) -> Path:
        """``knn/vocab_cand.npz``: candidate lists of the vocabulary network."""

        return self.knn_dir / "vocab_cand.npz"

    def vocab_nbr(self, k: int) -> Path:
        """Neighbour sets of the vocabulary network for one ``k``.

        Args:
            k: Number of neighbours.

        Returns:
            ``knn/vocab_k{k}.npz``.
        """

        return self.knn_dir / vocab_neighbors_name(k)

    @property
    def metrics_dir(self) -> Path:
        """Directory of the ``metrics`` stage."""

        return self.stage_dir("metrics")

    def metrics_table(self, name: str) -> Path:
        """CSV table of the ``metrics`` stage.

        Args:
            name: Table name without extension, one of :data:`METRICS_TABLES`.

        Returns:
            ``metrics/{name}.csv``.

        Raises:
            ValueError: If the name is not a declared metrics table.
        """

        return self.metrics_dir / f"{_listed(name, METRICS_TABLES, 'metrics')}.csv"

    @property
    def communities_dir(self) -> Path:
        """``metrics/communities``: one ``.npz`` partition per graph, resolution and method."""

        return self.metrics_dir / "communities"

    @property
    def communities_index(self) -> Path:
        """``metrics/communities/index.csv``: the list of the stored partitions."""

        return self.communities_dir / "index.csv"

    def community(self, gid: str, resolution: float, method: str = "leiden") -> Path:
        """Partition file of one graph.

        Args:
            gid: Graph id from :func:`graph_id`.
            resolution: Modularity resolution.
            method: ``leiden`` or another method name (see :func:`community_name`).

        Returns:
            Path inside :attr:`communities_dir`.
        """

        return self.communities_dir / community_name(gid, resolution, method)

    @property
    def analysis_dir(self) -> Path:
        """Directory of the ``analyze`` stage (``analysis``)."""

        return self.stage_dir("analysis")

    def analysis_table(self, name: str) -> Path:
        """CSV table of the ``analyze`` stage.

        Args:
            name: Table name without extension, one of :data:`ANALYSIS_TABLES`.

        Returns:
            ``analysis/{name}.csv``.

        Raises:
            ValueError: If the name is not a declared analysis table.
        """

        return self.analysis_dir / f"{_listed(name, ANALYSIS_TABLES, 'analysis')}.csv"

    @property
    def lens_dir(self) -> Path:
        """Directory of the ``lens`` stage."""

        return self.stage_dir("lens")

    def lens_file(self, name: str) -> Path:
        """File of the ``lens`` stage.

        Args:
            name: File name with extension, one of :data:`LENS_FILES`.

        Returns:
            ``lens/{name}``.

        Raises:
            ValueError: If the name is not a declared lens file.
        """

        return self.lens_dir / _listed(name, LENS_FILES, "lens")

    def lens_topk(self, rep: str) -> Path:
        """Vocabulary neighbours of every occurrence under one representation.

        Args:
            rep: Representation name.

        Returns:
            ``lens/topk_{rep}.npy``.
        """

        return self.lens_dir / f"topk_{rep}.npy"

    # generated report material
    @property
    def figures_dir(self) -> Path:
        """``report_dir/figuras``: generated figures (PDF)."""

        return self.report_dir / "figuras"

    @property
    def tables_dir(self) -> Path:
        """``report_dir/tabelas``: generated tables and ``numeros.tex``."""

        return self.report_dir / "tabelas"


def ensure_dir(path: Path) -> Path:
    """Create a directory, with its parents, when it does not exist.

    Args:
        path: Directory to create.

    Returns:
        ``path``, so the call can be chained.

    Side effects:
        Creates directories on disk.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_occurrences(
    path: Path, columns: Iterable[str] | None = None, dtype: dict[str, Any] | None = None
) -> pd.DataFrame:
    """``occurrences.csv`` (only ``columns`` when given), checked for row alignment.

    Empty cells stay empty strings (``keep_default_na=False``): token texts such as ``NA`` or
    ``nan`` are data, not missing values. When ``occurrence_id`` is read it must equal the row
    index, the vertex order of every representation and network.

    Args:
        path: The occurrences file of a run.
        columns: Columns to read; ``None`` reads them all. Names missing from the file are
            ignored without error. Include ``occurrence_id`` to get the alignment check.
        dtype: Column to dtype, passed to ``pandas.read_csv``.

    Returns:
        DataFrame with one row per vertex, in file order, and the columns in file order.

    Raises:
        ValueError: If ``occurrence_id`` was read and is not ``0, 1, ..., N - 1``.
    """

    wanted = None if columns is None else set(columns)
    frame = pd.read_csv(
        path,
        usecols=None if wanted is None else (lambda column: column in wanted),
        dtype=dtype,
        keep_default_na=False,
        low_memory=False,
    )
    if "occurrence_id" in frame:
        occurrence = pd.to_numeric(frame["occurrence_id"]).to_numpy(dtype=np.int64)
        if not np.array_equal(occurrence, np.arange(len(frame))):
            raise ValueError(f"{path}: occurrence_id must equal the row index")
    return frame


def token_text(token_repr: str) -> str:
    """Token text from the ``token_repr`` column of ``vocab_types.csv`` (``"' banco'"``).

    Args:
        token_repr: ``repr()`` of the decoded token text, as written by the ``sample`` stage.

    Returns:
        The decoded text. A value that is not a Python literal is returned unchanged, and a
        literal that is not a string (a bare number) is converted with ``str``.
    """

    try:
        value = ast.literal_eval(token_repr)
    except (ValueError, SyntaxError):
        return token_repr
    return value if isinstance(value, str) else str(value)


def read_token_ids(path: Path) -> np.ndarray:
    """``token_id`` of every occurrence, in vertex order.

    Args:
        path: The occurrences file of a run.

    Returns:
        int64 array of shape ``[N]``; entry ``i`` is the token id of occurrence ``i``.

    Raises:
        ValueError: If ``occurrence_id`` is not the row index (see :func:`read_occurrences`).
    """

    frame = read_occurrences(path, ["occurrence_id", "token_id"])
    return frame["token_id"].to_numpy(dtype=np.int64)


def bf16_tensor(bits: np.ndarray) -> torch.Tensor:
    """bfloat16 tensor from raw bfloat16 bits stored as uint16 (``all_layers.npy``).

    The bits are copied first: memmap slices are read-only, and torch refuses to wrap them.

    Args:
        bits: uint16 array of any shape whose values are bfloat16 bit patterns.

    Returns:
        ``torch.bfloat16`` tensor of the same shape; the bits are reinterpreted, not converted.
    """

    import torch

    array = np.array(bits, dtype=np.uint16, copy=True, order="C")
    return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)


# Spellings :func:`parse_bool` accepts, compared after ``str``, ``strip`` and ``lower``. ``1.0``
# and ``0.0`` cover boolean columns that pandas wrote as floats.
TRUE_TEXT = frozenset({"true", "1", "1.0", "yes"})
FALSE_TEXT = frozenset({"false", "0", "0.0", "no"})


def parse_bool(values: Iterable[Any]) -> np.ndarray:
    """Booleans written to CSV (``True``/``False``, ``1``/``0``...); other values are an error.

    Args:
        values: Cells of a boolean column, as text, numbers or booleans. Each is converted
            with ``str``, stripped and lower-cased before the comparison with
            :data:`TRUE_TEXT` and :data:`FALSE_TEXT`.

    Returns:
        Boolean array of shape ``[len(values)]``.

    Raises:
        ValueError: If a value is in neither set (an empty cell included).
    """

    text = pd.Series(list(values), dtype=object).astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - TRUE_TEXT - FALSE_TEXT)
    if unknown:
        raise ValueError(f"Unexpected boolean values: {unknown[:5]}")
    return text.isin(TRUE_TEXT).to_numpy()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSON lines (UTF-8, non-ASCII kept as is).

    Args:
        path: Output file; its directory is created.
        records: One JSON-serializable dictionary per line, in iteration order.

    Returns:
        Number of records written.

    Side effects:
        Replaces the file at ``path``.
    """

    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate over the records of a JSON-lines file, skipping blank lines.

    Args:
        path: File written by :func:`write_jsonl`.

    Yields:
        One dictionary per non-blank line, in file order. The file stays open until the
        iterator is exhausted or closed.
    """

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    """Write dictionaries as a CSV file with a header.

    Args:
        path: Output file; its directory is created.
        rows: One dictionary per row.
        fields: Column names, in output order.

    Returns:
        Number of data rows written.

    Raises:
        ValueError: From ``csv.DictWriter`` when a row has a key that is not in ``fields``
            (missing keys are written as empty cells).

    Side effects:
        Replaces the file at ``path``.
    """

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
    """Write a JSON file (UTF-8, two-space indent, trailing newline).

    Args:
        path: Output file; its directory is created.
        payload: JSON-serializable value. Dictionary keys keep their insertion order.

    Side effects:
        Replaces the file at ``path``.
    """

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded value.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def read_json_gz(path: Path) -> Any:
    """Read a gzip-compressed JSON file (the raw API cache).

    Args:
        path: File written by :func:`write_json_gz`.

    Returns:
        The decoded value.
    """

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_gz(path: Path, payload: Any) -> None:
    """Write a gzip-compressed JSON file (the raw API cache).

    Args:
        path: Output file; its directory is created.
        payload: JSON-serializable value.

    Side effects:
        Replaces the file at ``path``.
    """

    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)


def _git_describe(root: Path) -> str | None:
    """``git describe --always --dirty`` of the repository, for the manifests.

    Args:
        root: Directory the command runs in.

    Returns:
        The commit description (with ``-dirty`` when tracked files have uncommitted changes),
        or ``None`` when git is missing, fails, times out after 10 s or prints nothing.
    """

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
    """Versions of Python and of the libraries the results depend on.

    Returns:
        ``{"python": ..., "numpy": ..., ...}`` for numpy, torch, transformers, igraph,
        networkx and pandas; a library that cannot be imported is left out.

    Side effects:
        Imports every listed library, torch and transformers included, whatever the stage.
    """

    versions = {"python": platform.python_version()}
    for name in ("numpy", "torch", "transformers", "igraph", "networkx", "pandas"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def file_fingerprint(paths: Iterable[Path]) -> dict[str, list[int] | None]:
    """Size and modification time of each input (None when missing), keyed by path.

    Args:
        paths: Files to describe.

    Returns:
        ``{str(path): [size in bytes, mtime in ns]}``, or ``None`` for a missing file. The key
        is the path as given (relative or absolute), and the contents are not hashed.
    """

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
    """Copy with every mapping key as a string (``max_memory`` mixes ``0`` and ``"cpu"``).

    Args:
        value: Nested dictionaries, lists, tuples and scalars.

    Returns:
        The same structure with string keys; tuples become lists.
    """

    if isinstance(value, dict):
        return {str(key): _string_keys(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_string_keys(item) for item in value]
    return value


# Settings sections each stage reads, its own and those of the stages before it: changing any
# other section (say ``analysis.permutations``) does not make the stage stale. Stages not listed
# depend on the whole configuration.
_UPSTREAM = ("corpus", "sample", "model")
STAGE_SETTINGS: dict[str, tuple[str, ...]] = {
    "corpus": ("corpus",),
    "sample": ("corpus", "sample"),
    "extract": _UPSTREAM,
    "knn": (*_UPSTREAM, "networks"),
    "metrics": (*_UPSTREAM, "networks", "analysis"),
    "analyze": (*_UPSTREAM, "networks", "analysis"),
    "lens": (*_UPSTREAM, "lens"),
}


def settings_digest(settings: Settings, stage: str | None = None) -> str:
    """Hash of the configuration a stage depends on (the whole run when ``stage`` is not one of
    :data:`STAGE_SETTINGS`); the source file path never counts.

    Args:
        settings: Run configuration.
        stage: CLI stage name; ``None`` or an unlisted name (``report``) hashes every section
            and the run name.

    Returns:
        SHA-256 hex digest of the selected sections, serialized as JSON with sorted keys.
    """

    payload = _string_keys(settings.to_dict())
    payload.pop("source", None)
    sections = STAGE_SETTINGS.get(stage or "")
    if sections is not None:
        payload = {key: payload[key] for key in sections}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_is_fresh(
    stage_dir: Path, settings: Settings, inputs: Iterable[Path], outputs: Iterable[Path]
) -> bool:
    """True when the stage's outputs exist and were built from the same inputs and settings.

    Stages skip only when this holds, so rerunning an upstream stage (for example ``sample
    --force``) makes every downstream stage recompute instead of reusing stale artifacts.

    Args:
        stage_dir: Directory holding the stage's ``_manifest.json``.
        settings: Current run configuration.
        inputs: Upstream files the stage reads.
        outputs: Files the stage must have written.

    Returns:
        True when the manifest and every output exist, the settings digest recorded in the
        manifest equals the current one (for the stage named in the manifest) and the recorded
        input fingerprint equals the current one. An unreadable manifest counts as stale.

    Notes:
        The code version is not part of the test: after a code change the stage is still
        fresh, and only ``--force`` recomputes it. Inputs are compared by size and
        modification time under the path as written, not by content.
    """

    manifest = stage_dir / MANIFEST
    if not manifest.exists() or not all(Path(path).exists() for path in outputs):
        return False
    try:
        recorded = read_json(manifest)
    except (OSError, json.JSONDecodeError):
        return False
    digest = settings_digest(settings, recorded.get("stage"))
    return recorded.get("settings_digest") == digest and recorded.get(
        "inputs"
    ) == file_fingerprint(inputs)


def up_to_date(
    stage: str,
    paths: RunPaths,
    settings: Settings,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    force: bool = False,
) -> bool:
    """True, and logged, when a stage can be skipped (see :func:`stage_is_fresh`).

    Args:
        stage: CLI stage name, a key of :data:`STAGE_DIRS`.
        paths: Paths of the run.
        settings: Current run configuration.
        inputs: Upstream files the stage reads.
        outputs: Files the stage must have written.
        force: When true the stage is never skipped.

    Returns:
        True when the caller should return without doing work.
    """

    stage_dir = paths.stage_dir(STAGE_DIRS[stage])
    if force or not stage_is_fresh(stage_dir, settings, inputs, outputs):
        return False
    LOGGER.info("Etapa %s já está atualizada em %s; use --force para refazer", stage, stage_dir)
    return True


def require_inputs(paths: Iterable[Path], earlier: str) -> None:
    """Stop before any work when an input of a stage is missing.

    Args:
        paths: Files the stage needs.
        earlier: Name of the stage (or stages) that writes them, for the message.

    Raises:
        FileNotFoundError: Listing every missing file.
    """

    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Rode {earlier} antes; faltam: " + ", ".join(missing))


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

    Args:
        stage_dir: Directory of the stage; the manifest is written inside it.
        stage: CLI stage name, stored and later used to pick the settings sections.
        settings: Run configuration; stored whole, with its digest for ``stage``.
        started: ``time.time()`` at the start of the stage.
        extra: Stage-specific facts merged at the top level; a key equal to a standard one
            replaces it.
        root: Repository root, where ``git describe`` runs.
        inputs: Upstream files to fingerprint; ``None`` records no inputs.

    Returns:
        Path of the written ``_manifest.json``.

    Side effects:
        Writes the manifest, runs ``git describe`` and imports the libraries whose versions
        are recorded. Stages call this last, so an interrupted stage leaves no manifest and
        is redone.
    """

    path = stage_dir / MANIFEST
    write_json(
        path,
        {
            "stage": stage,
            "run": settings.name,
            "config_source": settings.source,
            "settings_digest": settings_digest(settings, stage),
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
