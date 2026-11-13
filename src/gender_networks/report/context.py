"""What every report item reads: the run's settings, paths and tables (``Context``)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gender_networks.artifacts import (
    RunPaths,
    read_json,
    read_jsonl,
    token_text,
)
from gender_networks.labels import (
    ordered,
)
from gender_networks.representations import is_lexical, try_parse
from gender_networks.sampling import band_names
from gender_networks.settings import Settings

STAGE = "report"


class Skip(Exception):
    """An item cannot be produced from the artifacts on disk (the message says why)."""


def read_table(path: Path) -> pd.DataFrame:
    """CSV with only empty cells as missing (tokens like ``NA`` stay text); empty -> empty.

    Args:
        path: CSV file.

    Returns:
        The table; an empty DataFrame for a file without columns.
    """

    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def band_order(settings: Settings, open_top: bool = False) -> list[str]:
    """Frequency bands of the run in order (open on top for the corpus bands).

    Args:
        settings: Run configuration.
        open_top: Label the last band as open-ended.

    Returns:
        The band labels of ``sample.bands``.
    """

    return band_names(settings.sample.bands, open_top)


class Context:
    """Lazy, cached access to the artifacts of one run.

    Every reader raises :class:`Skip` when its file is missing, which is how an item reports
    that it cannot be built yet.

    Attributes:
        settings: Run configuration.
        paths: Paths of the run.
        k: ``networks.k_main``.
        read: Every file read so far, recorded in the stage manifest.
    """

    def __init__(self, settings: Settings, paths: RunPaths) -> None:
        """Create a context; nothing is read yet.

        Args:
            settings: Run configuration.
            paths: Paths of the run.
        """

        self.settings = settings
        self.paths = paths
        self.k = settings.networks.k_main
        self._tables: dict[Path, pd.DataFrame] = {}
        self._npz: dict[Path, dict[str, np.ndarray]] = {}
        self.read: set[Path] = set()

    # generic readers ----------------------------------------------------------------------

    def table(self, path: Path) -> pd.DataFrame:
        """A CSV table, read once and cached.

        Args:
            path: CSV file.

        Returns:
            The cached DataFrame; callers that modify it change the cached copy.

        Raises:
            Skip: If the file is missing.
        """

        if path not in self._tables:
            if not path.exists():
                raise Skip(f"falta {path}")
            self._tables[path] = read_table(path)
            self.read.add(path)
        return self._tables[path]

    def nonempty(self, path: Path) -> pd.DataFrame:
        """A CSV table that must have rows.

        Args:
            path: CSV file.

        Returns:
            The cached DataFrame.

        Raises:
            Skip: If the file is missing or the table is empty.
        """

        frame = self.table(path)
        if frame.empty:
            raise Skip(f"{path} está vazio")
        return frame

    def analysis(self, name: str) -> pd.DataFrame:
        """A non-empty table of the ``analyze`` stage.

        Args:
            name: Table name, one of ``artifacts.ANALYSIS_TABLES``.

        Returns:
            The cached DataFrame.

        Raises:
            Skip: If the table is missing or empty.
        """

        return self.nonempty(self.paths.analysis_table(name))

    def metrics_table(self, name: str) -> pd.DataFrame:
        """A non-empty table of the ``metrics`` stage.

        Args:
            name: Table name, one of ``artifacts.METRICS_TABLES``.

        Returns:
            The cached DataFrame.

        Raises:
            Skip: If the table is missing or empty.
        """

        return self.nonempty(self.paths.metrics_table(name))

    def npz(self, path: Path) -> dict[str, np.ndarray]:
        """Every array of an ``.npz`` file, read once and cached.

        Args:
            path: File to read.

        Returns:
            Array name to array.

        Raises:
            Skip: If the file is missing.
        """

        if path not in self._npz:
            if not path.exists():
                raise Skip(f"falta {path}")
            with np.load(path) as data:
                self._npz[path] = {key: data[key] for key in data.files}
            self.read.add(path)
        return self._npz[path]

    def json(self, path: Path) -> Any:
        """A JSON file (not cached).

        Args:
            path: File to read.

        Returns:
            The decoded value.

        Raises:
            Skip: If the file is missing.
        """

        if not path.exists():
            raise Skip(f"falta {path}")
        self.read.add(path)
        return read_json(path)

    def manifest(self, directory: str | None) -> dict[str, Any] | None:
        """The ``_manifest.json`` of a stage, or ``None`` when it is missing or unreadable.

        Args:
            directory: Stage directory inside the run (``sample``, ``reps``...); ``None`` is the
                corpus manifest.

        Returns:
            The manifest dictionary, or ``None``.
        """

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
        """The ``_manifest.json`` of a stage that an item cannot do without.

        Args:
            directory: Stage directory inside the run; ``None`` is the corpus manifest.
            stage: Stage name for the message.

        Returns:
            The manifest dictionary.

        Raises:
            Skip: If the manifest is missing or unreadable.
        """

        data = self.manifest(directory)
        if data is None:
            raise Skip(f"falta o _manifest.json da etapa {stage}")
        return data

    # run artifacts ------------------------------------------------------------------------

    @cached_property
    def occurrences(self) -> pd.DataFrame:
        """``occurrences.csv`` with its text columns as strings (missing cells empty).

        Raises:
            Skip: If the file is missing or empty.
        """

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
        """``vocab_types.csv`` sorted by ``token_id``, so the row position is the token id.

        Raises:
            Skip: If the file is missing or empty.
        """

        frame = self.table(self.paths.vocab_types)
        if frame.empty:
            raise Skip("vocab_types.csv está vazio")
        frame = frame.sort_values("token_id").reset_index(drop=True)
        frame["token_repr"] = frame["token_repr"].fillna("").astype(str)
        return frame

    def vertex_measures(self) -> pd.DataFrame:
        """``vertex_measures.csv`` with the occurrence labels the P1 figures group by.

        Returns:
            A copy of the table with ``band_sample``, ``token_category``, ``stratum``,
            ``f_sample`` and ``pos_bucket`` of each occurrence.

        Raises:
            Skip: If a table is missing or the two do not have the same number of rows.
        """

        measures = self.analysis("vertex_measures")
        occ = self.occurrences
        if len(measures) != len(occ):
            raise Skip("vertex_measures.csv e occurrences.csv têm números de linhas diferentes")
        frame = measures.copy()
        for column in ("band_sample", "token_category", "stratum", "f_sample", "pos_bucket"):
            frame[column] = occ[column].to_numpy()
        return frame

    def vocab_text(self, token_id: int) -> str:
        """Decoded text of a vocabulary id.

        Args:
            token_id: Token id.

        Returns:
            The token text (see ``artifacts.token_text``).
        """

        return token_text(self.vocab_types["token_repr"].iat[int(token_id)])

    @cached_property
    def main_reps(self) -> list[str]:
        """Main representations whose neighbour file at ``k_main`` exists.

        Raises:
            Skip: If none exists.
        """

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
        """Main representations other than ``lex``, in configuration order."""

        return [rep for rep in self.settings.networks.representations if not is_lexical(rep)]

    def shows(self, name: str) -> bool:
        """Whether tables and per-representation panels include ``name``.

        Every name when ``report.representations`` is unset; otherwise the listed ones and their
        variants (``L36n`` and ``L36c`` follow ``L36``). Names that are not representations
        (``vocab``) always show. Figures that put the representations on an axis ignore this.

        Args:
            name: Representation name, or another network name.

        Returns:
            Whether the name is shown.
        """

        shown = self.settings.report.representations
        rep = try_parse(name)
        if shown is None or rep is None:
            return True
        return rep.raw().source.name in shown

    @property
    def shown_reps(self) -> list[str]:
        """Main representations that tables and panels show (see :meth:`shows`)."""

        return [rep for rep in self.settings.networks.representations if self.shows(rep)]

    @property
    def shown_contextual_reps(self) -> list[str]:
        """Contextual main representations that tables and panels show."""

        return [rep for rep in self.contextual_reps if self.shows(rep)]

    def nbr(self, rep: str, k: int | None = None) -> dict[str, np.ndarray]:
        """Arrays of the neighbour file of a representation.

        Args:
            rep: Representation name.
            k: Number of neighbours; ``None`` uses ``k_main``.

        Returns:
            Array name to array (``pos``, ``rand``, ``all_indptr``...).

        Raises:
            Skip: If the file is missing.
        """

        return self.npz(self.paths.nbr(rep, self.k if k is None else k))

    @cached_property
    def vocab_k(self) -> int:
        """``k`` of the vocabulary graph that has communities.

        Raises:
            Skip: If the vocabulary network is disabled.
        """

        from gender_networks.metrics import vocab_community_k

        value = vocab_community_k(self.settings)
        if value is None:
            raise Skip("networks.vocab.k_values está vazio")
        return value

    @cached_property
    def paragraph_stats(self) -> pd.DataFrame:
        """Theme and number of words of every corpus paragraph (columns ``theme``, ``words``).

        Raises:
            Skip: If ``paragraphs.jsonl`` is missing or empty.
        """

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
        """The records of ``articles.jsonl`` as a table.

        Raises:
            Skip: If the file is missing or empty.
        """

        if not self.paths.articles.exists():
            raise Skip(f"falta {self.paths.articles}")
        self.read.add(self.paths.articles)
        frame = pd.DataFrame(list(read_jsonl(self.paths.articles)))
        if frame.empty:
            raise Skip("articles.jsonl está vazio")
        return frame

    def theme_order(self, themes: Iterable[str]) -> list[str]:
        """Themes in the order of the configuration, unknown ones after.

        Args:
            themes: Theme codes found in the data.

        Returns:
            Each distinct theme once.
        """

        return ordered(list(themes), list(self.settings.corpus.themes))



