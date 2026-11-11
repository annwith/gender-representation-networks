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
    """CSV with only empty cells as missing (tokens like ``NA`` stay text); empty -> empty."""

    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def band_order(settings: Settings, open_top: bool = False) -> list[str]:
    """Frequency bands of the run in order (open on top for the corpus bands)."""

    return band_names(settings.sample.bands, open_top)


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
        return self.nonempty(self.paths.analysis_table(name))

    def metrics_table(self, name: str) -> pd.DataFrame:
        return self.nonempty(self.paths.metrics_table(name))

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

    def vertex_measures(self) -> pd.DataFrame:
        """``vertex_measures.csv`` with the occurrence labels the P1 figures group by."""

        measures = self.analysis("vertex_measures")
        occ = self.occurrences
        if len(measures) != len(occ):
            raise Skip("vertex_measures.csv e occurrences.csv têm números de linhas diferentes")
        frame = measures.copy()
        for column in ("band_sample", "token_category", "stratum", "f_sample", "pos_bucket"):
            frame[column] = occ[column].to_numpy()
        return frame

    def vocab_text(self, token_id: int) -> str:
        return token_text(self.vocab_types["token_repr"].iat[int(token_id)])

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
        return [rep for rep in self.settings.networks.representations if not is_lexical(rep)]

    def shows(self, name: str) -> bool:
        """Whether tables and per-representation panels include ``name``.

        Every name when ``report.representations`` is unset; otherwise the listed ones and their
        variants (``L36n`` and ``L36c`` follow ``L36``). Names that are not representations
        (``vocab``) always show. Figures that put the representations on an axis ignore this.
        """

        shown = self.settings.report.representations
        rep = try_parse(name)
        if shown is None or rep is None:
            return True
        return rep.raw().source.name in shown

    @property
    def shown_reps(self) -> list[str]:
        return [rep for rep in self.settings.networks.representations if self.shows(rep)]

    @property
    def shown_contextual_reps(self) -> list[str]:
        return [rep for rep in self.contextual_reps if self.shows(rep)]

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



