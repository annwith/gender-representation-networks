"""Typed settings for the lexical-to-contextual experiment.

One YAML file describes a whole run (``configs/experiment.yaml`` for the main run and
``configs/mini.yaml`` for a quick end-to-end rehearsal). Every stage of the CLI receives the
same :class:`Settings` object; sections that a stage does not use are simply ignored by it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsSettings:
    """Where shared inputs and per-run outputs live, relative to the repository root."""

    raw_dir: str = "data/raw/wiki"
    corpus_dir: str = "data/corpus"
    runs_dir: str = "outputs/experiment"
    report_dir: str = "report/relatorio"


@dataclass(frozen=True)
class ModelSettings:
    """Model loading and extraction choices."""

    name_or_path: str = "Qwen/Qwen3-4B-Base"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    max_memory: dict[int | str, str] | None = None
    prefix_token: str = "<|endoftext|>"
    layers: dict[str, int] = field(default_factory=lambda: {"L01": 1, "L18": 18, "L36": 36})
    capture_all_layers: bool = True
    verify_sequences: int = 8


@dataclass(frozen=True)
class ParagraphFilter:
    """Rules that keep only running prose paragraphs."""

    min_words: int = 40
    max_digit_ratio: float = 0.08
    dedup_chars: int = 80


@dataclass(frozen=True)
class CorpusSettings:
    """Wikipedia download and cleaning."""

    themes: dict[str, list[str]] = field(default_factory=dict)
    extra_categories: dict[str, list[str]] = field(default_factory=dict)
    bfs_depth: int = 2
    max_titles_per_theme: int = 600
    articles_per_theme: int = 150
    articles_per_extra_category: int = 60
    delay_s: float = 2.5
    batch_titles: int = 50
    user_agent: str = "gender-representation-networks/0.2 (academic project)"
    skip_category_patterns: list[str] = field(default_factory=list)
    exclude_infobox_patterns: list[str] = field(default_factory=list)
    drop_multi_theme: bool = True
    cut_sections: list[str] = field(default_factory=list)
    paragraph: ParagraphFilter = field(default_factory=ParagraphFilter)
    seed: int = 438


@dataclass(frozen=True)
class CoreSettings:
    """Dense core: whole paragraphs, used for P1–P3."""

    articles_per_theme: int = 5
    paragraphs_per_article: int = 2


@dataclass(frozen=True)
class MultithemeSettings:
    """Stratum of content words that occur in several themes (P4)."""

    n_types: int = 150
    per_type: int = 20
    min_themes: int = 3
    min_per_theme: int = 5


@dataclass(frozen=True)
class LimitSettings:
    """Diversity limits for sparse strata."""

    per_type_paragraph: int = 1
    per_type_article: int = 3


@dataclass(frozen=True)
class SampleSettings:
    """Hybrid occurrence sample."""

    seed: int = 438
    themes: list[str] | None = None
    f_max: int = 50
    bands: list[list[int]] = field(
        default_factory=lambda: [[1, 2], [3, 9], [10, 49], [50, 50]]
    )
    core: CoreSettings = field(default_factory=CoreSettings)
    targets: dict[str, dict[str, int]] = field(default_factory=dict)
    controls: dict[str, dict[str, int]] = field(default_factory=dict)
    min_per_sense: int = 10
    multitheme: MultithemeSettings = field(default_factory=MultithemeSettings)
    limits: LimitSettings = field(default_factory=LimitSettings)
    senses_per_target: int = 20


@dataclass(frozen=True)
class VocabNetworkSettings:
    """Lexical type network over the whole vocabulary."""

    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass(frozen=True)
class NetworkSettings:
    """k-NN construction."""

    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    k_main: int = 10
    eps: float = 1.0e-6
    eps_sensitivity: float = 1.0e-4
    seeds: int = 20
    base_seed: int = 438
    candidates: int = 128
    type_candidates: int = 64
    block_size: int = 1024
    representations: list[str] = field(default_factory=lambda: ["lex", "L01", "L18", "L36"])
    robust_representations: list[str] = field(default_factory=lambda: ["L36n"])
    centered: bool = True
    vocab: VocabNetworkSettings = field(default_factory=VocabNetworkSettings)


@dataclass(frozen=True)
class AnalysisSettings:
    """Metrics, communities and statistics."""

    leiden_runs: int = 10
    # Iterations per Leiden run (-1 = until no improvement). Unbounded runs did not finish on
    # the 151k-vertex vocabulary graph; a final extra iteration tells whether the run converged.
    leiden_iterations: int = 20
    resolution: float = 1.0
    resolution_sweep: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    permutations: int = 20
    distance_sample_sources: int = 500
    workers: int = 12
    exact_distances_vocab: bool = True


@dataclass(frozen=True)
class LensSettings:
    """Optional extension: nearest vocabulary rows of each occurrence."""

    k: int = 20
    batch_size: int = 512


@dataclass(frozen=True)
class Settings:
    """Complete configuration of one experiment run."""

    name: str
    paths: PathsSettings = field(default_factory=PathsSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    corpus: CorpusSettings = field(default_factory=CorpusSettings)
    sample: SampleSettings = field(default_factory=SampleSettings)
    networks: NetworkSettings = field(default_factory=NetworkSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    lens: LensSettings = field(default_factory=LensSettings)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary, suitable for JSON manifests."""

        return asdict(self)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{key}' must be a mapping")
    return value


def _build(cls: type, raw: dict[str, Any], nested: dict[str, type] | None = None, where: str = ""):
    """Instantiate a dataclass, rejecting unknown keys so typos fail loudly."""

    nested = nested or {}
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {where or cls.__name__}: {', '.join(sorted(unknown))}")
    values = dict(raw)
    for key, sub_cls in nested.items():
        if key in values and values[key] is not None:
            if not isinstance(values[key], dict):
                raise ValueError(f"{where}.{key} must be a mapping")
            values[key] = _build(sub_cls, values[key], where=f"{where}.{key}")
    return cls(**values)


def _validate(settings: Settings) -> None:
    corpus, sample, networks = settings.corpus, settings.sample, settings.networks
    if not settings.name or "/" in settings.name:
        raise ValueError("name must be a non-empty string without '/'")
    if not corpus.themes:
        raise ValueError("corpus.themes must list at least one theme")
    for theme in corpus.extra_categories:
        if theme not in corpus.themes:
            raise ValueError(f"corpus.extra_categories uses unknown theme '{theme}'")
    themes = set(sample.themes or corpus.themes)
    if not themes <= set(corpus.themes):
        raise ValueError("sample.themes must be a subset of corpus.themes")
    for label, table in (("targets", sample.targets), ("controls", sample.controls)):
        for word, quotas in table.items():
            if any(theme not in corpus.themes for theme in quotas):
                raise ValueError(f"sample.{label}.{word} uses an unknown theme")
            if any(not isinstance(q, int) or q <= 0 for q in quotas.values()):
                raise ValueError(f"sample.{label}.{word} quotas must be positive integers")
    if sample.f_max < 1:
        raise ValueError("sample.f_max must be positive")
    if networks.k_main not in networks.k_values:
        raise ValueError("networks.k_main must be one of networks.k_values")
    if networks.candidates <= max(networks.k_values):
        raise ValueError("networks.candidates must exceed the largest k")
    if networks.seeds < 2:
        raise ValueError("networks.seeds must be at least 2 (noise floor needs seed pairs)")
    unknown_layers = set(networks.representations + networks.robust_representations) - (
        {"lex", "L36n"} | set(settings.model.layers)
    )
    if unknown_layers:
        raise ValueError(f"Unknown representations: {', '.join(sorted(unknown_layers))}")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"The configuration root of {path} must be a mapping")
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    base = _read_yaml(path.parent / base_name)
    # Sections are merged key by key; a key given in the child replaces the base value whole
    # (so a child can restrict sample.targets without inheriting the base words).
    merged = dict(base)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = {**base[key], **value}
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path) -> Settings:
    """Load and validate a run configuration from YAML (optionally extending a ``base``)."""

    config_path = Path(path)
    raw = _read_yaml(config_path)
    allowed = set(Settings.__dataclass_fields__) - {"source"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")
    settings = Settings(
        name=raw.get("name", ""),
        paths=_build(PathsSettings, _section(raw, "paths"), where="paths"),
        model=_build(ModelSettings, _section(raw, "model"), where="model"),
        corpus=_build(
            CorpusSettings, _section(raw, "corpus"), {"paragraph": ParagraphFilter}, "corpus"
        ),
        sample=_build(
            SampleSettings,
            _section(raw, "sample"),
            {"core": CoreSettings, "multitheme": MultithemeSettings, "limits": LimitSettings},
            "sample",
        ),
        networks=_build(
            NetworkSettings, _section(raw, "networks"), {"vocab": VocabNetworkSettings}, "networks"
        ),
        analysis=_build(AnalysisSettings, _section(raw, "analysis"), where="analysis"),
        lens=_build(LensSettings, _section(raw, "lens"), where="lens"),
        source=str(config_path),
    )
    _validate(settings)
    return settings
