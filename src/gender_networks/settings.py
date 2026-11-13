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

from gender_networks.representations import FINAL_NORM, try_parse


@dataclass(frozen=True)
class PathsSettings:
    """Where shared inputs and per-run outputs live, relative to the repository root.

    Attributes:
        raw_dir: Cache of raw Wikipedia API responses (shared by every run).
        corpus_dir: ``articles.jsonl``, ``paragraphs.jsonl`` and ``manifest.csv`` (shared).
        runs_dir: Parent of the per-run directories; a run writes to ``runs_dir/<name>/``.
        report_dir: Where the ``report`` stage writes ``figuras/`` and ``tabelas/``.
    """

    raw_dir: str = "data/raw/wiki"
    corpus_dir: str = "data/corpus"
    runs_dir: str = "outputs/experiment"
    report_dir: str = "report/relatorio"


@dataclass(frozen=True)
class ModelSettings:
    """Model loading and extraction choices.

    Attributes:
        name_or_path: Hugging Face model id or local path.
        revision: Commit SHA, tag or branch of the model repository; the main run pins a SHA.
        dtype: Name of the torch dtype the weights are loaded in (``bfloat16``).
        device_map: ``device_map`` passed to ``from_pretrained``; ``None`` leaves the argument
            out, so the model loads on the library's default device.
        max_memory: Per-device memory budget for ``device_map="auto"`` (GPU index or ``"cpu"``
            to a size such as ``"20GiB"``); ``None`` leaves it to accelerate.
        prefix_token: Token placed at position 0 of every sequence; it is never a vertex.
        layers: Representation name to the 1-based block it is read from (``L18: 18``).
        capture_all_layers: Also write every block's output to ``reps/all_layers.npy``.
        verify_sequences: Number of sequences ``extract --verify`` runs through the model.
    """

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
    """Rules that keep only running prose paragraphs.

    Attributes:
        min_words: Minimum number of whitespace-separated words of a paragraph.
        max_digit_ratio: Largest allowed fraction of digit characters (drops result tables and
            chronologies).
        dedup_chars: Length of the lower-cased prefix used as the duplicate-paragraph key.
    """

    min_words: int = 40
    max_digit_ratio: float = 0.08
    dedup_chars: int = 80


@dataclass(frozen=True)
class CorpusSettings:
    """Wikipedia download and cleaning.

    Attributes:
        themes: Theme code to its root category titles (``fisica: [Categoria:Física]``).
        extra_categories: Theme code to extra categories walked one level deep, added to
            strengthen weak senses of the target words.
        bfs_depth: Depth of the breadth-first walk from the roots (0 = only the roots' own
            members).
        max_titles_per_theme: Title limit of each theme walk, or ``None`` for no limit.
        articles_per_theme: Articles kept from the root walk of each theme.
        articles_per_extra_category: Articles kept from each extra category.
        delay_s: Seconds to wait between API requests.
        batch_titles: Titles per content request (the API maximum is 50).
        user_agent: ``User-Agent`` header; Wikimedia policy asks for a contact.
        skip_category_patterns: Substrings of subcategory titles the walk does not open.
        exclude_infobox_patterns: Infobox template names that mark a biography.
        drop_multi_theme: Drop titles reached from more than one theme.
        cut_sections: Headings of support sections removed with their subsections.
        paragraph: Paragraph filter (see :class:`ParagraphFilter`).
        seed: Seed of the per-group shuffle that orders the candidates.
    """

    themes: dict[str, list[str]] = field(default_factory=dict)
    extra_categories: dict[str, list[str]] = field(default_factory=dict)
    bfs_depth: int = 2
    # Title limit of each theme walk (extra categories: 4 x articles_per_extra_category). The
    # walk stops at it, so it keeps the alphabetically first subcategories and titles. None
    # walks every tree whole, extras included, and the seeded shuffle does the choosing.
    max_titles_per_theme: int | None = 600
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
    """Dense core: whole paragraphs, used for P1–P3.

    Attributes:
        articles_per_theme: Articles drawn per theme.
        paragraphs_per_article: Paragraphs drawn from each of those articles (2 keeps article
            and paragraph distinct labels in P3).
        root_only: Restrict the draw to articles of the theme's root category.
    """

    articles_per_theme: int = 5
    paragraphs_per_article: int = 2
    # False draws the core articles from every article of the theme, extra categories included
    # (the main run of 05/10/2026). True draws them only from the theme's root category, so
    # every theme of P3 is the tree of its root; the corpus must then record each origin.
    root_only: bool = False


@dataclass(frozen=True)
class MultithemeSettings:
    """Stratum of content words that occur in several themes (P4).

    Attributes:
        n_types: Number of words drawn.
        per_type: Occurrences each word is completed to, counting those already in the sample.
        min_themes: Minimum number of themes a word must occur in to be eligible.
        min_per_theme: Minimum occurrences in each of those themes, outside the core.
    """

    n_types: int = 150
    per_type: int = 20
    min_themes: int = 3
    min_per_theme: int = 5


@dataclass(frozen=True)
class LimitSettings:
    """Diversity limits for sparse strata.

    Attributes:
        per_type_paragraph: Most occurrences of one type drawn from one paragraph.
        per_type_article: Most occurrences of one type drawn from one article.
    """

    per_type_paragraph: int = 1
    per_type_article: int = 3


@dataclass(frozen=True)
class SampleSettings:
    """Hybrid occurrence sample.

    Attributes:
        seed: Seed of the single generator that draws the four strata in order.
        themes: Themes sampled; ``None`` (or an empty list) means every corpus theme.
        f_max: Most occurrences of one token type kept in the sample.
        bands: Frequency bands as inclusive ``[low, high]`` pairs; the last one is open-ended
            when applied to corpus frequencies.
        core: Dense core (see :class:`CoreSettings`).
        targets: Target word to its quota per sense theme (``banco: {economia: 17, ...}``).
        controls: Control word to its quota per theme, with the same allocation as the targets.
        min_per_sense: Occurrences a target needs in a theme for that theme to count.
        multitheme: Multi-theme stratum (see :class:`MultithemeSettings`).
        limits: Diversity limits of the sparse strata (see :class:`LimitSettings`).
        senses_per_target: Rows per target written to the sense-annotation template.
    """

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
    """Lexical type network over the whole vocabulary.

    Attributes:
        k_values: Neighbour counts the vocabulary network is built for.
    """

    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass(frozen=True)
class NetworkSettings:
    """k-NN construction.

    Attributes:
        k_values: Neighbour counts of the occurrence networks.
        k_main: The ``k`` of the main configuration; must be one of ``k_values``.
        eps: Tie tolerance on the cosine: similarities within ``eps`` of the k-th are tied.
        eps_sensitivity: Larger tolerance used for the sensitivity neighbour sets (``_eps``).
        seeds: Number of random tie-break draws (variant a); at least 2.
        base_seed: First component of every tie-break generator seed.
        candidates: Candidates stored per vertex (``K``); must exceed the largest ``k``.
        type_candidates: Candidate types stored per row for variant (c).
        block_size: Most rows per similarity block (also capped by memory in ``knn``).
        representations: Main representations, in depth order.
        robust_representations: Robustness representations (``L36n``).
        centered: Also build the centered variant of every main block.
        vocab: Vocabulary network settings (see :class:`VocabNetworkSettings`).
    """

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
    """Metrics, communities and statistics.

    Attributes:
        leiden_runs: Leiden runs per graph (different seeds); the best modularity is kept.
        leiden_iterations: Iterations per Leiden run; ``-1`` runs until no improvement.
        resolution: Modularity resolution of the main configuration.
        resolution_sweep: Resolutions of the robustness sweep.
        permutations: Label permutations behind every NMI baseline.
        distance_sample_sources: BFS sources of the sampled distance estimate.
        workers: Worker processes of the ``metrics`` stage.
        exact_distances_vocab: Compute exact distances on the main vocabulary graph.
        layer_sweep: Compute the layer-by-layer curves of P2 in the ``analyze`` stage.
    """

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
    # Layer-by-layer J(l-1, l) and D(l) over the 36 blocks in the analyze stage (P2).
    layer_sweep: bool = True


@dataclass(frozen=True)
class LensSettings:
    """Optional extension: nearest vocabulary rows of each occurrence.

    Attributes:
        k: Vocabulary neighbours kept per occurrence and representation.
        batch_size: Occurrences per similarity batch.
    """

    k: int = 20
    batch_size: int = 512


@dataclass(frozen=True)
class ReportSettings:
    """What the report shows; changes nothing that the computing stages produce.

    Attributes:
        representations: Representations shown in tables and per-representation panels; a
            subset of ``networks.representations``, or ``None`` for all of them.
    """

    # Representations the tables and the per-representation panels show (a subset of
    # networks.representations); null = all of them. A run over every block lists a few here,
    # and the figures that put the representations on an axis still use every block.
    representations: list[str] | None = None


@dataclass(frozen=True)
class Settings:
    """Complete configuration of one experiment run.

    Attributes:
        name: Run name; also the directory of the run inside ``paths.runs_dir``.
        paths: Input and output locations.
        model: Model loading and extraction.
        corpus: Wikipedia download and cleaning.
        sample: Hybrid sample.
        networks: k-NN construction.
        analysis: Metrics, communities and statistics.
        lens: Optional vocabulary-lens extension.
        report: What the report shows.
        source: Path of the YAML file the settings were loaded from; never part of the
            settings digest.
    """

    name: str
    paths: PathsSettings = field(default_factory=PathsSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    corpus: CorpusSettings = field(default_factory=CorpusSettings)
    sample: SampleSettings = field(default_factory=SampleSettings)
    networks: NetworkSettings = field(default_factory=NetworkSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    lens: LensSettings = field(default_factory=LensSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary, suitable for JSON manifests.

        Returns:
            Nested dictionaries and lists built by :func:`dataclasses.asdict`. ``model.max_memory``
            keeps its mixed ``int``/``str`` keys; ``artifacts.settings_digest`` turns them into
            strings before hashing.
        """

        return asdict(self)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Top-level section of the raw configuration as a mapping.

    Args:
        raw: Parsed YAML root.
        key: Section name (``model``, ``corpus``...).

    Returns:
        The section, or an empty dictionary when it is absent or null.

    Raises:
        ValueError: If the section is present but is not a mapping.
    """

    value = raw.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{key}' must be a mapping")
    return value


def _build(cls: type, raw: dict[str, Any], nested: dict[str, type] | None = None, where: str = ""):
    """Instantiate a dataclass, rejecting unknown keys so typos fail loudly.

    Args:
        cls: Settings dataclass to build.
        raw: Mapping read from YAML; keys must be field names of ``cls``.
        nested: Field name to the dataclass that field holds; those values are built
            recursively (one level: the nested call receives no ``nested`` of its own).
        where: Dotted path of the section, used in error messages.

    Returns:
        An instance of ``cls``; fields missing from ``raw`` take the dataclass defaults.

    Raises:
        ValueError: If ``raw`` has a key that is not a field, or a nested value is not a
            mapping.

    Notes:
        Values are not type-checked: a string where a number is expected is accepted here and
        fails, if at all, where it is used.
    """

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
    """Cross-field checks of a loaded configuration.

    Checks, in order: the run name; at least one theme; extra categories and sample themes
    within the corpus themes; target and control quotas (known themes, positive integers);
    ``f_max``; ``k_main`` among ``k_values``; ``candidates`` above the largest ``k``; at least
    two seeds; every representation name known for the configured layers; and
    ``report.representations`` a non-empty subset of the main representations.

    Args:
        settings: Configuration to check.

    Raises:
        ValueError: On the first check that fails.

    Notes:
        Quota themes are checked against ``corpus.themes``, not ``sample.themes``. Fields not
        listed above (``eps``, ``bands``, ``type_candidates``, ``networks.vocab.k_values``...)
        are not validated.
    """

    corpus, sample, networks = settings.corpus, settings.sample, settings.networks
    if not settings.name or "/" in settings.name:
        raise ValueError("name must be a non-empty string without '/'")
    if not corpus.themes:
        raise ValueError("corpus.themes must list at least one theme")
    for theme in corpus.extra_categories:
        if theme not in corpus.themes:
            raise ValueError(f"corpus.extra_categories uses unknown theme '{theme}'")
    if corpus.max_titles_per_theme is not None and corpus.max_titles_per_theme < 1:
        raise ValueError("corpus.max_titles_per_theme must be positive or null")
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
    unknown_layers = {
        name
        for name in networks.representations + networks.robust_representations
        if not _known_representation(name, settings.model.layers)
    }
    if unknown_layers:
        raise ValueError(f"Unknown representations: {', '.join(sorted(unknown_layers))}")
    shown = settings.report.representations
    if shown is not None:
        if not shown:
            raise ValueError("report.representations must be null or list at least one name")
        outside = [name for name in shown if name not in networks.representations]
        if outside:
            raise ValueError(
                "report.representations must be a subset of networks.representations: "
                + ", ".join(outside)
            )


def _known_representation(name: str, layers: dict[str, int]) -> bool:
    """``lex``, the final RMSNorm output, or a configured block (uncentered).

    Args:
        name: Name from ``networks.representations`` or ``networks.robust_representations``.
        layers: ``model.layers``, the configured block names.

    Returns:
        True for ``lex``, for ``L36n`` (``representations.FINAL_NORM``) and for an uncentered
        block name that is a key of ``layers``. Centered names are always False: centering is
        switched on by ``networks.centered``, not listed.
    """

    rep = try_parse(name)
    if rep is None or rep.centered:
        return False
    return rep.is_lexical or rep == FINAL_NORM or (rep.kind == "layer" and name in layers)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a configuration file, merging it over its ``base`` when it names one.

    Args:
        path: YAML file. Its optional ``base`` key is a path relative to the file's directory.

    Returns:
        The merged mapping, without the ``base`` key. Bases may be chained.

    Raises:
        ValueError: If the root of a file is not a mapping.

    Notes:
        The merge is one level deep: a top-level section given in the child is merged key by
        key over the base section, and each key the child gives replaces the base value whole.
        A nested subsection (``sample.core``, ``networks.vocab``...) is therefore replaced, not
        merged: keys the child omits fall back to the dataclass defaults, not to the base
        values. A ``base`` cycle is not detected.
    """

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
    """Load and validate a run configuration from YAML (optionally extending a ``base``).

    Args:
        path: Configuration file.

    Returns:
        The validated :class:`Settings`, with ``source`` set to ``str(path)``.

    Raises:
        ValueError: On unknown keys at any level, a section that is not a mapping, or a failed
            check of :func:`_validate`.
        OSError: If the file or one of its bases cannot be read.
    """

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
        report=_build(ReportSettings, _section(raw, "report"), where="report"),
        source=str(config_path),
    )
    _validate(settings)
    return settings
