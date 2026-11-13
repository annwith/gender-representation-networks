"""Stage ``corpus``: download Wikipedia articles by theme and keep clean prose paragraphs.

Themes are defined by category trees. Every theme a title is reached from is recorded, and
titles reached from more than one theme are dropped (their vocabulary would blur the theme
labels used in P3 and as a proxy for sense in P4). Biographies are dropped for the same reason.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gender_networks.artifacts import (
    RunPaths,
    ensure_dir,
    write_csv,
    write_jsonl,
    write_manifest,
)
from gender_networks.settings import CorpusSettings, Settings
from gender_networks.textclean import (
    clean_article,
    has_excluded_infobox,
    is_disambiguation,
    split_sentences,
)
from gender_networks.wiki import Page, WikiClient

LOGGER = logging.getLogger(__name__)
OFFLINE_ENV = "GENDER_NETWORKS_OFFLINE"
MANIFEST_FIELDS = [
    "pageid", "revid", "title", "theme", "source_category", "origin", "kept", "dropped_reason",
]  # fmt: skip


@dataclass
class Candidate:
    """A title found by a category walk, before it is fetched.

    Attributes:
        title: Article title.
        theme: Theme of the walk that found it.
        source_category: Category that first reached the title in that walk.
        depth: Depth of that category (0 for a root).
        origin: ``"root"`` or the extra category the walk started from.
    """

    title: str
    theme: str
    source_category: str
    depth: int
    origin: str  # "root" or the extra category


@dataclass
class CorpusResult:
    """What the corpus stage produces, before it is written.

    Attributes:
        articles: One record per candidate that was handled (see ``articles.jsonl``).
        paragraphs: One record per kept paragraph (see ``paragraphs.jsonl``).
        fetch_batches: Number of content batches requested.
    """

    articles: list[dict[str, Any]] = field(default_factory=list)
    paragraphs: list[dict[str, Any]] = field(default_factory=list)
    fetch_batches: int = 0


@dataclass(frozen=True)
class Walk:
    """One breadth-first category walk of the corpus stage.

    Attributes:
        walk_id: ``<theme>::root`` or ``<theme>::<extra category>``; the candidate group.
        theme: Theme the found titles are assigned to.
        origin: ``"root"`` or the extra category.
        roots: Categories the walk starts from.
        depth: Deepest category level opened.
        max_titles: Title limit; ``None`` walks the whole tree down to ``depth``.
    """

    walk_id: str  # "<theme>::root" or "<theme>::<extra category>": the candidate group
    theme: str
    origin: str  # "root" or the extra category
    roots: tuple[str, ...]
    depth: int
    max_titles: int | None  # None: the whole tree down to ``depth``


def walks(settings: CorpusSettings) -> list[Walk]:
    """The walks of the corpus stage, in order: each theme from its roots, then each extra
    category (one level deep, with room for four times the articles kept from it).

    With ``max_titles_per_theme: null`` no walk has a title limit.

    Args:
        settings: The ``corpus`` section.

    Returns:
        One :class:`Walk` per theme, in configuration order, followed by one per extra
        category. Extra walks use depth ``min(1, bfs_depth)``.
    """

    full = settings.max_titles_per_theme is None
    out = [
        Walk(f"{theme}::root", theme, "root", tuple(roots), settings.bfs_depth,
             settings.max_titles_per_theme)
        for theme, roots in settings.themes.items()
    ]  # fmt: skip
    for theme, categories in settings.extra_categories.items():
        for extra in categories:
            out.append(
                Walk(f"{theme}::{extra}", theme, extra, (extra,), min(1, settings.bfs_depth),
                     None if full else settings.articles_per_extra_category * 4)
            )  # fmt: skip
    return out


def collect_candidates(
    client: WikiClient, settings: CorpusSettings
) -> tuple[dict[str, list[Candidate]], dict[str, set[str]]]:
    """Crawl every theme and extra category; also return the themes that reach each title.

    Args:
        client: API client.
        settings: The ``corpus`` section.

    Returns:
        ``(groups, reached)``: walk id to its candidates in discovery order, and title to the
        set of themes whose walks (root or extra) found it.

    Side effects:
        Performs (or reads from the cache) the category requests and logs one line per walk.
    """

    groups: dict[str, list[Candidate]] = {}
    reached: dict[str, set[str]] = {}
    for walk in walks(settings):
        found = client.crawl(
            walk.roots, walk.depth, walk.max_titles, settings.skip_category_patterns
        )
        groups[walk.walk_id] = [
            Candidate(title, walk.theme, category, depth, walk.origin)
            for title, (category, depth) in found.items()
        ]
        for title in found:
            reached.setdefault(title, set()).add(walk.theme)
        if walk.origin == "root":
            LOGGER.info("Tema %s: %d títulos nas categorias", walk.theme, len(found))
        else:
            LOGGER.info("Extra %s (%s): %d títulos", walk.origin, walk.theme, len(found))
    return groups, reached


def group_order(candidates: list[Candidate], seed: int, group: str) -> list[Candidate]:
    """Fixed random order of a group's candidates, independent of every other group.

    Each group has its own generator and shuffles its full candidate list (sorted by title),
    so the order, and therefore the fetch batches, depend only on the crawl and the seed, never
    on how earlier groups were cleaned. A cleaning fix then reuses the cached batches instead of
    requesting new ones (and newer revisions). ``str`` seeds are hashed deterministically.

    Args:
        candidates: Candidates of one group.
        seed: ``corpus.seed``.
        group: Walk id, mixed into the generator seed.

    Returns:
        New list with the candidates in their shuffled order.
    """

    order = sorted(candidates, key=lambda c: c.title)
    random.Random(f"{seed}:{group}").shuffle(order)
    return order


def build_corpus(client: WikiClient, settings: CorpusSettings) -> CorpusResult:
    """Choose, fetch and clean articles; deterministic given the seed and the API responses.

    Groups are handled in walk order. In each group, titles reached from more than one theme
    are recorded as ``multi_theme`` and left out; the rest are fetched in batches, in the
    order of :func:`group_order`, until the group's quota of kept articles is met. A fetched
    title is dropped when the API has no page for it (``missing``), when its page was already
    handled under another title (``duplicate``) or when :func:`clean_page` gives a reason.

    Args:
        client: API client.
        settings: The ``corpus`` section.

    Returns:
        The article records in handling order and the paragraph records of the kept
        articles. Titles of the last batch that come after the quota was met are fetched but
        get no record.

    Side effects:
        Performs (or reads from the cache) the content requests and logs one line per group,
        as a warning when the quota was not met.

    Notes:
        A title is handled once: when two groups share it, the first group in walk order
        takes it. The paragraph dedup runs across all articles, in handling order.
    """

    groups, reached = collect_candidates(client, settings)
    result = CorpusResult()
    seen_titles: set[str] = set()
    seen_pageids: set[int] = set()
    seen_paragraphs: set[str] = set()
    for group, candidates in groups.items():
        theme, origin = group.split("::", 1)
        quota = (
            settings.articles_per_theme
            if origin == "root"
            else settings.articles_per_extra_category
        )
        pool: list[Candidate] = []
        for candidate in group_order(candidates, settings.seed, group):
            themes = reached.get(candidate.title, {candidate.theme})
            if settings.drop_multi_theme and len(themes) > 1:
                if candidate.title not in seen_titles:
                    seen_titles.add(candidate.title)
                    result.articles.append(_record(candidate, None, themes, "multi_theme"))
                continue
            # Titles already handled by an earlier group stay in the pool so that the batches
            # (and their cache keys) do not depend on earlier groups; they are skipped below.
            pool.append(candidate)
        kept = 0
        position = 0
        while kept < quota and position < len(pool):
            chunk = pool[position : position + settings.batch_titles]
            position += len(chunk)
            result.fetch_batches += 1
            pages = client.fetch_pages([c.title for c in chunk], settings.batch_titles)
            for candidate in chunk:
                if kept >= quota:
                    break
                if candidate.title in seen_titles:
                    continue
                seen_titles.add(candidate.title)
                themes = reached.get(candidate.title, {candidate.theme})
                page = pages.get(candidate.title)
                if page is None:
                    result.articles.append(_record(candidate, None, themes, "missing"))
                    continue
                if page.pageid in seen_pageids:
                    result.articles.append(_record(candidate, page, themes, "duplicate"))
                    continue
                seen_pageids.add(page.pageid)
                reason, paragraphs = clean_page(page, settings, seen_paragraphs)
                record = _record(candidate, page, themes, reason)
                record["is_biography"] = reason == "biography"
                result.articles.append(record)
                if reason is not None:
                    continue
                result.paragraphs.extend(
                    paragraph_records(
                        page, candidate.theme, candidate.source_category, candidate.origin,
                        paragraphs,
                    )
                )
                kept += 1
        log = LOGGER.warning if kept < quota else LOGGER.info
        log("Grupo %s: %d artigos mantidos (cota %d)", group, kept, quota)
    return result


def clean_page(
    page: Page, settings: CorpusSettings, seen_paragraphs: set[str]
) -> tuple[str | None, list[str]]:
    """Content filters and cleaning of one fetched page: ``(dropped_reason, paragraphs)``.

    ``seen_paragraphs`` holds the dedup keys of the paragraphs kept so far, across articles; the
    keys of this page's paragraphs are added to it, so pages must be cleaned in corpus order.

    Args:
        page: Fetched page.
        settings: The ``corpus`` section (infobox patterns, sections to cut, paragraph
            filter).
        seen_paragraphs: Dedup keys seen so far, updated in place. A key is the first
            ``paragraph.dedup_chars`` characters of a paragraph, lower-cased.

    Returns:
        ``("disambiguation", [])``, ``("biography", [])``, ``("no_paragraphs", [])`` or
        ``(None, paragraphs)`` with the kept paragraphs in article order. A page whose
        paragraphs all repeat earlier ones is ``no_paragraphs``.
    """

    if page.disambiguation or is_disambiguation(page.wikitext, page.title):
        return "disambiguation", []
    if has_excluded_infobox(page.wikitext, settings.exclude_infobox_patterns):
        return "biography", []
    paragraphs = []
    for text in clean_article(
        page.wikitext,
        settings.cut_sections,
        settings.paragraph.min_words,
        settings.paragraph.max_digit_ratio,
    ):
        key = text[: settings.paragraph.dedup_chars].lower()
        if key in seen_paragraphs:
            continue
        seen_paragraphs.add(key)
        paragraphs.append(text)
    if not paragraphs:
        return "no_paragraphs", []
    return None, paragraphs


def paragraph_records(
    page: Page, theme: str, source_category: str, origin: str | None, paragraphs: list[str]
) -> list[dict[str, Any]]:
    """Records of ``paragraphs.jsonl`` for the kept paragraphs of one article.

    Args:
        page: Fetched page (id, revision and title come from it).
        theme: Theme of the article.
        source_category: Category that reached the article.
        origin: ``"root"``, an extra category, or ``None`` when unknown (old manifests).
        paragraphs: Kept paragraphs, in article order.

    Returns:
        One record per paragraph. ``paragraph_id`` is ``<pageid>-<index>``, with the index
        counting the kept paragraphs of the article; ``sentences`` holds the ``[start, end)``
        character spans of :func:`~gender_networks.textclean.split_sentences`.
    """

    return [
        {
            "paragraph_id": f"{page.pageid}-{index}",
            "pageid": page.pageid,
            "revid": page.revid,
            "title": page.title,
            "theme": theme,
            "source_category": source_category,
            "origin": origin,
            "paragraph_idx": index,
            "text": text,
            "sentences": [list(span) for span in split_sentences(text)],
        }
        for index, text in enumerate(paragraphs)
    ]


def _record(
    candidate: Candidate, page: Any, themes: set[str], dropped_reason: str | None
) -> dict[str, Any]:
    """Record of ``articles.jsonl`` for one candidate.

    Args:
        candidate: The candidate.
        page: Fetched page, or ``None`` when it was not fetched or is missing.
        themes: Themes whose walks reached the title.
        dropped_reason: Why the article was dropped; ``None`` when it was kept.

    Returns:
        The record, with ``is_biography`` set to False (the caller updates it).
    """

    return {
        "pageid": page.pageid if page else None,
        "revid": page.revid if page else None,
        "title": candidate.title,
        "theme": candidate.theme,
        "themes_reached": sorted(themes),
        "source_category": candidate.source_category,
        "depth": candidate.depth,
        "origin": candidate.origin,
        "is_biography": False,
        "dropped_reason": dropped_reason,
    }


def summarize(result: CorpusResult) -> dict[str, Any]:
    """Counts of a corpus, for the stage manifest.

    Args:
        result: Corpus as built or rebuilt.

    Returns:
        Articles considered and kept, kept articles by theme and by origin, dropped articles
        by reason, paragraphs in total and by theme, and words (whitespace-separated).
    """

    kept = [a for a in result.articles if a["dropped_reason"] is None]
    return {
        "articles_considered": len(result.articles),
        "articles_kept": len(kept),
        "kept_by_theme": dict(Counter(a["theme"] for a in kept)),
        "kept_by_origin": dict(Counter(a["origin"] for a in kept)),
        "dropped_by_reason": dict(
            Counter(a["dropped_reason"] for a in result.articles if a["dropped_reason"])
        ),
        "paragraphs": len(result.paragraphs),
        "paragraphs_by_theme": dict(Counter(p["theme"] for p in result.paragraphs)),
        "words": sum(len(p["text"].split()) for p in result.paragraphs),
    }


def offline_from_env() -> bool:
    """True when ``GENDER_NETWORKS_OFFLINE`` is ``1``, ``true`` or ``yes`` (any case)."""

    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes"}


def run(
    settings: Settings,
    paths: RunPaths,
    force: bool = False,
    offline: bool | None = None,
    **_: object,
) -> None:
    """Download (or rebuild from the cache) and clean the corpus.

    ``offline`` (default: environment variable ``GENDER_NETWORKS_OFFLINE=1``) forbids network
    requests, so a re-clean after a cleaning fix fails loudly instead of fetching new revisions.

    Args:
        settings: Run configuration; the ``corpus`` section is used.
        paths: Paths of the run.
        force: Rebuild even when ``paragraphs.jsonl`` exists.
        offline: Forbid network requests; ``None`` reads the environment variable.
        **_: Ignored; accepted so every stage has the same call.

    Side effects:
        Writes ``articles.jsonl``, ``paragraphs.jsonl``, ``manifest.csv`` (only articles
        that have a page id) and ``_manifest.json`` in ``corpus_dir``, and fills the API
        cache.

    Notes:
        Unlike the other stages, the skip test is only the presence of ``paragraphs.jsonl``:
        a change in the ``corpus`` section does not rebuild the corpus without ``force``.
    """

    if paths.paragraphs.exists() and not force:
        LOGGER.info("Corpus já existe em %s; use --force para refazer", paths.corpus_dir)
        return
    if offline is None:
        offline = offline_from_env()
    started = time.time()
    ensure_dir(paths.corpus_dir)
    client = WikiClient(
        paths.raw_dir, settings.corpus.user_agent, settings.corpus.delay_s, offline=offline
    )
    result = build_corpus(client, settings.corpus)
    write_jsonl(paths.articles, result.articles)
    write_jsonl(paths.paragraphs, result.paragraphs)
    write_csv(
        paths.corpus_manifest_csv,
        (
            {
                "pageid": a["pageid"],
                "revid": a["revid"],
                "title": a["title"],
                "theme": a["theme"],
                "source_category": a["source_category"],
                "origin": a["origin"],
                "kept": a["dropped_reason"] is None,
                "dropped_reason": a["dropped_reason"] or "",
            }
            for a in result.articles
            if a["pageid"] is not None
        ),
        MANIFEST_FIELDS,
    )
    stats = summarize(result)
    stats.update(
        offline=offline,
        fetch_batches=result.fetch_batches,
        network_requests=client.network_requests,
        cache_hits=client.cache_hits,
        skipped_categories=list(client.skipped_categories),
    )
    write_manifest(paths.corpus_dir, "corpus", settings, started, stats, root=paths.root)
    LOGGER.info("Corpus: %s", stats)
