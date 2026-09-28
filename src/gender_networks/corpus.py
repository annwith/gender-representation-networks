"""Stage ``corpus``: download Wikipedia articles by theme and keep clean prose paragraphs.

Themes are defined by category trees. Every theme a title is reached from is recorded, and
titles reached from more than one theme are dropped (their vocabulary would blur the theme
labels used in P3 and as a proxy for sense in P4). Biographies are dropped for the same reason.
"""

from __future__ import annotations

import logging
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
from gender_networks.wiki import WikiClient

LOGGER = logging.getLogger(__name__)
MANIFEST_FIELDS = ["pageid", "revid", "title", "theme", "source_category", "kept", "dropped_reason"]


@dataclass
class Candidate:
    title: str
    theme: str
    source_category: str
    depth: int
    origin: str  # "root" or the extra category


@dataclass
class CorpusResult:
    articles: list[dict[str, Any]] = field(default_factory=list)
    paragraphs: list[dict[str, Any]] = field(default_factory=list)


def collect_candidates(
    client: WikiClient, settings: CorpusSettings
) -> tuple[dict[str, list[Candidate]], dict[str, set[str]]]:
    """Crawl every theme and extra category; also return the themes that reach each title."""

    groups: dict[str, list[Candidate]] = {}
    reached: dict[str, set[str]] = {}
    for theme, roots in settings.themes.items():
        found = client.crawl(
            roots,
            settings.bfs_depth,
            settings.max_titles_per_theme,
            settings.skip_category_patterns,
        )
        groups[f"{theme}::root"] = [
            Candidate(title, theme, category, depth, "root")
            for title, (category, depth) in found.items()
        ]
        for title in found:
            reached.setdefault(title, set()).add(theme)
        LOGGER.info("Tema %s: %d títulos nas categorias", theme, len(found))
    for theme, categories in settings.extra_categories.items():
        for extra in categories:
            found = client.crawl(
                [extra],
                min(1, settings.bfs_depth),
                settings.articles_per_extra_category * 4,
                settings.skip_category_patterns,
            )
            groups[f"{theme}::{extra}"] = [
                Candidate(title, theme, category, depth, extra)
                for title, (category, depth) in found.items()
            ]
            for title in found:
                reached.setdefault(title, set()).add(theme)
            LOGGER.info("Extra %s (%s): %d títulos", extra, theme, len(found))
    return groups, reached


def build_corpus(client: WikiClient, settings: CorpusSettings) -> CorpusResult:
    """Choose, fetch and clean articles; deterministic given the seed and the API responses."""

    rng = random.Random(settings.seed)
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
        for candidate in sorted(candidates, key=lambda c: c.title):
            if candidate.title in seen_titles:
                continue
            themes = reached.get(candidate.title, {candidate.theme})
            if settings.drop_multi_theme and len(themes) > 1:
                seen_titles.add(candidate.title)
                result.articles.append(_record(candidate, None, themes, "multi_theme"))
                continue
            pool.append(candidate)
        rng.shuffle(pool)
        kept = 0
        position = 0
        while kept < quota and position < len(pool):
            chunk = pool[position : position + settings.batch_titles]
            position += len(chunk)
            pages = client.fetch_pages([c.title for c in chunk], settings.batch_titles)
            for candidate in chunk:
                seen_titles.add(candidate.title)
                if kept >= quota:
                    break
                themes = reached.get(candidate.title, {candidate.theme})
                page = pages.get(candidate.title)
                if page is None:
                    result.articles.append(_record(candidate, None, themes, "missing"))
                    continue
                if page.pageid in seen_pageids:
                    result.articles.append(_record(candidate, page, themes, "duplicate"))
                    continue
                seen_pageids.add(page.pageid)
                if is_disambiguation(page.wikitext):
                    result.articles.append(_record(candidate, page, themes, "disambiguation"))
                    continue
                if has_excluded_infobox(page.wikitext, settings.exclude_infobox_patterns):
                    record = _record(candidate, page, themes, "biography")
                    record["is_biography"] = True
                    result.articles.append(record)
                    continue
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
                    result.articles.append(_record(candidate, page, themes, "no_paragraphs"))
                    continue
                result.articles.append(_record(candidate, page, themes, None))
                for index, text in enumerate(paragraphs):
                    result.paragraphs.append(
                        {
                            "paragraph_id": f"{page.pageid}-{index}",
                            "pageid": page.pageid,
                            "revid": page.revid,
                            "title": page.title,
                            "theme": candidate.theme,
                            "source_category": candidate.source_category,
                            "paragraph_idx": index,
                            "text": text,
                            "sentences": [list(span) for span in split_sentences(text)],
                        }
                    )
                kept += 1
        LOGGER.info("Grupo %s: %d artigos mantidos (cota %d)", group, kept, quota)
    return result


def _record(
    candidate: Candidate, page: Any, themes: set[str], dropped_reason: str | None
) -> dict[str, Any]:
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


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    if paths.paragraphs.exists() and not force:
        LOGGER.info("Corpus já existe em %s; use --force para refazer", paths.corpus_dir)
        return
    started = time.time()
    ensure_dir(paths.corpus_dir)
    client = WikiClient(paths.raw_dir, settings.corpus.user_agent, settings.corpus.delay_s)
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
                "kept": a["dropped_reason"] is None,
                "dropped_reason": a["dropped_reason"] or "",
            }
            for a in result.articles
            if a["pageid"] is not None
        ),
        MANIFEST_FIELDS,
    )
    stats = summarize(result)
    stats.update(network_requests=client.network_requests, cache_hits=client.cache_hits)
    write_manifest(paths.corpus_dir, "corpus", settings, started, stats)
    LOGGER.info("Corpus: %s", stats)
