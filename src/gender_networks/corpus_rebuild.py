"""Command ``corpus-rebuild``: rebuild the corpus from the exact revisions in ``manifest.csv``.

``data/corpus/manifest.csv`` is the versioned record of the corpus: one row per candidate article
that was fetched (``pageid``, ``revid``, title, theme, source category, origin, kept, dropped
reason), in the order ``corpus`` handled them. This command fetches the kept revisions by
``revid`` (so later edits to the articles do not matter), cleans them with the same code as
``corpus`` and in the same order (the paragraph dedup runs across articles), and rewrites
``articles.jsonl``, ``paragraphs.jsonl`` and ``_manifest.json``. ``manifest.csv`` itself is the
input and is never rewritten.

Rows that were not kept come back with the recorded reason, without being fetched. The manifest
records the origin of each title (``root`` or the extra category whose walk reached it), which
``sample`` needs to draw the core from the root categories only; a manifest written before that
column existed rebuilds with ``origin`` set to ``None``. It does not record the rest of the
category crawl, so the rebuilt ``articles.jsonl`` has ``themes_reached`` and ``depth`` set to
``None``, and titles dropped as ``multi_theme`` (which have no ``pageid``) are absent. Paragraph
titles are the page titles the API returns now (an article renamed since the download shows its
new title).

A kept revision the API no longer serves (deleted or hidden) becomes ``missing``, and a kept
article that no longer yields paragraphs keeps the new reason; both are logged and listed in
``_manifest.json`` (``changed``), since they mean the text or the cleaning code changed.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from gender_networks.artifacts import RunPaths, ensure_dir, write_jsonl, write_manifest
from gender_networks.corpus import (
    CorpusResult,
    clean_page,
    offline_from_env,
    paragraph_records,
    summarize,
)
from gender_networks.settings import CorpusSettings, Settings
from gender_networks.wiki import WikiClient

LOGGER = logging.getLogger(__name__)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Rows of ``manifest.csv`` in file order, with typed ids and ``kept``.

    Args:
        path: The versioned manifest.

    Returns:
        One dictionary per row with every column, ``pageid`` and ``revid`` as integers,
        ``kept`` as a boolean (true only for the exact text ``True``), and ``origin`` and
        ``dropped_reason`` as ``None`` when empty.
    """

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            **row,
            "pageid": int(row["pageid"]),
            "revid": int(row["revid"]),
            "origin": row.get("origin") or None,  # absent from manifests written before it
            "kept": row["kept"] == "True",
            "dropped_reason": row["dropped_reason"] or None,
        }
        for row in rows
    ]


def rebuild_corpus(
    client: WikiClient, settings: CorpusSettings, rows: list[dict[str, Any]]
) -> tuple[CorpusResult, list[dict[str, Any]]]:
    """Corpus of the manifest rows, and the kept rows whose outcome changed.

    Kept revisions are fetched by id and cleaned in manifest order with a single dedup set,
    as the ``corpus`` stage did; rows that were not kept are copied with their recorded
    reason and are not fetched.

    Args:
        client: API client.
        settings: The ``corpus`` section.
        rows: Rows from :func:`read_manifest`.

    Returns:
        ``(result, changed)``: the rebuilt corpus, and one ``{"title", "revid", "reason"}``
        entry per kept row that is now missing or dropped by the cleaning.

    Side effects:
        Performs (or reads from the cache) the revision requests and logs a warning for
        every changed row.
    """

    revids = list(dict.fromkeys(row["revid"] for row in rows if row["kept"]))
    pages = {page.revid: page for page in client.fetch_revisions(revids, settings.batch_titles)}
    result = CorpusResult(fetch_batches=-(-len(revids) // settings.batch_titles))
    changed: list[dict[str, Any]] = []
    seen_paragraphs: set[str] = set()
    for row in rows:
        reason = row["dropped_reason"]
        paragraphs: list[str] = []
        page = pages.get(row["revid"]) if row["kept"] else None
        if row["kept"]:
            if page is None:
                reason = "missing"
            else:
                reason, paragraphs = clean_page(page, settings, seen_paragraphs)
            if reason is not None:
                LOGGER.warning("Artigo mantido no manifest virou %s: %s", reason, row["title"])
                changed.append({"title": row["title"], "revid": row["revid"], "reason": reason})
        result.articles.append(
            {
                "pageid": row["pageid"],
                "revid": row["revid"],
                "title": row["title"],
                "theme": row["theme"],
                "themes_reached": None,
                "source_category": row["source_category"],
                "depth": None,
                "origin": row["origin"],
                "is_biography": reason == "biography",
                "dropped_reason": reason,
            }
        )
        if page is not None and reason is None:
            result.paragraphs.extend(
                paragraph_records(
                    page, row["theme"], row["source_category"], row["origin"], paragraphs
                )
            )
    return result, changed


def run(
    settings: Settings,
    paths: RunPaths,
    force: bool = False,
    offline: bool | None = None,
    **_: object,
) -> None:
    """Rebuild ``articles.jsonl`` and ``paragraphs.jsonl`` from ``manifest.csv``.

    Existing outputs are kept unless ``force``: they may hold the original download. The cache
    is keyed by request, and ``corpus`` requests pages by title, so the first rebuild needs the
    network even with a full cache; ``offline`` (default: ``GENDER_NETWORKS_OFFLINE=1``) then
    fails on the first missing response.

    Args:
        settings: Run configuration; the ``corpus`` section is used.
        paths: Paths of the run.
        force: Rebuild even when ``paragraphs.jsonl`` exists.
        offline: Forbid network requests; ``None`` reads the environment variable.
        **_: Ignored; accepted so every stage has the same call.

    Side effects:
        Writes ``articles.jsonl``, ``paragraphs.jsonl`` and ``_manifest.json`` (recorded as
        stage ``corpus``, with ``rebuilt_from`` and ``changed``) in ``corpus_dir``.
        ``manifest.csv`` is only read.
    """

    if paths.paragraphs.exists() and not force:
        LOGGER.info("Corpus já existe em %s; use --force para reconstruir", paths.corpus_dir)
        return
    if offline is None:
        offline = offline_from_env()
    started = time.time()
    rows = read_manifest(paths.corpus_manifest_csv)
    ensure_dir(paths.corpus_dir)
    client = WikiClient(
        paths.raw_dir, settings.corpus.user_agent, settings.corpus.delay_s, offline=offline
    )
    result, changed = rebuild_corpus(client, settings.corpus, rows)
    write_jsonl(paths.articles, result.articles)
    write_jsonl(paths.paragraphs, result.paragraphs)
    stats = summarize(result)
    if any(row["origin"] is None for row in rows):
        stats.pop("kept_by_origin")  # a manifest written before the origin column
    stats.update(
        rebuilt_from=str(paths.corpus_manifest_csv.relative_to(paths.root)),
        manifest_rows=len(rows),
        manifest_kept=sum(row["kept"] for row in rows),
        changed=changed,
        offline=offline,
        fetch_batches=result.fetch_batches,
        network_requests=client.network_requests,
        cache_hits=client.cache_hits,
    )
    write_manifest(paths.corpus_dir, "corpus", settings, started, stats, root=paths.root)
    log = LOGGER.warning if changed else LOGGER.info
    log(
        "Corpus reconstruído: %d de %d artigos mantidos, %d parágrafos",
        stats["articles_kept"],
        stats["manifest_kept"],
        stats["paragraphs"],
    )
