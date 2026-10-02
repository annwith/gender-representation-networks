import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gender_networks import artifacts, corpus
from gender_networks.artifacts import RunPaths, read_json, read_jsonl
from gender_networks.corpus import build_corpus, summarize
from gender_networks.settings import CorpusSettings, ParagraphFilter, Settings
from gender_networks.wiki import Page

PROSE = " ".join(["texto"] * 45)


class FakeWiki:
    """In-memory stand-in for WikiClient."""

    def __init__(self, trees, pages) -> None:
        self.trees = trees
        self.pages = pages
        self.fetched: list[str] = []
        self.batches: list[list[str]] = []
        self.network_requests = 0
        self.cache_hits = 0
        self.skipped_categories: list[str] = []

    def crawl(self, roots, depth, max_titles, skip_patterns=()):
        found = {}
        for root in roots:
            for title in self.trees.get(root, []):
                found.setdefault(title, (root, 0))
        return dict(list(found.items())[:max_titles])

    def fetch_pages(self, titles, batch=50):
        self.fetched.extend(titles)
        self.batches.append(list(titles))
        return {t: self.pages[t] for t in titles if t in self.pages}


def page(title: str, pageid: int, body: str) -> Page:
    return Page(title=title, pageid=pageid, revid=pageid * 10, wikitext=body)


def settings(**kwargs) -> CorpusSettings:
    base = CorpusSettings(
        themes={"fisica": ["Categoria:Física"], "musica": ["Categoria:Música"]},
        extra_categories={"musica": ["Categoria:Teoria musical"]},
        articles_per_theme=10,
        articles_per_extra_category=10,
        exclude_infobox_patterns=["Info/Biografia"],
        cut_sections=["Referências"],
        paragraph=ParagraphFilter(min_words=40, max_digit_ratio=0.08, dedup_chars=30),
    )
    return replace(base, **kwargs)


def test_build_corpus_applies_theme_and_content_filters() -> None:
    trees = {
        "Categoria:Física": ["Átomo", "Som", "Einstein", "Banco (desambiguação)", "Vazio"],
        "Categoria:Música": ["Som", "Nota"],
        "Categoria:Teoria musical": ["Escala", "Nota"],
    }
    pages = {
        "Átomo": page("Átomo", 1, f"Átomo {PROSE}.\n\nÁtomo {PROSE}.\n\nOutro átomo {PROSE}."),
        "Som": page("Som", 2, f"Som {PROSE}."),
        "Einstein": page("Einstein", 3, "{{Info/Biografia}}\n" + f"Einstein {PROSE}."),
        "Banco (desambiguação)": page("Banco (desambiguação)", 4, "{{desambiguação}}\nX."),
        "Vazio": page("Vazio", 5, "Curto."),
        "Nota": page("Nota", 6, f"Nota musical {PROSE}.\n== Referências ==\nRef {PROSE}."),
        "Escala": page("Escala", 7, f"Escala {PROSE}."),
    }
    wiki = FakeWiki(trees, pages)

    result = build_corpus(wiki, settings())

    reasons = {a["title"]: a["dropped_reason"] for a in result.articles}
    assert reasons["Som"] == "multi_theme"  # reached from física and música
    assert reasons["Einstein"] == "biography"
    assert reasons["Banco (desambiguação)"] == "disambiguation"
    assert reasons["Vazio"] == "no_paragraphs"
    assert reasons["Átomo"] is None and reasons["Nota"] is None and reasons["Escala"] is None
    assert "Som" not in wiki.fetched  # multi-theme titles are never downloaded

    by_title = {}
    for paragraph in result.paragraphs:
        by_title.setdefault(paragraph["title"], []).append(paragraph)
    assert len(by_title["Átomo"]) == 2  # the duplicate opening is removed
    assert [p["paragraph_id"] for p in by_title["Átomo"]] == ["1-0", "1-1"]
    assert len(by_title["Nota"]) == 1  # the section after Referências is cut
    assert by_title["Escala"][0]["source_category"] == "Categoria:Teoria musical"
    assert by_title["Nota"][0]["theme"] == "musica"
    assert all(p["sentences"] for p in result.paragraphs)

    stats = summarize(result)
    assert stats["kept_by_theme"] == {"fisica": 1, "musica": 2}
    assert stats["dropped_by_reason"]["multi_theme"] == 1


def test_build_corpus_is_deterministic_and_respects_quota() -> None:
    titles = [f"Artigo {i}" for i in range(30)]
    trees = {"Categoria:Física": titles, "Categoria:Música": []}
    pages = {t: page(t, i + 1, f"{t} único {i} {PROSE}.") for i, t in enumerate(titles)}
    config = settings(extra_categories={}, articles_per_theme=5, batch_titles=4)

    first = build_corpus(FakeWiki(trees, pages), config)
    second = build_corpus(FakeWiki(trees, pages), config)

    kept = [a["title"] for a in first.articles if a["dropped_reason"] is None]
    assert len(kept) == 5
    assert kept == [a["title"] for a in second.articles if a["dropped_reason"] is None]


def kept_titles(result, origin: str | None = None) -> list[str]:
    return [
        a["title"]
        for a in result.articles
        if a["dropped_reason"] is None and (origin is None or a["origin"] == origin)
    ]


def batches_by_group(monkeypatch, wiki, config):
    """Run build_corpus and split the fetched batches by the group that requested them."""

    original = corpus.group_order
    markers: list[tuple[int, str]] = []

    def tracking(candidates, seed, group):
        markers.append((len(wiki.batches), group))
        return original(candidates, seed, group)

    monkeypatch.setattr(corpus, "group_order", tracking)
    result = build_corpus(wiki, config)
    bounds = markers + [(len(wiki.batches), "")]
    groups = {g: wiki.batches[i:j] for (i, g), (j, _) in zip(bounds, bounds[1:], strict=False)}
    return result, groups


def test_cleaning_changes_in_one_group_do_not_reshuffle_other_groups(monkeypatch) -> None:
    a_titles = [f"A{i:02d}" for i in range(16)]
    b_titles = [f"B{i}" for i in range(6)]
    y_titles = [f"Y{i}" for i in range(10)]
    trees = {
        "Categoria:Física": a_titles,
        "Categoria:Música": b_titles,
        "Categoria:Física aplicada": a_titles[4:],  # overlaps the física root group
        "Categoria:Teoria musical": y_titles,
    }
    ids = {t: i + 1 for i, t in enumerate(a_titles + b_titles + y_titles)}
    good = {t: page(t, ids[t], f"{t} único {PROSE}.") for t in ids}
    # Variant: some física articles lose their paragraphs (as after a cleaning fix), so the
    # física root group reads further into its order and marks more titles as seen.
    worse = dict(good)
    for title in a_titles[::2]:
        worse[title] = page(title, ids[title], "Curto.")
    config = settings(
        extra_categories={
            "fisica": ["Categoria:Física aplicada"],
            "musica": ["Categoria:Teoria musical"],
        },
        articles_per_theme=3,
        articles_per_extra_category=4,
        batch_titles=2,
    )

    first, first_batches = batches_by_group(monkeypatch, FakeWiki(trees, good), config)
    second, second_batches = batches_by_group(monkeypatch, FakeWiki(trees, worse), config)

    # Groups of other themes pick the same articles through the same (cached) batches.
    for group in ("musica::root", "musica::Categoria:Teoria musical"):
        assert first_batches[group] == second_batches[group]
    assert kept_titles(first, "Categoria:Teoria musical") == kept_titles(
        second, "Categoria:Teoria musical"
    )
    # The changed theme's groups reuse their earlier batches and at most append new ones.
    for group in ("fisica::root", "fisica::Categoria:Física aplicada"):
        shorter, longer = sorted((first_batches[group], second_batches[group]), key=len)
        assert longer[: len(shorter)] == shorter
    assert len(second_batches["fisica::root"]) > len(first_batches["fisica::root"])


def test_group_order_is_independent_of_input_order_and_other_groups() -> None:
    candidates = [corpus.Candidate(f"T{i}", "t", "C", 0, "root") for i in range(20)]

    order = corpus.group_order(candidates, 438, "t::root")

    assert order == corpus.group_order(list(reversed(candidates)), 438, "t::root")
    assert order != corpus.group_order(candidates, 438, "u::root")
    assert sorted(c.title for c in order) == sorted(c.title for c in candidates)


def test_candidate_after_quota_is_not_marked_seen() -> None:
    titles = [f"T{i}" for i in range(4)]
    trees = {
        "Categoria:Física": titles,
        "Categoria:Música": [],
        "Categoria:Física aplicada": titles,
    }
    pages = {t: page(t, i + 1, f"{t} único {PROSE}.") for i, t in enumerate(titles)}
    config = settings(
        extra_categories={"fisica": ["Categoria:Física aplicada"]},
        articles_per_theme=1,
        articles_per_extra_category=4,
        batch_titles=4,
    )

    result = build_corpus(FakeWiki(trees, pages), config)

    recorded = [a["title"] for a in result.articles]
    assert sorted(recorded) == titles and len(recorded) == len(set(recorded))
    assert len(kept_titles(result, "root")) == 1
    assert len(kept_titles(result, "Categoria:Física aplicada")) == 3


def test_redirect_duplicate_and_disambiguation_pageprop() -> None:
    trees = {"Categoria:Física": ["Física", "Fisica", "Nota"], "Categoria:Música": []}
    fisica = page("Física", 1, f"Física {PROSE}.")
    pages = {
        "Física": fisica,
        "Fisica": fisica,  # redirect resolved to the same page
        "Nota": Page("Nota", 2, 20, f"Nota {PROSE}.", disambiguation=True),
    }

    result = build_corpus(FakeWiki(trees, pages), settings(extra_categories={}))

    reasons = sorted((a["title"], a["dropped_reason"]) for a in result.articles)
    assert ("Nota", "disambiguation") in reasons
    assert sorted(r for t, r in reasons if t in {"Física", "Fisica"} and r) == ["duplicate"]


@pytest.fixture
def corpus_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trees = {
        "Categoria:Física": ["Átomo", "Einstein"],
        "Categoria:Música": ["Nota"],
        "Categoria:Teoria musical": ["Escala"],
    }
    pages = {
        "Átomo": page("Átomo", 1, f"Átomo {PROSE}. Segunda frase {PROSE}."),
        "Einstein": page("Einstein", 3, "{{Info/Biografia}}\n" + f"Einstein {PROSE}."),
        "Nota": page("Nota", 6, f"Nota musical {PROSE}."),
        "Escala": page("Escala", 7, f"Escala {PROSE}."),
    }
    created: list[dict] = []

    def fake_client(cache_dir, user_agent, delay_s, offline=False):
        created.append({"cache_dir": cache_dir, "offline": offline})
        return FakeWiki(trees, pages)

    monkeypatch.setattr(corpus, "WikiClient", fake_client)
    monkeypatch.setattr(artifacts, "library_versions", lambda: {})
    monkeypatch.delenv(corpus.OFFLINE_ENV, raising=False)
    config = Settings(name="teste", corpus=settings())
    return config, RunPaths.from_settings(config, tmp_path), created


def test_run_writes_documented_outputs_and_manifest(corpus_run) -> None:
    config, paths, created = corpus_run

    corpus.run(config, paths)

    articles = list(read_jsonl(paths.articles))
    documented = {
        "pageid", "revid", "title", "theme", "themes_reached", "source_category", "depth",
        "is_biography", "dropped_reason",
    }  # fmt: skip
    assert all(documented <= set(a) for a in articles)
    assert {a["title"]: a["dropped_reason"] for a in articles}["Einstein"] == "biography"
    paragraphs = list(read_jsonl(paths.paragraphs))
    assert set(paragraphs[0]) == {
        "paragraph_id", "pageid", "revid", "title", "theme", "source_category", "paragraph_idx",
        "text", "sentences",
    }  # fmt: skip
    assert paragraphs[0]["sentences"] == [list(s) for s in map(tuple, paragraphs[0]["sentences"])]
    with paths.corpus_manifest_csv.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == corpus.MANIFEST_FIELDS and len(rows) == 4
    manifest = read_json(paths.corpus_dir / "_manifest.json")
    assert manifest["stage"] == "corpus" and manifest["articles_kept"] == 3
    assert manifest["offline"] is False and manifest["skipped_categories"] == []
    assert manifest["fetch_batches"] >= 3
    assert created == [{"cache_dir": paths.raw_dir, "offline": False}]


def test_run_skips_existing_outputs_unless_forced(corpus_run, monkeypatch) -> None:
    config, paths, created = corpus_run
    corpus.run(config, paths)
    before = paths.paragraphs.read_text(encoding="utf-8")

    corpus.run(config, paths)
    assert len(created) == 1  # skipped: no client was even built

    monkeypatch.setenv(corpus.OFFLINE_ENV, "1")
    corpus.run(config, paths, force=True)
    assert len(created) == 2 and created[-1]["offline"] is True
    assert paths.paragraphs.read_text(encoding="utf-8") == before
    assert json.loads((paths.corpus_dir / "_manifest.json").read_text())["offline"] is True
