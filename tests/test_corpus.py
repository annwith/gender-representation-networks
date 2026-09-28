from dataclasses import replace

from gender_networks.corpus import build_corpus, summarize
from gender_networks.settings import CorpusSettings, ParagraphFilter
from gender_networks.wiki import Page

PROSE = " ".join(["texto"] * 45)


class FakeWiki:
    """In-memory stand-in for WikiClient."""

    def __init__(self, trees, pages) -> None:
        self.trees = trees
        self.pages = pages
        self.fetched: list[str] = []

    def crawl(self, roots, depth, max_titles, skip_patterns=()):
        found = {}
        for root in roots:
            for title in self.trees.get(root, []):
                found.setdefault(title, (root, 0))
        return dict(list(found.items())[:max_titles])

    def fetch_pages(self, titles, batch=50):
        self.fetched.extend(titles)
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
