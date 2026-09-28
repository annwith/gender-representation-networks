"""Wikitext cleaning: keep running prose paragraphs and split them into sentences."""

from __future__ import annotations

import re
from collections.abc import Iterable

import mwparserfromhell

_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)
_TEMPLATE_NAME = re.compile(r"\{\{\s*([^|}\n]+)")
_DISAMBIGUATION = ("desambiguação", "desambig", "disambig", "dab")
_CLOSERS = "\"'»”’)]"
# Abbreviations after which a period does not end a sentence (lowercase, without the period).
_ABBREVIATIONS = {
    "sr", "sra", "srs", "dr", "dra", "drs", "prof", "profa", "profs", "sto", "sta", "st", "jr",
    "av", "séc", "sec", "p.ex", "ex", "etc", "a.c", "d.c", "n.º", "nº", "n", "pág", "págs", "p",
    "pp", "vol", "vols", "cap", "caps", "ed", "eds", "fig", "figs", "cf", "aprox", "máx", "mín",
    "s.a", "ltda", "km", "kg", "cm", "mm", "hab", "sq", "gen", "cel", "ten", "sgt", "mr",
    "mrs", "ms", "vs", "op", "cit", "ibid", "apud", "art", "arts", "inc", "col", "cols", "org",
    "trad", "i.e", "e.g", "c", "ca",
}
_INVISIBLE_TAGS = {
    "ref", "references", "math", "chem", "ce", "gallery", "timeline", "score", "syntaxhighlight",
    "source", "pre", "code", "imagemap", "graph", "hiero", "mapframe", "templatedata",
}
_NON_TEXT_LINKS = (
    "ficheiro:", "arquivo:", "imagem:", "file:", "image:", "categoria:", "category:", "media:",
)
_SENTENCE_END = re.compile(r"[.!?…]+[" + re.escape(_CLOSERS) + r"]*(?=\s+)")


def template_names(wikitext: str) -> list[str]:
    """Names of the templates used in the wikitext, normalized to lowercase."""

    return [match.group(1).strip().lower().replace("_", " ") for match in
            _TEMPLATE_NAME.finditer(wikitext)]


def has_excluded_infobox(wikitext: str, patterns: Iterable[str]) -> bool:
    """True when a template name starts with one of the patterns (e.g. ``Info/Biografia``)."""

    wanted = tuple(pattern.strip().lower().replace("_", " ") for pattern in patterns)
    return any(name.startswith(wanted) for name in template_names(wikitext)) if wanted else False


def is_disambiguation(wikitext: str) -> bool:
    return any(name.startswith(_DISAMBIGUATION) for name in template_names(wikitext))


def cut_sections(wikitext: str, section_names: Iterable[str]) -> str:
    """Drop everything from the first heading whose title is one of ``section_names``."""

    names = {name.strip().lower() for name in section_names}
    for match in _HEADING.finditer(wikitext):
        if match.group(2).strip().lower() in names:
            return wikitext[: match.start()]
    return wikitext


def _remove_invisible(code: mwparserfromhell.wikicode.Wikicode) -> None:
    """Drop nodes whose content is not running text (strip_code would keep ref contents)."""

    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() in _INVISIBLE_TAGS:
            try:
                code.remove(tag)
            except ValueError:  # already removed together with an enclosing tag
                pass
    for link in code.filter_wikilinks(recursive=True):
        if str(link.title).strip().lower().startswith(_NON_TEXT_LINKS):
            try:
                code.remove(link)
            except ValueError:
                pass


def wikitext_to_text(wikitext: str) -> str:
    """Plain text with templates, references, tables and markup removed."""

    code = mwparserfromhell.parse(wikitext)
    _remove_invisible(code)
    text = code.strip_code(normalize=True, collapse=True)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if _HEADING.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def split_paragraphs(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def keep_paragraph(paragraph: str, min_words: int, max_digit_ratio: float) -> bool:
    """Running prose only: long enough, ends like a sentence, and not a table of numbers."""

    if len(paragraph.split()) < min_words:
        return False
    if not paragraph.rstrip(_CLOSERS).endswith((".", "!", "?")):
        return False
    if paragraph.startswith(("|", "!", "{", "*", "#", ":", ";")):
        return False
    digits = sum(character.isdigit() for character in paragraph)
    return digits / len(paragraph) <= max_digit_ratio


def _ends_with_abbreviation(text: str) -> bool:
    words = text.rstrip(_CLOSERS).rsplit(None, 1)
    if not words:
        return False
    token = words[-1].lower().rstrip(".!?…")
    token = token.lstrip("(\"'«“")
    if len(token) == 1 and token.isalpha():  # initials such as "J. Smith"
        return True
    return token in _ABBREVIATIONS


def split_sentences(paragraph: str) -> list[tuple[int, int]]:
    """Sentence spans ``[start, end)`` over the paragraph, trimmed of surrounding spaces.

    A sentence ends at ``.``, ``!``, ``?`` or ``…`` (plus closing quotes or brackets) followed by
    whitespace and an uppercase letter, digit, or opening quote, except after abbreviations.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end()
        rest = paragraph[end:].lstrip()
        if not rest:
            break
        first = rest[0]
        if not (first.isupper() or first.isdigit() or first in "\"'«“(—–-"):
            continue
        if paragraph[match.start()] == "." and _ends_with_abbreviation(paragraph[start:end]):
            continue
        spans.append((start, end))
        start = end + (len(paragraph[end:]) - len(rest))
    if start < len(paragraph):
        spans.append((start, len(paragraph.rstrip())))
    return [(s, e) for s, e in spans if e > s]


def clean_article(
    wikitext: str,
    cut: Iterable[str],
    min_words: int,
    max_digit_ratio: float,
) -> list[str]:
    """Kept paragraphs of one article, in order."""

    text = wikitext_to_text(cut_sections(wikitext, cut))
    return [
        paragraph
        for paragraph in split_paragraphs(text)
        if keep_paragraph(paragraph, min_words, max_digit_ratio)
    ]
