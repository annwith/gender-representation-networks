"""Wikitext cleaning: keep running prose paragraphs and split them into sentences."""

from __future__ import annotations

import re
from collections.abc import Iterable

import mwparserfromhell
from mwparserfromhell.nodes import Template

_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)
_TEMPLATE_NAME = re.compile(r"\{\{\s*([^|}\n]+)")
# Exact names: prefixes would also match hatnotes such as {{Dablink}} on regular articles.
_DISAMBIGUATION = {"desambiguação", "desambiguacao", "desambig", "disambig", "dab"}
_CLOSERS = "\"'»”’)]"
# Abbreviations after which a period does not end a sentence (lowercase, without the period).
# Unit symbols (km, kg, m...) are absent on purpose: they never take a period in Portuguese, so
# "300 km. Sua" is a sentence end.
_ABBREVIATIONS = {
    "sr", "sra", "srs", "dr", "dra", "drs", "prof", "profa", "profs", "sto", "sta", "st", "jr",
    "av", "séc", "sec", "p.ex", "ex", "etc", "a.c", "d.c", "n.º", "nº", "n", "pág", "págs", "p",
    "pp", "vol", "vols", "cap", "caps", "ed", "eds", "fig", "figs", "cf", "aprox", "máx", "mín",
    "s.a", "ltda", "hab", "sq", "gen", "cel", "ten", "sgt", "mr", "mrs", "ms", "vs", "op", "cit",
    "ibid", "apud", "art", "arts", "inc", "col", "cols", "org", "trad", "i.e", "e.g", "c", "ca",
}  # fmt: skip
# Block content that is not running text: removed without a trace.
_REMOVED_TAGS = {
    "ref", "references", "gallery", "timeline", "syntaxhighlight", "source", "pre", "imagemap",
    "graph", "mapframe", "templatedata", "table",
}  # fmt: skip
# Inline content that cannot be rendered as words: the paragraph that contains it is dropped,
# because removing it would leave a hole in the sentence ("a energia é dada por , onde...").
# Trivial formulas (a symbol or a number, e.g. <math>v</math>) are kept as plain text instead.
_POISON_TAGS = {"math", "chem", "ce", "score", "hiero"}
_PLAIN_FORMULA = re.compile(r"[A-Za-z]{1,2}\d?|\d+(?:[.,]\d+)?")
# Wiki list and indentation markup (*, #, :, ;): list items are not running prose.
_LIST_TAGS = {"li", "dd", "dt"}
_NON_TEXT_LINKS = (
    "ficheiro:", "arquivo:", "imagem:", "file:", "image:", "categoria:", "category:", "media:",
)  # fmt: skip
# Inline templates whose visible text is one of their arguments (the rest are dropped).
_FIRST_ARG_TEMPLATES = {
    "formatnum", "fmtn", "fmt", "nowrap", "nobr", "small", "sm", "versalete", "smallcaps",
    "nobreak", "número", "numero",
}  # fmt: skip
_CONVERT_TEMPLATES = {"converter", "convert", "conv", "cvt"}
_POISON = "\ue000"  # private-use character; the parser drops NUL
_SENTENCE_END = re.compile(r"[.!?…]+[" + re.escape(_CLOSERS) + r"]*(?=\s+)")


def template_names(wikitext: str) -> list[str]:
    """Names of the templates used in the wikitext, normalized to lowercase."""

    return [
        match.group(1).strip().lower().replace("_", " ")
        for match in _TEMPLATE_NAME.finditer(wikitext)
    ]


def has_excluded_infobox(wikitext: str, patterns: Iterable[str]) -> bool:
    """True when a template name starts with one of the patterns (e.g. ``Info/Biografia``)."""

    wanted = tuple(pattern.strip().lower().replace("_", " ") for pattern in patterns)
    return any(name.startswith(wanted) for name in template_names(wikitext)) if wanted else False


def is_disambiguation(wikitext: str, title: str = "") -> bool:
    """Disambiguation page: a disambiguation template or a ``(desambiguação)`` title."""

    if title.strip().lower().endswith("(desambiguação)"):
        return True
    return any(name in _DISAMBIGUATION for name in template_names(wikitext))


def cut_sections(wikitext: str, section_names: Iterable[str]) -> str:
    """Remove the sections titled with one of ``section_names`` (and their subsections).

    A removed section ends at the next heading of the same or a higher level, so content
    sections after it (e.g. a music article's ``== Notas ==`` followed by ``== História ==``)
    are kept.
    """

    names = {name.strip().lower() for name in section_names}
    pieces: list[str] = []
    position = 0
    cutting_level: int | None = None
    for match in _HEADING.finditer(wikitext):
        level = len(match.group(1))
        if cutting_level is not None and level > cutting_level:
            continue  # subsection of a removed section
        if cutting_level is None:
            pieces.append(wikitext[position : match.start()])
        cutting_level = None
        if match.group(2).strip().lower() in names:
            cutting_level = level
        else:
            position = match.start()
    if cutting_level is None:
        pieces.append(wikitext[position:])
    return "".join(pieces)


def _positional(template: Template) -> list[str]:
    return [str(param.value).strip() for param in template.params if not param.showkey]


def _template_text(template: Template) -> str | None:
    """Visible text of a whitelisted inline template, or None to let strip_code drop it."""

    name = str(template.name).strip().lower().replace("_", " ")
    if name.startswith("formatnum:"):
        return str(template.name).split(":", 1)[1].strip()
    args = _positional(template)
    if not args:
        return None
    if name in _FIRST_ARG_TEMPLATES:
        return args[0]
    if name in {"lang", "llang", "langx"} or name.startswith("lang-"):
        return args[-1]
    if name in _CONVERT_TEMPLATES:
        unit = args[1] if len(args) > 1 and not _is_number(args[1]) else ""
        return f"{args[0]} {unit}".strip()
    return None


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"[\d.,]+", text))


def _safe_replace(code: mwparserfromhell.wikicode.Wikicode, node: object, value: str) -> None:
    try:
        code.replace(node, value)
    except ValueError:  # already removed together with an enclosing node
        pass


def _prepare(code: mwparserfromhell.wikicode.Wikicode) -> None:
    """Rewrite nodes that strip_code would render badly (it keeps ref and table contents,
    glues words around ``<br>`` and drops text-bearing templates)."""

    for template in reversed(code.filter_templates(recursive=True)):
        text = _template_text(template)
        if text is not None:
            _safe_replace(code, template, text)
    for tag in code.filter_tags(recursive=True):
        name = str(tag.tag).strip().lower()
        if name in _REMOVED_TAGS:
            _safe_replace(code, tag, "")
        elif name in _POISON_TAGS:
            content = str(tag.contents or "").strip() if name == "math" else ""
            plain = _PLAIN_FORMULA.fullmatch(content) is not None
            _safe_replace(code, tag, content if plain else _POISON)
        elif name == "br":
            _safe_replace(code, tag, " ")
        elif name in _LIST_TAGS and tag.wiki_markup:
            _safe_replace(code, tag, _POISON)
    for link in code.filter_wikilinks(recursive=True):
        if str(link.title).strip().lower().startswith(_NON_TEXT_LINKS):
            _safe_replace(code, link, "")


def _tidy(line: str) -> str:
    """Remove the residue that dropped templates and references leave in a line."""

    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\(\s*[\s,;:–—-]*\)", "", line)  # "( )", "(; )"
    line = re.sub(r"\[\s*\]", "", line)
    line = re.sub(r"\(\s*[,;:]\s*", "(", line)  # "(; grego)" -> "(grego)"
    line = re.sub(r"\s*[,;:]\s*\)", ")", line)
    line = re.sub(r"\s+([.,;:!?…)\]»”])", r"\1", line)  # "pequeno ." -> "pequeno."
    line = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", line)  # ", ," -> ","
    return re.sub(r"\s+", " ", line).strip()


def wikitext_to_text(wikitext: str) -> str:
    """Plain text with templates, references, tables, lists and markup removed.

    Lines that contained inline math or chemistry, or were list items, are dropped entirely.
    """

    code = mwparserfromhell.parse(wikitext)
    _prepare(code)
    text = code.strip_code(normalize=True, collapse=True)
    lines = []
    for line in text.splitlines():
        if _POISON in line or _HEADING.match(line.strip()):
            continue
        lines.append(_tidy(line))
    return "\n".join(lines)


def split_paragraphs(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def keep_paragraph(paragraph: str, min_words: int, max_digit_ratio: float) -> bool:
    """Running prose only: long enough, ends like a sentence, and not a table of numbers."""

    if len(paragraph.split()) < min_words:
        return False
    if not paragraph.rstrip(_CLOSERS).endswith((".", "!", "?")):
        return False
    if paragraph.startswith(("|", "!", "{", "*", "#", ":", ";")):  # markup residue
        return False
    digits = sum(character.isdigit() for character in paragraph)
    return digits / len(paragraph) <= max_digit_ratio


def _ends_with_abbreviation(text: str) -> bool:
    words = text.rstrip(_CLOSERS).rsplit(None, 1)
    if not words:
        return False
    raw = words[-1].rstrip(".!?…").lstrip("(\"'«“")
    if len(raw) == 1 and raw.isalpha() and raw.isupper():  # initials such as "J. Smith"
        return True
    return raw.lower() in _ABBREVIATIONS


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
