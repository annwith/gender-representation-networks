"""Portuguese labels and pt-BR numbers shared by the figures, the tables and the web pages.

Codes in the artifacts are English (``whole_word``, ``core``, ``fisica``); what a reader sees is
Portuguese, and it comes from here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Representations are shown by their name (lex, L01, L36n...); only the vocabulary is translated.
REP_LABELS = {"vocab": "vocabulário"}
CATEGORY_ORDER = ["whole_word", "word_start", "continuation", "punctuation", "number"]
CATEGORY_LABELS = {
    "whole_word": "palavra inteira",
    "word_start": "início de palavra",
    "continuation": "continuação",
    "punctuation": "pontuação",
    "number": "número",
    "whitespace": "espaço",
    "all": "todos",
}
STRATUM_ORDER = ["core", "target", "control", "multitheme"]
STRATUM_LABELS = {
    "core": "núcleo",
    "target": "alvo",
    "control": "controle",
    "multitheme": "multitema",
    "all": "todos",
}
GROUP_ORDER = ["target", "control", "multitheme", "function"]
GROUP_LABELS = {
    "target": "alvos",
    "control": "controles",
    "multitheme": "multitema",
    "function": "funcionais",
    "other": "outros",
}
SCRIPT_ORDER = [
    "latin",
    "cjk",
    "mixed",
    "punctuation",
    "other_script",
    "byte_fragment",
    "whitespace",
    "digits",
    "special",
]
SCRIPT_LABELS = {
    "latin": "latina",
    "cjk": "chinês/japonês/coreano",
    "mixed": "código/misto",
    "punctuation": "pontuação",
    "other_script": "outras escritas",
    "byte_fragment": "pedaços de bytes",
    "whitespace": "espaços",
    "digits": "dígitos",
    "special": "especiais",
    "other": "outras",
}
THEME_LABELS = {
    "fisica": "Física",
    "biologia": "Biologia",
    "economia": "Economia",
    "politica": "Política",
    "computacao": "Computação",
    "musica": "Música",
    "esporte": "Esporte",
    "geografia": "Geografia",
}
REASON_ORDER = ["multi_theme", "biography", "disambiguation", "no_paragraphs", "duplicate"]
REASON_LABELS = {
    "multi_theme": "multitema",
    "biography": "biografia",
    "disambiguation": "desambiguação",
    "no_paragraphs": "sem parágrafos",
    "duplicate": "duplicada",
}
LABEL_LABELS = {
    "token_id": "token",
    "theme": "tema",
    "pageid": "artigo",
    "paragraph_id": "parágrafo",
    "sentence_id": "sentença",
    "next_token_id": "próximo token",
    "pred_next": "token previsto",
    "pos_bucket": "faixa de posição",
    "stratum": "estrato",
    "token_category": "categoria",
}


def label(mapping: Mapping[str, str], key: Any) -> str:
    """Portuguese label of a code, falling back to the code itself."""

    return mapping.get(str(key), str(key))


def theme_label(theme: str) -> str:
    return THEME_LABELS.get(theme, theme.capitalize())


def band_display(band: str) -> str:
    """``10-49`` -> ``10–49`` (en dash), ``50`` -> ``50``."""

    return str(band).replace("-", "–")


def ordered(values: Sequence[Any], order: Sequence[str]) -> list[str]:
    """Distinct values, known ones in ``order`` first, the others sorted after them."""

    present = [str(v) for v in dict.fromkeys(values)]
    known = [v for v in order if v in present]
    return known + sorted(v for v in present if v not in known)


# pt-BR numbers


def fmt_int(value: float | int) -> str:
    """``15104`` -> ``15.104``."""

    return f"{int(round(float(value))):,}".replace(",", ".")


def fmt_dec(value: float, digits: int = 2) -> str:
    """``1234.5`` -> ``1.234,50`` (thousands with a dot, decimal comma)."""

    text = f"{float(value):,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")
