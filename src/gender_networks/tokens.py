"""Token labelling from character offsets, and script classes of vocabulary entries.

Categories are derived from the exact paragraph slice of each token (its character offsets),
never from decoding a single id: byte-level BPE pieces of one character can decode to U+FFFD,
while their offsets still point at the character they belong to. Words are maximal runs of
Unicode word characters, so "tornou-se" has the words "tornou" and "se", and the token "-se" is
the whole word "se".

Deviation from the stage specification (deliberate): number tokens are assigned to the word
they overlap, like letter tokens, so they count in ``word_n_tokens`` and ``pos_in_word``. A
letter piece glued to digits is therefore not a whole word: in "1990s" the "s" is a
continuation (word "1990s" of five tokens), in "H2O" the "O" is a continuation. Treating "s" or
"O" as whole words would put suffix and formula pieces among the whole-word vertices of P1/P4.
The number tokens themselves keep the category ``number``.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Categories of a token inside a paragraph. The order matters: ``CATEGORY_CODES`` is used by
# the sampler to store categories compactly, and ``whole_word`` comes last so that argmax over
# category counts breaks ties against it (a type is a whole word only by a strict majority).
CATEGORIES = ("whitespace", "punctuation", "number", "continuation", "word_start", "whole_word")
CATEGORY_CODES = {name: code for code, name in enumerate(CATEGORIES)}
WORD_CATEGORIES = frozenset({"whole_word", "word_start", "continuation"})

SCRIPT_CLASSES = (
    "latin",
    "cjk",
    "mixed",
    "punctuation",
    "other_script",
    "byte_fragment",
    "whitespace",
    "digits",
    "special",
)

# Portuguese function words: articles, prepositions and their contractions, pronouns,
# conjunctions, frequent adverbs, quantifiers and auxiliary/modal verb forms. It extends the
# STOP set of the feasibility study (scripts/feasibility/simulate_sample.py). Target and control
# words (estado, primeiro, grande, ...) are deliberately absent.
STOPWORDS = frozenset(
    """
    a à ao aos as às o os um uma uns umas
    ante após até com contra de desde em entre para perante por sem sob sobre trás per
    da das do dos dum duma duns dumas na nas no nos num numa nuns numas
    pela pelas pelo pelos pra pro pras pros
    deste desta destes destas disto desse dessa desses dessas disso
    daquele daquela daqueles daquelas daquilo
    neste nesta nestes nestas nisto nesse nessa nesses nessas nisso
    naquele naquela naqueles naquelas naquilo àquele àquela àqueles àquelas àquilo
    dele dela deles delas nele nela neles nelas daí dali donde aonde
    eu tu ele ela nós vós eles elas me te se lhe lhes vos lo la los las
    mim ti si comigo contigo consigo conosco convosco
    meu minha meus minhas teu tua teus tuas seu sua seus suas
    nosso nossa nossos nossas vosso vossa vossos vossas
    este esta estes estas isto esse essa esses essas isso aquele aquela aqueles aquelas aquilo
    que quem qual quais cujo cuja cujos cujas onde quanto quanta quantos quantas
    algum alguma alguns algumas nenhum nenhuma outro outra outros outras
    todo toda todos todas tudo nada algo alguém ninguém cada mesmo mesma mesmos mesmas
    tal tais vários várias muitos muitas muito muita pouco pouca poucos poucas
    tanto tanta tantos tantas qualquer quaisquer demais ambos ambas
    e ou mas porém contudo todavia entretanto pois porque porquanto como quando embora
    conforme enquanto logo portanto nem senão
    não sim mais menos bem apenas cerca também já ainda assim então lá aqui aí ali cá
    sempre nunca só somente quase tão depois antes durante mediante através além
    vez parte você vocês
    ser sou é és somos são era eras éramos eram fui foi fomos foram fora será serão seria
    seriam seja sejam sejamos fosse fossem for forem sido sendo serei seremos
    estar estou está estás estamos estão estava estavam estive esteve estivemos estiveram
    estará estarão estaria estariam esteja estejam estivesse estivessem estiver estiverem
    estando
    ter tenho tem tens temos têm tinha tinham tive teve tivemos tiveram terá terão teria
    teriam tenha tenham tivesse tivessem tiver tiverem tido tendo terei teremos
    haver há havia haviam houve houveram haverá haveria haja hajam havendo havido hão
    houver houverem
    pode podem podia podiam pôde puderam poderá poderão poderia poderiam possa possam
    vai vão ia iam
    """.split()
)

_WORD = re.compile(r"\w+")
_CJK_NAMES = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")


def is_function_word(word: str) -> bool:
    """True when ``word`` (any case, surrounding spaces ignored) is a Portuguese function word."""

    return word.strip().lower() in STOPWORDS


def _script_of(character: str) -> str:
    name = unicodedata.name(character, "")
    return name.split(" ", 1)[0] if name else "UNKNOWN"


def script_class(decoded_text: str, is_special: bool = False) -> str:
    """Writing-system class of one decoded vocabulary entry.

    The checks run in a fixed order so each entry gets exactly one class; the letter checks use
    the text without surrounding whitespace, since most entries carry a leading space.
    """

    if is_special:
        return "special"
    if "�" in decoded_text:
        return "byte_fragment"
    text = decoded_text.strip()
    if not text:
        return "whitespace"
    if all(character.isdigit() for character in text):
        return "digits"
    if not any(character.isalnum() for character in text):
        return "punctuation"
    if all(character.isalpha() for character in text):
        names = [unicodedata.name(character, "") for character in text]
        if all("LATIN" in name for name in names):
            return "latin"
        if all(any(key in name for key in _CJK_NAMES) for name in names):
            return "cjk"
        if len({_script_of(character) for character in text}) == 1:
            return "other_script"
    return "mixed"


def special_token_ids(tokenizer: Any) -> set[int]:
    """Special ids: the declared special tokens plus every added token of the tokenizer."""

    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids.update(int(key) for key in (getattr(tokenizer, "added_tokens_decoder", {}) or {}))
    return ids


def decode_vocabulary(tokenizer: Any) -> list[str]:
    """Decoded text of every tokenizer id ``0..len(tokenizer)-1``."""

    return [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))]


def vocab_script_classes(tokenizer: Any, decoded: Sequence[str] | None = None) -> list[str]:
    """Script class of every tokenizer id ``0..len(tokenizer)-1``."""

    special = special_token_ids(tokenizer)
    texts = decoded if decoded is not None else decode_vocabulary(tokenizer)
    return [script_class(text, token_id in special) for token_id, text in enumerate(texts)]


@dataclass(frozen=True)
class TokenInfo:
    """One token of a paragraph, with its word and category derived from character offsets.

    ``word``/``word_idx``/``word_n_tokens``/``pos_in_word`` are empty (``""``, -1, 0, -1) for
    whitespace and punctuation tokens. Number tokens keep the word they belong to (so "1990" is
    one word of four number tokens), which also makes the letters after digits ("1990s") a
    continuation instead of a separate whole word.
    """

    index: int
    token_id: int
    char_start: int
    char_end: int
    text: str
    has_leading_space: bool
    category: str
    word: str
    word_idx: int
    word_n_tokens: int
    pos_in_word: int
    is_function_word: bool


@dataclass(frozen=True)
class _Words:
    spans: list[tuple[int, int]]
    texts: list[str]
    starts: list[int]


def _find_words(text: str) -> _Words:
    spans, texts = [], []
    for match in _WORD.finditer(text):
        spans.append((match.start(), match.end()))
        texts.append(match.group())
    return _Words(spans, texts, [start for start, _ in spans])


def _best_word(text: str, words: _Words, start: int, end: int) -> int:
    """Index of the word sharing the most alphanumeric characters with ``text[start:end]``."""

    best, best_overlap = -1, 0
    index = max(bisect_right(words.starts, start) - 1, 0)
    while index < len(words.spans) and words.spans[index][0] < end:
        word_start, word_end = words.spans[index]
        low, high = max(start, word_start), min(end, word_end)
        if high > low:
            overlap = sum(character.isalnum() for character in text[low:high])
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        index += 1
    return best


def token_categories(
    text: str, offsets: Sequence[tuple[int, int]]
) -> tuple[list[str], list[int], list[int], list[int], _Words]:
    """Category, word index, rank in the word and word length of each token.

    This is the light-weight core of :func:`label_tokens`, used directly when only the
    categories of a whole corpus are needed.
    """

    words = _find_words(text)
    base: list[str] = []
    word_of: list[int] = []
    for start, end in offsets:
        piece = text[start:end]
        if not piece.strip():
            base.append("whitespace")
            word_of.append(-1)
            continue
        alnum = [character for character in piece if character.isalnum()]
        if not alnum:
            base.append("punctuation")
            word_of.append(-1)
            continue
        index = _best_word(text, words, start, end)
        if index < 0:  # defensive: every alphanumeric character belongs to a word
            base.append("punctuation")
            word_of.append(-1)
            continue
        base.append("number" if all(character.isdigit() for character in alnum) else "word")
        word_of.append(index)

    members: dict[int, list[int]] = {}
    for token_index, word_index in enumerate(word_of):
        if word_index >= 0:
            members.setdefault(word_index, []).append(token_index)
    rank = [-1] * len(offsets)
    size = [0] * len(offsets)
    for tokens in members.values():
        for position, token_index in enumerate(tokens):
            rank[token_index] = position
            size[token_index] = len(tokens)

    categories: list[str] = []
    for token_index, kind in enumerate(base):
        if kind != "word":
            categories.append(kind)
        elif size[token_index] == 1:
            categories.append("whole_word")
        elif rank[token_index] == 0:
            categories.append("word_start")
        else:
            categories.append("continuation")
    return categories, word_of, rank, size, words


def label_tokens(
    text: str, token_ids: Sequence[int], offsets: Sequence[tuple[int, int]]
) -> list[TokenInfo]:
    """Build :class:`TokenInfo` records from token ids and their character offsets."""

    if len(token_ids) != len(offsets):
        raise ValueError("token_ids and offsets must have the same length")
    categories, word_of, rank, size, words = token_categories(text, offsets)
    infos = []
    for index, (token_id, (start, end)) in enumerate(zip(token_ids, offsets, strict=True)):
        piece = text[start:end]
        word_index = word_of[index]
        word = words.texts[word_index] if word_index >= 0 else ""
        category = categories[index]
        infos.append(
            TokenInfo(
                index=index,
                token_id=int(token_id),
                char_start=int(start),
                char_end=int(end),
                text=piece,
                has_leading_space=piece[:1].isspace(),
                category=category,
                word=word,
                word_idx=word_index,
                word_n_tokens=size[index],
                pos_in_word=rank[index],
                is_function_word=category in WORD_CATEGORIES and is_function_word(word),
            )
        )
    return infos


def encode_with_offsets(
    tokenizer: Any, texts: Sequence[str]
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Ids and character offsets of each text, without special tokens (fast tokenizer only)."""

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Character offsets need a fast tokenizer")
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    return [
        (list(ids), [(int(a), int(b)) for a, b in offsets])
        for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
    ]


def tokenize_paragraph(tokenizer: Any, text: str) -> list[TokenInfo]:
    """Tokenize one paragraph (no special tokens) and label each token."""

    ((ids, offsets),) = encode_with_offsets(tokenizer, [text])
    return label_tokens(text, ids, offsets)


def tokenize_paragraphs(
    tokenizer: Any, texts: Iterable[str], batch_size: int = 256
) -> list[list[TokenInfo]]:
    """Batched :func:`tokenize_paragraph` (the fast tokenizer encodes batches in parallel)."""

    items = list(texts)
    result: list[list[TokenInfo]] = []
    for begin in range(0, len(items), batch_size):
        chunk = items[begin : begin + batch_size]
        for text, (ids, offsets) in zip(chunk, encode_with_offsets(tokenizer, chunk), strict=True):
            result.append(label_tokens(text, ids, offsets))
    return result
