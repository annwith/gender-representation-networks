"""Stage ``sample``: the hybrid occurrence sample (plan decisions 4 and 6).

Vertices are token occurrences. Whole paragraphs alone leave too few repeated content words for
P4, so the sample has four strata, selected in this order with one seeded generator:

1. ``core``: per theme, a few articles and a few paragraphs of each, with every non-whitespace
   token as a vertex. The per-type cap ``f_max`` is applied by uniform random subsampling of the
   over-cap types after pooling, so the kept occurrences are spread over the paragraphs instead
   of being the first ones to arrive.
2. ``target``: polysemous target words, with a quota per sense theme, drawn from non-core
   paragraphs under per-paragraph and per-article limits. A target that does not reach
   ``min_per_sense`` occurrences in at least two of its themes loses its drawn occurrences.
3. ``control``: the same procedure for the control words.
4. ``multitheme``: content words that occur in several themes, filled round-robin over themes.

In the sparse strata only the drawn positions are vertices; the rest of the paragraph is
context. Each paragraph becomes one sequence ``[prefix] + tokens``, truncated after its last
selected position (the model is causal, so later tokens cannot change earlier states).

A studied word (target, control or multitheme type) counts only where its id is the whole word
(plan decision 4: ``Ġbanco``, not a prefix): ``' capital' + 'ismo'`` is the word "capitalismo",
so that position is never drawn, never fills a quota, never makes a type multitheme and gets an
empty ``target_word``. Core vertices keep every token, whatever its category.
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
import time
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from gender_networks.artifacts import (
    OCCURRENCE_COLUMNS,
    VOCAB_COLUMNS,
    RunPaths,
    ensure_dir,
    position_bucket,
    read_jsonl,
    stage_is_fresh,
    write_csv,
    write_jsonl,
    write_manifest,
)
from gender_networks.settings import ModelSettings, SampleSettings, Settings
from gender_networks.tokens import (
    CATEGORIES,
    CATEGORY_CODES,
    TokenInfo,
    decode_vocabulary,
    encode_with_offsets,
    is_function_word,
    script_class,
    special_token_ids,
    token_categories,
    tokenize_paragraphs,
)

LOGGER = logging.getLogger(__name__)

SENSES_COLUMNS = ["occurrence_id", "target_word", "sense_theme", "title", "local_context", "sense"]
STRATA = ("core", "target", "control", "multitheme")
# English function words and bibliography pieces that pass the alphabetic filter of the
# multitheme stratum (citations, inline English titles); they are not Portuguese content words.
ENGLISH_STOPWORDS = frozenset(
    """
    the and from for with that this which are was were has have had not but its his her their
    into over under between about after before other than then there these those what when who
    how also can will would may one two new
    www http https html htm org net pdf isbn doi issn retrieved edition vol pp
    """.split()
)
# Documented deviations from the stage specification, copied into the manifest.
DEVIATIONS = [
    "target_word/sense_theme are set only when the target or control id is a whole word; "
    "word_start pieces (' capital' + 'ismo') are not the word",
    "target, control and multitheme draws, core quota counts and multitheme eligibility use "
    "whole-word positions only",
    "multitheme types must also be lowercase and outside ENGLISH_STOPWORDS (no proper nouns, "
    "no English citation words)",
    "number tokens join the word they belong to, so letters after digits are continuations "
    "('1990s' -> 's' continuation, 'H2O' -> 'O' continuation)",
    "paragraphs whose text encodes to a special id never enter the sample (still counted in "
    "f_corpus)",
]
MIN_SENSE_THEMES = 2  # a target must reach min_per_sense in at least this many quota themes
CONTEXT_TOKENS = 8
WHITESPACE = CATEGORY_CODES["whitespace"]
WHOLE_WORD = CATEGORY_CODES["whole_word"]

T = TypeVar("T")


# ---------------------------------------------------------------------------------------------
# Corpus tokenization
# ---------------------------------------------------------------------------------------------


@dataclass
class CorpusParagraph:
    """A kept corpus paragraph with its token ids and compact category codes."""

    index: int
    record: Mapping[str, Any]
    ids: np.ndarray  # int64 token ids of the paragraph (no special tokens)
    codes: np.ndarray  # int8 codes of tokens.CATEGORIES

    @property
    def theme(self) -> str:
        return str(self.record["theme"])

    @property
    def pageid(self) -> int:
        return int(self.record["pageid"])

    def order_key(self, theme_rank: Mapping[str, int]) -> tuple[int, int, int]:
        return (
            theme_rank.get(self.theme, len(theme_rank)),
            self.pageid,
            int(self.record["paragraph_idx"]),
        )

    def candidates(self) -> np.ndarray:
        """Positions of non-whitespace tokens, the only positions that can become vertices."""

        return np.flatnonzero(self.codes != WHITESPACE)


def tokenize_corpus(
    paragraphs: Sequence[Mapping[str, Any]], tokenizer: Any, batch_size: int = 256
) -> list[CorpusParagraph]:
    """Token ids and categories of every paragraph (batched, offsets-based categories)."""

    result: list[CorpusParagraph] = []
    for begin in range(0, len(paragraphs), batch_size):
        chunk = paragraphs[begin : begin + batch_size]
        encoded = encode_with_offsets(tokenizer, [record["text"] for record in chunk])
        for record, (ids, offsets) in zip(chunk, encoded, strict=True):
            categories = token_categories(record["text"], offsets)[0]
            result.append(
                CorpusParagraph(
                    index=len(result),
                    record=record,
                    ids=np.asarray(ids, dtype=np.int64),
                    codes=np.asarray([CATEGORY_CODES[c] for c in categories], dtype=np.int8),
                )
            )
    return result


def corpus_frequencies(corpus: Sequence[CorpusParagraph], vocab_size: int) -> np.ndarray:
    """Count of every token id over all paragraphs of the corpus (whitespace tokens included)."""

    if not corpus:
        return np.zeros(vocab_size, dtype=np.int64)
    return np.bincount(np.concatenate([p.ids for p in corpus]), minlength=vocab_size).astype(
        np.int64
    )


# ---------------------------------------------------------------------------------------------
# Frequency bands and small helpers
# ---------------------------------------------------------------------------------------------


def validate_bands(bands: Sequence[Sequence[int]], f_max: int) -> None:
    """Bands must be contiguous from 1 and cover ``f_max`` (every f_sample needs a label)."""

    expected = 1
    for band in bands:
        if len(band) != 2 or band[0] != expected or band[1] < band[0]:
            raise ValueError(f"sample.bands must be contiguous intervals from 1, got {bands}")
        expected = band[1] + 1
    if not bands or bands[-1][1] < f_max:
        raise ValueError(f"sample.bands must cover f_max = {f_max}, got {bands}")


def band_label(value: int, bands: Sequence[Sequence[int]], open_top: bool = False) -> str:
    """Label of the frequency band of ``value`` (``'1-2'``, ``'50'``; ``'50+'`` if open)."""

    for number, (low, high) in enumerate(bands):
        last = number == len(bands) - 1
        if last and open_top and value >= low:
            return f"{low}+"
        if low <= value <= high:
            return f"{low}" if low == high else f"{low}-{high}"
    raise ValueError(f"Frequency {value} is outside the bands {bands}")


def single_token_ids(
    tokenizer: Any, words: Sequence[str]
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Id of ``' ' + word`` for each word that is one token; the others with their pieces."""

    ids: dict[str, int] = {}
    rejected: dict[str, list[int]] = {}
    for word in words:
        encoded = list(tokenizer.encode(" " + word, add_special_tokens=False))
        if len(encoded) == 1:
            ids[word] = int(encoded[0])
        else:
            rejected[word] = [int(token_id) for token_id in encoded]
    return ids, rejected


def sentence_index(spans: Sequence[Sequence[int]], char_start: int, token_text: str) -> int:
    """Index of the sentence span holding the token's first non-space character.

    Sentence spans are trimmed, so the space before a sentence belongs to no span; the last span
    starting at or before the character is used, which also covers trailing punctuation.
    """

    if not spans:
        return 0
    position = char_start + (len(token_text) - len(token_text.lstrip()))
    starts = [int(span[0]) for span in spans]
    return max(bisect_right(starts, position) - 1, 0)


def local_context(text: str, infos: Sequence[TokenInfo], index: int) -> str:
    """Up to 8 tokens before and after, with the token between « and ».

    Built from character offsets rather than by decoding ids, so byte-split characters at the
    window edges never show up as U+FFFD. A leading space stays outside the marks.
    """

    low = max(0, index - CONTEXT_TOKENS)
    high = min(len(infos) - 1, index + CONTEXT_TOKENS)
    token = infos[index]
    left = text[infos[low].char_start : token.char_start] if index > low else ""
    right = text[token.char_end : infos[high].char_end] if high > index else ""
    lead = token.text[: len(token.text) - len(token.text.lstrip())]
    return f"{left}{lead}«{token.text[len(lead):]}»{right}"


def is_content_type(decoded: str) -> bool:
    """Whether a decoded vocabulary entry can be a multitheme content word.

    Alphabetic, at least three letters, lowercase (capitalized entries are proper nouns or
    sentence starts, a different type anyway), not a Portuguese function word and not an
    English word or bibliography piece left in the text by citations.
    """

    text = decoded.strip()
    return (
        text.isalpha()
        and len(text) >= 3
        and text == text.lower()
        and not is_function_word(text)
        and text not in ENGLISH_STOPWORDS
    )


def cap_uniform(
    groups: Mapping[int, Sequence[T]], f_max: int, rng: random.Random
) -> tuple[list[T], int, int]:
    """Keep at most ``f_max`` items per key, drawn uniformly among that key's items.

    Keys are visited in sorted order so the result depends only on the generator state.
    Returns the kept items, the number of capped keys and the number of removed items.
    """

    kept: list[T] = []
    capped = removed = 0
    for key in sorted(groups):
        items = list(groups[key])
        if len(items) > f_max:
            capped += 1
            removed += len(items) - f_max
            items = rng.sample(items, f_max)
        kept.extend(items)
    return kept, capped, removed


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Everything the stage writes (except ``vocab_types.csv``) and the manifest statistics."""

    occurrences: list[dict[str, Any]]
    sequences: list[dict[str, Any]]
    senses: list[dict[str, Any]]
    f_corpus: np.ndarray  # int64 [len(tokenizer)]
    f_sample: Counter[int]
    stats: dict[str, Any] = field(default_factory=dict)


class _Selector:
    """Mutable selection state shared by the four strata."""

    def __init__(
        self,
        corpus: list[CorpusParagraph],
        settings: SampleSettings,
        sample_themes: list[str],
        rng: random.Random,
    ) -> None:
        self.corpus = corpus
        self.settings = settings
        self.limits = settings.limits
        self.f_max = settings.f_max
        self.rng = rng
        self.sample_themes = sample_themes
        self.core: dict[int, set[int]] = {}  # paragraph -> kept core positions
        self.strata: dict[int, dict[int, str]] = {}  # paragraph -> position -> stratum
        self.total: Counter[int] = Counter()  # id -> selected occurrences
        self.per_paragraph: Counter[tuple[int, int]] = Counter()  # (id, paragraph)
        self.per_article: Counter[tuple[int, int]] = Counter()  # (id, pageid), core included
        # (id, theme) whole-word occurrences in the core, which count toward the quotas
        self.core_theme: Counter[tuple[int, str]] = Counter()
        self.drawn: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
        self.index: dict[int, dict[int, list[int]]] = {}  # id -> non-core paragraph -> positions
        self.core_articles: dict[str, list[int]] = {}
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        LOGGER.warning(message)
        self.warnings.append(message)

    # -- core --------------------------------------------------------------------------------

    def select_core(self) -> dict[str, Any]:
        core = self.settings.core
        by_article: dict[str, dict[int, list[int]]] = {
            theme: defaultdict(list) for theme in self.sample_themes
        }
        for paragraph in self.corpus:
            if paragraph.theme in by_article:
                by_article[paragraph.theme][paragraph.pageid].append(paragraph.index)
        chosen: list[int] = []
        for theme in self.sample_themes:
            articles = by_article[theme]
            eligible = sorted(
                pageid
                for pageid, members in articles.items()
                if len(members) >= core.paragraphs_per_article
            )
            if len(eligible) < core.articles_per_theme:
                self.warn(
                    f"Tema {theme}: só {len(eligible)} artigos com ≥ "
                    f"{core.paragraphs_per_article} parágrafos (pedidos {core.articles_per_theme})"
                )
            pageids = sorted(self.rng.sample(eligible, min(core.articles_per_theme, len(eligible))))
            self.core_articles[theme] = pageids
            for pageid in pageids:
                members = sorted(
                    articles[pageid], key=lambda i: int(self.corpus[i].record["paragraph_idx"])
                )
                picked = self.rng.sample(members, core.paragraphs_per_article)
                chosen.extend(sorted(picked, key=members.index))

        pooled: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for p in chosen:
            paragraph = self.corpus[p]
            for j in paragraph.candidates().tolist():
                pooled[int(paragraph.ids[j])].append((p, j))
        kept, capped_types, removed = cap_uniform(pooled, self.f_max, self.rng)
        self.core = {p: set() for p in chosen}
        for p, j in kept:
            self.core[p].add(j)
            paragraph = self.corpus[p]
            token_id = int(paragraph.ids[j])
            self.total[token_id] += 1
            self.per_paragraph[(token_id, p)] += 1
            self.per_article[(token_id, paragraph.pageid)] += 1
            if paragraph.codes[j] == WHOLE_WORD:
                self.core_theme[(token_id, paragraph.theme)] += 1
        LOGGER.info(
            "Núcleo: %d parágrafos, %d vértices (%d tipos acima de f_max, %d ocorrências cortadas)",
            len(chosen),
            len(kept),
            capped_types,
            removed,
        )
        return {
            "core_paragraphs": len(chosen),
            "core_articles": {theme: ids for theme, ids in self.core_articles.items()},
            "core_vertices_before_cap": sum(len(v) for v in pooled.values()),
            "core_vertices": len(kept),
            "core_capped_types": capped_types,
            "core_removed_by_cap": removed,
        }

    # -- indexes of non-core paragraphs ------------------------------------------------------

    def noncore_paragraphs(self) -> list[CorpusParagraph]:
        themes = set(self.sample_themes)
        return [p for p in self.corpus if p.theme in themes and p.index not in self.core]

    def build_index(self, token_ids: set[int]) -> None:
        """Whole-word positions of the given ids in every non-core sample paragraph.

        Only whole-word positions can be drawn: a word_start piece belongs to a longer word.
        """

        wanted = np.asarray(sorted(token_ids - set(self.index)), dtype=np.int64)
        for token_id in wanted.tolist():
            self.index[token_id] = {}
        if wanted.size == 0:
            return
        for paragraph in self.noncore_paragraphs():
            mask = np.isin(paragraph.ids, wanted) & (paragraph.codes == WHOLE_WORD)
            for j in np.flatnonzero(mask).tolist():
                self.index[int(paragraph.ids[j])].setdefault(paragraph.index, []).append(j)

    def queue(self, token_id: int, theme: str) -> deque[int]:
        """Shuffled non-core paragraphs of ``theme`` that contain ``token_id``."""

        members = sorted(p for p in self.index[token_id] if self.corpus[p].theme == theme)
        self.rng.shuffle(members)
        return deque(members)

    # -- drawing under the diversity limits --------------------------------------------------

    def _available(self, token_id: int, p: int) -> list[int]:
        taken = self.strata.get(p, {})
        return [j for j in self.index[token_id].get(p, []) if j not in taken]

    def _room(self, token_id: int, p: int) -> int:
        return min(
            self.limits.per_type_paragraph - self.per_paragraph[(token_id, p)],
            self.limits.per_type_article - self.per_article[(token_id, self.corpus[p].pageid)],
            self.f_max - self.total[token_id],
        )

    def take(self, token_id: int, p: int, k: int, stratum: str) -> int:
        """Draw up to ``k`` occurrences of ``token_id`` from paragraph ``p``."""

        allowed = min(k, self._room(token_id, p))
        if allowed <= 0:
            return 0
        available = self._available(token_id, p)
        chosen = sorted(self.rng.sample(available, min(allowed, len(available))))
        pageid = self.corpus[p].pageid
        for j in chosen:
            self.strata.setdefault(p, {})[j] = stratum
            self.total[token_id] += 1
            self.per_paragraph[(token_id, p)] += 1
            self.per_article[(token_id, pageid)] += 1
            self.drawn[(token_id, stratum)].append((p, j))
        return len(chosen)

    def fill(self, token_id: int, queue: deque[int], need: int, stratum: str) -> int:
        """Draw up to ``need`` occurrences from the paragraphs in ``queue`` (consumed).

        A paragraph stays at the head of the queue only while it can still give more, so a
        round-robin caller asking for one occurrence at a time can come back to it.
        """

        got = 0
        while got < need and queue and self.total[token_id] < self.f_max:
            p = queue[0]
            got += self.take(token_id, p, need - got, stratum)
            if got < need or self._room(token_id, p) <= 0 or not self._available(token_id, p):
                queue.popleft()
        return got

    def undo(self, token_id: int, stratum: str) -> int:
        """Remove every draw of ``token_id`` made for ``stratum``."""

        draws = self.drawn.pop((token_id, stratum), [])
        for p, j in draws:
            del self.strata[p][j]
            if not self.strata[p]:
                del self.strata[p]
            self.total[token_id] -= 1
            self.per_paragraph[(token_id, p)] -= 1
            self.per_article[(token_id, self.corpus[p].pageid)] -= 1
        return len(draws)

    # -- quota strata (targets and controls) ------------------------------------------------

    def select_quota_stratum(
        self, table: Mapping[str, Mapping[str, int]], word_ids: Mapping[str, int], stratum: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Fill the per-theme quotas of each word and apply the min_per_sense rule."""

        report: dict[str, Any] = {}
        dropped: list[dict[str, Any]] = []
        kept: list[str] = []
        sample_themes = set(self.sample_themes)
        self.build_index({word_ids[w] for w in table if w in word_ids})
        for word, quotas in table.items():
            if word not in word_ids:
                continue
            token_id = word_ids[word]
            themes: dict[str, dict[str, Any]] = {}
            for theme, quota in quotas.items():
                if theme not in sample_themes:
                    self.warn(f"{stratum} {word}: tema {theme} fora dos temas da amostra")
                    themes[theme] = {
                        "requested": quota, "core": 0, "drawn": 0, "obtained": 0,
                        "in_sample": False,
                    }
                    continue
                core_count = self.core_theme[(token_id, theme)]
                need = min(quota - core_count, self.f_max - self.total[token_id])
                got = 0
                if need > 0:
                    got = self.fill(token_id, self.queue(token_id, theme), need, stratum)
                themes[theme] = {
                    "requested": quota, "core": core_count, "drawn": got,
                    "obtained": core_count + got, "in_sample": True,
                }
            reached = [
                theme
                for theme, info in themes.items()
                if info["in_sample"] and info["obtained"] >= self.settings.min_per_sense
            ]
            keep = len(reached) >= MIN_SENSE_THEMES
            if keep:
                kept.append(word)
            else:
                removed = self.undo(token_id, stratum)
                dropped.append(
                    {"word": word, "themes_reaching_min": reached, "removed_occurrences": removed}
                )
                self.warn(
                    f"{stratum} {word} descartada: só {len(reached)} tema(s) com ≥ "
                    f"{self.settings.min_per_sense} ocorrências"
                )
            report[word] = {"token_id": token_id, "kept": keep, "themes": themes}
        return report, dropped, kept

    # -- multitheme ----------------------------------------------------------------------------

    def theme_counts(self, vocab_size: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Raw and drawable whole-word occurrences of each id per theme, over non-core paragraphs.

        Only whole-word positions count, since only those can be drawn. Drawable counts apply
        the per-paragraph and per-article limits, so they say how many occurrences the
        round-robin fill can actually take from a theme.
        """

        limits = self.limits
        raw = {theme: np.zeros(vocab_size, dtype=np.int64) for theme in self.sample_themes}
        drawable = {theme: np.zeros(vocab_size, dtype=np.int64) for theme in self.sample_themes}
        by_article: dict[tuple[str, int], list[CorpusParagraph]] = defaultdict(list)
        for paragraph in self.noncore_paragraphs():
            by_article[(paragraph.theme, paragraph.pageid)].append(paragraph)
        for (theme, _), members in by_article.items():
            uniques, counts = [], []
            for paragraph in members:
                values, value_counts = np.unique(
                    paragraph.ids[paragraph.codes == WHOLE_WORD], return_counts=True
                )
                np.add.at(raw[theme], values, value_counts)
                uniques.append(values)
                counts.append(np.minimum(value_counts, limits.per_type_paragraph))
            if not any(len(values) for values in uniques):
                continue
            article_ids, inverse = np.unique(np.concatenate(uniques), return_inverse=True)
            sums = np.bincount(inverse, weights=np.concatenate(counts)).astype(np.int64)
            np.add.at(drawable[theme], article_ids, np.minimum(sums, limits.per_type_article))
        return raw, drawable

    def majority_categories(self, vocab_size: int) -> np.ndarray:
        """Most frequent category code of each id over the pool (all sample-theme paragraphs).

        Ties go to the category listed first in ``tokens.CATEGORIES``, so ``whole_word`` (last)
        wins only with more occurrences than any other category; ids never seen get -1.
        """

        n_categories = len(CATEGORIES)
        themes = set(self.sample_themes)
        keys = [
            p.ids[p.codes != WHITESPACE] * n_categories + p.codes[p.codes != WHITESPACE]
            for p in self.corpus
            if p.theme in themes
        ]
        counts = np.bincount(
            np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64),
            minlength=vocab_size * n_categories,
        ).reshape(vocab_size, n_categories)
        majority = counts.argmax(axis=1)
        majority[counts.sum(axis=1) == 0] = -1
        return majority

    def select_multitheme(
        self, tokenizer: Any, excluded: set[int], vocab_size: int
    ) -> dict[str, Any]:
        settings = self.settings.multitheme
        raw, drawable = self.theme_counts(vocab_size)
        majority = self.majority_categories(vocab_size)

        def eligible_ids(counts: dict[str, np.ndarray]) -> list[int]:
            spread = (np.stack([counts[t] for t in self.sample_themes]) >= settings.min_per_theme)
            candidates = np.flatnonzero(
                (spread.sum(axis=0) >= settings.min_themes) & (majority == WHOLE_WORD)
            )
            result = []
            for token_id in candidates.tolist():
                if token_id in excluded:
                    continue
                if is_content_type(tokenizer.decode([token_id])):
                    result.append(token_id)
            return result

        eligible = eligible_ids(drawable)
        eligible_raw = len(eligible_ids(raw))
        if len(eligible) < settings.n_types:
            self.warn(f"Multitema: só {len(eligible)} tipos elegíveis (pedidos {settings.n_types})")
        chosen = sorted(self.rng.sample(eligible, min(settings.n_types, len(eligible))))
        self.build_index(set(chosen))
        goal = min(settings.per_type, self.f_max)
        types: list[dict[str, Any]] = []
        saturated = 0
        for token_id in chosen:
            themes = [
                t for t in self.sample_themes if drawable[t][token_id] >= settings.min_per_theme
            ]
            before = self.total[token_id]
            need = goal - before
            got = 0
            if need <= 0:
                saturated += 1
            else:
                queues = {theme: self.queue(token_id, theme) for theme in themes}
                active = list(themes)
                while got < need and active:
                    for theme in list(active):
                        if got >= need:
                            break
                        taken = self.fill(token_id, queues[theme], 1, "multitheme")
                        got += taken
                        if taken == 0:
                            active.remove(theme)
            types.append(
                {
                    "token_id": token_id,
                    "text": tokenizer.decode([token_id]),
                    "themes": themes,
                    "before": before,
                    "drawn": got,
                }
            )
        LOGGER.info(
            "Multitema: %d elegíveis, %d sorteados, %d já saturados, %d ocorrências",
            len(eligible),
            len(chosen),
            saturated,
            sum(t["drawn"] for t in types),
        )
        return {
            "multitheme_eligible": len(eligible),
            "multitheme_eligible_raw_counts": eligible_raw,
            "multitheme_drawn_types": len(chosen),
            "multitheme_saturated_types": saturated,
            "multitheme_occurrences_drawn": sum(t["drawn"] for t in types),
            "multitheme_types": types,
        }


# ---------------------------------------------------------------------------------------------
# Assembly of the artifacts
# ---------------------------------------------------------------------------------------------


def assign_prefix_groups(
    positions: Sequence[tuple[int, int]], sequences: Sequence[Sequence[int]]
) -> list[int]:
    """Group vertices whose sequences share the identical prefix up to and including them.

    ``positions`` holds ``(sequence_id, pos_in_sequence)`` per vertex and ``sequences`` the
    input ids indexed by sequence id. Prefixes are interned in a trie (one node per distinct
    prefix), so the cost is linear in the tokens. Groups are numbered by their first member;
    vertices with a unique prefix get -1.
    """

    node_of: dict[tuple[int, int], int] = {}
    nodes_of_sequence: dict[int, list[int]] = {}
    vertex_nodes: list[int] = []
    for sequence_id, position in positions:
        nodes = nodes_of_sequence.get(sequence_id)
        if nodes is None:
            nodes, node = [], -1
            for token_id in sequences[sequence_id]:
                node = node_of.setdefault((node, int(token_id)), len(node_of))
                nodes.append(node)
            nodes_of_sequence[sequence_id] = nodes
        vertex_nodes.append(nodes[position])
    sizes = Counter(vertex_nodes)
    group_of: dict[int, int] = {}
    groups: list[int] = []
    for node in vertex_nodes:
        if sizes[node] < 2:
            groups.append(-1)
            continue
        groups.append(group_of.setdefault(node, len(group_of)))
    return groups


def order_paragraphs(
    corpus: Sequence[CorpusParagraph],
    selection: Mapping[int, Mapping[int, str]],
    core: set[int],
    corpus_themes: Sequence[str],
) -> list[int]:
    """Core paragraphs first, then the strata paragraphs, each by (theme, pageid, index)."""

    theme_rank = {theme: rank for rank, theme in enumerate(corpus_themes)}

    def key(p: int) -> tuple[int, int, int]:
        return corpus[p].order_key(theme_rank)

    with_vertices = [p for p, chosen in selection.items() if chosen]
    first = sorted((p for p in with_vertices if p in core), key=key)
    second = sorted((p for p in with_vertices if p not in core), key=key)
    return first + second


def assemble_sample(
    corpus: Sequence[CorpusParagraph],
    ordered: Sequence[int],
    selection: Mapping[int, Mapping[int, str]],
    tokenizer: Any,
    prefix_id: int,
    word_of_id: Mapping[int, str],
    f_corpus: np.ndarray,
    bands: Sequence[Sequence[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Occurrence rows and sequences, in the final order, with every column filled."""

    texts = [corpus[p].record["text"] for p in ordered]
    labelled = tokenize_paragraphs(tokenizer, texts)
    rows: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for sequence_id, (p, infos) in enumerate(zip(ordered, labelled, strict=True)):
        paragraph = corpus[p]
        record = paragraph.record
        ids = paragraph.ids.tolist()
        if [info.token_id for info in infos] != ids:
            raise RuntimeError(f"Tokenization of paragraph {record['paragraph_id']} changed")
        chosen = selection[p]
        input_ids = [int(prefix_id)] + ids[: max(chosen) + 1]
        sequences.append(
            {
                "sequence_id": sequence_id,
                "paragraph_id": record["paragraph_id"],
                "input_ids": input_ids,
            }
        )
        spans = record.get("sentences") or []
        for j in sorted(chosen):
            info = infos[j]
            position = j + 1
            # a studied id is its word only as a whole word (" capital" + "ismo" is not it)
            word = word_of_id.get(info.token_id, "") if info.category == "whole_word" else ""
            rows.append(
                {
                    "occurrence_id": len(rows),
                    "stratum": chosen[j],
                    "theme": paragraph.theme,
                    "source_category": record.get("source_category", ""),
                    "pageid": paragraph.pageid,
                    "revid": record.get("revid", ""),
                    "title": record.get("title", ""),
                    "paragraph_id": record["paragraph_id"],
                    "paragraph_idx": int(record["paragraph_idx"]),
                    "sentence_id": (
                        f"{record['paragraph_id']}-s"
                        f"{sentence_index(spans, info.char_start, info.text)}"
                    ),
                    "sequence_id": sequence_id,
                    "pos_in_sequence": position,
                    "pos_bucket": position_bucket(position),
                    "prefix_group": -1,
                    "token_id": info.token_id,
                    "token_text": info.text,
                    "char_start": info.char_start,
                    "char_end": info.char_end,
                    "has_leading_space": info.has_leading_space,
                    "prev_token_id": input_ids[position - 1],  # the prefix id at position 1
                    "next_token_id": ids[j + 1] if j + 1 < len(ids) else -1,
                    "word": info.word,
                    "word_idx": info.word_idx,
                    "word_n_tokens": info.word_n_tokens,
                    "pos_in_word": info.pos_in_word,
                    "token_category": info.category,
                    "is_function_word": info.is_function_word,
                    "f_sample": 0,
                    "band_sample": "",
                    "f_corpus": int(f_corpus[info.token_id]),
                    "band_corpus": band_label(int(f_corpus[info.token_id]), bands, open_top=True),
                    "target_word": word,
                    "sense_theme": paragraph.theme if word else "",
                    "local_context": local_context(record["text"], infos, j),
                }
            )
    counts = Counter(int(row["token_id"]) for row in rows)
    for row in rows:
        row["f_sample"] = counts[row["token_id"]]
        row["band_sample"] = band_label(row["f_sample"], bands)
    groups = assign_prefix_groups(
        [(row["sequence_id"], row["pos_in_sequence"]) for row in rows],
        [sequence["input_ids"] for sequence in sequences],
    )
    for row, group in zip(rows, groups, strict=True):
        row["prefix_group"] = group
    return rows, sequences


def build_senses(
    occurrences: Sequence[Mapping[str, Any]],
    sense_themes: Mapping[str, Sequence[str]],
    per_target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Annotation template: up to ``per_target`` occurrences per word, balanced over themes.

    Each theme's occurrences are shuffled and the themes are visited round-robin, so a weak
    sense gets as many rows as a strong one until it runs out.
    """

    rows: list[dict[str, Any]] = []
    for word, themes in sense_themes.items():
        pools = {theme: [] for theme in themes}
        for row in occurrences:
            if row["target_word"] == word and row["sense_theme"] in pools:
                pools[row["sense_theme"]].append(row)
        for pool in pools.values():
            rng.shuffle(pool)
        picked: list[Mapping[str, Any]] = []
        while len(picked) < per_target and any(pools.values()):
            for theme in themes:
                if pools[theme] and len(picked) < per_target:
                    picked.append(pools[theme].pop())
        for row in sorted(picked, key=lambda r: int(r["occurrence_id"])):
            rows.append(
                {
                    "occurrence_id": row["occurrence_id"],
                    "target_word": word,
                    "sense_theme": row["sense_theme"],
                    "title": row["title"],
                    "local_context": row["local_context"],
                    "sense": "",
                }
            )
    return rows


def validate_sample(
    occurrences: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    f_max: int,
    prefix_id: int | None = None,
) -> None:
    """Invariants every later stage relies on; raises ValueError on the first violation."""

    counts = Counter(int(row["token_id"]) for row in occurrences)
    over = {token_id: count for token_id, count in counts.items() if count > f_max}
    if over:
        raise ValueError(f"Types above f_max = {f_max}: {sorted(over.items())[:10]}")
    for index, sequence in enumerate(sequences):
        if int(sequence["sequence_id"]) != index:
            raise ValueError(f"sequence_id {sequence['sequence_id']} != its index {index}")
        if prefix_id is not None and int(sequence["input_ids"][0]) != prefix_id:
            raise ValueError(f"Sequence {index} does not start with the prefix id {prefix_id}")
    last_position: dict[int, int] = {}
    previous = (-1, -1)
    for row_index, row in enumerate(occurrences):
        if int(row["occurrence_id"]) != row_index:
            raise ValueError(f"occurrence_id {row['occurrence_id']} != row index {row_index}")
        category = row["token_category"]
        if category not in CATEGORIES or category == "whitespace":
            raise ValueError(f"Occurrence {row_index} has category {category!r}")
        if row["stratum"] not in STRATA:
            raise ValueError(f"Occurrence {row_index} has stratum {row['stratum']!r}")
        sequence_id, position = int(row["sequence_id"]), int(row["pos_in_sequence"])
        if position < 1:
            raise ValueError(f"Occurrence {row_index} sits at position {position}")
        if not 0 <= sequence_id < len(sequences):
            raise ValueError(f"Occurrence {row_index} refers to unknown sequence {sequence_id}")
        input_ids = sequences[sequence_id]["input_ids"]
        if position >= len(input_ids) or int(input_ids[position]) != int(row["token_id"]):
            raise ValueError(f"Occurrence {row_index}: token_id differs from input_ids")
        if (sequence_id, position) <= previous:
            raise ValueError("Occurrences must be ordered by (sequence_id, pos_in_sequence)")
        if int(row["f_sample"]) != counts[int(row["token_id"])]:
            raise ValueError(f"Occurrence {row_index}: wrong f_sample")
        previous = (sequence_id, position)
        last_position[sequence_id] = position
    for index, sequence in enumerate(sequences):
        if index not in last_position:
            raise ValueError(f"Sequence {index} has no vertex")
        if len(sequence["input_ids"]) != last_position[index] + 1:
            raise ValueError(f"Sequence {index} is not truncated after its last vertex")


# ---------------------------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------------------------


def _counts(values: Sequence[Any]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items())}


def _quota_report(
    report: Mapping[str, Any], final: Counter[tuple[int, str]]
) -> dict[str, dict[str, Any]]:
    """Requested and obtained occurrences per (word, theme), with the count in the sample."""

    out: dict[str, dict[str, Any]] = {}
    for word, info in report.items():
        themes = {}
        for theme, entry in info["themes"].items():
            themes[theme] = {**entry, "final": int(final[(info["token_id"], theme)])}
        out[word] = {"token_id": info["token_id"], "kept": info["kept"], "themes": themes}
    return out


def build_sample(
    paragraphs: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    sample: SampleSettings,
    corpus_themes: Sequence[str],
    prefix_id: int,
) -> SampleResult:
    """Select the hybrid sample from the corpus paragraphs; deterministic given the seed.

    ``paragraphs`` are records of ``paragraphs.jsonl``; ``corpus_themes`` is the theme order of
    ``settings.corpus.themes`` (used for the sample themes by default and for ordering).
    """

    validate_bands(sample.bands, sample.f_max)
    rng = random.Random(sample.seed)
    vocab_size = len(tokenizer)
    special = sorted(special_token_ids(tokenizer))

    started = time.time()
    tokenized = tokenize_corpus(paragraphs, tokenizer)
    f_corpus = corpus_frequencies(tokenized, vocab_size)
    LOGGER.info(
        "Corpus tokenizado: %d parágrafos, %d tokens em %.1f s",
        len(tokenized),
        int(f_corpus.sum()),
        time.time() - started,
    )
    # A paragraph whose text encodes to a special id (a literal "<|endoftext|>", say) would put
    # a special token among the vertices or the context, so it never enters the sample.
    usable = [p for p in tokenized if not np.isin(p.ids, special).any()]
    corpus = [replace(p, index=index) for index, p in enumerate(usable)]
    sample_themes = list(sample.themes or corpus_themes)

    target_ids, target_rejected = single_token_ids(tokenizer, list(sample.targets))
    control_ids, control_rejected = single_token_ids(tokenizer, list(sample.controls))
    shared = sorted(w for w, i in control_ids.items() if i in set(target_ids.values()))
    for word in shared:  # one id cannot be both a target and a control
        del control_ids[word]
    rejected = {**target_rejected, **control_rejected}
    for word, pieces in rejected.items():
        LOGGER.warning("Palavra %s não é um token só (%s); fica de fora", word, pieces)

    selector = _Selector(corpus, sample, sample_themes, rng)
    stats: dict[str, Any] = {"sample_themes": sample_themes}
    stats.update(selector.select_core())
    target_report, dropped_targets, kept_targets = selector.select_quota_stratum(
        sample.targets, target_ids, "target"
    )
    control_report, dropped_controls, kept_controls = selector.select_quota_stratum(
        sample.controls, control_ids, "control"
    )
    excluded = set(target_ids.values()) | set(control_ids.values())
    multitheme = selector.select_multitheme(tokenizer, excluded, vocab_size)

    selection: dict[int, dict[int, str]] = defaultdict(dict)
    for p, positions in selector.core.items():
        for j in positions:
            selection[p][j] = "core"
    for p, positions in selector.strata.items():
        selection[p].update(positions)
    ordered = order_paragraphs(corpus, selection, set(selector.core), corpus_themes)
    word_of_id = {token_id: word for word, token_id in {**target_ids, **control_ids}.items()}
    occurrences, sequences = assemble_sample(
        corpus, ordered, selection, tokenizer, prefix_id, word_of_id, f_corpus, sample.bands
    )
    validate_sample(occurrences, sequences, sample.f_max, prefix_id)

    sense_themes = {
        word: [t for t in sample.targets[word] if t in set(sample_themes)] for word in kept_targets
    }
    senses = build_senses(occurrences, sense_themes, sample.senses_per_target, rng)

    f_sample: Counter[int] = Counter(int(row["token_id"]) for row in occurrences)
    final = Counter((int(row["token_id"]), row["theme"]) for row in occurrences)
    groups = Counter(int(row["prefix_group"]) for row in occurrences if row["prefix_group"] >= 0)
    stats.update(
        {
            "n_vertices": len(occurrences),
            "n_types": len(f_sample),
            "n_sequences": len(sequences),
            "tokens_to_process": sum(len(s["input_ids"]) for s in sequences),
            "corpus_paragraphs": len(tokenized),
            "corpus_tokens": int(f_corpus.sum()),
            "paragraphs_with_special_tokens": len(tokenized) - len(usable),
            "prefix_id": int(prefix_id),
            "by_stratum": _counts([row["stratum"] for row in occurrences]),
            "by_theme": _counts([row["theme"] for row in occurrences]),
            "by_band_sample": _counts([row["band_sample"] for row in occurrences]),
            "by_band_corpus": _counts([row["band_corpus"] for row in occurrences]),
            "by_token_category": _counts([row["token_category"] for row in occurrences]),
            "by_pos_bucket": _counts([row["pos_bucket"] for row in occurrences]),
            "function_word_vertices": sum(bool(r["is_function_word"]) for r in occurrences),
            "types_f_sample_ge_3": sum(count >= 3 for count in f_sample.values()),
            "types_f_sample_ge_10": sum(count >= 10 for count in f_sample.values()),
            "types_at_f_max": sum(count == sample.f_max for count in f_sample.values()),
            "prefix_groups": len(groups),
            "prefix_group_largest": max(groups.values(), default=0),
            "prefix_group_vertices": sum(groups.values()),
            "targets": _quota_report(target_report, final),
            "controls": _quota_report(control_report, final),
            "kept_targets": kept_targets,
            "kept_controls": kept_controls,
            "dropped_targets": dropped_targets,
            "dropped_controls": dropped_controls,
            "not_single_token": rejected,
            "controls_sharing_target_ids": shared,
            "senses_rows": len(senses),
            "studied_id_pieces": sum(
                int(row["token_id"]) in word_of_id and row["token_category"] != "whole_word"
                for row in occurrences
            ),
            "deviations": DEVIATIONS,
            "warnings": selector.warnings,
            **multitheme,
        }
    )
    LOGGER.info(
        "Amostra: %d vértices, %d tipos, %d sequências, %d tokens a processar; estratos %s",
        stats["n_vertices"],
        stats["n_types"],
        stats["n_sequences"],
        stats["tokens_to_process"],
        stats["by_stratum"],
    )
    return SampleResult(occurrences, sequences, senses, f_corpus, f_sample, stats)


# ---------------------------------------------------------------------------------------------
# Vocabulary table and I/O
# ---------------------------------------------------------------------------------------------


def vocab_rows(
    tokenizer: Any, f_corpus: np.ndarray, f_sample: Mapping[int, int]
) -> list[dict[str, Any]]:
    """One row per tokenizer id ``0..len(tokenizer)-1`` (see ``artifacts.VOCAB_COLUMNS``)."""

    special = special_token_ids(tokenizer)
    rows = []
    for token_id, text in enumerate(decode_vocabulary(tokenizer)):
        is_special = token_id in special
        frequency = int(f_corpus[token_id]) if token_id < len(f_corpus) else 0
        rows.append(
            {
                "token_id": token_id,
                "token_repr": repr(text),
                "script_class": script_class(text, is_special),
                "is_special": is_special,
                "f_corpus": frequency,
                "f_sample": int(f_sample.get(token_id, 0)),
                "in_corpus": frequency > 0,
            }
        )
    return rows


def load_tokenizer(model: ModelSettings) -> Any:
    """Fast tokenizer of the configured model (imported lazily to keep the CLI light)."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, revision=model.revision)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{model.name_or_path} has no fast tokenizer (offsets are required)")
    return tokenizer


def resolve_prefix_id(tokenizer: Any, prefix_token: str) -> int:
    """Id of the prefix token, refusing a silent fallback to the unknown token."""

    prefix_id = tokenizer.convert_tokens_to_ids(prefix_token)
    if prefix_id is None or tokenizer.convert_ids_to_tokens(int(prefix_id)) != prefix_token:
        raise ValueError(f"The tokenizer has no token {prefix_token!r}")
    return int(prefix_id)


def _backup_annotations(path: Path) -> None:
    """Keep a copy of a senses file that already holds manual annotations before replacing it."""

    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        annotated = any((row.get("sense") or "").strip() for row in csv.DictReader(stream))
    if annotated:
        backup = path.with_name(f"{path.stem}.annotated-{int(time.time())}{path.suffix}")
        shutil.copy2(path, backup)
        LOGGER.warning("%s tinha anotações; cópia guardada em %s", path, backup)


def sample_inputs(paths: RunPaths) -> list[Path]:
    """Upstream files whose change makes the sample stale."""

    return [paths.paragraphs, paths.corpus_manifest]


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    outputs = [paths.occurrences, paths.sequences, paths.vocab_types, paths.senses]
    inputs = sample_inputs(paths)
    if not force and stage_is_fresh(paths.sample_dir, settings, inputs, outputs):
        LOGGER.info("Amostra já existe em %s; use --force para refazer", paths.sample_dir)
        return
    if not paths.paragraphs.exists():
        raise FileNotFoundError(f"{paths.paragraphs} não existe; rode a etapa corpus antes")
    started = time.time()
    tokenizer = load_tokenizer(settings.model)
    prefix_id = resolve_prefix_id(tokenizer, settings.model.prefix_token)
    paragraphs = list(read_jsonl(paths.paragraphs))
    LOGGER.info("Lidos %d parágrafos de %s", len(paragraphs), paths.paragraphs)
    result = build_sample(
        paragraphs, tokenizer, settings.sample, list(settings.corpus.themes), prefix_id
    )

    ensure_dir(paths.sample_dir)
    write_csv(paths.occurrences, result.occurrences, OCCURRENCE_COLUMNS)
    write_jsonl(paths.sequences, result.sequences)
    vocab = vocab_rows(tokenizer, result.f_corpus, result.f_sample)
    write_csv(paths.vocab_types, vocab, VOCAB_COLUMNS)
    _backup_annotations(paths.senses)
    write_csv(paths.senses, result.senses, SENSES_COLUMNS)
    extra = dict(result.stats)
    extra["vocab"] = {
        "size": len(vocab),
        "by_script_class": _counts([row["script_class"] for row in vocab]),
        "in_corpus": sum(row["in_corpus"] for row in vocab),
        "in_sample": sum(row["f_sample"] > 0 for row in vocab),
    }
    write_manifest(
        paths.sample_dir, "sample", settings, started, extra, root=paths.root, inputs=inputs
    )
    LOGGER.info("Amostra gravada em %s", paths.sample_dir)
