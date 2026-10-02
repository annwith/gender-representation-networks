import re
from collections import Counter
from pathlib import Path

import pytest

from gender_networks.tokens import (
    STOPWORDS,
    TokenInfo,
    is_function_word,
    label_tokens,
    script_class,
    special_token_ids,
    tokenize_paragraph,
    vocab_script_classes,
)

MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
FEASIBILITY = Path(__file__).resolve().parents[1] / "scripts" / "feasibility" / "simulate_sample.py"
TARGETS_AND_CONTROLS = (
    "banco campo órgão estado carga rede nota capital família massa matriz planta "
    "ano século grande primeiro importante água"
).split()


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    except Exception as error:  # any loading failure means "not cached here"
        pytest.skip(f"Qwen3 tokenizer not available offline: {error}")


def labels(infos: list[TokenInfo]) -> list[tuple[str, str]]:
    return [(info.text, info.category) for info in infos]


# ---------------------------------------------------------------------------------------------
# Function words and script classes (no tokenizer needed)
# ---------------------------------------------------------------------------------------------


def test_function_words_are_case_insensitive_and_spare_the_studied_words() -> None:
    assert is_function_word("De") and is_function_word(" que ") and is_function_word("NÃO")
    assert is_function_word("pelas") and is_function_word("foi") and is_function_word("se")
    assert not is_function_word("banco")
    assert not any(is_function_word(word) for word in TARGETS_AND_CONTROLS)


def test_stopwords_extend_the_feasibility_stop_set() -> None:
    if not FEASIBILITY.exists():
        pytest.skip("feasibility script not present")
    match = re.search(r'STOP = set\("""(.*?)"""', FEASIBILITY.read_text(encoding="utf-8"), re.S)
    assert match is not None
    stop = set(match.group(1).split())
    assert stop <= STOPWORDS
    assert len(STOPWORDS) > len(stop)


@pytest.mark.parametrize(
    ("text", "special", "expected"),
    [
        ("<|endoftext|>", True, "special"),
        ("�", False, "byte_fragment"),
        (" a�", False, "byte_fragment"),
        ("  \n", False, "whitespace"),
        ("", False, "whitespace"),
        ("123", False, "digits"),
        (" (", False, "punctuation"),
        ("…»", False, "punctuation"),
        (" banco", False, "latin"),
        ("ação", False, "latin"),
        ("本", False, "cjk"),
        ("日本の", False, "cjk"),
        ("カタカナ", False, "cjk"),
        ("한국", False, "cjk"),
        (" Москва", False, "other_script"),
        ("αβγ", False, "other_script"),
        (" كتاب", False, "other_script"),
        ("aб", False, "mixed"),
        ("x2", False, "mixed"),
        ("_init", False, "mixed"),
        ("foo()", False, "mixed"),
    ],
)
def test_script_class(text: str, special: bool, expected: str) -> None:
    assert script_class(text, special) == expected


class FakeTokenizer:
    """Tiny stand-in with the attributes vocab_script_classes reads."""

    pieces = [" casa", "本", "�", " ", "7", "!", "<|endoftext|>", "<extra>"]
    all_special_ids = [6]
    added_tokens_decoder = {7: object()}

    def __len__(self) -> int:
        return len(self.pieces)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids)


def test_vocab_script_classes_marks_special_and_added_tokens() -> None:
    fake = FakeTokenizer()
    assert special_token_ids(fake) == {6, 7}
    assert vocab_script_classes(fake) == [
        "latin", "cjk", "byte_fragment", "whitespace", "digits", "punctuation", "special",
        "special",
    ]


def test_label_tokens_handles_overlapping_offsets() -> None:
    # Byte-level BPE splits one rare character into pieces that share its offsets.
    text = "a 龘b"
    infos = label_tokens(text, [1, 2, 3, 4, 5], [(0, 1), (1, 3), (2, 3), (2, 3), (3, 4)])
    assert [info.category for info in infos] == [
        "whole_word", "word_start", "continuation", "continuation", "continuation",
    ]
    assert {info.word for info in infos[1:]} == {"龘b"}
    assert [info.pos_in_word for info in infos[1:]] == [0, 1, 2, 3]
    assert all(info.word_n_tokens == 4 for info in infos[1:])
    assert infos[1].text == " 龘" and infos[1].has_leading_space
    assert infos[2].text == "龘" and not infos[2].has_leading_space


def test_label_tokens_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        label_tokens("ab", [1, 2], [(0, 2)])


# ---------------------------------------------------------------------------------------------
# Real tokenizer
# ---------------------------------------------------------------------------------------------


def test_space_before_digits_is_whitespace_then_numbers(tokenizer) -> None:
    infos = tokenize_paragraph(tokenizer, " 1990")
    assert [info.token_id for info in infos] == [220, 16, 24, 24, 15]
    assert labels(infos) == [
        (" ", "whitespace"), ("1", "number"), ("9", "number"), ("9", "number"), ("0", "number"),
    ]
    assert infos[0].word == "" and infos[0].word_idx == -1
    assert {info.word for info in infos[1:]} == {"1990"}


def test_number_tokens_join_their_word_documented_deviation(tokenizer) -> None:
    # letters glued to digits are continuations of the digit word, not whole words
    infos = [i for i in tokenize_paragraph(tokenizer, "Nos anos 1990s a H2O") if i.text.strip()]
    by_text = {info.text: info for info in infos}
    suffix = by_text["s"]
    assert (suffix.category, suffix.word, suffix.word_n_tokens) == ("continuation", "1990s", 5)
    assert [by_text[d].category for d in "19"] == ["number", "number"]
    assert by_text["O"].category == "continuation" and by_text["O"].word == "H2O"
    assert by_text["2"].category == "number" and by_text["2"].word == "H2O"
    assert by_text[" H"].category == "word_start" and by_text[" H"].word_n_tokens == 3


def test_number_deviation_is_documented() -> None:
    import gender_networks.tokens as tokens_module
    from gender_networks.sampling import DEVIATIONS

    assert "1990s" in (tokens_module.__doc__ or "")
    assert any("number tokens" in note for note in DEVIATIONS)


def test_clitic_is_a_whole_word(tokenizer) -> None:
    infos = tokenize_paragraph(tokenizer, "Ele tornou-se rei.")
    by_text = {info.text: info for info in infos}
    assert [info.text for info in infos[1:4]] == [" torn", "ou", "-se"]
    assert by_text[" torn"].category == "word_start"
    assert by_text["ou"].category == "continuation"
    assert by_text[" torn"].word == by_text["ou"].word == "tornou"
    assert by_text[" torn"].word_n_tokens == 2 and by_text["ou"].pos_in_word == 1
    clitic = by_text["-se"]
    assert (clitic.category, clitic.word, clitic.word_n_tokens) == ("whole_word", "se", 1)
    assert clitic.is_function_word
    assert by_text["."].category == "punctuation"


def test_word_inside_parentheses_is_split(tokenizer) -> None:
    infos = tokenize_paragraph(tokenizer, "O (banco) da praça")
    assert labels(infos) == [
        ("O", "whole_word"),
        (" (", "punctuation"),
        ("ban", "word_start"),
        ("co", "continuation"),
        (")", "punctuation"),
        (" da", "whole_word"),
        (" pra", "word_start"),
        ("ça", "continuation"),
    ]
    assert infos[2].word == infos[3].word == "banco"
    assert infos[6].word == infos[7].word == "praça" and infos[7].word_n_tokens == 2
    assert infos[0].is_function_word and infos[5].is_function_word
    assert not infos[6].is_function_word  # "pra" is a function word, "praça" is not


def test_target_word_is_the_single_spaced_id(tokenizer) -> None:
    (target_id,) = tokenizer.encode(" banco", add_special_tokens=False)
    infos = tokenize_paragraph(tokenizer, "O banco central emitiu uma nota.")
    banco = next(info for info in infos if info.word == "banco")
    assert banco.token_id == target_id
    assert banco.text == " banco" and banco.has_leading_space
    assert banco.category == "whole_word" and banco.word_n_tokens == 1
    assert not banco.is_function_word


def test_token_texts_are_exact_slices(tokenizer) -> None:
    text = "Em 1990, a Física (e a química) mudou: São Paulo — 日本 — ação!"
    infos = tokenize_paragraph(tokenizer, text)
    assert [info.index for info in infos] == list(range(len(infos)))
    for info in infos:
        assert info.text == text[info.char_start : info.char_end]
        assert info.has_leading_space == info.text[:1].isspace()
        if info.category in {"whole_word", "word_start", "continuation"}:
            assert info.word and 0 <= info.pos_in_word < info.word_n_tokens
    assert "".join(info.text for info in infos) == text


def test_vocabulary_script_classes_match_reference_counts(tokenizer) -> None:
    classes = vocab_script_classes(tokenizer)
    assert len(classes) == len(tokenizer) == 151_669
    counts = Counter(classes)
    assert counts["latin"] == 76_107
    assert counts["cjk"] == 31_148
    assert counts["mixed"] == 20_282
    assert counts["punctuation"] == 8_254
    assert counts["byte_fragment"] == 1_457
    assert counts["whitespace"] == 441
    assert counts["digits"] == 28
    assert counts["special"] == 26
    known = {
        52465: "latin",  # " banco"
        151643: "special",  # <|endoftext|>
        220: "whitespace",  # " "
        16: "digits",  # "1"
        8: "punctuation",  # ")"
        21894: "cjk",  # "本"
        127: "byte_fragment",  # lone byte 0xC3
        137639: "other_script",  # " Москва"
        6137: "mixed",  # "_init"
    }
    for token_id, expected in known.items():
        assert classes[token_id] == expected, tokenizer.decode([token_id])
