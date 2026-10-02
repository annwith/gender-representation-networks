import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pytest

from gender_networks import sampling
from gender_networks.artifacts import OCCURRENCE_COLUMNS, VOCAB_COLUMNS, RunPaths, write_jsonl
from gender_networks.sampling import (
    CorpusParagraph,
    _Selector,
    assign_prefix_groups,
    band_label,
    build_sample,
    cap_uniform,
    is_content_type,
    sentence_index,
    validate_bands,
    validate_sample,
)
from gender_networks.settings import (
    CoreSettings,
    CorpusSettings,
    LimitSettings,
    MultithemeSettings,
    PathsSettings,
    SampleSettings,
    Settings,
)
from gender_networks.tokens import CATEGORY_CODES, is_function_word

MODEL = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
PREFIX_ID = 151643
THEMES = ["fisica", "economia", "geografia"]
THEME_WORDS = {
    "fisica": ["energia", "força", "partícula", "luz", "onda"],
    "economia": ["mercado", "preço", "moeda", "custo", "venda"],
    "geografia": ["rio", "montanha", "clima", "região", "solo"],
}
COMMON = ["sistema", "forma", "processo"]
RARE = ["modelo", "método", "valor", "tempo", "área"]
FUNCTION = ["a", "o", "de", "que", "em", "um", "para", "com", "do", "da"]
# probability, per sentence, of inserting each studied word in each theme
INSERT = {
    "banco": {"economia": 0.15, "geografia": 0.12},
    "carga": {"fisica": 0.15},
    "grande": {"fisica": 0.15, "economia": 0.15},
}
OPENERS = ["O sistema", "A forma", "O processo"]
F_MAX = 20


def make_corpus(
    articles: int = 8, paragraphs: int = 4, seed: int = 2024, insert: dict | None = None
) -> list[dict]:
    """Portuguese-like paragraphs in the paragraphs.jsonl format, deterministic."""

    insert = INSERT if insert is None else insert
    rng = random.Random(seed)
    records = []
    for theme_rank, theme in enumerate(THEMES):
        for article in range(articles):
            pageid = 1000 * (theme_rank + 1) + article
            for index in range(paragraphs):
                sentences = [f"{rng.choice(OPENERS)} de {rng.choice(THEME_WORDS[theme])}"]
                for _ in range(6):
                    words = []
                    for _ in range(9):
                        draw = rng.random()
                        if draw < 0.5:
                            words.append(rng.choice(FUNCTION))
                        elif draw < 0.8:
                            words.append(rng.choice(THEME_WORDS[theme]))
                        else:
                            words.append(rng.choice(COMMON))
                    if rng.random() < 0.2:
                        words.insert(rng.randrange(1, len(words)), rng.choice(RARE))
                    for word, probabilities in insert.items():
                        if rng.random() < probabilities.get(theme, 0.0):
                            words.insert(rng.randrange(1, len(words)), word)
                    sentences.append(" ".join(words))
                text, spans = "", []
                for number, sentence in enumerate(sentences):
                    sentence = sentence[0].upper() + sentence[1:] + "."
                    start = len(text) + (1 if number else 0)
                    text = f"{text} {sentence}" if number else sentence
                    spans.append([start, len(text)])
                records.append(
                    {
                        "paragraph_id": f"{pageid}-{index}",
                        "pageid": pageid,
                        "revid": pageid * 10,
                        "title": f"Artigo {theme} {article}",
                        "theme": theme,
                        "source_category": f"Categoria:{theme}",
                        "paragraph_idx": index,
                        "text": text,
                        "sentences": spans,
                    }
                )
    return records


def make_settings(**changes) -> SampleSettings:
    base = SampleSettings(
        seed=7,
        themes=None,
        f_max=F_MAX,
        bands=[[1, 2], [3, 9], [10, 19], [20, 20]],
        core=CoreSettings(articles_per_theme=2, paragraphs_per_article=2),
        targets={
            "banco": {"economia": 8, "geografia": 8},
            "carga": {"fisica": 8, "economia": 8},
            "partícula": {"fisica": 5, "economia": 5},
        },
        controls={"grande": {"fisica": 6, "economia": 6}},
        min_per_sense=3,
        multitheme=MultithemeSettings(n_types=3, per_type=8, min_themes=2, min_per_theme=3),
        limits=LimitSettings(per_type_paragraph=1, per_type_article=2),
        senses_per_target=4,
    )
    return replace(base, **changes)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    except Exception as error:  # any loading failure means "not cached here"
        pytest.skip(f"Qwen3 tokenizer not available offline: {error}")


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return make_corpus()


@pytest.fixture(scope="module")
def result(tokenizer, corpus):
    return build_sample(corpus, tokenizer, make_settings(), THEMES, PREFIX_ID)


def token_id(tokenizer, text: str) -> int:
    (value,) = tokenizer.encode(text, add_special_tokens=False)
    return value


def paragraph_ids(tokenizer, corpus) -> dict[str, list[int]]:
    return {
        r["paragraph_id"]: tokenizer.encode(r["text"], add_special_tokens=False) for r in corpus
    }


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_cap_uniform_draws_uniformly_not_first_come() -> None:
    groups = {5: list(range(100)), 2: ["a", "b"]}
    kept, capped, removed = cap_uniform(groups, 10, random.Random(0))
    assert (capped, removed) == (1, 90)
    assert kept[:2] == ["a", "b"]  # keys in sorted order, small groups untouched
    drawn = kept[2:]
    assert len(drawn) == len(set(drawn)) == 10
    assert drawn != list(range(10)) and max(drawn) > 50
    means = []
    for seed in range(200):
        values = cap_uniform({0: list(range(100))}, 10, random.Random(seed))[0]
        means.append(sum(values) / len(values))
    assert abs(sum(means) / len(means) - 49.5) < 3  # uniform over the whole group


def test_assign_prefix_groups_uses_the_prefix_up_to_the_vertex() -> None:
    sequences = [[9, 1, 2, 3], [9, 1, 2, 4], [9, 5]]
    positions = [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1)]
    assert assign_prefix_groups(positions, sequences) == [0, -1, 0, -1, -1, -1]
    positions = [(0, 2), (1, 2), (0, 3), (1, 1), (0, 1)]
    assert assign_prefix_groups(positions, sequences) == [0, 0, -1, 1, 1]


def test_bands_and_labels() -> None:
    bands = [[1, 2], [3, 9], [10, 49], [50, 50]]
    validate_bands(bands, 50)
    assert [band_label(v, bands) for v in (1, 2, 3, 49, 50)] == ["1-2", "1-2", "3-9", "10-49", "50"]
    assert band_label(7000, bands, open_top=True) == "50+"
    assert band_label(50, bands, open_top=True) == "50+"
    with pytest.raises(ValueError):
        band_label(51, bands)
    with pytest.raises(ValueError):
        validate_bands([[1, 2], [4, 50]], 50)
    with pytest.raises(ValueError):
        validate_bands([[1, 2], [3, 9]], 50)


def test_sentence_index_uses_the_first_visible_character() -> None:
    spans = [[0, 10], [11, 20], [21, 30]]
    assert sentence_index(spans, 0, "Abc") == 0
    assert sentence_index(spans, 10, " Nova") == 1  # the space belongs to no sentence
    assert sentence_index(spans, 9, ".") == 0
    assert sentence_index(spans, 25, " fim") == 2
    assert sentence_index([], 5, "x") == 0


def fake_rows(sequences):
    rows = []
    for sequence in sequences:
        for position in range(1, len(sequence["input_ids"])):
            rows.append(
                {
                    "occurrence_id": len(rows),
                    "stratum": "core",
                    "token_category": "whole_word",
                    "sequence_id": sequence["sequence_id"],
                    "pos_in_sequence": position,
                    "token_id": sequence["input_ids"][position],
                    "f_sample": 0,
                }
            )
    counts = Counter(row["token_id"] for row in rows)
    for row in rows:
        row["f_sample"] = counts[row["token_id"]]
    return rows


def test_validate_sample_rejects_broken_invariants() -> None:
    sequences = [
        {"sequence_id": 0, "input_ids": [PREFIX_ID, 10, 11]},
        {"sequence_id": 1, "input_ids": [PREFIX_ID, 10, 12]},
    ]
    rows = fake_rows(sequences)
    validate_sample(rows, sequences, f_max=2, prefix_id=PREFIX_ID)
    with pytest.raises(ValueError, match="f_max"):
        validate_sample(rows, sequences, f_max=1)

    def broken(**change):
        copy = [dict(row) for row in rows]
        copy[1].update(change)
        return copy

    with pytest.raises(ValueError, match="input_ids"):
        validate_sample(broken(token_id=99, f_sample=1), sequences, f_max=2)
    with pytest.raises(ValueError, match="category"):
        validate_sample(broken(token_category="whitespace"), sequences, f_max=2)
    with pytest.raises(ValueError, match="position"):
        validate_sample(broken(pos_in_sequence=0), sequences, f_max=2)
    swapped = [dict(row, occurrence_id=i) for i, row in enumerate(reversed(rows))]
    with pytest.raises(ValueError, match="ordered"):
        validate_sample(swapped, sequences, f_max=2)
    with pytest.raises(ValueError, match="row index"):
        validate_sample(list(reversed(rows)), sequences, f_max=2)
    long = [dict(sequences[0], input_ids=[PREFIX_ID, 10, 11, 13]), sequences[1]]
    with pytest.raises(ValueError, match="truncated"):
        validate_sample(rows, long, f_max=2)
    with pytest.raises(ValueError, match="prefix"):
        validate_sample(rows, [dict(sequences[0], input_ids=[0, 10, 11]), sequences[1]], 2, 7)


# ---------------------------------------------------------------------------------------------
# build_sample on a synthetic corpus with the real tokenizer
# ---------------------------------------------------------------------------------------------


def test_sample_columns_and_basic_invariants(result, tokenizer) -> None:
    rows = result.occurrences
    assert rows and all(list(row) == OCCURRENCE_COLUMNS for row in rows)
    assert [row["occurrence_id"] for row in rows] == list(range(len(rows)))
    counts = Counter(row["token_id"] for row in rows)
    assert max(counts.values()) <= F_MAX
    assert all(row["f_sample"] == counts[row["token_id"]] for row in rows)
    assert all(row["pos_in_sequence"] >= 1 for row in rows)
    assert {row["token_category"] for row in rows} <= {
        "whole_word", "word_start", "continuation", "punctuation", "number",
    }
    assert {row["stratum"] for row in rows} == {"core", "target", "control", "multitheme"}
    assert all(PREFIX_ID not in [row["token_id"]] for row in rows)
    assert result.stats["n_vertices"] == len(rows)
    assert result.stats["n_types"] == len(counts)
    assert result.stats["tokens_to_process"] == sum(len(s["input_ids"]) for s in result.sequences)
    first = rows[0]
    assert first["prev_token_id"] == PREFIX_ID and first["pos_in_sequence"] == 1
    assert first["local_context"].startswith("«")


def test_occurrences_are_ordered_and_match_input_ids(result, tokenizer, corpus) -> None:
    rows, sequences = result.occurrences, result.sequences
    keys = [(row["sequence_id"], row["pos_in_sequence"]) for row in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    full = paragraph_ids(tokenizer, corpus)
    text_of = {r["paragraph_id"]: r["text"] for r in corpus}
    for row in rows:
        sequence = sequences[row["sequence_id"]]
        assert sequence["sequence_id"] == row["sequence_id"]
        assert sequence["paragraph_id"] == row["paragraph_id"]
        assert sequence["input_ids"][row["pos_in_sequence"]] == row["token_id"]
        ids = full[row["paragraph_id"]]
        j = row["pos_in_sequence"] - 1
        assert row["next_token_id"] == (ids[j + 1] if j + 1 < len(ids) else -1)
        assert row["prev_token_id"] == ([PREFIX_ID] + ids)[j]
        text = text_of[row["paragraph_id"]]
        assert row["token_text"] == text[row["char_start"] : row["char_end"]]
        assert row["local_context"].count("«") == 1


def test_sequences_are_truncated_after_their_last_vertex(result, tokenizer, corpus) -> None:
    full = paragraph_ids(tokenizer, corpus)
    last = defaultdict(int)
    for row in result.occurrences:
        last[row["sequence_id"]] = max(last[row["sequence_id"]], row["pos_in_sequence"])
    for sequence in result.sequences:
        ids = sequence["input_ids"]
        assert ids[0] == PREFIX_ID
        assert len(ids) == last[sequence["sequence_id"]] + 1
        assert ids[1:] == full[sequence["paragraph_id"]][: len(ids) - 1]
    # sparse strata use only a few positions, so their sequences stop early
    sparse = {r["sequence_id"] for r in result.occurrences if r["stratum"] != "core"}
    assert any(
        len(result.sequences[s]["input_ids"]) - 1 < len(full[result.sequences[s]["paragraph_id"]])
        for s in sparse
    )


def test_core_comes_first_in_theme_order(result, corpus) -> None:
    record = {r["paragraph_id"]: r for r in corpus}
    stratum_of = defaultdict(set)
    for row in result.occurrences:
        stratum_of[row["sequence_id"]].add(row["stratum"])
    core = [s for s in result.sequences if "core" in stratum_of[s["sequence_id"]]]
    assert len(core) == len(THEMES) * 2 * 2
    assert result.sequences[: len(core)] == core
    assert all(stratum_of[s["sequence_id"]] == {"core"} for s in core)

    def key(sequence):
        r = record[sequence["paragraph_id"]]
        return (THEMES.index(r["theme"]), r["pageid"], r["paragraph_idx"])

    rest = result.sequences[len(core) :]
    assert [key(s) for s in core] == sorted(key(s) for s in core)
    assert [key(s) for s in rest] == sorted(key(s) for s in rest)
    articles = Counter(record[s["paragraph_id"]]["pageid"] for s in core)
    assert set(articles.values()) == {2} and len(articles) == len(THEMES) * 2


def test_core_cap_is_a_uniform_draw_over_the_paragraphs(result, tokenizer, corpus) -> None:
    de = token_id(tokenizer, " de")
    full = paragraph_ids(tokenizer, corpus)
    core_rows = [r for r in result.occurrences if r["stratum"] == "core"]
    core_sequences = sorted({r["sequence_id"] for r in core_rows})
    pool = [
        (s, j + 1)
        for s in core_sequences
        for j, value in enumerate(full[result.sequences[s]["paragraph_id"]])
        if value == de
    ]
    kept = [
        (r["sequence_id"], r["pos_in_sequence"]) for r in result.occurrences if r["token_id"] == de
    ]
    assert len(pool) >= 2 * F_MAX  # " de" is far above the cap in the core
    assert len(kept) == F_MAX
    assert all(r["stratum"] == "core" for r in result.occurrences if r["token_id"] == de)
    assert kept != pool[:F_MAX]  # not first-come
    assert len({s for s, _ in kept}) >= 5  # spread over many paragraphs
    assert max(s for s, _ in kept) > core_sequences[len(core_sequences) // 2]
    assert result.stats["core_capped_types"] > 0 and result.stats["core_removed_by_cap"] > 0


def test_quotas_and_diversity_limits(result, tokenizer) -> None:
    rows = result.occurrences
    limits = make_settings().limits
    report = result.stats["targets"]["banco"]
    assert report["kept"]
    banco = token_id(tokenizer, " banco")
    for theme, entry in report["themes"].items():
        assert entry["requested"] == make_settings().targets["banco"][theme]
        assert entry["drawn"] <= max(0, entry["requested"] - entry["core"])
        assert entry["obtained"] == entry["core"] + entry["drawn"] == entry["final"]
        drawn_rows = [
            r for r in rows if r["stratum"] == "target" and r["sense_theme"] == theme
            and r["token_id"] == banco
        ]
        assert len(drawn_rows) == entry["drawn"] and entry["drawn"] > 0
    assert report["themes"]["economia"]["obtained"] == 8  # plenty of "banco" in economia

    for stratum in ("target", "control", "multitheme"):
        sparse = [r for r in rows if r["stratum"] == stratum]
        per_paragraph = Counter((r["token_id"], r["paragraph_id"]) for r in sparse)
        assert max(per_paragraph.values()) <= limits.per_type_paragraph
    per_article = Counter((r["token_id"], r["pageid"]) for r in rows)
    sparse_keys = {(r["token_id"], r["pageid"]) for r in rows if r["stratum"] != "core"}
    assert all(per_article[key] <= limits.per_type_article for key in sparse_keys)


def test_min_per_sense_drops_a_one_theme_target(result, tokenizer) -> None:
    carga = token_id(tokenizer, " carga")
    dropped = {d["word"]: d for d in result.stats["dropped_targets"]}
    assert "carga" in dropped and "banco" not in dropped
    assert dropped["carga"]["themes_reaching_min"] == ["fisica"]
    assert dropped["carga"]["removed_occurrences"] > 0
    assert result.stats["targets"]["carga"]["kept"] is False
    carga_rows = [r for r in result.occurrences if r["token_id"] == carga]
    assert all(r["stratum"] == "core" for r in carga_rows)  # only the core keeps it
    assert all(r["target_word"] == "carga" and r["sense_theme"] == r["theme"] for r in carga_rows)
    assert result.stats["kept_targets"] == ["banco"]
    assert result.stats["kept_controls"] == ["grande"]
    assert "partícula" in result.stats["not_single_token"]
    assert "partícula" not in result.stats["targets"]
    assert not any(r["target_word"] == "partícula" for r in result.occurrences)


def test_target_word_is_set_for_whole_words_in_every_stratum(result, tokenizer) -> None:
    ids = {token_id(tokenizer, " " + w): w for w in ("banco", "carga", "grande")}
    for row in result.occurrences:
        whole = row["token_category"] == "whole_word"
        expected = ids.get(row["token_id"], "") if whole else ""
        assert row["target_word"] == expected
        assert row["sense_theme"] == (row["theme"] if expected else "")
        if row["stratum"] in ("target", "control"):
            assert expected


def test_multitheme_draws_content_words_round_robin(result, tokenizer) -> None:
    stats = result.stats
    assert stats["multitheme_drawn_types"] == 3
    assert stats["multitheme_eligible"] >= 3
    excluded = {token_id(tokenizer, " " + w) for w in ("banco", "carga", "grande")}
    types = {t["token_id"]: t for t in stats["multitheme_types"]}
    rows = [r for r in result.occurrences if r["stratum"] == "multitheme"]
    assert rows and {r["token_id"] for r in rows} <= set(types)
    for value, info in types.items():
        text = tokenizer.decode([value]).strip()
        assert value not in excluded
        assert text.isalpha() and len(text) >= 3 and not is_function_word(text)
        assert len(info["themes"]) >= 2
        mine = [r for r in rows if r["token_id"] == value]
        assert len(mine) == info["drawn"]
        assert info["before"] + info["drawn"] <= max(info["before"], 8)
        if info["drawn"] >= 2:  # round-robin spreads the draws over the themes
            assert len({r["theme"] for r in mine}) >= 2


def test_senses_template_is_balanced(result) -> None:
    senses = result.senses
    assert senses and {s["target_word"] for s in senses} == {"banco"}
    assert len(senses) == 4
    themes = Counter(s["sense_theme"] for s in senses)
    assert themes == {"economia": 2, "geografia": 2}
    by_id = {r["occurrence_id"]: r for r in result.occurrences}
    for sense in senses:
        row = by_id[sense["occurrence_id"]]
        assert row["target_word"] == "banco" and sense["sense"] == ""
        assert sense["local_context"] == row["local_context"]


def test_prefix_groups_follow_identical_prefixes(result) -> None:
    rows = result.occurrences
    prefix = {
        r["occurrence_id"]: tuple(
            result.sequences[r["sequence_id"]]["input_ids"][: r["pos_in_sequence"] + 1]
        )
        for r in rows
    }
    members = defaultdict(list)
    for r in rows:
        members[prefix[r["occurrence_id"]]].append(r["occurrence_id"])
    shared = [ids for ids in members.values() if len(ids) > 1]
    assert shared  # several core paragraphs start with the same opener
    for ids in members.values():
        groups = {rows[i]["prefix_group"] for i in ids}
        assert len(groups) == 1
        assert (groups == {-1}) == (len(ids) == 1)
    assert result.stats["prefix_groups"] == len(shared)
    assert result.stats["prefix_group_largest"] == max(len(ids) for ids in shared)
    openers = [r for r in rows if r["pos_in_sequence"] == 1 and r["prefix_group"] >= 0]
    assert len({r["sequence_id"] for r in openers}) >= 2


def test_sample_is_deterministic_by_seed(result, tokenizer, corpus) -> None:
    again = build_sample(corpus, tokenizer, make_settings(), THEMES, PREFIX_ID)
    assert again.occurrences == result.occurrences
    assert again.sequences == result.sequences
    assert again.senses == result.senses
    other = build_sample(corpus, tokenizer, make_settings(seed=8), THEMES, PREFIX_ID)
    assert [r["paragraph_id"] for r in other.occurrences] != [
        r["paragraph_id"] for r in result.occurrences
    ]


def test_corpus_frequency_counts_every_paragraph(tokenizer, corpus) -> None:
    subset = build_sample(
        corpus, tokenizer, make_settings(themes=["fisica", "economia"]), THEMES, PREFIX_ID
    )
    rio = token_id(tokenizer, " rio")
    full = paragraph_ids(tokenizer, corpus)
    assert subset.f_corpus[rio] == sum(ids.count(rio) for ids in full.values()) > 0
    assert {r["theme"] for r in subset.occurrences} == {"fisica", "economia"}
    banco = subset.stats["targets"]["banco"]
    assert banco["themes"]["geografia"]["in_sample"] is False
    assert banco["kept"] is False  # only economia is left for banco
    for row in subset.occurrences:
        assert row["f_corpus"] == subset.f_corpus[row["token_id"]]
        assert row["band_corpus"] == band_label(row["f_corpus"], make_settings().bands, True)


def test_run_writes_every_artifact_and_skips_when_done(
    tmp_path, tokenizer, corpus, monkeypatch
) -> None:
    settings = Settings(
        name="teste",
        paths=PathsSettings(
            raw_dir="raw", corpus_dir="corpus", runs_dir="runs", report_dir="relatorio"
        ),
        corpus=CorpusSettings(themes={theme: [f"Categoria:{theme}"] for theme in THEMES}),
        sample=make_settings(),
    )
    paths = RunPaths.from_settings(settings, tmp_path)
    write_jsonl(paths.paragraphs, corpus)
    monkeypatch.setattr(sampling, "load_tokenizer", lambda model: tokenizer)

    sampling.run(settings, paths)

    with paths.occurrences.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == OCCURRENCE_COLUMNS
        occurrences = list(reader)
    with paths.vocab_types.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == VOCAB_COLUMNS
        vocab = list(reader)
    sequences = [json.loads(line) for line in paths.sequences.read_text().splitlines()]
    manifest = json.loads((paths.sample_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert len(occurrences) == manifest["n_vertices"]
    assert len(sequences) == manifest["n_sequences"]
    assert list(sequences[0]) == ["sequence_id", "paragraph_id", "input_ids"]
    assert len(vocab) == len(tokenizer)
    assert vocab[PREFIX_ID]["script_class"] == "special" and vocab[PREFIX_ID]["is_special"]
    banco = token_id(tokenizer, " banco")
    assert int(vocab[banco]["f_sample"]) == sum(int(r["token_id"]) == banco for r in occurrences)
    assert vocab[banco]["in_corpus"] == "True"
    assert manifest["stage"] == "sample"
    assert manifest["vocab"]["by_script_class"]["special"] == 26
    assert paths.senses.read_text(encoding="utf-8").startswith("occurrence_id,target_word")

    before = paths.occurrences.stat().st_mtime_ns
    sampling.run(settings, paths)  # outputs exist: nothing is redone
    assert paths.occurrences.stat().st_mtime_ns == before

    # manual annotations survive a forced rerun as a backup copy
    lines = paths.senses.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1] + "instituição"
    paths.senses.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sampling.run(settings, paths, force=True)
    backups = list(paths.sample_dir.glob("senses.annotated-*.csv"))
    assert len(backups) == 1 and "instituição" in backups[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Regressions: whole-word draws, the f_max ceiling and special-token paragraphs
# ---------------------------------------------------------------------------------------------

WW = CATEGORY_CODES["whole_word"]
WS = CATEGORY_CODES["word_start"]
CO = CATEGORY_CODES["continuation"]
# "grandeza" is " grande" + "za" and "capitalismo" is " capital" + "ismo" in Qwen3: the studied
# id is only the first piece of a longer word there. "capital" itself is only in economia.
PREFIX_INSERT = {
    **INSERT,
    "grandeza": {"fisica": 0.3, "economia": 0.3, "geografia": 0.3},
    "capitalismo": {"fisica": 0.4, "economia": 0.2},
    "capital": {"economia": 0.2},
    "valor": {"fisica": 0.2, "economia": 0.2, "geografia": 0.2},
    "valorização": {"fisica": 0.1, "economia": 0.1, "geografia": 0.1},
}


def prefix_settings() -> SampleSettings:
    base = make_settings()
    return replace(
        base,
        targets={**base.targets, "capital": {"economia": 6, "fisica": 6}},
        multitheme=replace(base.multitheme, n_types=50),
    )


@pytest.fixture(scope="module")
def prefix_result(tokenizer):
    corpus = make_corpus(insert=PREFIX_INSERT)
    return build_sample(corpus, tokenizer, prefix_settings(), THEMES, PREFIX_ID)


def fake_paragraph(index: int, theme: str, pageid: int, ids, codes) -> CorpusParagraph:
    record = {"theme": theme, "pageid": pageid, "paragraph_idx": index, "paragraph_id": str(index)}
    return CorpusParagraph(
        index, record, np.asarray(ids, dtype=np.int64), np.asarray(codes, dtype=np.int8)
    )


def test_index_and_theme_counts_use_whole_words_only() -> None:
    corpus = [
        fake_paragraph(0, "fisica", 1, [5, 6, 5, 7], [WS, CO, WW, WW]),
        fake_paragraph(1, "fisica", 2, [5, 6, 7], [WS, CO, WW]),
        fake_paragraph(2, "economia", 3, [5, 6], [WS, CO]),
    ]
    selector = _Selector(corpus, make_settings(), THEMES, random.Random(0))
    selector.build_index({5, 7})
    assert selector.index[5] == {0: [2]}  # the word_start pieces are never candidates
    assert selector.index[7] == {0: [3], 1: [2]}
    raw, drawable = selector.theme_counts(10)
    assert raw["fisica"][5] == 1 and raw["economia"][5] == 0
    assert raw["fisica"][7] == 2 and raw["fisica"][6] == 0


def test_core_quota_counts_ignore_word_pieces() -> None:
    corpus = [
        fake_paragraph(i, "fisica", 1 + i // 2, [5, 6, 5, 9], [WS, CO, WW, WW]) for i in range(4)
    ]
    settings = replace(make_settings(), themes=["fisica"])
    selector = _Selector(corpus, settings, ["fisica"], random.Random(0))
    selector.select_core()
    assert selector.total[5] == 8  # every piece stays a core vertex
    assert selector.core_theme[(5, "fisica")] == 4  # only whole words fill quotas


def test_prefix_pieces_are_never_drawn_or_labelled(prefix_result, tokenizer) -> None:
    rows = prefix_result.occurrences
    grande, capital = token_id(tokenizer, " grande"), token_id(tokenizer, " capital")
    pieces = [
        r for r in rows
        if r["token_id"] in (grande, capital) and r["token_category"] != "whole_word"
    ]
    assert pieces and all(r["stratum"] == "core" for r in pieces)  # the test is meaningful
    assert all(r["target_word"] == "" and r["sense_theme"] == "" for r in pieces)
    assert prefix_result.stats["studied_id_pieces"] == len(pieces)
    for r in rows:
        if r["stratum"] != "core":
            assert r["token_category"] == "whole_word", r
    senses = {s["occurrence_id"] for s in prefix_result.senses}
    assert all(rows[i]["token_category"] == "whole_word" for i in senses)
    # core counts in the quota report are whole-word counts
    for group in ("targets", "controls"):
        for word, info in prefix_result.stats[group].items():
            for theme, entry in info["themes"].items():
                core = sum(
                    r["token_id"] == info["token_id"] and r["theme"] == theme
                    and r["stratum"] == "core" and r["token_category"] == "whole_word"
                    for r in rows
                )
                assert entry["core"] == core, (word, theme)


def test_target_kept_only_by_longer_words_is_dropped(prefix_result) -> None:
    # "capitalismo" is everywhere in fisica, but "capital" itself only in economia
    dropped = {d["word"]: d for d in prefix_result.stats["dropped_targets"]}
    assert "capital" in dropped
    assert dropped["capital"]["themes_reaching_min"] == ["economia"]
    assert prefix_result.stats["targets"]["capital"]["themes"]["fisica"]["obtained"] == 0
    assert "banco" in prefix_result.stats["kept_targets"]


def test_multitheme_types_are_lowercase_portuguese_content_words() -> None:
    assert is_content_type(" valor") and is_content_type("energia")
    for text in (" The", " from", "www", " John", " Harvard", " de", " é", "ab", " 1990", "-se"):
        assert not is_content_type(text), text


def test_quota_draws_stop_at_f_max(tokenizer, corpus) -> None:
    settings = make_settings(
        f_max=12,
        bands=[[1, 2], [3, 9], [10, 11], [12, 12]],
        targets={"banco": {"economia": 8, "geografia": 8}},
        controls={"grande": {"fisica": 6, "economia": 6}},
    )
    result = build_sample(corpus, tokenizer, settings, THEMES, PREFIX_ID)
    banco = token_id(tokenizer, " banco")
    themes = result.stats["targets"]["banco"]["themes"]
    assert result.stats["targets"]["banco"]["kept"]
    assert themes["economia"]["obtained"] == 8
    geografia = themes["geografia"]
    total_after_economia = themes["economia"]["obtained"] + geografia["core"]
    # need = min(quota - core, f_max - total): the f_max term is the binding one here
    assert geografia["drawn"] == 12 - total_after_economia < 8 - geografia["core"]
    assert result.f_sample[banco] == 12 == max(result.f_sample.values())


def test_paragraph_with_special_token_is_excluded_but_counted(tokenizer, corpus) -> None:
    text = "O sistema de <|endoftext|> zircônio e banco."
    special = {
        "paragraph_id": "1000-99", "pageid": 1000, "revid": 1, "title": "Artigo fisica 0",
        "theme": "fisica", "source_category": "Categoria:fisica", "paragraph_idx": 99,
        "text": text, "sentences": [[0, len(text)]],
    }
    result = build_sample(corpus + [special], tokenizer, make_settings(), THEMES, PREFIX_ID)
    assert result.stats["paragraphs_with_special_tokens"] == 1
    assert "1000-99" not in {s["paragraph_id"] for s in result.sequences}
    assert "1000-99" not in {r["paragraph_id"] for r in result.occurrences}
    assert result.f_corpus[PREFIX_ID] == 1
    full = paragraph_ids(tokenizer, corpus + [special])
    for value in set(full["1000-99"]):
        assert result.f_corpus[value] == sum(ids.count(value) for ids in full.values())
