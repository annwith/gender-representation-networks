from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from gender_networks import analysis, knn, metrics
from gender_networks.artifacts import OCCURRENCE_COLUMNS, VOCAB_COLUMNS, RunPaths, write_csv
from gender_networks.settings import (
    AnalysisSettings,
    CorpusSettings,
    ModelSettings,
    NetworkSettings,
    PathsSettings,
    SampleSettings,
    Settings,
    VocabNetworkSettings,
)

N_LAYERS = 3


def tiny_settings() -> Settings:
    return Settings(
        name="tiny",
        paths=PathsSettings(runs_dir="runs", report_dir="relatorio"),
        model=ModelSettings(layers={"L01": 1, "L18": 2, "L36": 3}),
        corpus=CorpusSettings(themes={"a": ["Categoria:A"], "b": ["Categoria:B"]}),
        sample=SampleSettings(targets={"alvo": {"a": 6, "b": 6}}),
        networks=NetworkSettings(
            k_values=[2, 3],
            k_main=3,
            seeds=2,
            candidates=8,
            type_candidates=6,
            block_size=16,
            representations=["lex", "L01", "L18", "L36"],
            robust_representations=["L36n"],
            centered=False,
            vocab=VocabNetworkSettings(k_values=[3]),
        ),
        analysis=AnalysisSettings(
            leiden_runs=2,
            leiden_iterations=10,
            resolution_sweep=[1.0],
            permutations=2,
            distance_sample_sources=5,
            workers=1,
            exact_distances_vocab=False,
        ),
    )


def bf16(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32).to(torch.bfloat16)


def write_fake_run(root: Path, settings: Settings) -> tuple[RunPaths, pd.DataFrame]:
    rng = np.random.default_rng(7)
    paths = RunPaths.from_settings(settings, root)
    vocab_size, d = 60, 16
    weight = rng.standard_normal((vocab_size, d))
    frequencies = {3: 12, 4: 12, 5: 10, 6: 4, 7: 4, 8: 3, 9: 2, 10: 1, 11: 1, 12: 1}
    token_ids = np.concatenate([np.full(f, t) for t, f in frequencies.items()])
    token_ids = token_ids[rng.permutation(token_ids.size)]
    n = token_ids.size
    counts = pd.Series(token_ids).map(pd.Series(token_ids).value_counts()).to_numpy()
    rows = []
    for i, token in enumerate(token_ids):
        f = int(counts[i])
        band = "1-2" if f <= 2 else "3-9" if f <= 9 else "10-49"
        row = dict.fromkeys(OCCURRENCE_COLUMNS, "")
        row.update(
            occurrence_id=i,
            stratum="core" if i % 5 else "multitheme",
            theme="a" if i % 2 else "b",
            pageid=100 + i % 6,
            paragraph_id=f"{100 + i % 6}-{i % 3}",
            sentence_id=f"{100 + i % 6}-{i % 3}-s{i % 2}",
            sequence_id=i // 8,
            pos_in_sequence=i % 8 + 1,
            pos_bucket="1-4" if i % 8 < 4 else "5-16",
            prefix_group=-1,
            token_id=int(token),
            token_text=f" t{token}",
            next_token_id=int(token_ids[i + 1]) if i + 1 < n else -1,
            token_category="whole_word" if token % 2 else "word_start",
            is_function_word=token == 6,
            f_sample=f,
            band_sample=band,
            target_word="alvo" if token == 3 else "",
            sense_theme=("a" if i % 2 else "b") if token == 3 else "",
        )
        rows.append(row)
    write_csv(paths.occurrences, rows, OCCURRENCE_COLUMNS)
    vocab_rows = []
    for token in range(vocab_size):
        row = dict.fromkeys(VOCAB_COLUMNS, "")
        row.update(
            token_id=token,
            token_repr=repr(f" t{token}"),
            script_class="latin" if token % 3 else "cjk",
            is_special=token >= 58,
            f_corpus=int(frequencies.get(token, 0)),
            f_sample=int(frequencies.get(token, 0)),
            in_corpus=token in frequencies,
        )
        vocab_rows.append(row)
    write_csv(paths.vocab_types, vocab_rows, VOCAB_COLUMNS)
    paths.reps_dir.mkdir(parents=True)
    save_file({"weight": bf16(weight)}, str(paths.embeddings))
    theme_shift = np.where(np.arange(n) % 2, 1.0, -1.0)[:, None] * rng.standard_normal(d)
    layers = []
    for layer in range(1, N_LAYERS + 1):
        x = weight[token_ids] + layer * (0.5 * rng.standard_normal((n, d)) + 0.3 * theme_shift)
        layers.append(bf16(x))
    for name, layer in settings.model.layers.items():
        save_file({"x": layers[layer - 1]}, str(paths.rep(name)))
    save_file({"x": bf16(rng.standard_normal((n, d)))}, str(paths.rep("L36n")))
    stack = torch.stack(layers).view(torch.int16).numpy().view(np.uint16)
    np.save(paths.all_layers, stack)
    np.save(paths.pred_next, rng.integers(0, 58, size=n).astype(np.int32))
    return paths, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("analysis")
    settings = tiny_settings()
    paths, occurrences = write_fake_run(root, settings)
    knn.run(settings, paths)
    metrics.run(settings, paths)
    analysis.run(settings, paths)
    return settings, paths, occurrences


def read(paths: RunPaths, name: str) -> pd.DataFrame:
    return pd.read_csv(paths.analysis_dir / name, keep_default_na=False)


def test_lexical_dominance_reaches_its_ceiling(analyzed) -> None:
    _, paths, occurrences = analyzed
    measures = pd.read_csv(paths.analysis_dir / "vertex_measures.csv")

    assert len(measures) == len(occurrences)
    repeated = measures["ceiling"] > 0
    # The exact lexical network puts every other occurrence of the type first.
    assert np.allclose(measures.loc[repeated, "Dn_lex"], 1.0)
    assert np.allclose(measures["D_lex"], measures["ceiling"])
    for column in ("J_lex_L01", "J_L01_L18", "J_L18_L36", "Jw_lex_L36", "Jc_lex_L36", "NF_lex"):
        values = measures[column].dropna()
        assert ((values >= 0) & (values <= 1)).all(), column


def test_summary_and_transitions(analyzed) -> None:
    _, paths, _ = analyzed
    summary = read(paths, "p1_summary.csv")
    transitions = read(paths, "p2_transitions.csv")

    overall = summary[(summary["group"] == "all") & (summary["measure"] == "D_lex")]
    assert len(overall) == 1 and overall["count"].iat[0] > 0
    assert set(summary["group"]) >= {"band_sample", "token_category", "stratum", "pos_bucket"}
    consecutive = transitions[transitions["consecutive"].astype(str) == "True"]
    assert list(consecutive["transition"]) == ["lex->L01", "L01->L18", "L18->L36"]
    assert np.isclose(consecutive["share_largest_change"].astype(float).sum(), 1.0)
    assert not read(paths, "p2_transitions_groups.csv").empty


def test_layer_sweep_covers_every_block(analyzed) -> None:
    _, paths, _ = analyzed
    layers = read(paths, "p2_layers.csv")

    overall = layers[layers["token_category"] == "all"]
    assert list(overall["layer"]) == list(range(1, N_LAYERS + 1))
    assert overall["J_prev_mean"].astype(float).between(0, 1).all()


def test_global_hubs_and_robustness_tables(analyzed) -> None:
    settings, paths, _ = analyzed
    global_table = read(paths, "p1_global.csv")
    robustness = read(paths, "robustness_neighbors.csv")
    core = read(paths, "core_metrics.csv")

    assert set(global_table["rep"]) == set(settings.networks.representations)
    assert {"union", "directed"} <= set(global_table["sym"])
    lex_union = global_table[(global_table["rep"] == "lex") & (global_table["sym"] == "union")]
    assert int(lex_union["seeds"].iat[0]) == settings.networks.seeds
    assert set(robustness["tie"]) == {"pos", "rand", "all"}
    assert set(robustness["rep"]) >= {"lex", "L01", "L36n"}
    assert (core["subset"] == "core").all() and len(core) == 4
    assert not read(paths, "p1_hubs.csv").empty


def test_p3_agreement_uses_core_labels(analyzed) -> None:
    _, paths, occurrences = analyzed
    table = read(paths, "p3_agreement.csv")

    core = table[table["subset"] == "core"]
    assert {"token_id", "theme", "paragraph_id", "pred_next"} <= set(core["label"])
    assert (core["n"].astype(int) <= int((occurrences["stratum"] == "core").sum())).all()
    assert not read(paths, "p3_layer_nmi.csv").empty


def test_p4_types_groups_and_vocab(analyzed) -> None:
    _, paths, _ = analyzed
    types = read(paths, "p4_types.csv")

    assert set(types["token_id"].astype(int)) == {3, 4, 5}  # f_sample >= 10
    lex = types[types["rep"] == "lex"]
    assert np.allclose(lex["self_similarity"].astype(float), 1.0)
    assert types.loc[types["token_id"].astype(int) == 3, "group"].iloc[0] == "target"
    assert not read(paths, "p4_groups.csv").empty
    summary = read(paths, "vocab_summary.csv")
    assert "script_class" in set(summary["label"])
    neighbors = read(paths, "vocab_target_neighbors.csv")
    assert list(neighbors["word"]) == ["alvo"]


def test_rerun_is_skipped_until_an_input_changes(analyzed, caplog) -> None:
    settings, paths, _ = analyzed
    caplog.set_level("INFO")

    analysis.run(settings, paths)

    assert "já existem" in caplog.text
