from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from gender_networks import artifacts
from gender_networks.artifacts import OCCURRENCE_COLUMNS, RunPaths, ensure_dir
from gender_networks.settings import LensSettings, PathsSettings, Settings
from gender_networks.vocab_lens import (
    PRED_NEXT_FILE,
    RANK_COLUMNS,
    LensResult,
    VocabIndex,
    crossover_layer,
    lens_ranks,
    lexical_sanity,
    pred_next_agreement,
    rank_frame,
    raw_logit_argmax,
    representations,
    run,
    summarize_ranks,
    topk_jaccard,
    transitions,
)


def make_index(rows) -> VocabIndex:
    return VocabIndex.from_weight(torch.tensor(rows, dtype=torch.float32), "cpu")


def brute_force(weight: np.ndarray, queries: np.ndarray, own: np.ndarray, nxt: np.ndarray, k: int):
    """float64 reference for ranks, top-k and the raw-logit argmax."""

    unit = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    sims = q @ unit.T
    rows = np.arange(len(own))
    own_rank = (sims > sims[rows, own][:, None]).sum(1)
    next_rank = np.where(nxt >= 0, (sims > sims[rows, np.maximum(nxt, 0)][:, None]).sum(1), -1)
    topk = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return own_rank, next_rank, topk, (queries @ weight.T).argmax(1)


# ---------------------------------------------------------------------------------------------
# ranks and top-k
# ---------------------------------------------------------------------------------------------


def test_hand_built_ranks_topk_and_ties() -> None:
    # row 2 is scaled by 3 to show that norms do not affect the cosine ranking
    index = make_index([[1, 0, 0], [0, 1, 0], [0, 0, 3], [1, 1, 0], [-1, 0, 0]])
    queries = torch.tensor([[2.0, 1.0, 0.0], [0.1, 0.0, 5.0], [1.0, 1.0, 0.2]])
    # q0: cos order e3 (0.949) > e0 (0.894) > e1 > e2 > e4
    # q1: e2 > e0 > e3 > e1 > e4
    # q2: e3 > e0 = e1 (exact tie) > e2 > e4
    result = lens_ranks(index, queries, [1, 2, 3], [4, -1, 0], k=2, batch_size=8)

    np.testing.assert_array_equal(result.topk[:2], [[3, 0], [2, 0]])
    assert result.topk.dtype == np.int32
    assert result.topk[2, 0] == 3 and result.topk[2, 1] in {0, 1}  # tie order is arbitrary
    np.testing.assert_array_equal(result.own_rank, [2, 0, 0])
    np.testing.assert_array_equal(result.next_rank, [4, -1, 1])  # the tie e0 = e1 shares rank 1
    np.testing.assert_array_equal(result.own_in_topk(), [False, True, True])
    np.testing.assert_array_equal(result.next_in_topk(), [False, False, True])
    assert result.own_cos[0] == pytest.approx(1 / np.sqrt(5), abs=1e-6)
    assert np.isnan(result.next_cos[1]) and result.next_cos[0] == pytest.approx(-2 / np.sqrt(5))
    np.testing.assert_allclose(result.topk_cos[0], [1.5 / np.sqrt(2.5), 2 / np.sqrt(5)], atol=1e-6)
    assert result.logit_argmax is None


def test_next_rank_minus_one_when_there_is_no_next_token() -> None:
    rng = np.random.default_rng(3)
    index = make_index(rng.standard_normal((20, 6)))
    queries = torch.from_numpy(rng.standard_normal((5, 6)).astype(np.float32))
    result = lens_ranks(index, queries, [0, 1, 2, 3, 4], [-1, 5, -1, -1, 19], k=4, batch_size=2)

    np.testing.assert_array_equal(result.next_rank[[0, 2, 3]], [-1, -1, -1])
    assert (result.next_rank[[1, 4]] >= 0).all()
    assert not result.next_in_topk()[[0, 2, 3]].any()
    assert np.isnan(result.next_cos[[0, 2, 3]]).all()
    assert not np.isnan(result.next_cos[[1, 4]]).any()


def test_batching_matches_single_batch_and_float64_reference() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((60, 16)) * rng.uniform(0.5, 2.0, (60, 1))
    queries = rng.standard_normal((45, 16))
    own = rng.integers(0, 60, 45)
    nxt = np.where(rng.random(45) < 0.2, -1, rng.integers(0, 60, 45))
    index = make_index(weight)
    q = torch.from_numpy(queries.astype(np.float32))

    runs = [lens_ranks(index, q, own, nxt, 5, bs, logit_argmax=True) for bs in (1, 7, 45, 100)]
    for other in runs[1:]:
        for field in ("topk", "own_rank", "next_rank", "logit_argmax"):
            np.testing.assert_array_equal(getattr(other, field), getattr(runs[0], field))
        np.testing.assert_allclose(other.topk_cos, runs[0].topk_cos, atol=1e-6)
        np.testing.assert_allclose(other.own_cos, runs[0].own_cos, atol=1e-6)

    own_rank, next_rank, topk, argmax = brute_force(weight, queries, own, nxt, 5)
    np.testing.assert_array_equal(runs[0].own_rank, own_rank)
    np.testing.assert_array_equal(runs[0].next_rank, next_rank)
    np.testing.assert_array_equal(runs[0].topk, topk)
    np.testing.assert_array_equal(runs[0].logit_argmax, argmax)


def test_invalid_inputs_are_rejected() -> None:
    index = make_index(np.eye(4))
    q = torch.eye(4)
    with pytest.raises(ValueError, match="k must"):
        lens_ranks(index, q, [0, 1, 2, 3], [1, 2, 3, -1], k=5, batch_size=2)
    with pytest.raises(ValueError, match="own token ids"):
        lens_ranks(index, q, [0, 1, 2, 4], [1, 2, 3, -1], k=2, batch_size=2)
    with pytest.raises(ValueError, match="next token ids"):
        lens_ranks(index, q, [0, 1, 2, 3], [1, 2, 3, -2], k=2, batch_size=2)
    with pytest.raises(ValueError, match="queries must have shape"):
        lens_ranks(index, q[:3], [0, 1, 2, 3], [1, 2, 3, -1], k=2, batch_size=2)


def test_from_weight_does_not_modify_the_callers_tensor() -> None:
    weight = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    index = VocabIndex.from_weight(weight, "cpu")

    assert weight.tolist() == [[3.0, 4.0], [0.0, 2.0]]
    np.testing.assert_allclose(index.norms.numpy(), [5.0, 2.0])
    np.testing.assert_allclose(index.unit.numpy(), [[0.6, 0.8], [0.0, 1.0]])
    assert index.weight.tolist() == [[3.0, 4.0], [0.0, 2.0]]


def test_from_weight_keeps_the_stored_dtype_for_the_logit_check() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.bfloat16)
    index = VocabIndex.from_weight(weight, "cpu")

    assert index.unit.dtype == torch.float32 and index.weight.dtype == torch.bfloat16
    assert torch.equal(index.weight, weight)


# ---------------------------------------------------------------------------------------------
# lexical sanity and the logit check
# ---------------------------------------------------------------------------------------------


def test_own_token_is_rank_zero_at_lex() -> None:
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((80, 12)) * rng.uniform(0.1, 3.0, (80, 1))
    index = make_index(weight)
    own = rng.integers(0, 80, 30)
    nxt = rng.integers(0, 80, 30)

    lexical = lens_ranks(index, None, own, nxt, 4, 8)
    explicit = lens_ranks(index, torch.from_numpy(weight[own].astype(np.float32)), own, nxt, 4, 8)

    assert (lexical.own_rank == 0).all()
    np.testing.assert_array_equal(lexical.topk[:, 0], own)
    np.testing.assert_allclose(lexical.own_cos, 1.0, atol=1e-6)
    np.testing.assert_array_equal(lexical.next_rank, explicit.next_rank)
    info = lexical_sanity(lexical, own, index)
    assert info["own_rank_nonzero"] == 0 and info["unexplained"] == 0 and info["top1_not_own"] == 0


def test_duplicated_rows_never_count_as_unexplained_at_lex() -> None:
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((30, 8))
    weight[7] = weight[3]  # a duplicated embedding row ties with the own row
    index = make_index(weight)

    result = lens_ranks(index, None, [3, 7, 10], [7, 3, -1], 3, 2)
    info = lexical_sanity(result, [3, 7, 10], index)

    assert (result.own_rank <= 1).all()
    assert info["unexplained"] == 0
    assert info["identical_rows"] == info["own_rank_nonzero"]
    assert set(result.topk[0, :2]) == {3, 7} and set(result.topk[1, :2]) == {3, 7}


def test_lexical_sanity_classifies_nonzero_ranks() -> None:
    index = make_index([[1, 0], [1, 0], [0, 1], [1, 1]])
    fabricated = LensResult(
        topk=np.array([[1, 0], [2, 3], [1, 3]], dtype=np.int32),
        topk_cos=np.array([[1.0000001, 1.0], [1.0, 0.7], [1.2, 1.0]], dtype=np.float32),
        own_rank=np.array([1, 0, 1]),
        next_rank=np.array([-1, -1, -1]),
        own_cos=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        next_cos=np.full(3, np.nan, dtype=np.float32),
    )
    info = lexical_sanity(fabricated, [0, 2, 3], index)

    assert info["own_rank_nonzero"] == 2 and info["types_nonzero"] == 2
    assert info["identical_rows"] == 1  # row 1 duplicates row 0
    assert info["unexplained"] == 1  # row 1 beating row 3 by 0.2 cannot be rounding
    assert info["top1_not_own"] == 2
    assert info["example_token_ids"] == [0, 3]
    assert info["max_gap"] == pytest.approx(0.2, abs=1e-6)


def test_logit_argmax_uses_raw_dot_product_not_cosine() -> None:
    index = make_index([[1.0, 0.0], [6.0, 8.0]])  # row 1 is less aligned but ten times longer
    result = lens_ranks(index, torch.tensor([[1.0, 0.2]]), [0], [1], 2, 1, logit_argmax=True)

    assert result.topk[0, 0] == 0  # nearest by cosine
    assert result.logit_argmax.tolist() == [1]  # largest logit: 7.6 against 1.0


def test_raw_logit_argmax_chunks_and_ties() -> None:
    rng = np.random.default_rng(8)
    weight = rng.standard_normal((50, 6)) * rng.uniform(0.2, 3.0, (50, 1))
    rows = rng.standard_normal((9, 6))
    expected = (rows @ weight.T).argmax(1)
    w, x = torch.from_numpy(weight.astype(np.float32)), torch.from_numpy(rows.astype(np.float32))

    for chunk in (1, 7, 50, 1000):
        np.testing.assert_array_equal(raw_logit_argmax(w, x, chunk).numpy(), expected)
    # exact ties across chunks keep the lowest id, as torch.argmax does within one row
    tied = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    query = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    for chunk in (1, 2, 3, 4):
        assert raw_logit_argmax(tied, query, chunk).tolist() == [0, 1]
    with pytest.raises(ValueError, match="chunk_rows"):
        raw_logit_argmax(tied, query, 0)


def test_logit_argmax_at_lex_uses_the_embedding_row() -> None:
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((25, 5)) * rng.uniform(0.2, 3.0, (25, 1))
    index = make_index(weight)
    own = np.array([0, 4, 9, 24])

    result = lens_ranks(index, None, own, [-1, -1, -1, -1], 3, 3, logit_argmax=True)

    np.testing.assert_array_equal(result.logit_argmax, (weight[own] @ weight.T).argmax(1))


def test_pred_next_agreement() -> None:
    info = pred_next_agreement(np.array([1, 2, 3, 4]), np.array([1, 0, 3, 4], dtype=np.int32))

    assert info == {"n": 4, "agree": 3, "rate": 0.75, "disagreeing_occurrences": [1]}
    with pytest.raises(ValueError, match="shape"):
        pred_next_agreement(np.array([1, 2]), np.array([1, 2, 3]))


def test_pred_next_agreement_reports_the_cosine_top1_rate() -> None:
    pred = np.array([5, 6, 7, 8], dtype=np.int32)
    info = pred_next_agreement(pred, pred, cosine_top1=np.array([5, 0, 0, 8], dtype=np.int32))

    assert info["rate"] == 1.0 and info["disagreeing_occurrences"] == []
    assert info["cosine_top1_rate"] == 0.5
    with pytest.raises(ValueError, match="cosine_top1"):
        pred_next_agreement(pred, pred, cosine_top1=np.array([1]))


# ---------------------------------------------------------------------------------------------
# Jaccard, chain and summaries
# ---------------------------------------------------------------------------------------------


def test_topk_jaccard() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[3, 2, 10], [7, 8, 9], [9, 8, 7]])

    np.testing.assert_allclose(topk_jaccard(a, b), [0.5, 0.0, 1.0])


def test_default_representations_and_transitions() -> None:
    reps = representations(Settings(name="x"))

    assert reps == ["lex", "L01", "L18", "L36", "L36n"]
    assert transitions(reps) == [("lex", "L01"), ("L01", "L18"), ("L18", "L36"), ("L36", "L36n")]


def test_summarize_ranks_by_group_skips_missing_next_tokens() -> None:
    result = LensResult(
        topk=np.zeros((4, 2), dtype=np.int32),
        topk_cos=np.zeros((4, 2), dtype=np.float32),
        own_rank=np.array([0, 3, 1, 10]),
        next_rank=np.array([5, -1, 0, 2]),
        own_cos=np.array([0.9, 0.5, 0.7, 0.1], dtype=np.float32),
        next_cos=np.array([0.2, np.nan, 0.8, 0.4], dtype=np.float32),
    )
    frame = rank_frame(result, "L18")
    frame["token_category"] = ["whole_word", "whole_word", "continuation", "continuation"]
    frame["stratum"] = ["core", "core", "core", "target"]

    summary = summarize_ranks(frame)["L18"]
    overall = summary["all"]

    assert list(frame.columns[: len(RANK_COLUMNS)]) == RANK_COLUMNS
    assert overall["n"] == 4 and overall["next_n"] == 3
    assert overall["own_rank_median"] == 2.0 and overall["own_rank0"] == 0.25
    assert overall["own_in_topk"] == 0.5  # ranks 0 and 1 are below k = 2
    assert overall["next_rank_median"] == 2.0 and overall["next_rank0"] == pytest.approx(1 / 3)
    assert overall["next_closer_than_own"] == pytest.approx(2 / 3)
    assert overall["next_cos_mean"] == pytest.approx(0.4666667, abs=1e-6)
    assert summary["by_token_category"]["whole_word"]["next_n"] == 1
    assert summary["by_token_category"]["whole_word"]["next_rank_mean"] == 5.0
    assert summary["by_stratum"]["target"]["own_rank_mean"] == 10.0


def test_crossover_layer() -> None:
    curve = pd.DataFrame(
        {
            "layer": [0, 1, 2, 3, 2],
            "group_by": ["all", "all", "all", "all", "stratum"],
            "next_closer_than_own": [0.0, 0.3, 0.6, 0.9, 1.0],
        }
    )
    assert crossover_layer(curve) == 2
    assert crossover_layer(curve.assign(next_closer_than_own=0.1)) is None


# ---------------------------------------------------------------------------------------------
# run() end to end on a tiny fake run directory
# ---------------------------------------------------------------------------------------------

N, V, D, LAYERS, K = 14, 40, 8, 36, 3


def to_bf16(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).contiguous()


def make_run(tmp_path: Path) -> tuple[Settings, RunPaths, dict[str, np.ndarray]]:
    settings = Settings(
        name="tiny",
        paths=PathsSettings(runs_dir="runs", report_dir="relatorio"),
        lens=LensSettings(k=K, batch_size=4),
    )
    paths = RunPaths.from_settings(settings, tmp_path)
    rng = np.random.default_rng(11)
    weight = to_bf16(rng.standard_normal((V, D)) * rng.uniform(0.5, 2.0, (V, 1)))
    token_ids = rng.integers(0, V, N)
    next_ids = np.append(token_ids[1:], -1)
    layers = to_bf16(rng.standard_normal((LAYERS, N, D)))
    normalized = to_bf16(rng.standard_normal((N, D)))
    pred_next = (normalized.float() @ weight.float().T).argmax(1).numpy().astype(np.int32)

    ensure_dir(paths.sample_dir)
    ensure_dir(paths.reps_dir)
    categories = ["whole_word", "word_start", "continuation", "punctuation", "number"]
    rows = []
    for i in range(N):
        row = dict.fromkeys(OCCURRENCE_COLUMNS, "")
        row.update(
            occurrence_id=i,
            stratum=["core", "target"][i % 2],
            token_id=int(token_ids[i]),
            next_token_id=int(next_ids[i]),
            token_category=categories[i % len(categories)],
            pos_in_sequence=i + 1,
            local_context="um «texto», com vírgula",
        )
        rows.append(row)
    pd.DataFrame(rows, columns=OCCURRENCE_COLUMNS).to_csv(paths.occurrences, index=False)
    save_file({"weight": weight}, str(paths.embeddings))
    for rep, layer in settings.model.layers.items():
        save_file({"x": layers[layer - 1].contiguous()}, str(paths.rep(rep)))
    save_file({"x": normalized}, str(paths.rep("L36n")))
    np.save(paths.all_layers, layers.view(torch.int16).numpy().view(np.uint16))
    np.save(paths.reps_dir / PRED_NEXT_FILE, pred_next)
    data = {"weight": weight.float().numpy(), "token_ids": token_ids, "next_ids": next_ids}
    data["layers"] = layers.float().numpy()
    data["L18"] = data["layers"][17]
    return settings, paths, data


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifacts, "_git_describe", lambda root: None)


def test_run_end_to_end(tmp_path: Path, no_git: None, caplog: pytest.LogCaptureFixture) -> None:
    settings, paths, data = make_run(tmp_path)
    run(settings, paths, device="cpu")
    out = paths.lens_dir

    reps = ["lex", "L01", "L18", "L36", "L36n"]
    for rep in reps:
        topk = np.load(out / f"topk_{rep}.npy")
        assert topk.shape == (N, K) and topk.dtype == np.int32
    ranks = pd.read_csv(out / "ranks.csv")
    assert list(ranks.columns) == RANK_COLUMNS
    assert len(ranks) == N * len(reps) and list(ranks["rep"].unique()) == reps
    lex = ranks[ranks["rep"] == "lex"]
    assert (lex["own_rank"] == 0).all() and lex["own_in_topk"].all()
    last = ranks[ranks["occurrence_id"] == N - 1]
    assert (last["next_rank"] == -1).all() and not last["next_in_topk"].any()

    own_rank, next_rank, topk, _ = brute_force(
        data["weight"], data["L18"], data["token_ids"], data["next_ids"], K
    )
    l18 = ranks[ranks["rep"] == "L18"]
    np.testing.assert_array_equal(l18["own_rank"], own_rank)
    np.testing.assert_array_equal(l18["next_rank"], next_rank)
    np.testing.assert_array_equal(np.load(out / "topk_L18.npy"), topk)

    jaccard = pd.read_csv(out / "topk_jaccard.csv")
    assert len(jaccard) == 4 * N and jaccard["jaccard"].between(0, 1).all()
    expected = topk_jaccard(np.load(out / "topk_L01.npy"), np.load(out / "topk_L18.npy"))
    got = jaccard[(jaccard["source"] == "L01") & (jaccard["target"] == "L18")]["jaccard"]
    np.testing.assert_allclose(got, expected, atol=1e-6)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_occurrences"] == N and summary["vocab_size"] == V and summary["k"] == K
    assert summary["pred_next_agreement"]["rate"] == 1.0
    assert summary["lex_sanity"]["own_rank_nonzero"] == 0
    assert summary["ranks"]["lex"]["all"]["own_rank0"] == 1.0
    assert set(summary["ranks"]["L36"]["by_token_category"]) == {
        "whole_word",
        "word_start",
        "continuation",
        "punctuation",
        "number",
    }
    assert set(summary["ranks"]["L36"]["by_stratum"]) == {"core", "target"}
    assert summary["ranks"]["L36"]["all"]["next_n"] == N - 1
    assert set(summary["topk_jaccard"]) == {"lex->L01", "L01->L18", "L18->L36", "L36->L36n"}

    curve = pd.read_csv(out / "layer_curve.csv")
    overall = curve[curve["group_by"] == "all"]
    assert overall["layer"].tolist() == list(range(LAYERS + 1))
    l18_curve = overall[overall["layer"] == 18].iloc[0]
    l18_summary = summary["ranks"]["L18"]["all"]
    assert l18_curve["own_rank_mean"] == pytest.approx(l18_summary["own_rank_mean"])
    assert l18_curve["next_rank_median"] == pytest.approx(l18_summary["next_rank_median"])
    matches = summary["layer_curve"]["all_layers_match_reps"]
    assert matches == {"L01": True, "L18": True, "L36": True}
    assert summary["layer_curve"]["reused_layers"] == [1, 18, 36]
    # a layer computed from the memmap agrees with the float64 reference too
    own7, next7, _, _ = brute_force(
        data["weight"], data["layers"][6], data["token_ids"], data["next_ids"], K
    )
    l07 = overall[overall["layer"] == 7].iloc[0]
    assert l07["own_rank_mean"] == pytest.approx(own7.mean())
    assert l07["next_rank_median"] == pytest.approx(np.median(next7[:-1]))
    assert 0.0 <= summary["pred_next_agreement"]["cosine_top1_rate"] <= 1.0

    manifest = json.loads((out / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "lens" and manifest["device"] == "cpu"
    assert manifest["pred_next_agreement_rate"] == 1.0

    # cached: a second run is skipped; a missing output or --force recomputes
    stamp = (out / "_manifest.json").stat().st_mtime_ns
    with caplog.at_level(logging.INFO, logger="gender_networks.vocab_lens"):
        run(settings, paths, device="cpu")
    assert "já existem" in caplog.text
    assert (out / "_manifest.json").stat().st_mtime_ns == stamp
    (out / "summary.json").unlink()
    run(settings, paths, device="cpu")
    assert (out / "summary.json").exists()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gender_networks.vocab_lens"):
        run(settings, paths, force=True, device="cpu", layer_curve=False)
    assert "já existem" not in caplog.text


def test_run_without_all_layers_or_pred_next(tmp_path: Path, no_git: None) -> None:
    settings, paths, _ = make_run(tmp_path)
    paths.all_layers.unlink()
    (paths.reps_dir / PRED_NEXT_FILE).unlink()
    run(settings, paths, device="cpu")

    summary = json.loads((paths.lens_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pred_next_agreement"] is None and summary["layer_curve"] is None
    assert not (paths.lens_dir / "layer_curve.csv").exists()


def test_run_requires_the_representations(tmp_path: Path, no_git: None) -> None:
    settings, paths, _ = make_run(tmp_path)
    paths.rep("L18").unlink()

    with pytest.raises(FileNotFoundError, match="L18"):
        run(settings, paths, device="cpu")
