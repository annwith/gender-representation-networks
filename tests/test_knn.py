from __future__ import annotations

import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from gender_networks.artifacts import OCCURRENCE_COLUMNS, VOCAB_COLUMNS, RunPaths, write_csv
from gender_networks.knn import (
    BoundarySelector,
    Candidates,
    SampleTypes,
    _block_candidates,
    candidates_path,
    knn_candidates,
    lexical_candidates,
    lexical_type_candidates,
    load_layer,
    load_representation,
    neighbors_path,
    normalize_rows,
    output_paths,
    rep_rng,
    representation_specs,
    run,
    select_neighbors,
    select_row,
    type_similarity,
    vocab_candidates_path,
    vocab_neighbors_path,
)
from gender_networks.network_analysis import build_union_knn_graph
from gender_networks.settings import (
    NetworkSettings,
    PathsSettings,
    Settings,
    VocabNetworkSettings,
)

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def dyadic_vectors(rng: np.random.Generator, n: int, d: int = 8) -> np.ndarray:
    """Integer vectors of norm 1, 2 or 4.

    Normalized entries are 0, +-1/2 or +-1, so every cosine is an exact multiple of 1/4 in
    floating point: ties are exact and every implementation computes identical values.
    """

    out = np.zeros((n, d))
    for i in range(n):
        kind = rng.integers(4)
        width = 1 if kind < 2 else 4
        scale = 1.0 if kind % 2 == 0 else 2.0
        cols = rng.choice(d, size=width, replace=False)
        out[i, cols] = scale * rng.choice([-1.0, 1.0], size=width)
    return out


def with_duplicates(rng: np.random.Generator, base: np.ndarray, n: int) -> np.ndarray:
    rows = np.concatenate([np.arange(len(base)), rng.integers(0, len(base), n - len(base))])
    return base[rng.permutation(rows)]


def brute_select(sims: np.ndarray, ids: np.ndarray, k: int, eps: float, tie: str) -> np.ndarray:
    """The selector written directly from its definition (reference)."""

    order = np.lexsort((ids, -sims))
    ids, sims = ids[order], sims[order]
    s_k = sims[k - 1]
    in_f = sims - s_k > eps
    in_b = np.abs(sims - s_k) <= eps
    if tie == "all":
        return ids[in_f | in_b]
    r = k - int(in_f.sum())
    chosen = set(np.sort(ids[in_b])[:r].tolist())
    return ids[in_f | np.isin(ids, list(chosen))]


def brute_rows(sim: np.ndarray, k: int, eps: float, tie: str) -> list[np.ndarray]:
    n = sim.shape[0]
    rows = []
    for i in range(n):
        others = np.delete(np.arange(n), i)
        rows.append(brute_select(sim[i, others], others, k, eps, tie))
    return rows


def csr_rows(indptr: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    return [idx[indptr[i] : indptr[i + 1]] for i in range(len(indptr) - 1)]


def exact_cosine(x: np.ndarray) -> np.ndarray:
    unit = normalize_rows(x)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def type_maxima(sim: np.ndarray, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force variant (c) scores: best occurrence j != i of every type, own type masked."""

    ids = np.unique(token_ids)
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    best = np.stack([s[:, token_ids == u].max(axis=1) for u in ids], axis=1)
    best[np.arange(len(token_ids)), np.searchsorted(ids, token_ids)] = -np.inf
    return ids, best


def make_candidates(rows: list[tuple[np.ndarray, np.ndarray]]) -> Candidates:
    """CSR candidates from (ids, sims) rows, each sorted by (-sim, id)."""

    idx, sim, lengths = [], [], []
    for ids, sims in rows:
        order = np.lexsort((ids, -sims))
        idx.append(ids[order])
        sim.append(sims[order])
        lengths.append(len(ids))
    return Candidates(
        np.concatenate([[0], np.cumsum(lengths)]), np.concatenate(idx), np.concatenate(sim)
    )


# ---------------------------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------------------------


def test_normalize_rows_float64_unit_and_centering() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    unit = normalize_rows(x)
    assert unit.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)
    np.testing.assert_allclose(unit[0], [0.6, 0.8])
    centered = normalize_rows(x, center=True)
    reference = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(centered, reference / np.linalg.norm(reference, axis=1)[:, None])
    np.testing.assert_allclose(normalize_rows(torch.tensor(x)), unit)
    with pytest.raises(ValueError, match="zero"):
        normalize_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="zero"):
        normalize_rows(np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]]), center=True)


# ---------------------------------------------------------------------------------------------
# golden regression against the frozen pilot
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_candidates", [6, 64])
def test_position_rule_reproduces_the_pilot(seed: int, n_candidates: int) -> None:
    rng = np.random.default_rng(seed)
    vectors = with_duplicates(rng, dyadic_vectors(rng, 30), 50)
    candidates, _ = knn_candidates(normalize_rows(vectors), 5, n_candidates, 0.0, block_size=7)
    for k in (1, 3, 5):
        pilot = build_union_knn_graph(vectors, k)
        pos = select_neighbors(candidates, k, 0.0, "position")
        assert pos.dtype == np.int32
        np.testing.assert_array_equal(pos, pilot.neighbors)
        ours = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(pos) for j in row}
        theirs = {(min(a, b), max(a, b)) for a, b in pilot.graph.edges()}
        assert ours == theirs


# ---------------------------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.0, 0.3])
def test_candidates_sorted_complete_and_selector_matches_brute_force(eps: float) -> None:
    rng = np.random.default_rng(5)
    vectors = with_duplicates(rng, dyadic_vectors(rng, 25), 45)
    sim = exact_cosine(vectors)
    k_max = 4
    candidates, _ = knn_candidates(normalize_rows(vectors), k_max, 6, eps, block_size=8)
    assert candidates.fallback is not None and candidates.fallback.any()
    for i in range(len(vectors)):
        ids, sims = candidates.row(i)
        assert i not in ids and len(set(ids.tolist())) == ids.size
        np.testing.assert_array_equal(np.lexsort((ids, -sims)), np.arange(ids.size))
        np.testing.assert_array_equal(sims, sim[i, ids])
        others = np.delete(sim[i], i)
        s_kmax = np.sort(others)[::-1][k_max - 1]
        needed = np.flatnonzero(sim[i] - s_kmax >= -eps)
        assert set(needed.tolist()) - {i} <= set(ids.tolist())
    for k in range(1, k_max + 1):
        for tie in ("position", "all"):
            expected = brute_rows(sim, k, eps, tie)
            got = select_neighbors(candidates, k, eps, tie)
            got_rows = csr_rows(*got) if tie == "all" else list(got)
            for row, ref in zip(got_rows, expected, strict=True):
                np.testing.assert_array_equal(row, ref)


def test_completeness_fallback_when_tie_block_exceeds_candidates() -> None:
    rng = np.random.default_rng(3)
    v = rng.standard_normal(6)
    vectors = np.vstack([np.tile(v, (25, 1)), rng.standard_normal((15, 6))])
    candidates, _ = knn_candidates(normalize_rows(vectors), 3, 8, 1e-6, block_size=16)
    assert candidates.fallback is not None
    assert candidates.fallback[:25].all() and (candidates.lengths[:25] == 24).all()
    # other rows fall back too when the 25 equal columns reach their 3rd neighbour
    sim = exact_cosine(vectors)
    np.fill_diagonal(sim, -np.inf)
    s3 = np.sort(sim, axis=1)[:, ::-1][:, 2]
    needed = (sim - s3[:, None] >= -1e-6).sum(axis=1)
    fallback = needed >= 8
    np.testing.assert_array_equal(candidates.fallback, fallback)
    # elsewhere the 25 equal columns may straddle the 8th position: that group is trimmed
    s8 = np.sort(sim, axis=1)[:, ::-1][:, 7]
    straddle = (sim >= s8[:, None]).sum(axis=1) > 8
    trimmed = np.where(straddle, (sim > s8[:, None]).sum(axis=1), 8)
    assert (straddle & ~fallback).any()
    np.testing.assert_array_equal(candidates.lengths, np.where(fallback, needed, trimmed))
    assert candidates.fallback_rows(1e-6) == int(fallback.sum())
    assert 25 < fallback.sum() < 40
    pos = select_neighbors(candidates, 3, 1e-6, "position")
    indptr, idx = select_neighbors(candidates, 3, 1e-6, "all")
    for i in range(25):
        np.testing.assert_array_equal(pos[i], [j for j in range(25) if j != i][:3])
        assert set(idx[indptr[i] : indptr[i + 1]].tolist()) == set(range(25)) - {i}
    rand = select_neighbors(candidates, 3, 1e-6, "random", np.random.default_rng(0))
    assert all(set(rand[i].tolist()) <= set(range(25)) - {i} for i in range(25))


def test_block_candidates_trims_a_tie_group_cut_by_the_kth_position() -> None:
    s = np.array(
        [
            [-np.inf, 0.9, 0.8, 0.5, 0.3, 0.3, 0.3, 0.1],
            [0.9, -np.inf, 0.2, 0.7, 0.6, 0.5, 0.4, 0.3],
        ]
    )
    block = _block_candidates(s, k_max=2, n_candidates=4, eps=1e-6)
    assert block.lengths.tolist() == [3, 4]
    assert block.idx.tolist() == [1, 2, 3, 0, 3, 4, 5]
    assert not block.fallback.any()
    np.testing.assert_allclose(block.kth_gap, [0.3 - 0.8, 0.5 - 0.7])


# ---------------------------------------------------------------------------------------------
# exact lexical derivation
# ---------------------------------------------------------------------------------------------


def lexical_case(weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = [1, 1, 2, 3, 5, 8, 1, 4, 2, 6, 1, 3]
    used = rng.choice(np.arange(3, len(weight)), size=len(counts), replace=False)
    used[:2] = [0, 1]  # token ids 0 and 1 share one embedding row below
    token_ids = np.repeat(used, counts)
    return token_ids[rng.permutation(token_ids.size)]


def test_lexical_candidates_equal_brute_force_on_materialized_vectors() -> None:
    rng = np.random.default_rng(11)
    weight = dyadic_vectors(rng, 30)
    weight[1] = weight[0]
    token_ids = lexical_case(weight, rng)
    types = SampleTypes.from_token_ids(token_ids)
    sim_types = type_similarity(normalize_rows(weight[types.ids]))
    sim = exact_cosine(weight[token_ids])
    k_max = 5
    candidates = lexical_candidates(types, sim_types, k_max, 0.0)
    for i in range(token_ids.size):
        ids, sims = candidates.row(i)
        assert i not in ids
        np.testing.assert_array_equal(np.lexsort((ids, -sims)), np.arange(ids.size))
        np.testing.assert_array_equal(sims, sim[i, ids])
        s_kmax = np.sort(np.delete(sim[i], i))[::-1][k_max - 1]
        assert set(np.flatnonzero(sim[i] >= s_kmax).tolist()) - {i} == set(ids.tolist())
    for k in (1, 2, 3, 5):
        for tie in ("position", "all"):
            expected = brute_rows(sim, k, 0.0, tie)
            got = select_neighbors(candidates, k, 0.0, tie)
            got_rows = csr_rows(*got) if tie == "all" else list(got)
            for row, ref in zip(got_rows, expected, strict=True):
                np.testing.assert_array_equal(row, ref)


@pytest.mark.parametrize("eps", [1e-9, 1e-6])
def test_lexical_candidates_with_random_embeddings(eps: float) -> None:
    rng = np.random.default_rng(12)
    weight = rng.standard_normal((30, 16))
    weight[1] = weight[0]
    token_ids = lexical_case(weight, rng)
    types = SampleTypes.from_token_ids(token_ids)
    sim_types = type_similarity(normalize_rows(weight[types.ids]))
    candidates = lexical_candidates(types, sim_types, 5, eps)
    sim = exact_cosine(weight[token_ids])  # duplicates differ here by rounding only
    for k in (2, 5):
        for tie in ("position", "all"):
            expected = brute_rows(sim, k, eps, tie)
            got = select_neighbors(candidates, k, eps, tie)
            got_rows = csr_rows(*got) if tie == "all" else list(got)
            for row, ref in zip(got_rows, expected, strict=True):
                assert sorted(row.tolist()) == sorted(ref.tolist())


# ---------------------------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------------------------


def test_selector_invariants() -> None:
    rng = np.random.default_rng(21)
    levels = np.array([0.9, 0.7, 0.5, 0.3])
    rows = []
    for _ in range(300):
        m = int(rng.integers(8, 16))
        sims = rng.choice(levels, size=m) + rng.choice([0.0, 2e-4, -3e-4], size=m)
        rows.append((rng.choice(1000, size=m, replace=False), sims))
    candidates = make_candidates(rows)
    for k in range(1, 7):
        for eps in (0.0, 1e-3):
            selector = BoundarySelector(candidates, k, eps)
            pos = selector.pick("position")
            rand = selector.pick("random", np.random.default_rng(k))
            indptr, idx = selector.pick("all")
            for i in range(candidates.n_rows):
                ids, sims = candidates.row(i)
                s_k = sims[k - 1]
                f = set(ids[sims - s_k > eps].tolist())
                b = set(ids[np.abs(sims - s_k) <= eps].tolist())
                assert (
                    set(idx[indptr[i] : indptr[i + 1]].tolist())
                    == set(ids[sims >= s_k - eps].tolist())
                    == f | b
                )
                for row in (pos[i], rand[i]):
                    chosen = set(row.tolist())
                    assert len(chosen) == k and f <= chosen <= f | b
                    position = {int(j): p for p, j in enumerate(ids)}
                    assert [position[int(j)] for j in row] == sorted(position[int(j)] for j in row)
                np.testing.assert_array_equal(pos[i], brute_select(sims, ids, k, eps, "position"))
            assert selector.stats()["tied_rows"] == int((selector.n_b > k - selector.n_f).sum())


def test_random_ties_are_uniform_and_reproducible() -> None:
    ids = np.array([7, 3, 10, 11, 12, 13, 14, 1])
    sims = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1])
    draws = 20_000
    candidates = make_candidates([(ids, sims)] * draws)
    rand = select_neighbors(candidates, 4, 1e-6, "random", np.random.default_rng(123))
    assert (rand[:, :2] == [7, 3]).all()
    counts: dict[tuple[int, ...], int] = {}
    for row in rand[:, 2:]:
        key = tuple(sorted(row.tolist()))
        counts[key] = counts.get(key, 0) + 1
    subsets = list(itertools.combinations([10, 11, 12, 13, 14], 2))
    assert set(counts) == set(subsets)
    expected = draws / len(subsets)
    chi2 = sum((counts[s] - expected) ** 2 / expected for s in subsets)
    assert chi2 < 33.7  # 99.99% quantile of chi-square with 9 degrees of freedom

    again = select_neighbors(candidates, 4, 1e-6, "random", np.random.default_rng(123))
    other = select_neighbors(candidates, 4, 1e-6, "random", np.random.default_rng(124))
    np.testing.assert_array_equal(rand, again)
    assert not np.array_equal(rand, other)
    # one vectorized draw equals the rows drawn one after the other from the same generator
    rng = np.random.default_rng(123)
    sequential = [
        select_row(
            sims[np.lexsort((ids, -sims))], ids[np.lexsort((ids, -sims))], 4, 1e-6, "random", rng
        )
        for _ in range(50)
    ]
    np.testing.assert_array_equal(np.stack(sequential), rand[:50])


def test_eps_boundary() -> None:
    eps = 1e-6
    ids = np.array([5, 9, 2, 7])
    tie = select_row(np.array([0.9, 0.8 + 0.5 * eps, 0.8, 0.7]), ids, 2, eps, "position")
    assert tie.tolist() == [5, 2]  # id 2 wins the eps-tie despite its lower similarity
    gap = select_row(np.array([0.9, 0.8 + 2 * eps, 0.8, 0.7]), ids, 2, eps, "position")
    assert gap.tolist() == [5, 9]
    all_tie = select_row(np.array([0.9, 0.8 + 0.5 * eps, 0.8, 0.7]), ids, 2, eps, "all")
    assert all_tie.tolist() == [5, 9, 2]
    # F and B are measured from s_k: a chain of eps/2 steps does not keep growing B
    chain = np.array([0.9, 0.8 + eps, 0.8 + 0.5 * eps, 0.8, 0.8 - 0.5 * eps, 0.8 - 1.5 * eps])
    chain_ids = np.arange(6)
    assert select_row(chain, chain_ids, 3, eps, "all").tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        select_row(chain, chain_ids, 3, eps, "random")
    with pytest.raises(ValueError):
        select_row(chain, chain_ids, 3, eps, "nearest")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# variant (c)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dyadic", [True, False])
def test_variant_c_best_occurrence_per_type(dyadic: bool) -> None:
    rng = np.random.default_rng(31)
    n = 44
    vectors = (
        with_duplicates(rng, dyadic_vectors(rng, 30), n)
        if dyadic
        else (rng.standard_normal((n, 6)))
    )
    token_ids = rng.choice([4, 9, 15, 16, 23, 42, 50, 61, 77], size=n)
    types = SampleTypes.from_token_ids(token_ids)
    eps = 0.0 if dyadic else 1e-9
    _, type_candidates = knn_candidates(
        normalize_rows(vectors), 4, 10, eps, block_size=6, types=types, n_type_candidates=5
    )
    assert type_candidates is not None
    ids, best = type_maxima(exact_cosine(vectors), token_ids)
    for k in (1, 2, 4):
        pos = select_neighbors(type_candidates, k, eps, "position")
        rand = select_neighbors(type_candidates, k, eps, "random", np.random.default_rng(1))
        for i in range(n):
            others = np.flatnonzero(ids != token_ids[i])
            expected = brute_select(best[i, others], ids[others], k, eps, "position")
            np.testing.assert_array_equal(pos[i], expected)
            for row in (pos[i], rand[i]):
                assert token_ids[i] not in row and len(set(row.tolist())) == k


def test_variant_c_for_lex_is_the_knn_among_sample_types() -> None:
    rng = np.random.default_rng(41)
    weight = dyadic_vectors(rng, 30)
    weight[1] = weight[0]
    token_ids = lexical_case(weight, rng)
    types = SampleTypes.from_token_ids(token_ids)
    unit_types = normalize_rows(weight[types.ids])
    sim_types = type_similarity(unit_types)
    lex_types = lexical_type_candidates(types, sim_types, 5, 6, 0.0)
    # the same scores from the materialized lexical occurrence vectors
    _, from_vectors = knn_candidates(
        normalize_rows(weight[token_ids]),
        5,
        8,
        0.0,
        block_size=9,
        types=types,
        n_type_candidates=6,
    )
    assert from_vectors is not None
    type_sim = exact_cosine(weight[types.ids])
    for k in (1, 3, 5):
        pos = select_neighbors(lex_types, k, 0.0, "position")
        np.testing.assert_array_equal(pos, select_neighbors(from_vectors, k, 0.0, "position"))
        for i, t in enumerate(types.inverse):
            others = np.delete(np.arange(types.n_types), t)
            expected = brute_select(type_sim[t, others], types.ids[others], k, 0.0, "position")
            np.testing.assert_array_equal(pos[i], expected)
            assert token_ids[i] not in pos[i]


# ---------------------------------------------------------------------------------------------
# centered variant
# ---------------------------------------------------------------------------------------------


def test_centered_variant_uses_vectors_minus_their_mean() -> None:
    rng = np.random.default_rng(51)
    x = rng.standard_normal((40, 5)) + np.array([30.0, -20.0, 0.0, 10.0, 5.0])
    centered, _ = knn_candidates(normalize_rows(x, center=True), 3, 8, 1e-9, block_size=11)
    raw, _ = knn_candidates(normalize_rows(x), 3, 8, 1e-9, block_size=11)
    pos = select_neighbors(centered, 3, 1e-9, "position")
    expected = brute_rows(exact_cosine(x - x.mean(axis=0)), 3, 1e-9, "position")
    np.testing.assert_array_equal(pos, np.stack(expected))
    assert not np.array_equal(pos, select_neighbors(raw, 3, 1e-9, "position"))


# ---------------------------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------------------------


def tiny_settings() -> Settings:
    return Settings(
        name="tiny",
        paths=PathsSettings(
            raw_dir="raw", corpus_dir="corpus", runs_dir="runs", report_dir="report"
        ),
        networks=NetworkSettings(
            k_values=[2, 3],
            k_main=3,
            seeds=2,
            candidates=6,
            type_candidates=5,
            block_size=7,
            vocab=VocabNetworkSettings(k_values=[2, 3]),
        ),
    )


def bf16(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32).to(torch.bfloat16)


def write_fake_run(root: Path, settings: Settings) -> tuple[RunPaths, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(61)
    paths = RunPaths.from_settings(settings, root)
    vocab_size, d, special = 40, 16, [38, 39]
    weight = bf16(rng.standard_normal((vocab_size, d)))
    types = rng.choice(38, size=12, replace=False)
    token_ids = np.repeat(types, [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7])
    token_ids = token_ids[rng.permutation(token_ids.size)]
    n = token_ids.size
    rows = []
    for i, token in enumerate(token_ids):
        row = dict.fromkeys(OCCURRENCE_COLUMNS, "")
        row.update(occurrence_id=i, token_id=int(token), stratum="core", pos_in_sequence=i + 1)
        rows.append(row)
    write_csv(paths.occurrences, rows, OCCURRENCE_COLUMNS)
    vocab_rows = []
    for token in range(vocab_size):
        row = dict.fromkeys(VOCAB_COLUMNS, "")
        row.update(token_id=token, is_special=token in special, script_class="latin")
        vocab_rows.append(row)
    write_csv(paths.vocab_types, vocab_rows, VOCAB_COLUMNS)
    paths.reps_dir.mkdir(parents=True)
    save_file({"weight": weight}, str(paths.embeddings))
    for name in ("L01", "L18", "L36", "L36n"):
        offset = 5.0 * rng.standard_normal(d)
        save_file({"x": bf16(rng.standard_normal((n, d)) + offset)}, str(paths.rep(name)))
    return paths, token_ids, weight.to(torch.float64).numpy()


def test_run_end_to_end(tmp_path: Path) -> None:
    settings = tiny_settings()
    paths, token_ids, weight = write_fake_run(tmp_path, settings)
    n = token_ids.size
    run(settings, paths)

    reps = list(representation_specs(settings))
    assert reps == ["lex", "L01", "L18", "L36", "L36n", "L01c", "L18c", "L36c"]
    assert all(path.exists() for path in output_paths(settings, paths))
    for rep in reps:
        with np.load(candidates_path(paths, rep)) as data:
            assert set(data.files) == {"indptr", "idx", "sim"}
            assert data["indptr"].shape == (n + 1,) and data["indptr"].dtype == np.int64
            assert data["idx"].dtype == np.int32 and data["sim"].dtype == np.float64
        for k in (2, 3):
            with np.load(neighbors_path(paths, rep, k)) as data:
                assert set(data.files) == {
                    "pos",
                    "rand",
                    "all_indptr",
                    "all_idx",
                    "types_pos",
                    "types_rand",
                }
                assert data["pos"].shape == (n, k) and data["pos"].dtype == np.int32
                assert data["rand"].shape == (2, n, k) and data["rand"].dtype == np.int32
                assert data["all_indptr"].shape == (n + 1,)
                assert data["all_indptr"].dtype == np.int64 and data["all_idx"].dtype == np.int32
                assert np.diff(data["all_indptr"]).min() >= k
                assert data["types_pos"].shape == (n, k)
                assert data["types_rand"].shape == (2, n, k)
                assert (data["pos"] != np.arange(n)[:, None]).all()
                assert (data["types_pos"] != token_ids[:, None]).all()
                assert np.isin(data["types_pos"], token_ids).all()
        with np.load(neighbors_path(paths, rep, 3, sensitivity=True)) as data:
            assert set(data.files) == {"pos", "rand"}
            assert data["pos"].shape == (n, 3) and data["rand"].shape == (2, n, 3)

    # lexical: occurrences of a frequent type see the smallest other occurrence ids of the type
    with np.load(neighbors_path(paths, "lex", 3)) as data:
        lex_pos, lex_rand = data["pos"], data["rand"]
    for i in range(n):
        same = np.flatnonzero(token_ids == token_ids[i])
        same = same[same != i]
        if same.size >= 3:
            assert lex_pos[i].tolist() == same[:3].tolist()
            assert np.isin(lex_rand[:, i], same).all()
    lex_cand = Candidates.load(candidates_path(paths, "lex"))
    for r in range(2):
        rng = np.random.default_rng([438, zlib.crc32(b"lex"), 3, r])
        np.testing.assert_array_equal(
            lex_rand[r], select_neighbors(lex_cand, 3, settings.networks.eps, "random", rng)
        )
    assert not np.array_equal(lex_rand[0], lex_rand[1])

    # contextual and centered neighbour sets follow from the stored representation
    x = load_representation(paths, settings, "L01")
    for rep, center in (("L01", False), ("L01c", True)):
        cand, _ = knn_candidates(normalize_rows(x, center=center), 3, 6, 1e-4, block_size=7)
        with np.load(neighbors_path(paths, rep, 3)) as data:
            np.testing.assert_array_equal(data["pos"], select_neighbors(cand, 3, 1e-6))
            rng = rep_rng(438, rep, 3, 1)
            np.testing.assert_array_equal(
                data["rand"][1], select_neighbors(cand, 3, 1e-6, "random", rng)
            )

    # vocabulary network: special ids excluded, rows are indices into vocab_ids
    with np.load(vocab_candidates_path(paths)) as data:
        assert set(data.files) == {"vocab_ids", "indptr", "idx", "sim"}
        vocab_ids = data["vocab_ids"]
    np.testing.assert_array_equal(vocab_ids, np.arange(38))
    expected = np.stack(brute_rows(exact_cosine(weight[:38]), 3, 1e-6, "position"))
    for k in (2, 3):
        with np.load(vocab_neighbors_path(paths, k)) as data:
            assert set(data.files) == {"vocab_ids", "pos", "rand"}
            assert data["pos"].shape == (38, k) and data["rand"].shape == (1, 38, k)
            np.testing.assert_array_equal(data["pos"], expected[:, :k])

    manifest = json.loads((paths.knn_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "knn" and manifest["n_occurrences"] == n
    assert set(manifest["representations"]) == set(reps)
    lex_stats = manifest["representations"]["lex"]["k"]["3"]
    assert lex_stats["tied_rows"] > 0 and lex_stats["max_tie_block"] >= 4
    assert {"eps_sensitivity", "types", "all_mean_degree"} <= set(lex_stats)
    assert "fallback_rows_eps_sensitivity" in manifest["representations"]["L01"]["candidates"]
    assert manifest["vocab"]["rows"] == 38 and manifest["vocab"]["special_excluded"] == 2

    # a second run is skipped; force recomputes
    stamp = candidates_path(paths, "L01").stat().st_mtime_ns
    run(settings, paths)
    assert candidates_path(paths, "L01").stat().st_mtime_ns == stamp
    run(settings, paths, force=True)
    assert candidates_path(paths, "L01").stat().st_mtime_ns != stamp


def test_run_requires_inputs(tmp_path: Path) -> None:
    settings = tiny_settings()
    with pytest.raises(FileNotFoundError, match="extract"):
        run(settings, RunPaths.from_settings(settings, tmp_path))


def test_load_layer_and_fallback_from_all_layers(tmp_path: Path) -> None:
    settings = tiny_settings()
    paths, _, _ = write_fake_run(tmp_path, settings)
    x18 = load_representation(paths, settings, "L18")
    stack = np.zeros((36, x18.shape[0], x18.shape[1]), dtype=np.uint16)
    stack[17] = torch.from_numpy(x18).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    np.save(paths.all_layers, stack)
    np.testing.assert_array_equal(load_layer(paths.all_layers, 18), x18)
    paths.rep("L18").unlink()
    np.testing.assert_array_equal(load_representation(paths, settings, "L18"), x18)
    with pytest.raises(FileNotFoundError):
        load_representation(paths, settings, "L99")
