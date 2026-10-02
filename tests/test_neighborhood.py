from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from gender_networks.neighborhood import (
    as_csr,
    dominance,
    dominance_ceiling,
    jaccard,
    jaccard_csr,
    jaccard_fixed,
    noise_floor,
    normalized_dominance,
    set_sizes,
    type_composition_jaccard,
    type_set_jaccard,
)
from gender_networks.network_analysis import (
    jaccard_per_occurrence,
    same_token_neighbor_fraction,
)


def _set_jaccard(a, b) -> float:
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


def _random_rows(rng: np.random.Generator, n: int, k: int, pool: int) -> np.ndarray:
    return np.stack([rng.choice(pool, size=k, replace=False) for _ in range(n)])


def _random_csr(rng: np.random.Generator, n: int, pool: int, max_size: int):
    sizes = rng.integers(0, max_size + 1, size=n)
    rows = [rng.choice(pool, size=s, replace=False) for s in sizes]
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    idx = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    return indptr, idx.astype(np.int32), rows


def test_jaccard_fixed_matches_set_reference_same_and_different_k() -> None:
    rng = np.random.default_rng(0)
    a = _random_rows(rng, 200, 10, 40)
    b = _random_rows(rng, 200, 10, 40)
    c = _random_rows(rng, 200, 5, 40)
    expected_ab = [_set_jaccard(x, y) for x, y in zip(a, b, strict=True)]
    expected_ac = [_set_jaccard(x, y) for x, y in zip(a, c, strict=True)]
    np.testing.assert_allclose(jaccard_fixed(a, b), expected_ab)
    np.testing.assert_allclose(jaccard_fixed(a, c), expected_ac)
    np.testing.assert_allclose(jaccard_fixed(a, b), jaccard_per_occurrence(a, b))
    np.testing.assert_array_equal(jaccard_fixed(a, a), np.ones(200))
    with pytest.raises(ValueError):
        jaccard_fixed(a, b[:10])


def test_jaccard_csr_matches_set_reference_with_empty_rows() -> None:
    rng = np.random.default_rng(1)
    indptr_a, idx_a, rows_a = _random_csr(rng, 150, 60, 12)
    indptr_b, idx_b, rows_b = _random_csr(rng, 150, 60, 12)
    expected = np.array([_set_jaccard(x, y) for x, y in zip(rows_a, rows_b, strict=True)])
    got = jaccard_csr(indptr_a, idx_a, indptr_b, idx_b)
    np.testing.assert_allclose(got, expected, equal_nan=True)
    both_empty = (np.diff(indptr_a) == 0) & (np.diff(indptr_b) == 0)
    assert np.isnan(got[both_empty]).all()


def test_jaccard_mixes_dense_and_csr() -> None:
    rng = np.random.default_rng(2)
    dense = _random_rows(rng, 80, 6, 30)
    indptr, idx, rows = _random_csr(rng, 80, 30, 9)
    expected = [_set_jaccard(x, y) for x, y in zip(dense, rows, strict=True)]
    np.testing.assert_allclose(jaccard(dense, (indptr, idx)), expected, equal_nan=True)
    np.testing.assert_allclose(jaccard(dense, as_csr(dense)), np.ones(80))


def test_type_set_jaccard_on_token_id_sets() -> None:
    a = np.array([[151643, 5, 9], [1, 2, 3]])
    b = np.array([[9, 5, 7], [4, 5, 6]])
    np.testing.assert_allclose(type_set_jaccard(a, b), [2 / 4, 0.0])


def test_set_sizes_counts_distinct_ids() -> None:
    indptr = np.array([0, 3, 3, 5])
    idx = np.array([4, 4, 1, 2, 3])
    np.testing.assert_array_equal(set_sizes((indptr, idx)), [2, 0, 2])


def test_dominance_dense_matches_pilot_and_divides_by_variable_sizes() -> None:
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 6, size=100)
    dense = _random_rows(rng, 100, 8, 100)
    np.testing.assert_allclose(
        dominance(dense, tokens), same_token_neighbor_fraction(dense, tokens)
    )
    tokens = np.array([7, 7, 7, 8, 8])
    indptr = np.array([0, 3, 4, 4, 6, 10])
    idx = np.array([1, 2, 3, 0, 4, 1, 0, 1, 2, 3])
    got = dominance((indptr, idx), tokens)
    np.testing.assert_allclose(got[[0, 1, 3, 4]], [2 / 3, 1.0, 0.5, 1 / 4])
    assert np.isnan(got[2])


def test_dominance_ceiling_and_normalization() -> None:
    tokens = np.array([1, 2, 2, 3, 3, 3, 3, 3, 3])
    np.testing.assert_allclose(
        dominance_ceiling(tokens, None, 4), [0, 1 / 4, 1 / 4] + [4 / 4] * 6
    )
    f_sample = np.array([1, 5, 50])
    np.testing.assert_allclose(dominance_ceiling(np.array([9, 8, 7]), f_sample, 10), [0, 0.4, 1])
    ceiling = np.array([0.0, 0.5, 1.0])
    normalized = normalized_dominance(np.array([0.0, 0.25, 0.3]), ceiling)
    assert np.isnan(normalized[0])
    np.testing.assert_allclose(normalized[1:], [0.5, 0.3])
    with pytest.raises(ValueError):
        dominance_ceiling(tokens, np.zeros(9, dtype=int), 4)


def _lexical_neighbors(
    token_ids: np.ndarray, type_vectors: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Exact lexical k-NN: occurrences ranked by type similarity, ties broken at random.

    Distinct random type vectors never tie, so every tie is between occurrences of one type
    (a block of identical vectors), as at the lexical stage.
    """

    rng = np.random.default_rng(seed)
    unit = type_vectors / np.linalg.norm(type_vectors, axis=1, keepdims=True)
    sims = (unit @ unit.T)[token_ids][:, token_ids]
    n = token_ids.size
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        candidates = np.delete(np.arange(n), i)
        order = np.lexsort((rng.random(candidates.size), -sims[i, candidates]))
        out[i] = candidates[order[:k]]
    return out


def test_lexical_dominance_equals_ceiling() -> None:
    rng = np.random.default_rng(4)
    tokens = rng.choice(12, size=90, p=np.linspace(1, 12, 12) / 78)
    vectors = rng.normal(size=(12, 16))
    for k in (3, 5):
        neighbors = _lexical_neighbors(tokens, vectors, k, seed=0)
        np.testing.assert_allclose(dominance(neighbors, tokens), dominance_ceiling(tokens, None, k))


def test_type_composition_jaccard_matches_multiset_reference() -> None:
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 5, size=60)
    a = _random_rows(rng, 60, 7, 60)
    indptr, idx, rows = _random_csr(rng, 60, 60, 9)
    expected = []
    for row_a, row_b in zip(a, rows, strict=True):
        ca, cb = Counter(tokens[row_a].tolist()), Counter(tokens[row_b].tolist())
        low = sum(min(ca[u], cb[u]) for u in ca | cb)
        high = sum(max(ca[u], cb[u]) for u in ca | cb)
        expected.append(low / high if high else float("nan"))
    got = type_composition_jaccard(a, (indptr, idx), tokens)
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_type_composition_jaccard_is_seed_invariant_at_the_lexical_stage() -> None:
    rng = np.random.default_rng(6)
    tokens = rng.choice(10, size=80, p=np.linspace(1, 10, 10) / 55)
    vectors = rng.normal(size=(10, 12))
    k = 6
    seeds = [_lexical_neighbors(tokens, vectors, k, seed=s) for s in range(3)]
    other = _random_rows(rng, 80, k, 80)  # stands for a contextual layer
    for s in seeds[1:]:
        np.testing.assert_allclose(type_composition_jaccard(seeds[0], s, tokens), 1.0)
        np.testing.assert_allclose(
            type_composition_jaccard(seeds[0], other, tokens),
            type_composition_jaccard(s, other, tokens),
        )
    # The plain Jaccard does depend on the tie seed, which is why J^w complements it.
    assert jaccard_fixed(seeds[0], seeds[1]).min() < 1.0


def test_noise_floor_uses_disjoint_seed_pairs() -> None:
    rng = np.random.default_rng(7)
    rand = np.stack([_random_rows(rng, 30, 4, 12) for _ in range(5)])
    expected = (jaccard_fixed(rand[0], rand[1]) + jaccard_fixed(rand[2], rand[3])) / 2
    np.testing.assert_allclose(noise_floor(rand), expected)  # seed 4 is left out
    same = np.repeat(rand[:1], 4, axis=0)
    np.testing.assert_allclose(noise_floor(same), 1.0)
    with pytest.raises(ValueError):
        noise_floor(rand[:1])
