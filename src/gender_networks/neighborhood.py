"""Neighbourhood measures on k-NN neighbour sets (plan decision 9).

Neighbour sets come from the ``knn`` stage either as a dense ``[N, k]`` array (variants (a) and
(c), fixed size) or as CSR ``(indptr, idx)`` (variant (b), every candidate tied with the k-th,
variable size). Every measure here accepts both, and all of them are vectorized: set operations
become sorted-key operations on ``row * M + id`` so 15 000 vertices x 20 seeds stay cheap.
"""

from __future__ import annotations

import numpy as np

Neighbors = np.ndarray | tuple[np.ndarray, np.ndarray]


def neighbor_rows(neighbors: Neighbors) -> tuple[np.ndarray, np.ndarray, int]:
    """Flatten neighbour sets into ``(rows, ids, n_rows)``.

    ``neighbors`` is a dense ``[N, k]`` integer array or a CSR pair ``(indptr, idx)``.

    Args:
        neighbors: Neighbour sets of ``N`` rows. Dense: row ``i`` lists the neighbours of
            ``i``. CSR: the neighbours of ``i`` are ``idx[indptr[i]:indptr[i + 1]]``.

    Returns:
        ``rows`` and ``ids`` as int64 arrays with one entry per (row, neighbour) pair, in row
        order, and the number of rows ``N``.

    Raises:
        ValueError: If the dense array is not 2-D or the CSR pair is malformed.
    """

    if isinstance(neighbors, tuple):
        indptr, idx = (np.asarray(part) for part in neighbors)
        if indptr.ndim != 1 or idx.ndim != 1 or indptr.size == 0:
            raise ValueError("CSR neighbours must be a pair of 1-D arrays (indptr, idx)")
        if indptr[0] != 0 or indptr[-1] != idx.size or np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must start at 0, be non-decreasing and end at len(idx)")
        n_rows = indptr.size - 1
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
        return rows, idx.astype(np.int64, copy=False), n_rows
    array = np.asarray(neighbors)
    if array.ndim != 2:
        raise ValueError("dense neighbours must have shape [N, k]")
    n_rows, k = array.shape
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    return rows, array.reshape(-1).astype(np.int64, copy=False), n_rows


def as_csr(neighbors: Neighbors) -> tuple[np.ndarray, np.ndarray]:
    """CSR form of any neighbour structure (dense rows keep their order).

    Args:
        neighbors: Dense ``[N, k]`` array or CSR pair.

    Returns:
        ``(indptr, idx)`` with ``indptr`` as int64 ``[N + 1]``; a dense array gives
        ``indptr = [0, k, 2k, ...]`` and its values row by row.
    """

    if isinstance(neighbors, tuple):
        indptr, idx = neighbors
        return np.asarray(indptr, dtype=np.int64), np.asarray(idx)
    array = np.asarray(neighbors)
    n_rows, k = array.shape
    return np.arange(0, n_rows * k + 1, k, dtype=np.int64), array.reshape(-1)


def set_sizes(neighbors: Neighbors) -> np.ndarray:
    """``|N_i|`` for every row, counting distinct ids.

    Args:
        neighbors: Dense ``[N, k]`` array or CSR pair.

    Returns:
        Int array ``[N]`` with the number of distinct neighbour ids of each row.
    """

    rows, ids, n_rows = neighbor_rows(neighbors)
    modulus = _modulus(ids)
    return np.bincount(_row_keys(rows, ids, modulus) // modulus, minlength=n_rows)


def _modulus(*ids: np.ndarray) -> int:
    """Row stride for ``row * M + id`` keys; ids must be non-negative.

    Args:
        *ids: Id arrays that will share the keys.

    Returns:
        One more than the largest id (1 when every array is empty), so each (row, id) pair
        maps to a distinct integer.

    Raises:
        ValueError: If an id is negative.
    """

    top = -1
    for part in ids:
        if part.size:
            if part.min() < 0:
                raise ValueError("neighbour ids must be non-negative")
            top = max(top, int(part.max()))
    return top + 1 if top >= 0 else 1


def _row_keys(rows: np.ndarray, ids: np.ndarray, modulus: int) -> np.ndarray:
    """Sorted distinct ``row * modulus + id`` keys (sets per row, duplicates dropped).

    Args:
        rows: Row of each entry.
        ids: Neighbour id of each entry.
        modulus: Stride from :func:`_modulus`.

    Returns:
        Sorted int64 array of the distinct keys; ``key // modulus`` recovers the row.
    """

    return np.unique(rows * modulus + ids)


def _jaccard_rows(a: Neighbors, b: Neighbors) -> np.ndarray:
    """Per-row Jaccard between two neighbour structures with the same number of rows.

    Both structures are turned into sorted ``row * M + id`` keys; the intersection size of a
    row is the number of keys the two key arrays share in that row, and the union size is
    ``|A_i| + |B_i| - |A_i & B_i|``.

    Args:
        a: Dense ``[N, k_a]`` array or CSR pair.
        b: Dense ``[N, k_b]`` array or CSR pair.

    Returns:
        float64 array ``[N]``; NaN where both rows are empty.

    Raises:
        ValueError: If the row counts differ or an id is negative.
    """

    rows_a, ids_a, n_a = neighbor_rows(a)
    rows_b, ids_b, n_b = neighbor_rows(b)
    if n_a != n_b:
        raise ValueError(f"neighbour structures have different row counts ({n_a} vs {n_b})")
    modulus = _modulus(ids_a, ids_b)
    keys_a = _row_keys(rows_a, ids_a, modulus)
    keys_b = _row_keys(rows_b, ids_b, modulus)
    size_a = np.bincount(keys_a // modulus, minlength=n_a)
    size_b = np.bincount(keys_b // modulus, minlength=n_a)
    common = np.intersect1d(keys_a, keys_b, assume_unique=True)
    inter = np.bincount(common // modulus, minlength=n_a)
    union = size_a + size_b - inter
    out = np.full(n_a, np.nan, dtype=np.float64)
    nonempty = union > 0
    out[nonempty] = inter[nonempty] / union[nonempty]
    return out


def jaccard(a: Neighbors, b: Neighbors) -> np.ndarray:
    """Per-row Jaccard ``|A_i & B_i| / |A_i | B_i|`` between any two neighbour structures.

    Rows are treated as sets (duplicates ignored). Two empty rows give NaN (undefined).

    Args:
        a: Dense ``[N, k_a]`` array or CSR pair.
        b: Dense ``[N, k_b]`` array or CSR pair.

    Returns:
        float64 array ``[N]`` with values in ``[0, 1]`` or NaN.
    """

    return _jaccard_rows(a, b)


def jaccard_fixed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard between dense ``[N, k_a]`` and ``[N, k_b]`` neighbour arrays (k may differ).

    Args:
        a: Dense neighbour array.
        b: Dense neighbour array with the same number of rows.

    Returns:
        float64 array ``[N]``. With equal sizes ``k`` and ``c`` common neighbours this is
        ``c / (2k - c)``.

    Raises:
        ValueError: If an argument is not 2-D.
    """

    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("jaccard_fixed expects two [N, k] arrays")
    return _jaccard_rows(a, b)


def jaccard_csr(
    indptr_a: np.ndarray, idx_a: np.ndarray, indptr_b: np.ndarray, idx_b: np.ndarray
) -> np.ndarray:
    """Jaccard between variable-size neighbour sets given in CSR form (variant (b)).

    Args:
        indptr_a: Row pointers of the first structure, ``[N + 1]``.
        idx_a: Neighbour ids of the first structure.
        indptr_b: Row pointers of the second structure, ``[N + 1]``.
        idx_b: Neighbour ids of the second structure.

    Returns:
        float64 array ``[N]``.
    """

    return _jaccard_rows((indptr_a, idx_a), (indptr_b, idx_b))


def type_set_jaccard(types_a: Neighbors, types_b: Neighbors) -> np.ndarray:
    """Jaccard between sets of neighbour token ids (variant (c), ``T_i``).

    Args:
        types_a: Neighbour token ids of every row, dense or CSR.
        types_b: Neighbour token ids of every row, dense or CSR.

    Returns:
        float64 array ``[N]``.
    """

    return _jaccard_rows(types_a, types_b)


def dominance(neighbors: Neighbors, token_ids: np.ndarray) -> np.ndarray:
    """Lexical dominance ``D_i``: fraction of neighbours sharing the token id of ``i``.

    Divides by ``|N_i|``, so variant (b), whose out-degree varies, is handled; an empty row
    gives NaN.

    Args:
        neighbors: Neighbour occurrence ids of every row, dense or CSR.
        token_ids: Token id of every occurrence, ``[N]``; the ids in ``neighbors`` index it.

    Returns:
        float64 array ``[N]`` with ``|{j in N_i : t_j = t_i}| / |N_i|``. The denominator
        counts the entries of the row, which are distinct in the sets the ``knn`` stage
        writes.

    Raises:
        ValueError: If ``token_ids`` does not have one entry per row.
    """

    tokens = np.asarray(token_ids)
    rows, ids, n_rows = neighbor_rows(neighbors)
    if n_rows != tokens.size:
        raise ValueError("token_ids must have one entry per neighbour row")
    matches = (tokens[ids] == tokens[rows]).astype(np.float64)
    same = np.bincount(rows, weights=matches, minlength=n_rows)
    sizes = np.bincount(rows, minlength=n_rows)
    out = np.full(n_rows, np.nan, dtype=np.float64)
    nonempty = sizes > 0
    out[nonempty] = same[nonempty] / sizes[nonempty]
    return out


def sample_frequencies(token_ids: np.ndarray) -> np.ndarray:
    """Per-occurrence count ``f_t`` of its token id among all occurrences.

    Args:
        token_ids: Token id of every occurrence, ``[N]``.

    Returns:
        Int array ``[N]``; entry ``i`` is the number of occurrences with the token id of
        ``i``.
    """

    _, inverse, counts = np.unique(np.asarray(token_ids), return_inverse=True, return_counts=True)
    return counts[inverse.reshape(-1)]


def dominance_ceiling(
    token_ids: np.ndarray, f_sample: np.ndarray | None, k: int
) -> np.ndarray:
    """Largest lexical dominance an occurrence can reach: ``min(f_t - 1, k) / k``.

    An occurrence with ``f_t = 1`` has ceiling 0 (no other occurrence of its type), so raw
    ``D`` mixes the network's behaviour with the sample design; dividing by the ceiling removes
    that. ``f_sample`` is the per-occurrence type count (the ``f_sample`` column of
    ``occurrences.csv``); when None it is counted from ``token_ids``.

    Args:
        token_ids: Token id of every occurrence, ``[N]``.
        f_sample: Type count of every occurrence, ``[N]``, or ``None``.
        k: Number of neighbours.

    Returns:
        float64 array ``[N]`` with values in ``[0, 1]``.

    Raises:
        ValueError: If ``k`` is below 1, the shapes differ or a frequency is below 1.
    """

    if k < 1:
        raise ValueError("k must be positive")
    tokens = np.asarray(token_ids)
    freq = sample_frequencies(tokens) if f_sample is None else np.asarray(f_sample)
    if freq.shape != tokens.shape:
        raise ValueError("f_sample must have one entry per occurrence")
    if np.any(freq < 1):
        raise ValueError("f_sample must be at least 1 for every occurrence")
    return np.minimum(freq - 1, k).astype(np.float64) / k


def normalized_dominance(dom: np.ndarray, ceiling: np.ndarray) -> np.ndarray:
    """``D / ceiling`` where the ceiling is positive, NaN otherwise.

    Args:
        dom: Dominance of every occurrence.
        ceiling: Ceiling from :func:`dominance_ceiling`, same shape.

    Returns:
        float64 array of the same shape; NaN for occurrences of types with ``f_t = 1``.
    """

    dom = np.asarray(dom, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    out = np.full(dom.shape, np.nan, dtype=np.float64)
    positive = ceiling > 0
    out[positive] = dom[positive] / ceiling[positive]
    return out


def _type_counts(
    neighbors: Neighbors, codes: np.ndarray, n_types: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Distinct ``row * n_types + type`` keys with their multiplicities.

    Args:
        neighbors: Neighbour occurrence ids, dense or CSR.
        codes: Type code (``0..n_types - 1``) of every occurrence.
        n_types: Number of distinct types, the key stride.

    Returns:
        ``(keys, counts, n_rows)``: the sorted keys, how many neighbours of the row have that
        type, and the number of rows.
    """

    rows, ids, n_rows = neighbor_rows(neighbors)
    keys, counts = np.unique(rows * n_types + codes[ids], return_counts=True)
    return keys, counts, n_rows


def type_composition_jaccard(
    neighbors_a: Neighbors, neighbors_b: Neighbors, token_ids: np.ndarray
) -> np.ndarray:
    """Weighted Jaccard ``J^w`` of the multisets of neighbour token ids.

    ``sum_u min(c_a(u), c_b(u)) / sum_u max(c_a(u), c_b(u))``. At the lexical stage a random
    tie only swaps occurrences of the same type inside a tie block, so the multiset (and
    ``J^w``) does not depend on the tie seed, unlike the plain Jaccard.

    Args:
        neighbors_a: Neighbour occurrence ids in the first network, dense or CSR.
        neighbors_b: Neighbour occurrence ids in the second network, dense or CSR.
        token_ids: Token id of every occurrence, ``[N]``.

    Returns:
        float64 array ``[N]``; NaN where both rows are empty.

    Raises:
        ValueError: If a structure does not have one row per occurrence.

    Notes:
        The seed invariance holds for rows whose boundary tie block has a single type; when
        two types share the block (identical embeddings, or cosines within ``eps``) the
        composition can change with the seed.
    """

    _, codes = np.unique(np.asarray(token_ids), return_inverse=True)
    codes = codes.reshape(-1).astype(np.int64)
    n_types = int(codes.max()) + 1 if codes.size else 1
    keys_a, counts_a, n_a = _type_counts(neighbors_a, codes, n_types)
    keys_b, counts_b, n_b = _type_counts(neighbors_b, codes, n_types)
    if n_a != n_b or n_a != codes.size:
        raise ValueError("both neighbour structures need one row per occurrence")
    keys = np.union1d(keys_a, keys_b)
    ca = np.zeros(keys.size, dtype=np.int64)
    cb = np.zeros(keys.size, dtype=np.int64)
    ca[np.searchsorted(keys, keys_a)] = counts_a
    cb[np.searchsorted(keys, keys_b)] = counts_b
    rows = keys // n_types
    low = np.bincount(rows, weights=np.minimum(ca, cb), minlength=n_a)
    high = np.bincount(rows, weights=np.maximum(ca, cb), minlength=n_a)
    out = np.full(n_a, np.nan, dtype=np.float64)
    nonempty = high > 0
    out[nonempty] = low[nonempty] / high[nonempty]
    return out


def noise_floor(rand: np.ndarray) -> np.ndarray:
    """Per-vertex mean Jaccard between tie seeds over disjoint pairs (0, 1), (2, 3), ...

    Disjoint pairs keep the pair estimates independent; with an odd number of seeds the last
    one is left out.

    Args:
        rand: Neighbour sets of ``R`` tie seeds, int array ``[R, N, k]``.

    Returns:
        float64 array ``[N]`` with the mean Jaccard ``J`` over the ``R // 2`` pairs (1 when
        the seeds agree). The change produced by the tie-break alone is ``1 - J``.

    Raises:
        ValueError: If ``rand`` is not 3-D or has fewer than two seeds.
    """

    rand = np.asarray(rand)
    if rand.ndim != 3 or rand.shape[0] < 2:
        raise ValueError("noise_floor expects an [R, N, k] array with R >= 2")
    pairs = [jaccard_fixed(rand[r], rand[r + 1]) for r in range(0, rand.shape[0] - 1, 2)]
    return np.mean(pairs, axis=0)
