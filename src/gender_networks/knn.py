"""Stage ``knn``: candidate lists and k-NN neighbour sets (plan decision 8, stage 4).

Every similarity is a cosine computed in **float64 on the CPU**: at d = 2560, float32 errors of
1e-6 to 1e-5 would swamp the tie tolerance ``eps = 1e-6``. Rows are processed in blocks, so the
``N x N`` matrix is never stored; only a CSR list of candidates per row survives.

Candidates (``cand_{rep}.npz``). Row ``i`` holds the top ``K = networks.candidates`` neighbours
of ``i`` sorted by ``(-sim, idx)``, never ``i`` itself, and is **complete** for
``k_max = max(k_values)``: whenever the K-th candidate is within ``eps`` of the k_max-th
similarity ``s(k_max)``, the row is taken from the full block row instead, keeping every ``j``
with ``s_ij >= s(k_max) - eps``. Completeness uses ``max(eps, eps_sensitivity)`` so that the same
lists also serve the eps-sensitivity analysis.

Selector (shared by every variant and every ``k <= k_max``). With ``s_k`` the k-th similarity of
a sorted candidate row, ``F = {s > s_k + eps}``, ``B = {|s - s_k| <= eps}`` and
``r = k - |F|``; ``position`` keeps F plus the r smallest ids of B, ``random`` keeps F plus r
entries of B drawn uniformly without replacement, and ``all`` keeps ``F | B`` (variant (b)).
Measuring F and B from ``s_k`` itself avoids chains of eps-ties. Rows without a boundary tie
(``|B| == r``, almost every contextual row) take a vectorized fast path.

Lexical network (``lex``). Lexical occurrence vectors are copies of their type's embedding, so the
network is derived exactly from the ``U x U`` cosine among the sample types (``S_T[t, t] = 1``):
every occurrence of ``t`` shares one template (the occurrences of the most similar types, each
with its single type-pair similarity), which makes the cosine-1 ties exact and never builds the
``N x d`` lexical matrix.

Variant (c) (``types_pos`` / ``types_rand``). Each other type ``u`` is scored by its occurrence
most similar to ``i`` (``numpy.maximum.reduceat`` over columns grouped by token id), the own type
is masked, and the same candidate and selector routines run over token ids (the position rule
prefers the smaller token id). For ``lex`` this is the k-NN among the sample types.

Vocabulary network (``vocab_*``): the same routine over every non-special tokenizer id.

Random tie-breaking seeds: ``rand`` of representation ``rep`` and seed ``r`` uses
``default_rng([base_seed, crc32(rep), k, r])`` (the eps-sensitivity file reuses them),
``types_rand`` uses ``crc32(rep + "/types")`` and the vocabulary uses ``crc32("vocab")``.
"""

from __future__ import annotations

import logging
import os
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from gender_networks.artifacts import RunPaths, ensure_dir, write_manifest
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)

Tie = Literal["position", "random", "all"]
TIE_RULES: tuple[str, ...] = ("position", "random", "all")
LEX = "lex"
VOCAB = "vocab"
TYPES_SUFFIX = "/types"
# Budget for one float64 block of similarities; with the vocabulary (151 643 columns) this gives
# about 440 rows per block, next to the 3.1 GB float64 matrix itself.
BLOCK_BYTES = 512 << 20
NORM_CHUNK_ROWS = 16_384


# ---------------------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------------------


@dataclass
class Candidates:
    """Candidate neighbours of every row in CSR form, each row sorted by ``(-sim, idx)``.

    ``kth_gap`` (similarity of the K-th candidate minus ``s(k_max)``, ``-inf`` when the row holds
    every other item) and ``fallback`` (row taken in full) are diagnostics kept in memory only.
    """

    indptr: np.ndarray  # int64 [n + 1]
    idx: np.ndarray  # int32
    sim: np.ndarray  # float64
    kth_gap: np.ndarray | None = None  # float64 [n]
    fallback: np.ndarray | None = None  # bool [n]

    def __post_init__(self) -> None:
        self.indptr = np.asarray(self.indptr, dtype=np.int64)
        self.idx = np.asarray(self.idx, dtype=np.int32)
        self.sim = np.asarray(self.sim, dtype=np.float64)
        if self.indptr.ndim != 1 or self.indptr.size == 0 or self.indptr[0] != 0:
            raise ValueError("indptr must be a 1-D array starting at 0")
        if self.indptr[-1] != self.idx.size or self.idx.size != self.sim.size:
            raise ValueError("indptr, idx and sim do not describe the same entries")

    @property
    def n_rows(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """``(idx, sim)`` of row ``i``."""

        lo, hi = self.indptr[i], self.indptr[i + 1]
        return self.idx[lo:hi], self.sim[lo:hi]

    def take_rows(self, rows: np.ndarray) -> Candidates:
        """Candidates whose row ``i`` is row ``rows[i]`` of this one."""

        rows = np.asarray(rows, dtype=np.int64)
        lengths = self.lengths[rows]
        flat = _ranges(self.indptr[:-1][rows], lengths)
        return Candidates(
            indptr=_indptr(lengths),
            idx=self.idx[flat],
            sim=self.sim[flat],
            kth_gap=None if self.kth_gap is None else self.kth_gap[rows],
            fallback=None if self.fallback is None else self.fallback[rows],
        )

    def fallback_rows(self, eps: float) -> int:
        """Rows that the completeness rule would take in full at tolerance ``eps``."""

        if self.kth_gap is None:
            return 0
        return int(np.count_nonzero(self.kth_gap >= -eps))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"indptr": self.indptr, "idx": self.idx, "sim": self.sim}

    @classmethod
    def load(cls, path: Path) -> Candidates:
        with np.load(path) as data:
            return cls(indptr=data["indptr"], idx=data["idx"], sim=data["sim"])


@dataclass(frozen=True)
class SampleTypes:
    """Distinct token ids of the sample and where their occurrences are."""

    ids: np.ndarray  # int64 [U], sorted
    inverse: np.ndarray  # int64 [N], type index of each occurrence
    counts: np.ndarray  # int64 [U]
    order: np.ndarray  # int64 [N], occurrence ids grouped by type, increasing within a type
    starts: np.ndarray  # int64 [U + 1], group boundaries in ``order``

    @classmethod
    def from_token_ids(cls, token_ids: Sequence[int] | np.ndarray) -> SampleTypes:
        values = np.asarray(token_ids, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("token_ids must be a non-empty 1-D array")
        ids, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        return cls(ids, inverse.astype(np.int64), counts, order, _indptr(counts))

    @property
    def n_types(self) -> int:
        return int(self.ids.size)

    @property
    def n_occurrences(self) -> int:
        return int(self.inverse.size)

    def members(self, t: int) -> np.ndarray:
        return self.order[self.starts[t] : self.starts[t + 1]]


@dataclass
class _Block:
    lengths: np.ndarray
    idx: np.ndarray
    sim: np.ndarray
    kth_gap: np.ndarray
    fallback: np.ndarray


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _indptr(lengths: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out[1:])
    return out


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, s + l)`` for every pair, without a Python loop."""

    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    offsets = np.cumsum(lengths) - lengths
    return np.repeat(np.asarray(starts, dtype=np.int64) - offsets, lengths) + np.arange(total)


def rep_rng(base_seed: int, name: str, k: int, seed: int) -> np.random.Generator:
    """Generator of the random tie-breaking of ``name`` at ``k`` and seed index ``seed``."""

    return np.random.default_rng([base_seed, zlib.crc32(name.encode()), k, seed])


def _normalize_inplace(x: np.ndarray, chunk_rows: int = NORM_CHUNK_ROWS) -> np.ndarray:
    """Scale float64 rows to unit norm in chunks (``np.linalg.norm`` would copy the matrix)."""

    for start in range(0, x.shape[0], chunk_rows):
        rows = x[start : start + chunk_rows]
        norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
        bad = np.flatnonzero(~(norms > 0) | ~np.isfinite(norms))
        if bad.size:
            raise ValueError(
                f"cosine is undefined for zero or non-finite rows: {(bad + start)[:10].tolist()}"
            )
        rows /= norms[:, None]
    return x


def normalize_rows(x: np.ndarray | torch.Tensor, center: bool = False) -> np.ndarray:
    """float64 copy of ``x`` with unit rows; ``center`` subtracts the column mean first.

    Raises on zero rows, where the cosine is undefined.
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().to("cpu", torch.float64).numpy()
    out = np.array(x, dtype=np.float64, copy=True, order="C")
    if out.ndim != 2 or out.shape[0] == 0:
        raise ValueError("vectors must have shape [n, d] with n > 0")
    if center:
        out -= out.mean(axis=0, keepdims=True)
    return _normalize_inplace(out)


def type_similarity(unit: np.ndarray) -> np.ndarray:
    """Cosine among unit type vectors, clipped to [-1, 1], with an exact 1.0 diagonal."""

    sim = np.asarray(unit, dtype=np.float64) @ np.asarray(unit, dtype=np.float64).T
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)
    return sim


# ---------------------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------------------


def _block_candidates(s: np.ndarray, k_max: int, n_candidates: int, eps: float) -> _Block:
    """Complete candidate rows of a block of similarities.

    Every row of ``s`` must hold exactly one excluded entry set to ``-inf`` (the row itself, or
    its own type) and finite values elsewhere.
    """

    b, m = s.shape
    valid = m - 1
    if valid < k_max:
        raise ValueError(f"k_max = {k_max} needs at least {k_max + 1} items, got {m}")
    if n_candidates < k_max:
        raise ValueError(f"candidates ({n_candidates}) must be at least k_max ({k_max})")
    kk = min(n_candidates, valid)
    if kk == valid:
        # Every other item fits: a stable sort of -s orders by (-sim, idx); -inf goes last.
        top = np.argsort(-s, axis=1, kind="stable")[:, :kk]
        sim = np.take_along_axis(s, top, axis=1)
        return _Block(
            lengths=np.full(b, kk, dtype=np.int64),
            idx=top.astype(np.int32).ravel(),
            sim=sim.ravel(),
            kth_gap=np.full(b, -np.inf),
            fallback=np.zeros(b, dtype=bool),
        )
    values, top_t = torch.topk(torch.from_numpy(s), kk, dim=1, largest=True, sorted=False)
    top, sim = top_t.numpy(), values.numpy()
    order = np.lexsort((top, -sim), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    sim = np.take_along_axis(sim, order, axis=1)
    s_kmax = sim[:, k_max - 1]
    kth = sim[:, kk - 1]
    gap = kth - s_kmax
    fallback = gap >= -eps
    # A tie group that crosses the K-th position is cut arbitrarily by topk; drop it so the
    # stored rows are deterministic (it lies below s(k_max) - eps, so completeness holds).
    trim = (np.count_nonzero(s >= kth[:, None], axis=1) > kk) & ~fallback
    lengths = np.full(b, kk, dtype=np.int64)
    special = np.flatnonzero(fallback | trim)
    if special.size == 0:
        return _Block(lengths, top.astype(np.int32).ravel(), sim.ravel(), gap, fallback)
    idx_rows: list[np.ndarray] = list(top)
    sim_rows: list[np.ndarray] = list(sim)
    for r in special:
        if fallback[r]:
            j = np.flatnonzero(s[r] - s_kmax[r] >= -eps)
            order_r = np.lexsort((j, -s[r, j]))
            idx_rows[r], sim_rows[r] = j[order_r], s[r, j[order_r]]
        else:
            keep = sim[r] > kth[r]
            idx_rows[r], sim_rows[r] = top[r, keep], sim[r, keep]
        lengths[r] = idx_rows[r].size
    return _Block(
        lengths,
        np.concatenate(idx_rows).astype(np.int32),
        np.concatenate(sim_rows),
        gap,
        fallback,
    )


def _assemble(blocks: Sequence[_Block], ids: np.ndarray | None = None) -> Candidates:
    idx = np.concatenate([block.idx for block in blocks])
    if ids is not None:
        idx = ids[idx]
    return Candidates(
        indptr=_indptr(np.concatenate([block.lengths for block in blocks])),
        idx=idx,
        sim=np.concatenate([block.sim for block in blocks]),
        kth_gap=np.concatenate([block.kth_gap for block in blocks]),
        fallback=np.concatenate([block.fallback for block in blocks]),
    )


def rows_per_block(block_size: int, n_columns: int, block_bytes: int = BLOCK_BYTES) -> int:
    """Rows of one float64 similarity block: ``block_size`` capped by the memory budget."""

    return max(1, min(int(block_size), int(block_bytes) // (8 * max(1, n_columns))))


def knn_candidates(
    x: np.ndarray,
    k_max: int,
    n_candidates: int,
    eps: float,
    block_size: int = 1024,
    *,
    types: SampleTypes | None = None,
    n_type_candidates: int | None = None,
    block_bytes: int = BLOCK_BYTES,
    label: str | None = None,
) -> tuple[Candidates, Candidates | None]:
    """Complete cosine candidates of every row of unit float64 ``x`` against all other rows.

    With ``types``, the same block loop also yields variant (c): per row, the best occurrence of
    every other type (``idx`` are token ids), with the same completeness rule over types.
    """

    if x.dtype != np.float64 or x.ndim != 2:
        raise ValueError("x must be a float64 matrix of unit rows (see normalize_rows)")
    x = np.ascontiguousarray(x)
    n = x.shape[0]
    if n - 1 < k_max:
        raise ValueError(f"k_max = {k_max} needs more than {n} rows")
    if types is not None:
        if types.n_occurrences != n:
            raise ValueError("types do not describe the rows of x")
        if types.n_types - 1 < k_max:
            raise ValueError(f"variant (c) needs more than {k_max} distinct types")
    step = rows_per_block(block_size, n, block_bytes)
    n_blocks = -(-n // step)
    report_every = max(1, n_blocks // 10)
    blocks: list[_Block] = []
    type_blocks: list[_Block] = []
    tick = time.perf_counter()
    for number, start in enumerate(range(0, n, step), start=1):
        stop = min(n, start + step)
        rows = np.arange(stop - start)
        s = x[start:stop] @ x.T
        np.clip(s, -1.0, 1.0, out=s)
        s[rows, np.arange(start, stop)] = -np.inf
        blocks.append(_block_candidates(s, k_max, n_candidates, eps))
        if types is not None:
            best = np.maximum.reduceat(s[:, types.order], types.starts[:-1], axis=1)
            best[rows, types.inverse[start:stop]] = -np.inf
            type_blocks.append(
                _block_candidates(best, k_max, n_type_candidates or n_candidates, eps)
            )
            del best
        del s
        if label and (number % report_every == 0 or number == n_blocks):
            LOGGER.info(
                "%s: bloco %d/%d (%.1f s)", label, number, n_blocks, time.perf_counter() - tick
            )
    candidates = _assemble(blocks)
    type_candidates = _assemble(type_blocks, types.ids) if types is not None else None
    return candidates, type_candidates


def lexical_candidates(
    types: SampleTypes, sim_types: np.ndarray, k_max: int, eps: float
) -> Candidates:
    """Exact complete candidates of the lexical occurrence network, derived from the types.

    For each type ``t`` the other types are taken by decreasing ``S_T[t, u]`` (ties by smaller
    token id) until, excluding one self occurrence, ``k_max`` occurrences are covered and the
    next type falls below ``s(k_max) - eps``. Each occurrence of ``t`` gets that template minus
    itself, sorted by ``(-sim, occurrence_id)``.
    """

    n, n_types = types.n_occurrences, types.n_types
    if n - 1 < k_max:
        raise ValueError(f"k_max = {k_max} needs more than {n} occurrences")
    if sim_types.shape != (n_types, n_types):
        raise ValueError("sim_types must be the U x U similarity of the sample types")
    counts = types.counts
    type_index = np.arange(n_types)
    lengths = np.zeros(n, dtype=np.int64)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(n_types):
        row = sim_types[t]
        order = np.lexsort((type_index, -row))
        sorted_sim = row[order]
        covered = counts[order].copy()
        covered[order == t] -= 1
        j_star = int(np.searchsorted(np.cumsum(covered), k_max))
        s_kmax = sorted_sim[j_star]
        keep = order[: int(np.count_nonzero(sorted_sim - s_kmax >= -eps))]
        occ = types.order[_ranges(types.starts[keep], counts[keep])]
        sim = np.repeat(row[keep], counts[keep])
        by_rank = np.lexsort((occ, -sim))
        occ, sim = occ[by_rank], sim[by_rank]
        own = types.members(t)
        others = occ[None, :] != own[:, None]
        width = occ.size - 1
        rows.append(
            (
                own,
                np.broadcast_to(occ, others.shape)[others].reshape(own.size, width),
                np.broadcast_to(sim, others.shape)[others].reshape(own.size, width),
            )
        )
        lengths[own] = width
    indptr = _indptr(lengths)
    idx = np.empty(indptr[-1], dtype=np.int32)
    sims = np.empty(indptr[-1], dtype=np.float64)
    for own, occ_rows, sim_rows in rows:
        flat = indptr[own][:, None] + np.arange(occ_rows.shape[1])
        idx[flat] = occ_rows
        sims[flat] = sim_rows
    return Candidates(indptr, idx, sims, kth_gap=np.full(n, -np.inf), fallback=np.zeros(n, bool))


def lexical_type_candidates(
    types: SampleTypes, sim_types: np.ndarray, k_max: int, n_candidates: int, eps: float
) -> Candidates:
    """Variant (c) for ``lex``: the k-NN among the sample types, one row per occurrence.

    ``idx`` are token ids; the rows of all occurrences of a type are identical.
    """

    if types.n_types - 1 < k_max:
        raise ValueError(f"variant (c) needs more than {k_max} distinct types")
    masked = np.array(sim_types, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, -np.inf)
    per_type = _assemble([_block_candidates(masked, k_max, n_candidates, eps)], types.ids)
    return per_type.take_rows(types.inverse)


# ---------------------------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------------------------


class BoundarySelector:
    """Neighbour sets of size ``k`` from sorted candidate rows, for any tie rule.

    ``n_f`` and ``n_b`` are ``|F|`` and ``|B|`` of every row; since rows are sorted by
    ``(-sim, idx)``, F is the first ``n_f`` entries and B the next ``n_b``.
    """

    def __init__(self, candidates: Candidates, k: int, eps: float) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        lengths = candidates.lengths
        if lengths.size and lengths.min() < k:
            raise ValueError(f"candidate rows are shorter than k = {k}")
        self.candidates, self.k, self.eps = candidates, int(k), float(eps)
        n = candidates.n_rows
        starts = candidates.indptr[:-1]
        rows = np.repeat(np.arange(n), lengths)
        diff = candidates.sim - candidates.sim[starts + k - 1][rows]
        self.n_f = np.bincount(rows[diff > eps], minlength=n)
        self.n_b = np.bincount(rows[np.abs(diff) <= eps], minlength=n)

    @property
    def r(self) -> np.ndarray:
        return self.k - self.n_f

    @property
    def tied(self) -> np.ndarray:
        """Rows with a boundary tie (``|B| > r``)."""

        return self.n_b > self.r

    def stats(self) -> dict[str, Any]:
        tied = self.tied
        degree = self.n_f + self.n_b
        return {
            "tied_rows": int(tied.sum()),
            "max_tie_block": int(self.n_b[tied].max()) if tied.any() else 0,
            "all_mean_degree": float(degree.mean()) if degree.size else 0.0,
            "all_max_degree": int(degree.max()) if degree.size else 0,
        }

    def pick(
        self, tie: Tie = "position", rng: np.random.Generator | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """``[n, k]`` int32 neighbours (``position``/``random``) or CSR ``(indptr, idx)`` (``all``).

        Rows are ordered by decreasing similarity, then by id. ``random`` draws one uniform key
        per entry of B in the tied rows only (row order), so it is reproducible from ``rng``.
        """

        cand, k = self.candidates, self.k
        starts = cand.indptr[:-1]
        if tie == "all":
            degree = self.n_f + self.n_b
            return _indptr(degree), cand.idx[_ranges(starts, degree)]
        if tie not in ("position", "random"):
            raise ValueError(f"Unknown tie rule '{tie}' (expected one of {TIE_RULES})")
        if tie == "random" and rng is None:
            raise ValueError("the random tie rule needs a numpy Generator")
        out = cand.idx[starts[:, None] + np.arange(k)]
        tied = np.flatnonzero(self.tied)
        if tied.size == 0:
            return out
        n_f, n_b = self.n_f[tied], self.n_b[tied]
        flat = _ranges(starts[tied] + n_f, n_b)
        group = np.repeat(np.arange(tied.size), n_b)
        keys = cand.idx[flat] if tie == "position" else rng.random(flat.size)
        ranked = np.lexsort((keys, group))
        rank = np.arange(flat.size) - np.repeat(np.cumsum(n_b) - n_b, n_b)
        chosen = np.sort(ranked[rank < np.repeat(k - n_f, n_b)])
        sub = out[tied]
        sub[np.arange(k)[None, :] >= n_f[:, None]] = cand.idx[flat[chosen]]
        out[tied] = sub
        return out


def select_neighbors(
    candidates: Candidates,
    k: int,
    eps: float,
    tie: Tie = "position",
    rng: np.random.Generator | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Neighbour sets of every row (see :meth:`BoundarySelector.pick`)."""

    return BoundarySelector(candidates, k, eps).pick(tie, rng)


def select_row(
    sim: np.ndarray,
    idx: np.ndarray,
    k: int,
    eps: float,
    tie: Tie = "position",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """The selector on one candidate row sorted by ``(-sim, idx)``."""

    single = Candidates(np.array([0, len(idx)]), idx, sim)
    result = select_neighbors(single, k, eps, tie, rng)
    return result[1] if tie == "all" else result[0]


# ---------------------------------------------------------------------------------------------
# Artifact names and loading
# ---------------------------------------------------------------------------------------------


def candidates_path(paths: RunPaths, rep: str) -> Path:
    return paths.knn_dir / f"cand_{rep}.npz"


def neighbors_path(paths: RunPaths, rep: str, k: int, sensitivity: bool = False) -> Path:
    return paths.knn_dir / f"nbr_{rep}_k{k}{'_eps' if sensitivity else ''}.npz"


def vocab_candidates_path(paths: RunPaths) -> Path:
    return paths.knn_dir / "vocab_cand.npz"


def vocab_neighbors_path(paths: RunPaths, k: int) -> Path:
    return paths.knn_dir / f"vocab_k{k}.npz"


def representation_specs(settings: Settings) -> dict[str, tuple[str, bool]]:
    """Occurrence representations of a run: ``name -> (source representation, centered)``.

    Main ones first, then the robust ones, then the centered variants of the main contextual
    layers (``L01c``...).
    """

    net = settings.networks
    specs: dict[str, tuple[str, bool]] = {}
    for rep in [*net.representations, *net.robust_representations]:
        specs.setdefault(rep, (rep, False))
    if net.centered:
        for rep in net.representations:
            if rep in settings.model.layers:
                specs.setdefault(f"{rep}c", (rep, True))
    return specs


def output_paths(settings: Settings, paths: RunPaths) -> list[Path]:
    """Every file of a complete ``knn`` stage (the manifest is written last)."""

    net = settings.networks
    files: list[Path] = []
    for rep in representation_specs(settings):
        files.append(candidates_path(paths, rep))
        files += [neighbors_path(paths, rep, k) for k in net.k_values]
        files.append(neighbors_path(paths, rep, net.k_main, sensitivity=True))
    files.append(vocab_candidates_path(paths))
    files += [vocab_neighbors_path(paths, k) for k in net.vocab.k_values]
    files.append(paths.knn_dir / "_manifest.json")
    return files


def read_token_ids(path: Path) -> np.ndarray:
    """``token_id`` of every occurrence, checking that ``occurrence_id`` is the row index."""

    frame = pd.read_csv(path, usecols=["occurrence_id", "token_id"])
    occurrence = frame["occurrence_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(occurrence, np.arange(len(frame))):
        raise ValueError(f"{path}: occurrence_id must equal the row index")
    return frame["token_id"].to_numpy(dtype=np.int64)


def bf16_bits_to_float64(bits: np.ndarray) -> np.ndarray:
    """float64 array from raw bfloat16 bits stored as uint16."""

    # A private copy: memmap slices are read-only and torch refuses to wrap them silently.
    raw = np.array(bits, dtype=np.uint16, copy=True, order="C").view(np.int16)
    return torch.from_numpy(raw).view(torch.bfloat16).to(torch.float64).numpy()


def load_layer(path: Path, layer: int) -> np.ndarray:
    """Block output ``layer`` (1-based) of ``all_layers.npy`` as float64 ``[N, d]``."""

    stack = np.load(path, mmap_mode="r")
    if not 1 <= layer <= stack.shape[0]:
        raise ValueError(f"layer {layer} is outside 1..{stack.shape[0]}")
    return bf16_bits_to_float64(stack[layer - 1])


def load_representation(paths: RunPaths, settings: Settings, name: str) -> np.ndarray:
    """float64 ``[N, d]`` vectors of a contextual representation.

    Reads ``reps/{name}.safetensors``; a configured block layer falls back to ``all_layers.npy``.
    """

    path = paths.rep(name)
    if path.exists():
        return load_file(str(path))["x"].to(torch.float64).numpy()
    if name in settings.model.layers and paths.all_layers.exists():
        LOGGER.warning("%s ausente; lendo a camada de %s", path, paths.all_layers)
        return load_layer(paths.all_layers, settings.model.layers[name])
    raise FileNotFoundError(f"Representation {name} not found at {path}")


def load_embeddings(path: Path) -> torch.Tensor:
    """Input embedding rows of every tokenizer id (bfloat16 ``[V, d]``)."""

    return load_file(str(path))["weight"]


def _as_bool(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - {"true", "false", "1", "0", "1.0", "0.0"})
    if unknown:
        raise ValueError(f"Unexpected boolean values: {unknown[:5]}")
    return text.isin(["true", "1", "1.0"]).to_numpy()


def special_mask(paths: RunPaths, settings: Settings, vocab_size: int) -> np.ndarray:
    """Boolean ``[vocab_size]``: True for special tokenizer ids (excluded from ``vocab``).

    Uses ``sample/vocab_types.csv``; without it, the tokenizer's special and added ids.
    """

    mask = np.zeros(vocab_size, dtype=bool)
    if paths.vocab_types.exists():
        frame = pd.read_csv(paths.vocab_types, usecols=["token_id", "is_special"])
        ids = frame["token_id"].to_numpy(dtype=np.int64)
        if ids.size != vocab_size or not np.array_equal(np.sort(ids), np.arange(vocab_size)):
            raise ValueError(
                f"{paths.vocab_types} must list the ids 0..{vocab_size - 1} of the embedding rows"
            )
        mask[ids] = _as_bool(frame["is_special"])
        return mask
    from transformers import AutoTokenizer

    from gender_networks.tokens import special_token_ids

    LOGGER.warning("%s ausente; tokens especiais lidos do tokenizer", paths.vocab_types)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model.name_or_path, revision=settings.model.revision
    )
    if len(tokenizer) != vocab_size:
        raise ValueError(f"tokenizer has {len(tokenizer)} ids but embeddings {vocab_size} rows")
    mask[[i for i in special_token_ids(tokenizer) if 0 <= i < vocab_size]] = True
    return mask


def vocab_unit_rows(weight: torch.Tensor, vocab_ids: np.ndarray) -> np.ndarray:
    """Unit float64 rows ``weight[vocab_ids]``, filled chunk by chunk to bound peak memory."""

    ids = np.asarray(vocab_ids, dtype=np.int64)
    out = np.empty((ids.size, int(weight.shape[1])), dtype=np.float64)
    for start in range(0, ids.size, NORM_CHUNK_ROWS):
        chunk = torch.from_numpy(ids[start : start + NORM_CHUNK_ROWS])
        out[start : start + chunk.numel()] = weight[chunk].to(torch.float64).numpy()
    return _normalize_inplace(out)


def _save_npz(path: Path, compressed: bool = True, **arrays: np.ndarray) -> None:
    """Write an ``.npz`` atomically, so an interrupted run never leaves a half file behind."""

    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        (np.savez_compressed if compressed else np.savez)(stream, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------


def candidate_stats(candidates: Candidates, tolerances: dict[str, float]) -> dict[str, Any]:
    """Size of the candidate lists and rows taken in full at each tolerance."""

    lengths = candidates.lengths
    stats: dict[str, Any] = {
        "rows": candidates.n_rows,
        "entries": int(lengths.sum()),
        "mean_length": float(lengths.mean()) if lengths.size else 0.0,
        "max_length": int(lengths.max()) if lengths.size else 0,
        "fallback_rows": int(candidates.fallback.sum()) if candidates.fallback is not None else 0,
    }
    for label, eps in tolerances.items():
        stats[f"fallback_rows_{label}"] = candidates.fallback_rows(eps)
    return stats


def _draws(
    selector: BoundarySelector, seeds: int, rng_for: Callable[[int], np.random.Generator]
) -> np.ndarray:
    return np.stack([selector.pick("random", rng_for(seed)) for seed in range(seeds)])


def write_neighbor_sets(
    rep: str,
    candidates: Candidates,
    type_candidates: Candidates,
    settings: Settings,
    paths: RunPaths,
) -> dict[str, Any]:
    """Write ``nbr_{rep}_k{k}.npz`` for every k and the eps-sensitivity file; return tie stats."""

    net = settings.networks
    per_k: dict[str, Any] = {}
    for k in net.k_values:
        main = BoundarySelector(candidates, k, net.eps)
        types = BoundarySelector(type_candidates, k, net.eps)
        all_indptr, all_idx = main.pick("all")
        _save_npz(
            neighbors_path(paths, rep, k),
            pos=main.pick("position"),
            rand=_draws(main, net.seeds, lambda r, k=k: rep_rng(net.base_seed, rep, k, r)),
            all_indptr=all_indptr,
            all_idx=all_idx,
            types_pos=types.pick("position"),
            types_rand=_draws(
                types, net.seeds, lambda r, k=k: rep_rng(net.base_seed, rep + TYPES_SUFFIX, k, r)
            ),
        )
        sensitivity = BoundarySelector(candidates, k, net.eps_sensitivity)
        if k == net.k_main:
            _save_npz(
                neighbors_path(paths, rep, k, sensitivity=True),
                pos=sensitivity.pick("position"),
                rand=_draws(
                    sensitivity, net.seeds, lambda r, k=k: rep_rng(net.base_seed, rep, k, r)
                ),
            )
        per_k[str(k)] = {
            **main.stats(),
            "eps_sensitivity": sensitivity.stats(),
            "types": types.stats(),
        }
    return per_k


def build_vocab_network(
    x: np.ndarray, vocab_ids: np.ndarray, settings: Settings, paths: RunPaths
) -> dict[str, Any]:
    """Candidates and neighbour sets of the vocabulary type network (rows = ``vocab_ids``)."""

    net = settings.networks
    k_values = list(net.vocab.k_values)
    stats: dict[str, Any] = {"rows": int(vocab_ids.size), "k_values": k_values}
    if not k_values:
        LOGGER.info("Rede do vocabulário desativada (vocab.k_values vazio)")
        return stats
    tick = time.perf_counter()
    candidates, _ = knn_candidates(
        x, max(k_values), net.candidates, net.eps, net.block_size, label="vocab"
    )
    stats["candidates_s"] = round(time.perf_counter() - tick, 3)
    vocab_ids = np.asarray(vocab_ids, dtype=np.int32)
    _save_npz(
        vocab_candidates_path(paths), compressed=False, vocab_ids=vocab_ids, **candidates.arrays()
    )
    stats["candidates"] = candidate_stats(candidates, {"eps": net.eps})
    stats["k"] = {}
    for k in k_values:
        selector = BoundarySelector(candidates, k, net.eps)
        _save_npz(
            vocab_neighbors_path(paths, k),
            vocab_ids=vocab_ids,
            pos=selector.pick("position"),
            rand=_draws(selector, 1, lambda r, k=k: rep_rng(net.base_seed, VOCAB, k, r)),
        )
        stats["k"][str(k)] = selector.stats()
    stats["total_s"] = round(time.perf_counter() - tick, 3)
    return stats


def _occurrence_candidates(
    rep: str,
    source: str,
    centered: bool,
    types: SampleTypes,
    weight: torch.Tensor,
    settings: Settings,
    paths: RunPaths,
) -> tuple[Candidates, Candidates]:
    net = settings.networks
    k_max = max(net.k_values)
    eps = max(net.eps, net.eps_sensitivity)
    if rep == LEX:
        unit = normalize_rows(weight[torch.from_numpy(types.ids)])
        sim_types = type_similarity(unit)
        return (
            lexical_candidates(types, sim_types, k_max, eps),
            lexical_type_candidates(types, sim_types, k_max, net.type_candidates, eps),
        )
    x = normalize_rows(load_representation(paths, settings, source), center=centered)
    if x.shape[0] != types.n_occurrences:
        raise ValueError(
            f"{source} has {x.shape[0]} rows, occurrences.csv has {types.n_occurrences}"
        )
    candidates, type_candidates = knn_candidates(
        x,
        k_max,
        net.candidates,
        eps,
        net.block_size,
        types=types,
        n_type_candidates=net.type_candidates,
    )
    assert type_candidates is not None
    return candidates, type_candidates


def check_settings(settings: Settings) -> None:
    """Fail before any computation when the candidate sizes cannot cover the largest k."""

    net = settings.networks
    if not net.k_values or min(net.k_values) < 1:
        raise ValueError("networks.k_values must list positive integers")
    if net.k_main not in net.k_values:
        raise ValueError("networks.k_main must be one of networks.k_values")
    if net.eps < 0 or net.eps_sensitivity < 0:
        raise ValueError("networks.eps and networks.eps_sensitivity must be non-negative")
    k_max = max(net.k_values)
    for label, size, needed in (
        ("candidates", net.candidates, k_max),
        ("type_candidates", net.type_candidates, k_max),
        ("candidates", net.candidates, max(net.vocab.k_values, default=0)),
    ):
        if size < needed:
            raise ValueError(f"networks.{label} = {size} is smaller than the largest k ({needed})")


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    net = settings.networks
    targets = output_paths(settings, paths)
    if not force and all(path.exists() for path in targets):
        LOGGER.info("k-NN já existe em %s; use --force para refazer", paths.knn_dir)
        return
    check_settings(settings)
    specs = representation_specs(settings)
    needed = [paths.occurrences, paths.embeddings]
    for rep, (source, _centered) in specs.items():
        if rep != LEX and not (
            paths.rep(source).exists()
            or (source in settings.model.layers and paths.all_layers.exists())
        ):
            needed.append(paths.rep(source))
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Rode as etapas sample e extract antes; faltam: " + ", ".join(missing)
        )
    started = time.time()
    ensure_dir(paths.knn_dir)
    token_ids = read_token_ids(paths.occurrences)
    types = SampleTypes.from_token_ids(token_ids)
    weight = load_embeddings(paths.embeddings)
    if token_ids.min() < 0 or token_ids.max() >= weight.shape[0]:
        raise ValueError("occurrences.csv has token ids outside the embedding rows")
    tolerances = {"eps": net.eps, "eps_sensitivity": net.eps_sensitivity}
    LOGGER.info(
        "k-NN: %d ocorrências, %d tipos, %d representações",
        types.n_occurrences,
        types.n_types,
        len(specs),
    )

    rep_stats: dict[str, Any] = {}
    for rep, (source, centered) in specs.items():
        tick = time.perf_counter()
        candidates, type_candidates = _occurrence_candidates(
            rep, source, centered, types, weight, settings, paths
        )
        candidates_s = time.perf_counter() - tick
        _save_npz(candidates_path(paths, rep), compressed=False, **candidates.arrays())
        tock = time.perf_counter()
        per_k = write_neighbor_sets(rep, candidates, type_candidates, settings, paths)
        neighbors_s = time.perf_counter() - tock
        rep_stats[rep] = {
            "source": source,
            "centered": centered,
            "exact_lexical": rep == LEX,
            "timings_s": {
                "candidates": round(candidates_s, 3),
                "neighbors": round(neighbors_s, 3),
                "total": round(time.perf_counter() - tick, 3),
            },
            "candidates": candidate_stats(candidates, tolerances),
            "type_candidates": candidate_stats(type_candidates, tolerances),
            "k": per_k,
        }
        main = per_k[str(net.k_main)]
        LOGGER.info(
            "%s: candidatos %.1f s, vizinhos %.1f s; k=%d: %d linhas com empate (maior bloco %d), "
            "%d linhas recalculadas",
            rep,
            candidates_s,
            neighbors_s,
            net.k_main,
            main["tied_rows"],
            main["max_tie_block"],
            rep_stats[rep]["candidates"]["fallback_rows"],
        )
        del candidates, type_candidates

    tick = time.perf_counter()
    special = special_mask(paths, settings, int(weight.shape[0]))
    vocab_ids = np.flatnonzero(~special).astype(np.int32)
    LOGGER.info(
        "Rede do vocabulário: %d linhas (%d tokens especiais excluídos)",
        vocab_ids.size,
        int(special.sum()),
    )
    x_vocab = vocab_unit_rows(weight, vocab_ids)
    del weight
    vocab_stats = build_vocab_network(x_vocab, vocab_ids, settings, paths)
    del x_vocab
    vocab_stats["special_excluded"] = int(special.sum())
    vocab_stats["total_with_loading_s"] = round(time.perf_counter() - tick, 3)
    LOGGER.info("Rede do vocabulário em %.1f s", vocab_stats["total_with_loading_s"])

    write_manifest(
        paths.knn_dir,
        "knn",
        settings,
        started,
        extra={
            "n_occurrences": types.n_occurrences,
            "n_types": types.n_types,
            "k_values": list(net.k_values),
            "k_main": net.k_main,
            "eps": net.eps,
            "eps_sensitivity": net.eps_sensitivity,
            "completeness_eps": max(net.eps, net.eps_sensitivity),
            "candidates": net.candidates,
            "type_candidates": net.type_candidates,
            "seeds": net.seeds,
            "representations": rep_stats,
            "vocab": vocab_stats,
        },
        root=paths.root,
    )
