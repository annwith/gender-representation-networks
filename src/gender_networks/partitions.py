"""Communities and agreement between partitions (plan decision 10).

Leiden is run with ``objective_function='modularity'`` (igraph's default is CPM, whose
resolution means something else) from several seeds; the best run by modularity is kept and
the mean pairwise NMI between runs is reported as its stability. igraph draws its random numbers
from Python's :mod:`random`, so every run reseeds it (and the caller's state is restored).

Agreement with vertex labels (token, theme, article, paragraph, ...) uses NMI and ARI from
:func:`igraph.compare_communities`, a permutation baseline for NMI (NMI grows with the number of
classes even for unrelated labellings) and purity computed in numpy.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np


@dataclass
class Partition:
    """A vertex partition with canonical labels (0 = largest community)."""

    membership: np.ndarray
    modularity: float  # standard modularity (resolution 1)
    quality: float  # modularity at ``resolution``, the objective that selected this run
    resolution: float
    method: str
    seed: int
    stability: float = math.nan
    memberships: np.ndarray | None = None  # every run, [runs, N]
    qualities: list[float] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)  # per run (Leiden with an iteration cap)
    max_iterations: int = -1

    @property
    def sizes(self) -> list[int]:
        return np.bincount(self.membership).tolist() if self.membership.size else []

    @property
    def n_communities(self) -> int:
        return len(self.sizes)


def canonical_labels(membership: Sequence[int] | np.ndarray) -> np.ndarray:
    """Relabel communities by decreasing size, ties by smallest vertex id (int32).

    Solvers return arbitrary label numbers; canonical labels make saved memberships comparable
    and diff-friendly across runs.
    """

    labels = np.asarray(membership)
    if labels.size == 0:
        return np.zeros(0, dtype=np.int32)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    first = np.full(counts.size, labels.size, dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(labels.size))
    order = np.lexsort((first, -counts))
    rank = np.empty(counts.size, dtype=np.int32)
    rank[order] = np.arange(counts.size, dtype=np.int32)
    return rank[inverse]


def _codes(labels: Sequence[Any] | np.ndarray) -> list[int]:
    """Labels remapped to ``0..m-1`` (igraph wants small non-negative integers)."""

    _, inverse = np.unique(np.asarray(labels), return_inverse=True)
    return inverse.reshape(-1).tolist()


def nmi(a: Sequence[Any] | np.ndarray, b: Sequence[Any] | np.ndarray) -> float:
    """Normalized mutual information between two labellings of the same vertices."""

    if len(a) != len(b):
        raise ValueError("labellings must have the same length")
    return float(ig.compare_communities(_codes(a), _codes(b), method="nmi"))


def ari(a: Sequence[Any] | np.ndarray, b: Sequence[Any] | np.ndarray) -> float:
    """Adjusted Rand index (NaN when both labellings are a single class, as in igraph)."""

    if len(a) != len(b):
        raise ValueError("labellings must have the same length")
    return float(ig.compare_communities(_codes(a), _codes(b), method="adjusted_rand"))


def purity(pred: Sequence[Any] | np.ndarray, true: Sequence[Any] | np.ndarray) -> float:
    """Fraction of vertices whose community's majority class equals their own class.

    Computed from the sparse contingency table, so thousands of classes (token ids) are fine.
    """

    pred_codes = np.asarray(_codes(pred), dtype=np.int64)
    true_codes = np.asarray(_codes(true), dtype=np.int64)
    if pred_codes.size != true_codes.size:
        raise ValueError("labellings must have the same length")
    if pred_codes.size == 0:
        return math.nan
    n_true = int(true_codes.max()) + 1
    cells, counts = np.unique(pred_codes * n_true + true_codes, return_counts=True)
    best = np.zeros(int(pred_codes.max()) + 1, dtype=np.int64)
    np.maximum.at(best, cells // n_true, counts)
    return float(best.sum() / pred_codes.size)


def inverse_purity(pred: Sequence[Any] | np.ndarray, true: Sequence[Any] | np.ndarray) -> float:
    """Purity of the classes with respect to the communities (penalizes over-splitting)."""

    return purity(true, pred)


def nmi_baseline(
    pred: Sequence[Any] | np.ndarray,
    true: Sequence[Any] | np.ndarray,
    permutations: int = 20,
    seed: int = 0,
) -> float:
    """Mean NMI after randomly permuting ``true``: the NMI expected from class sizes alone."""

    if permutations < 1:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(seed)
    true_codes = np.asarray(_codes(true))
    pred_codes = _codes(pred)
    values = [
        ig.compare_communities(pred_codes, rng.permutation(true_codes).tolist(), method="nmi")
        for _ in range(permutations)
    ]
    return float(np.mean(values))


def valid_labels(labels: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Mask of usable labels: NaN, None and empty strings mark vertices without a label."""

    array = np.asarray(labels)
    if array.dtype.kind == "f":
        return ~np.isnan(array)
    if array.dtype.kind in "US":
        return array != array.dtype.type()
    if array.dtype.kind == "O":
        return np.array(
            [
                not (value is None or value == "" or (isinstance(value, float) and value != value))
                for value in array.tolist()
            ],
            dtype=bool,
        )
    return np.ones(array.shape, dtype=bool)


def agreement(
    pred: Sequence[int] | np.ndarray,
    labels_by_name: Mapping[str, Sequence[Any] | np.ndarray],
    permutations: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Agreement of a partition with every labelling in ``labels_by_name``.

    Vertices without a label (see :func:`valid_labels`) are left out for that labelling only;
    ``n`` records how many were used.
    """

    pred = np.asarray(pred)
    rows = []
    for name, labels in labels_by_name.items():
        labels = np.asarray(labels)
        if labels.shape[0] != pred.shape[0]:
            raise ValueError(f"labels '{name}' have {labels.shape[0]} entries, not {pred.size}")
        mask = valid_labels(labels)
        p, t = pred[mask], labels[mask]
        if p.size == 0:
            value = baseline = adjusted = pur = inv = math.nan
        else:
            value = nmi(p, t)
            baseline = nmi_baseline(p, t, permutations, seed)
            adjusted = ari(p, t)
            pur = purity(p, t)
            inv = inverse_purity(p, t)
        rows.append(
            {
                "label": name,
                "n": int(p.size),
                "n_classes": int(np.unique(t).size),
                "n_communities": int(np.unique(p).size),
                "nmi": value,
                "nmi_baseline": baseline,
                "nmi_minus_baseline": value - baseline,
                "ari": adjusted,
                "purity": pur,
                "inverse_purity": inv,
            }
        )
    return rows


def mean_pairwise_nmi(memberships: Sequence[Sequence[int]] | np.ndarray) -> float:
    """Mean NMI over all pairs of runs; NaN with fewer than two runs."""

    runs = [np.asarray(m).tolist() for m in memberships]
    if len(runs) < 2:
        return math.nan
    values = [
        ig.compare_communities(a, b, method="nmi") for a, b in itertools.combinations(runs, 2)
    ]
    return float(np.mean(values))


def leiden(
    g: ig.Graph,
    runs: int = 10,
    resolution: float = 1.0,
    seed: int = 0,
    max_iterations: int = -1,
) -> Partition:
    """Best of ``runs`` Leiden runs (modularity objective), seeds ``seed .. seed + runs - 1``.

    Runs are compared by modularity at ``resolution`` (the optimized objective); the first
    best run wins ties. With ``max_iterations > 0`` each run stops after that many iterations
    and one more iteration from its result tells whether it had converged (unchanged
    membership); ``-1`` iterates until no further improvement.
    """

    if runs < 1:
        raise ValueError("runs must be positive")
    if g.is_directed():
        raise ValueError("communities are detected on undirected graphs")
    state = random.getstate()
    memberships, qualities, converged = [], [], []
    try:
        for r in range(runs):
            random.seed(seed + r)
            clustering = g.community_leiden(
                objective_function="modularity",
                resolution=resolution,
                n_iterations=max_iterations,
            )
            membership = canonical_labels(clustering.membership)
            if max_iterations > 0:
                again = g.community_leiden(
                    objective_function="modularity",
                    resolution=resolution,
                    n_iterations=1,
                    initial_membership=membership.tolist(),
                )
                same = np.array_equal(canonical_labels(again.membership), membership)
                converged.append(bool(same))
            else:
                converged.append(True)
            memberships.append(membership)
            qualities.append(float(g.modularity(membership.tolist(), resolution=resolution)))
    finally:
        random.setstate(state)
    best = int(np.argmax(qualities))
    stacked = np.vstack(memberships).astype(np.int32)
    return Partition(
        membership=memberships[best],
        modularity=float(g.modularity(memberships[best].tolist())),
        quality=qualities[best],
        resolution=float(resolution),
        method="leiden",
        seed=seed,
        stability=mean_pairwise_nmi(stacked),
        memberships=stacked,
        qualities=qualities,
        converged=converged,
        max_iterations=max_iterations,
    )


def louvain(g: ig.Graph, seed: int = 0, resolution: float = 1.0) -> Partition:
    """Louvain (multilevel) partition, a cross-check of the Leiden result."""

    if g.is_directed():
        raise ValueError("communities are detected on undirected graphs")
    state = random.getstate()
    try:
        random.seed(seed)
        clustering = g.community_multilevel(resolution=resolution)
    finally:
        random.setstate(state)
    membership = canonical_labels(clustering.membership)
    quality = float(g.modularity(membership.tolist(), resolution=resolution))
    return Partition(
        membership=membership,
        modularity=float(g.modularity(membership.tolist())),
        quality=quality,
        resolution=float(resolution),
        method="louvain",
        seed=seed,
        qualities=[quality],
    )
