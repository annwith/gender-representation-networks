"""Stage ``metrics``: structural metrics and communities of every k-NN graph (plan stage 5).

The stage reads only the neighbour sets written by the ``knn`` stage and measures one graph per
*graph specification* ``rep|k{k}|{sym}|{tie}|s{seed}``:

- main configuration: ``k = k_main``, union and directed graphs, random ties (every seed for
  ``lex``, whose ties are massive, seed 0 elsewhere) and the position rule as reference;
- robustness: the other k values, the mutual graph, variant (b) (``all``, every tied candidate),
  the robust (``L36n``) and centered representations, and the ``eps = 1e-4`` files
  (``rand_eps``);
- the lexical type network over the vocabulary (``vocab``) for each of its k values.

Many specifications give the same edge set (outside ``lex`` random ties are rare, so ``rand``
and ``pos`` usually coincide); graphs are fingerprinted first and each distinct edge set is
measured once, the others pointing to it through ``duplicate_of``. Distinct graphs run in
parallel processes that load the ``.npz`` files themselves.

Distances are exact (one all-pairs BFS in C) for every occurrence graph; on the vocabulary
network they are exact only for ``k = k_main`` (when ``analysis.exact_distances_vocab``) and
estimated from sampled BFS sources otherwise. Leiden communities (modularity objective) are
computed on the union graphs that the analyses of P3/P4 read, plus a resolution sweep and a
Louvain cross-check on the main configuration.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gender_networks.artifacts import (
    RunPaths,
    ensure_dir,
    graph_id,
    neighbors_name,
    read_token_ids,
    require_inputs,
    up_to_date,
    vocab_neighbors_name,
    write_csv,
    write_manifest,
)
from gender_networks.graphs import (
    directed_edges,
    directed_metrics_from_edges,
    edge_hash,
    graph_metrics,
    symmetrize,
    to_igraph,
)
from gender_networks.neighborhood import Neighbors, neighbor_rows
from gender_networks.partitions import Partition, leiden, louvain, nmi
from gender_networks.representations import is_lexical, occurrence_names
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)

VOCAB = "vocab"
SYMS = ("union", "directed", "mutual")
TIES = ("rand", "pos", "all", "rand_eps", "pos_eps")
MAIN_SYMS = ("union", "directed")
TOP_HUBS = 20
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

METRIC_COLUMNS = [
    "graph_id",
    "rep",
    "k",
    "sym",
    "tie",
    "seed",
    "group",  # main | robust | vocab
    "duplicate_of",  # graph_id whose identical edge set was measured (empty when computed)
    "edge_hash",
    "n_vertices",
    "n_edges",
    "mean_degree",
    "density",
    "transitivity",
    "avg_local_clustering",
    "n_components",
    "largest_component",
    "largest_fraction",
    "mean_distance",
    "diameter",  # lower bound when distance_method is 'sampled'
    "distance_method",  # exact | sampled | none (directed rows)
    "n_distance_sources",
    "reciprocity",
    "in_degree_mean",
    "in_degree_std",
    "in_degree_max",
    "in_degree_skewness",
    "in_degree_zero_fraction",
    "out_degree_min",
    "out_degree_max",
    "modularity",  # Leiden at analysis.resolution, when communities were computed
    "n_communities",
    "community_stability",
    "elapsed_s",
]
DEGREE_COLUMNS = ["graph_id", "kind", "degree", "count"]
COMPONENT_COLUMNS = ["graph_id", "size", "count"]
DISTANCE_COLUMNS = ["graph_id", "distance", "count"]
HUB_COLUMNS = ["graph_id", "rank", "vertex", "in_degree", "token_id"]
COMMUNITY_COLUMNS = [
    "graph_id",
    "method",  # leiden | louvain
    "resolution",
    "file",
    "duplicate_of",
    "n_vertices",
    "modularity",  # standard modularity (resolution 1)
    "quality",  # modularity at the run's resolution (the optimized objective)
    "n_communities",
    "largest_community",
    "stability",  # mean pairwise NMI between the Leiden runs
    "runs",
    "seed",
    "max_iterations",  # Leiden iteration cap per run (-1 = until no improvement)
    "converged_fraction",  # Leiden runs whose membership no longer changed at the cap
    "nmi_vs_leiden",  # Louvain rows: NMI with the Leiden partition of the same graph
]


# --------------------------------------------------------------------------------------------
# graph specifications


@dataclass(frozen=True)
class GraphSpec:
    """One graph to measure: which neighbour set, how it is symmetrized, where it lives.

    Attributes:
        rep: Representation name, or ``vocab``.
        k: Number of neighbours.
        sym: ``union``, ``directed`` or ``mutual``.
        tie: ``rand``, ``pos``, ``all``, ``rand_eps`` or ``pos_eps``.
        seed: Tie-break seed index (0 for the deterministic rules).
        source: Name of the ``.npz`` file inside the knn directory.
        group: ``main``, ``robust`` or ``vocab``.
    """

    rep: str
    k: int
    sym: str  # union | directed | mutual
    tie: str  # rand | pos | all | rand_eps | pos_eps
    seed: int
    source: str  # .npz file name inside the knn directory
    group: str  # main | robust | vocab

    def __post_init__(self) -> None:
        """Reject unknown symmetrizations and tie treatments.

        Raises:
            ValueError: If ``sym`` is not in :data:`SYMS` or ``tie`` is not in :data:`TIES`.
        """

        if self.sym not in SYMS:
            raise ValueError(f"Unknown symmetrization '{self.sym}'")
        if self.tie not in TIES:
            raise ValueError(f"Unknown tie treatment '{self.tie}'")

    @property
    def graph_id(self) -> str:
        """Identifier ``rep|k{k}|{sym}|{tie}|s{seed}`` of the specification."""

        return graph_id(self.rep, self.k, self.sym, self.tie, self.seed)

    @property
    def directed(self) -> bool:
        """True for the directed graph."""

        return self.sym == "directed"


@dataclass(frozen=True)
class CommunityRequest:
    """A partition to compute on one graph specification.

    Attributes:
        graph_id: Graph the partition is computed on.
        resolution: Modularity resolution.
        method: ``leiden`` or ``louvain``.
    """

    graph_id: str
    resolution: float
    method: str = "leiden"  # leiden | louvain


def vocab_community_k(settings: Settings) -> int | None:
    """k of the vocabulary graph that gets communities: ``k_main`` or the closest vocab k.

    Args:
        settings: Run configuration.

    Returns:
        The value of ``networks.vocab.k_values`` nearest to ``k_main`` (the smaller on a
        tie), or ``None`` when the vocabulary network is disabled.
    """

    values = settings.networks.vocab.k_values
    if not values:
        return None
    k_main = settings.networks.k_main
    return min(values, key=lambda k: (abs(k - k_main), k))


def plan_graphs(settings: Settings) -> list[GraphSpec]:
    """Every graph specification of a run, main configuration first (pure, no file access).

    Main group: for each main representation at ``k_main``, the union and directed graphs
    with random ties (every seed for ``lex``, seed 0 elsewhere) and with the position rule.
    Robust group: the other ``k`` values, the mutual graphs and variant (b) for every ``k``,
    the robust and centered representations at ``k_main``, and the eps-sensitivity graphs.
    Vocab group: union and directed graphs of every vocabulary ``k``, random and position.

    Args:
        settings: Run configuration.

    Returns:
        The specifications, without repeats, in planning order.
    """

    net = settings.networks
    k_main = net.k_main
    specs: list[GraphSpec] = []
    seen: set[str] = set()

    def add(rep: str, k: int, sym: str, tie: str, seed: int, source: str, group: str) -> None:
        """Append a specification unless one with the same graph id was already planned.

        Args:
            rep: Representation name, or ``vocab``.
            k: Number of neighbours.
            sym: Symmetrization.
            tie: Tie treatment.
            seed: Tie-break seed index.
            source: Neighbour file name.
            group: ``main``, ``robust`` or ``vocab``.
        """

        spec = GraphSpec(rep, k, sym, tie, seed, source, group)
        if spec.graph_id not in seen:
            seen.add(spec.graph_id)
            specs.append(spec)

    def block(rep: str, k: int, seeds: Sequence[int], group: str, with_main: bool) -> None:
        """Union/directed graphs (``with_main``) and, for robustness, mutual and variant (b).

        Args:
            rep: Representation name.
            k: Number of neighbours.
            seeds: Random tie-break seeds to plan.
            group: ``main`` or ``robust``.
            with_main: Plan the union and directed graphs (random seeds and position rule).
        """

        source = neighbors_name(rep, k)
        if with_main:
            for sym in MAIN_SYMS:
                for seed in seeds:
                    add(rep, k, sym, "rand", seed, source, group)
                add(rep, k, sym, "pos", 0, source, group)
        if group == "robust":
            add(rep, k, "mutual", "rand", 0, source, "robust")
            add(rep, k, "mutual", "pos", 0, source, "robust")
            add(rep, k, "union", "all", 0, source, "robust")

    for rep in net.representations:
        seeds = range(net.seeds) if is_lexical(rep) else [0]
        block(rep, k_main, list(seeds), "main", with_main=True)
    for rep in net.representations:
        for k in net.k_values:
            block(rep, k, [0], "robust", with_main=k != k_main)
    extra = [rep for rep in occurrence_names(settings) if rep not in net.representations]
    for rep in extra:
        block(rep, k_main, [0], "robust", with_main=True)
    for rep in occurrence_names(settings):
        eps_source = neighbors_name(rep, k_main, sensitivity=True)
        for sym in MAIN_SYMS:
            add(rep, k_main, sym, "rand_eps", 0, eps_source, "robust")
    for k in net.vocab.k_values:
        for sym in MAIN_SYMS:
            for tie in ("rand", "pos"):
                add(VOCAB, k, sym, tie, 0, vocab_neighbors_name(k), "vocab")
    return specs


def plan_communities(settings: Settings, specs: Sequence[GraphSpec]) -> list[CommunityRequest]:
    """Partitions to compute, restricted to specifications present in ``specs``.

    Leiden at the main resolution on the union graph at ``k_main`` of every occurrence
    representation (random seed 0 and the position rule; for ``lex`` also seed 1); the
    resolution sweep and a Louvain run on the main representations; and Leiden on one
    vocabulary graph.

    Args:
        settings: Run configuration.
        specs: Planned specifications whose files exist.

    Returns:
        The requests, without repeats.
    """

    net, analysis = settings.networks, settings.analysis
    k_main, res = net.k_main, float(analysis.resolution)
    available = {spec.graph_id for spec in specs}
    requests: list[CommunityRequest] = []

    def add(gid: str, resolution: float, method: str = "leiden") -> None:
        """Append a request when its graph is available and it is not already listed.

        Args:
            gid: Graph id.
            resolution: Modularity resolution.
            method: ``leiden`` or ``louvain``.
        """

        request = CommunityRequest(gid, float(resolution), method)
        if gid in available and request not in requests:
            requests.append(request)

    for rep in occurrence_names(settings):
        add(graph_id(rep, k_main, "union", "rand", 0), res)
        if is_lexical(rep) and net.seeds > 1:
            add(graph_id(rep, k_main, "union", "rand", 1), res)
        add(graph_id(rep, k_main, "union", "pos", 0), res)
    for rep in net.representations:
        main_id = graph_id(rep, k_main, "union", "rand", 0)
        for resolution in analysis.resolution_sweep:
            add(main_id, resolution)
        add(main_id, res, "louvain")
    vocab_k = vocab_community_k(settings)
    if vocab_k is not None:
        add(graph_id(VOCAB, vocab_k, "union", "rand", 0), res)
    return requests


def distance_mode(spec: GraphSpec, settings: Settings) -> str:
    """``none`` for digraphs, ``exact`` for occurrence graphs, exact or sampled for ``vocab``.

    Args:
        spec: Graph specification.
        settings: Run configuration.

    Returns:
        The distance mode. A vocabulary graph is exact only at ``k_main`` with
        ``analysis.exact_distances_vocab``.
    """

    if spec.directed:
        return "none"
    if spec.rep != VOCAB:
        return "exact"
    exact = settings.analysis.exact_distances_vocab and spec.k == settings.networks.k_main
    return "exact" if exact else "sampled"


# --------------------------------------------------------------------------------------------
# loading neighbour sets


def select_neighbors(data: Mapping[str, np.ndarray], tie: str, seed: int) -> Neighbors:
    """Neighbour structure of one tie treatment from a loaded ``nbr_*.npz`` / ``vocab_k*.npz``.

    Args:
        data: Arrays of the file, by name.
        tie: ``pos``/``pos_eps`` read ``pos``; ``rand``/``rand_eps`` read one seed of
            ``rand``; ``all`` reads the CSR pair of variant (b). The ``_eps`` treatments
            differ only in the file they are read from.
        seed: Seed index into ``rand``.

    Returns:
        Dense ``[N, k]`` array, or the CSR pair for ``all``.

    Raises:
        ValueError: If the treatment is unknown or the seed is not in the file.
    """

    if tie in ("pos", "pos_eps"):
        return np.asarray(data["pos"])
    if tie in ("rand", "rand_eps"):
        rand = np.asarray(data["rand"])
        if rand.ndim != 3 or not 0 <= seed < rand.shape[0]:
            raise ValueError(f"seed {seed} is not available in 'rand' with shape {rand.shape}")
        return rand[seed]
    if tie == "all":
        return np.asarray(data["all_indptr"]), np.asarray(data["all_idx"])
    raise ValueError(f"Unknown tie treatment '{tie}'")


class _NpzCache(Mapping[str, np.ndarray]):
    """Reads each array of an open ``.npz`` once (``NpzFile`` decompresses on every access)."""

    def __init__(self, npz: Mapping[str, np.ndarray]) -> None:
        """Wrap an open ``.npz`` file.

        Args:
            npz: The open file; it must stay open while the cache is used.
        """

        self._npz = npz
        self._arrays: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        """The array stored under ``key``, read on first use and kept in memory.

        Args:
            key: Array name.

        Returns:
            The array.
        """

        if key not in self._arrays:
            self._arrays[key] = np.asarray(self._npz[key])
        return self._arrays[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the array names of the file."""

        return iter(self._npz)

    def __len__(self) -> int:
        """Number of arrays in the file."""

        return len(self._npz)


def load_neighbors(knn_dir: Path, spec: GraphSpec) -> Neighbors:
    """Neighbour structure of a specification, read from its file.

    Args:
        knn_dir: Directory of the ``knn`` stage.
        spec: Graph specification.

    Returns:
        Dense ``[N, k]`` array, or the CSR pair for variant (b).
    """

    with np.load(knn_dir / spec.source) as data:
        return select_neighbors(data, spec.tie, spec.seed)


@dataclass(frozen=True)
class Fingerprint:
    """What the planner needs to know about a graph before measuring it.

    Attributes:
        edge_hash: Hash of the symmetrized edge set (see ``graphs.edge_hash``).
        n_vertices: Number of rows of the neighbour structure.
        n_edges: Edges of the symmetrized graph.
        n_self_loops: Neighbour entries equal to their own row.
        n_repeated: Neighbour entries that repeat another entry of the same row.
        out_degree_min: Smallest number of distinct neighbours of a row.
        out_degree_max: Largest number of distinct neighbours of a row.
    """

    edge_hash: str
    n_vertices: int
    n_edges: int
    n_self_loops: int
    n_repeated: int
    out_degree_min: int
    out_degree_max: int


def fingerprint(neighbors: Neighbors, sym: str) -> Fingerprint:
    """Edge-set hash and sanity counts (self loops, repeated ids, out-degree range).

    Args:
        neighbors: Neighbour sets, dense or CSR.
        sym: ``union``, ``directed`` or ``mutual``.

    Returns:
        The :class:`Fingerprint` of the graph.
    """

    rows, ids, n = neighbor_rows(neighbors)
    edges = directed_edges(neighbors, n)
    return _fingerprint_from_edges(edges, n, sym, rows, ids)


def _fingerprint_from_edges(
    edges: np.ndarray, n: int, sym: str, rows: np.ndarray, ids: np.ndarray
) -> Fingerprint:
    """:func:`fingerprint` from directed edges that were already computed.

    Args:
        edges: Directed edges from ``graphs.directed_edges``.
        n: Number of vertices.
        sym: Symmetrization of the graph to fingerprint.
        rows: Row of every neighbour entry (before removing loops and repeats).
        ids: Neighbour id of every entry.

    Returns:
        The :class:`Fingerprint`.
    """

    loops = int(np.count_nonzero(rows == ids))
    outdeg = np.bincount(edges[:, 0], minlength=n) if n else np.zeros(0, dtype=np.int64)
    graph_edges = symmetrize(edges, n, sym)
    return Fingerprint(
        edge_hash=edge_hash(graph_edges, n, directed=sym == "directed"),
        n_vertices=n,
        n_edges=int(graph_edges.shape[0]),
        n_self_loops=loops,
        n_repeated=int(ids.size - loops - edges.shape[0]),
        out_degree_min=int(outdeg.min()) if n else 0,
        out_degree_max=int(outdeg.max()) if n else 0,
    )


def fingerprint_specs(knn_dir: Path, specs: Sequence[GraphSpec]) -> dict[str, Fingerprint]:
    """Fingerprints of every specification, opening each ``.npz`` file once.

    Args:
        knn_dir: Directory of the ``knn`` stage.
        specs: Specifications whose files exist.

    Returns:
        Graph id to its :class:`Fingerprint`.
    """

    by_source: dict[str, list[GraphSpec]] = defaultdict(list)
    for spec in specs:
        by_source[spec.source].append(spec)
    out: dict[str, Fingerprint] = {}
    for source, group in by_source.items():
        with np.load(knn_dir / source) as npz:
            data = _NpzCache(npz)
            directed: dict[tuple[str, int], tuple[np.ndarray, int, np.ndarray, np.ndarray]] = {}
            for spec in group:
                key = (spec.tie, spec.seed)
                if key not in directed:
                    neighbors = select_neighbors(data, spec.tie, spec.seed)
                    rows, ids, n = neighbor_rows(neighbors)
                    directed[key] = (directed_edges(neighbors, n), n, rows, ids)
                edges, n, rows, ids = directed[key]
                out[spec.graph_id] = _fingerprint_from_edges(edges, n, spec.sym, rows, ids)
    return out


# --------------------------------------------------------------------------------------------
# measuring one graph (runs inside worker processes)


@dataclass(frozen=True)
class GraphJob:
    """Everything a worker needs to measure one distinct graph (arrays are loaded there).

    Attributes:
        spec: Graph specification.
        knn_dir: Directory of the ``knn`` stage, as a string (the job is pickled).
        distances: Distance mode: ``exact``, ``sampled`` or ``none``.
        sample_sources: BFS sources of the sampled mode.
        distance_seed: Seed of the source draw.
        leiden_resolutions: Resolutions to run Leiden at.
        louvain_resolutions: Resolutions to run Louvain at.
        leiden_runs: Leiden runs per resolution.
        leiden_iterations: Iteration cap per Leiden run (-1 for none).
        community_seed: First seed of the community runs.
        top_hubs: Hubs kept for directed graphs.
        cost: Rough cost estimate, used only to order the jobs.
    """

    spec: GraphSpec
    knn_dir: str
    distances: str
    sample_sources: int
    distance_seed: int
    leiden_resolutions: tuple[float, ...] = ()
    louvain_resolutions: tuple[float, ...] = ()
    leiden_runs: int = 10
    leiden_iterations: int = -1
    community_seed: int = 0
    top_hubs: int = TOP_HUBS
    cost: float = 0.0


@dataclass
class GraphResult:
    """Metrics, histograms and partitions of one measured graph.

    Attributes:
        graph_id: Graph that was measured.
        metrics: Scalar metrics (columns of ``graph_metrics.csv``).
        degree_hist: Kind (``undirected``, ``in``, ``out``) to its degree histogram.
        component_sizes: Component size to the number of components.
        distance_hist: Distance to the number of pairs.
        hubs: ``(vertex, in_degree)`` of the top hubs (directed graphs only).
        partitions: ``(method, resolution)`` to its :class:`Partition`.
        elapsed_s: Time spent measuring.
    """

    graph_id: str
    metrics: dict[str, Any]
    degree_hist: dict[str, dict[int, int]]
    component_sizes: dict[int, int]
    distance_hist: dict[int, int]
    hubs: list[tuple[int, int]] = field(default_factory=list)
    partitions: dict[tuple[str, float], Partition] = field(default_factory=dict)
    elapsed_s: float = 0.0


def measure_graph(neighbors: Neighbors, n: int, job: GraphJob) -> GraphResult:
    """Metrics (and partitions, for undirected graphs) of one neighbour structure.

    A directed specification gets the size metrics, the weak components and the in-degree
    and reciprocity statistics; an undirected one gets clustering, distances, components
    and the requested partitions.

    Args:
        neighbors: Neighbour sets, dense or CSR.
        n: Number of vertices.
        job: The job (specification, distance mode, community requests, seeds).

    Returns:
        The :class:`GraphResult`.
    """

    started = time.perf_counter()
    spec = job.spec
    edges = directed_edges(neighbors, n)
    hubs: list[tuple[int, int]] = []
    partitions: dict[tuple[str, float], Partition] = {}
    if spec.directed:
        graph = to_igraph(edges, n, directed=True)
        metrics = graph_metrics(graph, distances="none")
        directed = directed_metrics_from_edges(edges, n, job.top_hubs)
        hubs = directed.pop("hubs")
        directed.pop("in_degree_hist")
        directed.pop("out_degree_hist")
        for key in ("n_vertices", "n_edges", "out_degree_mean"):
            directed.pop(key)
        metrics.update(directed)
    else:
        graph = to_igraph(symmetrize(edges, n, spec.sym), n, directed=False)
        metrics = graph_metrics(graph, job.distances, job.sample_sources, job.distance_seed)
        for resolution in job.leiden_resolutions:
            partitions[("leiden", resolution)] = leiden(
                graph,
                runs=job.leiden_runs,
                resolution=resolution,
                seed=job.community_seed,
                max_iterations=job.leiden_iterations,
            )
        for resolution in job.louvain_resolutions:
            partitions[("louvain", resolution)] = louvain(
                graph, seed=job.community_seed, resolution=resolution
            )
    degree_hist = metrics.pop("degree_hist")
    component_sizes = metrics.pop("component_sizes")
    distance_hist = metrics.pop("distance_hist")
    return GraphResult(
        graph_id=spec.graph_id,
        metrics=metrics,
        degree_hist=degree_hist,
        component_sizes=component_sizes,
        distance_hist=distance_hist,
        hubs=hubs,
        partitions=partitions,
        elapsed_s=time.perf_counter() - started,
    )


def compute_graph(job: GraphJob) -> GraphResult:
    """Worker entry point: load the neighbour set from disk and measure it.

    Args:
        job: The job.

    Returns:
        The :class:`GraphResult`.
    """

    neighbors = load_neighbors(Path(job.knn_dir), job.spec)
    _, _, n = neighbor_rows(neighbors)
    return measure_graph(neighbors, n, job)


@contextmanager
def single_thread_blas() -> Iterator[None]:
    """Environment for worker processes: one BLAS/OpenMP thread each.

    Spawned children read these variables when numpy loads, so they must be set before the
    pool starts; the parent's values are restored afterwards.

    Yields:
        Nothing; the variables of :data:`BLAS_THREAD_VARS` are ``1`` inside the block.
    """

    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARS}
    os.environ.update({name: "1" for name in BLAS_THREAD_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute_jobs(jobs: Sequence[GraphJob], workers: int) -> Iterator[GraphResult]:
    """Run jobs inline (``workers <= 1``) or in a pool of spawned processes, as they finish.

    Args:
        jobs: Jobs to run.
        workers: Number of worker processes.

    Yields:
        One :class:`GraphResult` per job, in completion order (not in job order) when a
        pool is used.
    """

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield compute_graph(job)
        return
    context = mp.get_context("spawn")
    with single_thread_blas(), context.Pool(processes=workers) as pool:
        yield from pool.imap_unordered(compute_graph, jobs, chunksize=1)


# --------------------------------------------------------------------------------------------
# stage


@dataclass
class MetricsPlan:
    """Specifications grouped into distinct jobs, with duplicates and community requests.

    Attributes:
        specs: Every specification, in planning order.
        fingerprints: Graph id to its :class:`Fingerprint`.
        duplicate_of: Graph id to the graph id whose identical edge set is measured.
        requests: Partitions to compute.
        jobs: One job per distinct graph, slowest first.
    """

    specs: list[GraphSpec]
    fingerprints: dict[str, Fingerprint]
    duplicate_of: dict[str, str]  # graph_id -> canonical graph_id
    requests: list[CommunityRequest]
    jobs: list[GraphJob]

    def canonical(self, gid: str) -> str:
        """Graph id that was actually measured for ``gid`` (itself unless it is a duplicate).

        Args:
            gid: Graph id.

        Returns:
            The canonical graph id.
        """

        return self.duplicate_of.get(gid, gid)


def build_plan(
    settings: Settings,
    specs: Sequence[GraphSpec],
    fingerprints: Mapping[str, Fingerprint],
    knn_dir: Path,
) -> MetricsPlan:
    """Deduplicate specifications by edge set and attach communities to the distinct graphs.

    Two specifications are duplicates when they have the same edge hash and the same
    distance mode; the first in planning order is measured. Community requests of a
    duplicate are run on the graph it points to.

    Args:
        settings: Run configuration.
        specs: Specifications whose files exist.
        fingerprints: Graph id to its fingerprint.
        knn_dir: Directory of the ``knn`` stage.

    Returns:
        The :class:`MetricsPlan`. The distance seed of a job is
        ``crc32("{base_seed}|{graph_id}")`` and the community seed is ``base_seed``.
    """

    canonical_by_key: dict[tuple[str, str], str] = {}
    duplicate_of: dict[str, str] = {}
    for spec in specs:
        key = (fingerprints[spec.graph_id].edge_hash, distance_mode(spec, settings))
        if key in canonical_by_key:
            duplicate_of[spec.graph_id] = canonical_by_key[key]
        else:
            canonical_by_key[key] = spec.graph_id
    requests = plan_communities(settings, specs)
    leiden_res: dict[str, set[float]] = defaultdict(set)
    louvain_res: dict[str, set[float]] = defaultdict(set)
    for request in requests:
        target = duplicate_of.get(request.graph_id, request.graph_id)
        (leiden_res if request.method == "leiden" else louvain_res)[target].add(request.resolution)
    analysis, base_seed = settings.analysis, settings.networks.base_seed
    jobs = []
    for spec in specs:
        gid = spec.graph_id
        if gid in duplicate_of:
            continue
        fp = fingerprints[gid]
        mode = distance_mode(spec, settings)
        leiden_list = tuple(sorted(leiden_res.get(gid, ())))
        louvain_list = tuple(sorted(louvain_res.get(gid, ())))
        # Rough cost used only to start the slowest graphs first.
        bfs = {"exact": fp.n_vertices, "sampled": analysis.distance_sample_sources, "none": 0}
        cost = (fp.n_vertices + fp.n_edges) * (
            1 + bfs[mode] + 20 * analysis.leiden_runs * len(leiden_list)
        )
        jobs.append(
            GraphJob(
                spec=spec,
                knn_dir=str(knn_dir),
                distances=mode,
                sample_sources=analysis.distance_sample_sources,
                distance_seed=zlib.crc32(f"{base_seed}|{gid}".encode()),
                leiden_resolutions=leiden_list,
                louvain_resolutions=louvain_list,
                leiden_runs=analysis.leiden_runs,
                leiden_iterations=analysis.leiden_iterations,
                community_seed=base_seed,
                cost=float(cost),
            )
        )
    jobs.sort(key=lambda job: -job.cost)
    return MetricsPlan(list(specs), dict(fingerprints), duplicate_of, requests, jobs)


def _blank(value: Any) -> Any:
    """CSV cell: None and NaN become empty; numpy scalars become Python numbers.

    Args:
        value: Cell value.

    Returns:
        The value to write.
    """

    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def metric_row(
    spec: GraphSpec,
    fp: Fingerprint,
    result: GraphResult,
    duplicate_of: str | None,
    main_partition: Partition | None,
) -> dict[str, Any]:
    """One ``graph_metrics.csv`` row (not-applicable cells are empty).

    Args:
        spec: Specification of the row.
        fp: Its fingerprint.
        result: Result of the graph that was measured (its own, or the one it duplicates).
        duplicate_of: Graph id it duplicates, or ``None``.
        main_partition: Leiden partition at the main resolution, when there is one.

    Returns:
        Dictionary with the keys of :data:`METRIC_COLUMNS`.
    """

    row: dict[str, Any] = dict.fromkeys(METRIC_COLUMNS)
    row.update(
        graph_id=spec.graph_id,
        rep=spec.rep,
        k=spec.k,
        sym=spec.sym,
        tie=spec.tie,
        seed=spec.seed,
        group=spec.group,
        duplicate_of=duplicate_of,
        edge_hash=fp.edge_hash,
        elapsed_s=None if duplicate_of else round(result.elapsed_s, 3),
    )
    for key, value in result.metrics.items():
        if key in row and key not in {"graph_id", "elapsed_s"}:
            row[key] = value
    if main_partition is not None:
        row["modularity"] = main_partition.modularity
        row["n_communities"] = main_partition.n_communities
        row["community_stability"] = main_partition.stability
    return {key: _blank(value) for key, value in row.items()}


def _save_partition(path: Path, partition: Partition, vocab_ids: np.ndarray | None) -> None:
    """Write a partition to ``communities/*.npz``.

    Args:
        path: Output file.
        partition: The partition; ``memberships`` falls back to the single membership.
        vocab_ids: Token id of every vertex of a vocabulary graph, stored when given.

    Side effects:
        Writes the file with ``membership``, ``modularity``, ``quality``, ``stability``,
        ``resolution``, ``seed``, ``sizes``, ``qualities`` and ``memberships``.
    """

    arrays: dict[str, Any] = {
        "membership": partition.membership.astype(np.int32),
        "modularity": np.float64(partition.modularity),
        "quality": np.float64(partition.quality),
        "stability": np.float64(partition.stability),
        "resolution": np.float64(partition.resolution),
        "seed": np.int64(partition.seed),
        "sizes": np.asarray(partition.sizes, dtype=np.int64),
        "qualities": np.asarray(partition.qualities, dtype=np.float64),
        "memberships": (
            partition.memberships.astype(np.int32)
            if partition.memberships is not None
            else partition.membership.astype(np.int32)[None, :]
        ),
    }
    if vocab_ids is not None:
        arrays["vocab_ids"] = np.asarray(vocab_ids, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def _occurrence_token_ids(paths: RunPaths) -> np.ndarray | None:
    """Token id of every occurrence, or ``None`` when ``occurrences.csv`` is missing.

    Args:
        paths: Paths of the run.

    Returns:
        int64 array ``[N]`` or ``None``.
    """

    return read_token_ids(paths.occurrences) if paths.occurrences.exists() else None


def _vocab_ids(knn_dir: Path, specs: Sequence[GraphSpec]) -> dict[str, np.ndarray]:
    """Token ids of the rows of every vocabulary neighbour file.

    Args:
        knn_dir: Directory of the ``knn`` stage.
        specs: Specifications; only the ``vocab`` ones are read.

    Returns:
        File name to its ``vocab_ids`` array.
    """

    out: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.rep == VOCAB and spec.source not in out:
            with np.load(knn_dir / spec.source) as data:
                out[spec.source] = np.asarray(data["vocab_ids"], dtype=np.int64)
    return out


def _check_vertex_counts(
    specs: Sequence[GraphSpec],
    fingerprints: Mapping[str, Fingerprint],
    occurrence_tokens: np.ndarray | None,
    vocab_ids: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Every occurrence graph must have the same |V| (and match occurrences.csv).

    Args:
        specs: Specifications.
        fingerprints: Graph id to its fingerprint.
        occurrence_tokens: Token id of every occurrence, or ``None``.
        vocab_ids: File name to the token ids of its rows.

    Returns:
        ``n_occurrences`` and, per vocabulary ``k``, the number of rows.

    Raises:
        ValueError: If the occurrence graphs disagree on the number of vertices, differ
            from ``occurrences.csv``, or a vocabulary file disagrees with its id list.
    """

    occurrence_n = {fingerprints[s.graph_id].n_vertices for s in specs if s.rep != VOCAB}
    if len(occurrence_n) > 1:
        raise ValueError(f"Occurrence graphs disagree on |V|: {sorted(occurrence_n)}")
    n_occ = occurrence_n.pop() if occurrence_n else None
    if n_occ is not None and occurrence_tokens is not None and occurrence_tokens.size != n_occ:
        raise ValueError(
            f"knn neighbour sets have {n_occ} rows but occurrences.csv has {occurrence_tokens.size}"
        )
    vocab_n: dict[str, int] = {}
    for spec in specs:
        if spec.rep == VOCAB:
            n = fingerprints[spec.graph_id].n_vertices
            if n != vocab_ids[spec.source].size:
                raise ValueError(f"{spec.source}: {n} rows but {vocab_ids[spec.source].size} ids")
            vocab_n[f"k{spec.k}"] = n
    return {"n_occurrences": n_occ, "n_vocab_rows": vocab_n}


def _anomalies(
    specs: Sequence[GraphSpec], fingerprints: Mapping[str, Fingerprint]
) -> list[dict[str, Any]]:
    """Neighbour sets breaking the knn contract (loops, repeats, out-degree != k).

    Args:
        specs: Specifications.
        fingerprints: Graph id to its fingerprint.

    Returns:
        One entry per offending graph; variant (b) may exceed ``k`` but not fall below it.
    """

    found = []
    for spec in specs:
        fp = fingerprints[spec.graph_id]
        low_ok = fp.out_degree_min >= spec.k
        exact_ok = spec.tie == "all" or fp.out_degree_max == spec.k
        if fp.n_self_loops or fp.n_repeated or not low_ok or not exact_ok:
            found.append(
                {
                    "graph_id": spec.graph_id,
                    "self_loops": fp.n_self_loops,
                    "repeated": fp.n_repeated,
                    "out_degree_min": fp.out_degree_min,
                    "out_degree_max": fp.out_degree_max,
                }
            )
    return found


def metrics_inputs(paths: RunPaths) -> list[Path]:
    """Upstream files whose change makes the metrics stale.

    Args:
        paths: Paths of the run.

    Returns:
        The manifests of the ``knn`` and ``sample`` stages.
    """

    return [paths.manifest("knn"), paths.manifest("sample")]


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    """Measure every graph specification whose neighbour file exists in ``paths.knn_dir``.

    Args:
        settings: Run configuration; the ``networks`` and ``analysis`` sections are used.
        paths: Paths of the run.
        force: Redo the stage even when it is up to date.
        **_: Ignored; accepted so every stage has the same call.

    Raises:
        FileNotFoundError: If the knn directory or a main-configuration neighbour file is
            missing. Missing robustness files only skip their graphs, with a warning.
        ValueError: If the neighbour sets disagree on the number of vertices.

    Side effects:
        Starts worker processes, removes the old partition files and writes the tables of
        ``metrics/``, the partitions with their index and, last, the manifest.

    Notes:
        Only ``graph_metrics.csv`` is tested to decide whether the stage is up to date.
    """

    out_dir = paths.metrics_dir
    outputs = [paths.metrics_table("graph_metrics")]
    if up_to_date("metrics", paths, settings, metrics_inputs(paths), outputs, force):
        return
    started = time.time()
    knn_dir = paths.knn_dir
    require_inputs([knn_dir], "a etapa knn")

    planned = plan_graphs(settings)
    present = {source for source in {s.source for s in planned} if (knn_dir / source).exists()}
    specs = [spec for spec in planned if spec.source in present]
    missing = sorted({spec.source for spec in planned} - present)
    k_main = settings.networks.k_main
    required = {neighbors_name(rep, k_main) for rep in settings.networks.representations}
    if required - present:
        raise FileNotFoundError(
            f"Main-configuration neighbour files missing in {knn_dir}: "
            f"{', '.join(sorted(required - present))}"
        )
    for source in missing:
        LOGGER.warning("Arquivo %s ausente; grafos que dependem dele foram pulados", source)

    phase = time.time()
    fingerprints = fingerprint_specs(knn_dir, specs)
    occurrence_tokens = _occurrence_token_ids(paths)
    vocab_ids = _vocab_ids(knn_dir, specs)
    counts = _check_vertex_counts(specs, fingerprints, occurrence_tokens, vocab_ids)
    anomalies = _anomalies(specs, fingerprints)
    for anomaly in anomalies:
        LOGGER.warning("Vizinhança fora do contrato: %s", anomaly)
    plan = build_plan(settings, specs, fingerprints, knn_dir)
    fingerprint_s = time.time() - phase
    LOGGER.info(
        "%d especificações, %d grafos distintos, %d duplicados, %d pedidos de comunidades",
        len(specs),
        len(plan.jobs),
        len(plan.duplicate_of),
        len(plan.requests),
    )

    phase = time.time()
    workers = max(1, min(settings.analysis.workers, len(plan.jobs), os.cpu_count() or 1))
    results: dict[str, GraphResult] = {}
    for done, result in enumerate(execute_jobs(plan.jobs, workers), start=1):
        results[result.graph_id] = result
        LOGGER.info(
            "Grafo %d/%d: %s (%.1f s)", done, len(plan.jobs), result.graph_id, result.elapsed_s
        )
    compute_s = time.time() - phase

    phase = time.time()
    extra = _write_outputs(settings, paths, plan, results, occurrence_tokens, vocab_ids)
    write_s = time.time() - phase

    slowest = sorted(results.values(), key=lambda r: -r.elapsed_s)[:10]
    extra.update(
        n_specs=len(specs),
        n_computed=len(plan.jobs),
        n_duplicates=len(plan.duplicate_of),
        specs_by_group={
            group: sum(spec.group == group for spec in specs)
            for group in ("main", "robust", "vocab")
        },
        missing_sources=missing,
        anomalies=anomalies,
        workers=workers,
        leiden_seed=settings.networks.base_seed,
        distance_seed_rule="crc32('{base_seed}|{graph_id}')",
        timings_s={
            "fingerprint": round(fingerprint_s, 3),
            "compute_wall": round(compute_s, 3),
            "compute_cpu_sum": round(sum(r.elapsed_s for r in results.values()), 3),
            "write": round(write_s, 3),
        },
        slowest_graphs=[
            {"graph_id": r.graph_id, "elapsed_s": round(r.elapsed_s, 3)} for r in slowest
        ],
        **counts,
    )
    write_manifest(
        out_dir, "metrics", settings, started, extra, root=paths.root, inputs=metrics_inputs(paths)
    )
    LOGGER.info(
        "Métricas: %d grafos (%d calculados) e %d partições em %.1f s",
        len(specs),
        len(plan.jobs),
        extra["n_partition_files"],
        time.time() - started,
    )


def _write_outputs(
    settings: Settings,
    paths: RunPaths,
    plan: MetricsPlan,
    results: Mapping[str, GraphResult],
    occurrence_tokens: np.ndarray | None,
    vocab_ids: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write every table and partition file; return statistics for the manifest.

    Args:
        settings: Run configuration.
        paths: Paths of the run.
        plan: The plan that was executed.
        results: Canonical graph id to its result.
        occurrence_tokens: Token id of every occurrence, for the hub table.
        vocab_ids: File name to the token ids of its rows.

    Returns:
        The number of partition files, the Louvain cross-check and the main-configuration
        summary.

    Side effects:
        Deletes every ``.npz`` in ``metrics/communities`` before writing the new ones.
    """

    community_dir = ensure_dir(paths.communities_dir)
    for stale in community_dir.glob("*.npz"):
        stale.unlink()
    main_res = float(settings.analysis.resolution)

    metric_rows, degree_rows, component_rows, distance_rows, hub_rows = [], [], [], [], []
    for spec in plan.specs:
        gid = spec.graph_id
        canonical = plan.canonical(gid)
        result = results[canonical]
        main_partition = result.partitions.get(("leiden", main_res))
        metric_rows.append(
            metric_row(
                spec,
                plan.fingerprints[gid],
                result,
                canonical if canonical != gid else None,
                main_partition,
            )
        )
        for kind, hist in result.degree_hist.items():
            degree_rows.extend(
                {"graph_id": gid, "kind": kind, "degree": d, "count": c} for d, c in hist.items()
            )
        component_rows.extend(
            {"graph_id": gid, "size": s, "count": c} for s, c in result.component_sizes.items()
        )
        distance_rows.extend(
            {"graph_id": gid, "distance": d, "count": c} for d, c in result.distance_hist.items()
        )
        tokens = vocab_ids.get(spec.source) if spec.rep == VOCAB else occurrence_tokens
        for rank, (vertex, indeg) in enumerate(result.hubs, start=1):
            token = int(tokens[vertex]) if tokens is not None else -1
            hub_rows.append(
                {
                    "graph_id": gid,
                    "rank": rank,
                    "vertex": vertex,
                    "in_degree": indeg,
                    "token_id": token,
                }
            )
    write_csv(paths.metrics_table("graph_metrics"), metric_rows, METRIC_COLUMNS)
    write_csv(paths.metrics_table("degree_hist"), degree_rows, DEGREE_COLUMNS)
    write_csv(paths.metrics_table("components"), component_rows, COMPONENT_COLUMNS)
    write_csv(paths.metrics_table("distance_hist"), distance_rows, DISTANCE_COLUMNS)
    write_csv(paths.metrics_table("hubs"), hub_rows, HUB_COLUMNS)

    specs_by_id = {spec.graph_id: spec for spec in plan.specs}
    index_rows, louvain_checks = [], []
    for request in plan.requests:
        gid, res, method = request.graph_id, request.resolution, request.method
        canonical = plan.canonical(gid)
        partition = results[canonical].partitions[(method, res)]
        spec = specs_by_id[gid]
        path = paths.community(gid, res, method)
        ids = vocab_ids.get(spec.source) if spec.rep == VOCAB else None
        _save_partition(path, partition, ids)
        nmi_vs_leiden = None
        if method == "louvain":
            reference = results[canonical].partitions.get(("leiden", res))
            if reference is not None:
                nmi_vs_leiden = nmi(partition.membership, reference.membership)
                louvain_checks.append(
                    {
                        "graph_id": gid,
                        "resolution": res,
                        "leiden_modularity": reference.modularity,
                        "louvain_modularity": partition.modularity,
                        "leiden_n_communities": reference.n_communities,
                        "louvain_n_communities": partition.n_communities,
                        "nmi": nmi_vs_leiden,
                    }
                )
        sizes = partition.sizes
        index_rows.append(
            {
                key: _blank(value)
                for key, value in {
                    "graph_id": gid,
                    "method": method,
                    "resolution": res,
                    "file": path.relative_to(paths.metrics_dir).as_posix(),
                    "duplicate_of": canonical if canonical != gid else None,
                    "n_vertices": int(partition.membership.size),
                    "modularity": partition.modularity,
                    "quality": partition.quality,
                    "n_communities": partition.n_communities,
                    "largest_community": max(sizes) if sizes else 0,
                    "stability": None if method == "louvain" else partition.stability,
                    "runs": len(partition.qualities),
                    "seed": partition.seed,
                    "max_iterations": None if method == "louvain" else partition.max_iterations,
                    "converged_fraction": (
                        None
                        if method == "louvain" or not partition.converged
                        else float(np.mean(partition.converged))
                    ),
                    "nmi_vs_leiden": nmi_vs_leiden,
                }.items()
            }
        )
    write_csv(paths.communities_index, index_rows, COMMUNITY_COLUMNS)
    return {
        "n_partition_files": len(index_rows),
        "louvain_check": louvain_checks,
        "main_configuration": [
            {
                "graph_id": row["graph_id"],
                "n_edges": row["n_edges"],
                "mean_distance": row["mean_distance"],
                "diameter": row["diameter"],
                "avg_local_clustering": row["avg_local_clustering"],
                "modularity": row["modularity"],
            }
            for row in metric_rows
            if row["group"] == "main"
            and row["sym"] == "union"
            and row["tie"] == "rand"
            and row["seed"] == 0
        ],
    }
