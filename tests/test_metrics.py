from __future__ import annotations

import json
import zlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from gender_networks.artifacts import OCCURRENCE_COLUMNS, RunPaths
from gender_networks.metrics import (
    GraphJob,
    GraphSpec,
    centered_representations,
    community_file,
    distance_mode,
    file_safe,
    fingerprint,
    graph_id,
    measure_graph,
    plan_communities,
    plan_graphs,
    run,
    select_neighbors,
)
from gender_networks.settings import (
    AnalysisSettings,
    CorpusSettings,
    NetworkSettings,
    PathsSettings,
    Settings,
    VocabNetworkSettings,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6


def _settings(workers: int = 1) -> Settings:
    return Settings(
        name="tiny",
        paths=PathsSettings(runs_dir="runs", report_dir="relatorio"),
        corpus=CorpusSettings(themes={"a": ["Categoria:A"]}),
        networks=NetworkSettings(
            k_values=[2, 3],
            k_main=3,
            seeds=3,
            base_seed=438,
            representations=["lex", "L01"],
            robust_representations=["L36n"],
            centered=True,
            vocab=VocabNetworkSettings(k_values=[2, 3]),
        ),
        analysis=AnalysisSettings(
            leiden_runs=2,
            resolution=1.0,
            resolution_sweep=[1.0, 2.0],
            permutations=2,
            distance_sample_sources=5,
            workers=workers,
            exact_distances_vocab=False,
        ),
    )


def _select(row: np.ndarray, i: int, k: int, eps: float, rngs: list[np.random.Generator]):
    """Tie semantics of the knn contract for one row: pos, rand (one per rng) and all."""

    n = row.size
    order = np.lexsort((np.arange(n), -row))
    order = order[order != i]
    s_k = row[order[k - 1]]
    free = order[row[order] > s_k + eps]
    block = order[np.abs(row[order] - s_k) <= eps]
    r = k - free.size

    def ordered(picked: np.ndarray) -> np.ndarray:
        chosen = np.concatenate([free, picked])
        return chosen[np.lexsort((chosen, -row[chosen]))]

    pos = ordered(np.sort(block)[:r])
    rand = [ordered(rng.choice(block, size=r, replace=False)) for rng in rngs]
    return pos, rand, np.sort(np.concatenate([free, block]))


def _knn_arrays(vectors: np.ndarray, rep: str, k: int, seeds: int, eps: float, base_seed: int):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit.astype(np.float64) @ unit.astype(np.float64).T
    np.fill_diagonal(sims, -np.inf)
    n = vectors.shape[0]
    rngs = [
        np.random.default_rng([base_seed, zlib.crc32(rep.encode()), k, r]) for r in range(seeds)
    ]
    pos = np.empty((n, k), dtype=np.int32)
    rand = np.empty((seeds, n, k), dtype=np.int32)
    all_rows = []
    for i in range(n):
        p, rs, everything = _select(sims[i], i, k, eps, rngs)
        pos[i] = p
        for r, chosen in enumerate(rs):
            rand[r, i] = chosen
        all_rows.append(everything)
    indptr = np.concatenate([[0], np.cumsum([row.size for row in all_rows])]).astype(np.int64)
    return pos, rand, indptr, np.concatenate(all_rows).astype(np.int32)


def _write_fake_run(tmp_path: Path, settings: Settings) -> tuple[RunPaths, dict[str, object]]:
    """A knn directory (and occurrences.csv) in the format documented by the knn stage."""

    paths = RunPaths.from_settings(settings, tmp_path)
    knn = paths.knn_dir
    knn.mkdir(parents=True)
    net = settings.networks
    rng = np.random.default_rng(0)
    counts = [12, 8, 6, 5, 4, 3, 3, 2, 1, 1]
    token_ids = np.repeat(np.arange(len(counts)) * 7 + 3, counts)
    n = token_ids.size
    type_vectors = rng.normal(size=(token_ids.max() + 1, 6))
    l01 = rng.normal(size=(n, 6)) + 0.8 * type_vectors[token_ids]
    vectors = {
        "lex": type_vectors[token_ids],
        "L01": l01,
        "L36n": rng.normal(size=(n, 6)),
        "L01c": l01 - l01.mean(axis=0),
    }
    for rep, x in vectors.items():
        ks = net.k_values if rep in net.representations else [net.k_main]
        for k in ks:
            pos, rand, indptr, idx = _knn_arrays(x, rep, k, net.seeds, EPS, net.base_seed)
            np.savez_compressed(
                knn / f"nbr_{rep}_k{k}.npz",
                pos=pos,
                rand=rand,
                all_indptr=indptr,
                all_idx=idx,
                types_pos=token_ids[pos].astype(np.int32),
                types_rand=token_ids[rand].astype(np.int32),
            )
        if rep != "L01c":  # one eps file is missing on purpose
            pos, rand, _, _ = _knn_arrays(x, rep, net.k_main, net.seeds, 1e-4, net.base_seed)
            np.savez_compressed(knn / f"nbr_{rep}_k{net.k_main}_eps.npz", pos=pos, rand=rand)
    vocab_vectors = rng.normal(size=(30, 6))
    vocab_vectors[1] = vocab_vectors[0]  # an exact tie, like untrained rows
    vocab_ids = (np.arange(30) + 100).astype(np.int32)
    for k in net.vocab.k_values:
        pos, rand, _, _ = _knn_arrays(vocab_vectors, "vocab", k, 1, EPS, net.base_seed)
        np.savez_compressed(knn / f"vocab_k{k}.npz", vocab_ids=vocab_ids, pos=pos, rand=rand)
    paths.sample_dir.mkdir(parents=True)
    frame = pd.DataFrame({column: [""] * n for column in OCCURRENCE_COLUMNS})
    frame["occurrence_id"] = np.arange(n)
    frame["token_id"] = token_ids
    frame.to_csv(paths.occurrences, index=False)
    return paths, {"token_ids": token_ids, "vocab_ids": vocab_ids, "n": n}


def _nx_union(neighbors: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(neighbors.shape[0]))
    graph.add_edges_from((i, int(j)) for i, row in enumerate(neighbors) for j in row)
    return graph


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("metrics")
    settings = _settings()
    paths, info = _write_fake_run(tmp_path, settings)
    run(settings, paths)
    return settings, paths, info


def test_run_writes_one_row_per_available_spec(tiny_run) -> None:
    settings, paths, _ = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv", keep_default_na=False)
    expected = [s.graph_id for s in plan_graphs(settings) if (paths.knn_dir / s.source).exists()]
    assert frame["graph_id"].tolist() == expected
    assert "L01c|k3|union|rand_eps|s0" not in set(frame["graph_id"])
    manifest = json.loads((paths.metrics_dir / "_manifest.json").read_text())
    assert manifest["stage"] == "metrics"
    assert manifest["missing_sources"] == ["nbr_L01c_k3_eps.npz"]
    assert manifest["n_specs"] == len(expected)
    assert manifest["n_computed"] + manifest["n_duplicates"] == len(expected)
    assert manifest["n_occurrences"] == 45
    assert manifest["n_vocab_rows"] == {"k2": 30, "k3": 30}
    assert manifest["anomalies"] == []
    lex_main = [g for g in expected if g.startswith("lex|k3|union|rand|")]
    assert lex_main == [f"lex|k3|union|rand|s{s}" for s in range(3)]


def test_union_row_matches_networkx(tiny_run) -> None:
    _, paths, info = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv").set_index("graph_id")
    with np.load(paths.knn_dir / "nbr_lex_k3.npz") as data:
        neighbors = data["rand"][0]
    reference = _nx_union(neighbors)
    row = frame.loc["lex|k3|union|rand|s0"]
    assert row["n_vertices"] == info["n"]
    assert row["n_edges"] == reference.number_of_edges()
    assert row["transitivity"] == pytest.approx(nx.transitivity(reference))
    defined = [v for v, d in reference.degree() if d >= 2]
    expected = np.mean([nx.clustering(reference, v) for v in defined])
    assert row["avg_local_clustering"] == pytest.approx(expected)
    components = list(nx.connected_components(reference))
    assert row["n_components"] == len(components)
    lcc = reference.subgraph(max(components, key=len))
    assert row["largest_component"] == lcc.number_of_nodes()
    assert row["mean_distance"] == pytest.approx(nx.average_shortest_path_length(lcc))
    assert row["diameter"] == nx.diameter(lcc)
    assert row["distance_method"] == "exact"
    assert row["group"] == "main"
    assert not np.isnan(row["modularity"]) and row["n_communities"] >= 1


def test_directed_rows_and_hubs(tiny_run) -> None:
    _, paths, info = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv").set_index("graph_id")
    row = frame.loc["L01|k3|directed|rand|s0"]
    assert row["distance_method"] == "none"
    assert row["mean_degree"] == pytest.approx(3.0)
    assert row["out_degree_min"] == row["out_degree_max"] == 3
    assert 0.0 <= row["reciprocity"] <= 1.0
    assert np.isnan(frame.loc["L01|k3|union|rand|s0", "reciprocity"])
    hubs = pd.read_csv(paths.metrics_dir / "hubs.csv")
    top = hubs[hubs["graph_id"] == "L01|k3|directed|rand|s0"].sort_values("rank")
    assert top["rank"].tolist() == list(range(1, 21))
    assert top["in_degree"].iloc[0] == row["in_degree_max"]
    assert (top["token_id"].to_numpy() == info["token_ids"][top["vertex"].to_numpy()]).all()
    vocab_hubs = hubs[hubs["graph_id"] == "vocab|k3|directed|pos|s0"]
    assert (vocab_hubs["token_id"] == vocab_hubs["vertex"] + 100).all()
    assert set(hubs["graph_id"]) == {g for g in frame.index if "|directed|" in g}


def test_duplicates_point_to_the_measured_graph(tiny_run) -> None:
    _, paths, _ = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv", keep_default_na=False)
    frame = frame.set_index("graph_id")
    # Continuous vectors have no ties: the position rule gives the same graph as seed 0.
    assert frame.loc["L01|k3|union|pos|s0", "duplicate_of"] == "L01|k3|union|rand|s0"
    assert frame.loc["L01|k3|union|rand|s0", "duplicate_of"] == ""
    duplicated = frame[frame["duplicate_of"] != ""]
    for _, row in duplicated.iterrows():
        source = frame.loc[row["duplicate_of"]]
        assert source["duplicate_of"] == ""
        assert source["edge_hash"] == row["edge_hash"]
        assert source["n_edges"] == row["n_edges"]
        assert source["mean_distance"] == row["mean_distance"]
        assert row["elapsed_s"] == "" and source["elapsed_s"] != ""
    # Lexical ties are massive, so random seeds give different graphs.
    lex = frame.loc[[f"lex|k3|union|rand|s{s}" for s in range(3)], "edge_hash"]
    assert lex.nunique() > 1


def test_histograms_cover_every_graph(tiny_run) -> None:
    _, paths, _ = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv").set_index("graph_id")
    degree = pd.read_csv(paths.metrics_dir / "degree_hist.csv")
    sums = degree.groupby(["graph_id", "kind"])["count"].sum()
    for gid, row in frame.iterrows():
        kinds = ("in", "out") if row["sym"] == "directed" else ("undirected",)
        for kind in kinds:
            assert sums.loc[(gid, kind)] == row["n_vertices"]
    components = pd.read_csv(paths.metrics_dir / "components.csv")
    sizes = components.assign(total=components["size"] * components["count"])
    totals = sizes.groupby("graph_id")["total"].sum()
    assert (totals.reindex(frame.index) == frame["n_vertices"]).all()
    distances = pd.read_csv(paths.metrics_dir / "distance_hist.csv")
    row = frame.loc["lex|k3|union|rand|s0"]
    hist = distances[distances["graph_id"] == "lex|k3|union|rand|s0"]
    mean = (hist["distance"] * hist["count"]).sum() / hist["count"].sum()
    assert mean == pytest.approx(row["mean_distance"])
    assert hist["distance"].max() == row["diameter"]
    assert set(distances["graph_id"]) == {g for g in frame.index if "|directed|" not in g}


def test_vocab_rows_use_sampled_distances(tiny_run) -> None:
    _, paths, _ = tiny_run
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv").set_index("graph_id")
    vocab = frame[frame["rep"] == "vocab"]
    assert set(vocab["group"]) == {"vocab"}
    assert len(vocab) == 8
    union = vocab[vocab["sym"] == "union"]
    assert set(union["distance_method"]) == {"sampled"}
    assert (union["n_distance_sources"] == 5).all()
    assert (vocab["n_vertices"] == 30).all()


def test_communities_files_and_index(tiny_run) -> None:
    settings, paths, info = tiny_run
    community_dir = paths.metrics_dir / "communities"
    index = pd.read_csv(community_dir / "index.csv", keep_default_na=False)
    specs = [s for s in plan_graphs(settings) if (paths.knn_dir / s.source).exists()]
    requests = plan_communities(settings, specs)
    assert len(index) == len(requests)
    assert set(zip(index["graph_id"], index["resolution"], index["method"], strict=True)) == {
        (r.graph_id, r.resolution, r.method) for r in requests
    }
    for _, row in index.iterrows():
        path = paths.metrics_dir / row["file"]
        assert path.exists()
        with np.load(path) as data:
            assert data["membership"].dtype == np.int32
            n = 30 if row["graph_id"].startswith("vocab") else info["n"]
            assert data["membership"].shape == (n,)
            assert int(data["membership"].max()) + 1 == row["n_communities"]
            assert float(data["modularity"]) == pytest.approx(row["modularity"])
            if row["method"] == "leiden":
                assert data["memberships"].shape == (settings.analysis.leiden_runs, n)
    lex = index[(index["graph_id"] == "lex|k3|union|rand|s0") & (index["method"] == "leiden")]
    assert sorted(lex["resolution"]) == [1.0, 2.0]
    assert "lex|k3|union|rand|s1" in set(index["graph_id"])
    louvain = index[index["method"] == "louvain"]
    assert set(louvain["graph_id"]) == {"lex|k3|union|rand|s0", "L01|k3|union|rand|s0"}
    assert all(0.0 <= float(v) <= 1.0 for v in louvain["nmi_vs_leiden"])
    duplicated = index[index["graph_id"] == "L01|k3|union|pos|s0"]
    assert duplicated["duplicate_of"].tolist() == ["L01|k3|union|rand|s0"]
    name = community_file("vocab|k3|union|rand|s0", 1.0)
    assert name == "vocab__k3__union__rand__s0__res1.0.npz"
    with np.load(community_dir / name) as data:
        np.testing.assert_array_equal(data["vocab_ids"], info["vocab_ids"])
    frame = pd.read_csv(paths.metrics_dir / "graph_metrics.csv").set_index("graph_id")
    first = index[(index["graph_id"] == "lex|k3|union|rand|s0") & (index["resolution"] == 1.0)]
    leiden_row = first[first["method"] == "leiden"].iloc[0]
    assert frame.loc["lex|k3|union|rand|s0", "modularity"] == pytest.approx(
        float(leiden_row["modularity"])
    )


def test_rerun_skips_and_parallel_force_gives_the_same_tables(tiny_run, caplog) -> None:
    settings, paths, _ = tiny_run
    table = paths.metrics_dir / "graph_metrics.csv"
    before = table.stat().st_mtime_ns
    with caplog.at_level("INFO", logger="gender_networks.metrics"):
        run(settings, paths)
    assert table.stat().st_mtime_ns == before
    assert any("já existem" in message for message in caplog.messages)
    serial = pd.read_csv(table).drop(columns=["elapsed_s"])
    serial_index = pd.read_csv(paths.metrics_dir / "communities" / "index.csv")
    run(_settings(workers=2), paths, force=True)
    parallel = pd.read_csv(table).drop(columns=["elapsed_s"])
    pd.testing.assert_frame_equal(serial, parallel)
    pd.testing.assert_frame_equal(
        serial_index, pd.read_csv(paths.metrics_dir / "communities" / "index.csv")
    )
    manifest = json.loads((paths.metrics_dir / "_manifest.json").read_text())
    assert manifest["workers"] == 2


def test_missing_main_file_fails_loudly(tmp_path: Path) -> None:
    settings = _settings()
    paths, _ = _write_fake_run(tmp_path, settings)
    (paths.knn_dir / "nbr_L01_k3.npz").unlink()
    with pytest.raises(FileNotFoundError, match="nbr_L01_k3.npz"):
        run(settings, paths)


def test_plan_for_the_experiment_configs() -> None:
    settings = load_settings(ROOT / "configs" / "experiment.yaml")
    specs = plan_graphs(settings)
    ids = [s.graph_id for s in specs]
    assert len(ids) == len(set(ids))
    lex_main = [s for s in specs if s.rep == "lex" and s.group == "main" and s.tie == "rand"]
    assert len(lex_main) == 2 * settings.networks.seeds
    assert {s.seed for s in specs if s.rep == "L18" and s.tie == "rand"} == {0}
    assert centered_representations(settings) == ["L01c", "L18c", "L36c"]
    assert {s.rep for s in specs if s.tie == "rand_eps"} == {
        "lex", "L01", "L18", "L36", "L36n", "L01c", "L18c", "L36c"
    }
    assert {(s.k, s.sym, s.tie) for s in specs if s.rep == "vocab"} == {
        (k, sym, tie)
        for k in (5, 10, 20)
        for sym in ("union", "directed")
        for tie in ("rand", "pos")
    }
    assert not any(s.sym == "mutual" and s.group == "main" for s in specs)
    exact_vocab = [
        s.graph_id for s in specs if s.rep == "vocab" and distance_mode(s, settings) == "exact"
    ]
    assert exact_vocab == ["vocab|k10|union|rand|s0", "vocab|k10|union|pos|s0"]
    requests = plan_communities(settings, specs)
    leiden_ids = {r.graph_id for r in requests if r.method == "leiden"}
    assert "lex|k10|union|rand|s1" in leiden_ids and "vocab|k10|union|rand|s0" in leiden_ids
    assert "L36c|k10|union|pos|s0" in leiden_ids
    sweep = {r.resolution for r in requests if r.graph_id == "L18|k10|union|rand|s0"}
    assert sweep == {0.5, 1.0, 2.0}
    mini = load_settings(ROOT / "configs" / "mini.yaml")
    mini_specs = plan_graphs(mini)
    assert all(distance_mode(s, mini) != "exact" for s in mini_specs if s.rep == "vocab")
    assert {s.k for s in mini_specs if s.rep == "vocab"} == {10}


def test_graph_ids_and_neighbor_selection() -> None:
    gid = graph_id("L36n", 10, "union", "rand_eps", 0)
    assert gid == "L36n|k10|union|rand_eps|s0"
    assert file_safe(gid) == "L36n__k10__union__rand_eps__s0"
    data = {
        "pos": np.array([[1], [0]]),
        "rand": np.array([[[1], [0]], [[1], [1]]]),
        "all_indptr": np.array([0, 1, 2]),
        "all_idx": np.array([1, 0]),
    }
    assert select_neighbors(data, "pos", 0).tolist() == [[1], [0]]
    assert select_neighbors(data, "rand", 1).tolist() == [[1], [1]]
    indptr, idx = select_neighbors(data, "all", 0)
    assert indptr.tolist() == [0, 1, 2] and idx.tolist() == [1, 0]
    with pytest.raises(ValueError):
        select_neighbors(data, "rand", 2)
    with pytest.raises(ValueError):
        GraphSpec("lex", 10, "both", "rand", 0, "x.npz", "main")
    # Row 0 lists itself; rows 1 and 2 repeat an id: edges 0->1, 1->0, 2->1 remain.
    loopy = fingerprint(np.array([[0, 1], [0, 0], [1, 1]]), "directed")
    assert loopy.n_self_loops == 1 and loopy.n_repeated == 2
    assert loopy.out_degree_min == loopy.out_degree_max == 1 and loopy.n_edges == 3


def test_measure_graph_computes_requested_partitions() -> None:
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    spec = GraphSpec("lex", 2, "union", "pos", 0, "nbr_lex_k2.npz", "main")
    job = GraphJob(
        spec=spec,
        knn_dir=".",
        distances="exact",
        sample_sources=10,
        distance_seed=0,
        leiden_resolutions=(1.0,),
        louvain_resolutions=(1.0,),
        leiden_runs=3,
    )
    result = measure_graph(neighbors, 6, job)
    assert result.metrics["n_components"] == 2
    assert result.partitions[("leiden", 1.0)].membership.tolist() == [0, 0, 0, 1, 1, 1]
    assert result.partitions[("louvain", 1.0)].n_communities == 2
    assert result.degree_hist == {"undirected": {2: 6}}
