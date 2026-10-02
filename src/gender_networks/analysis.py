"""Stage ``analyze``: answer P1–P4 from neighbour sets, communities and representations.

Outputs in ``paths.analysis_dir`` (all CSV unless noted), read by the ``report`` stage:

- ``vertex_measures.csv``: one row per occurrence (main configuration, ``k = k_main``):
  ``D_{rep}`` and ``Dn_{rep}`` (dominance and dominance over its ceiling) for every main
  representation, ``NF_lex`` (tie noise floor), and for every pair ``a→b`` in :data:`PAIRS`
  the Jaccard ``J_{a}_{b}``, the type-composition Jaccard ``Jw_{a}_{b}`` and the
  distinct-type Jaccard ``Jc_{a}_{b}``. Lexical sets are averaged over the tie seeds.
- ``p1_summary.csv``: long table ``measure, group, value, count, mean, median, std, q25, q75``
  of the vertex measures by ``all``, band, category, stratum, position bucket and
  band x category.
- ``p1_global.csv``: course metrics of the main graphs (mean and sd over tie seeds).
- ``p1_hubs.csv``: top in-degree hubs of the main directed graphs with their tokens.
- ``p2_transitions.csv`` and ``p2_transitions_groups.csv``: change per transition.
- ``p2_layers.csv``: layer-by-layer ``J(l-1, l)`` and ``D(l)`` from ``all_layers.npy``.
- ``p3_agreement.csv``, ``p3_layer_nmi.csv``: community agreement with context labels.
- ``p4_types.csv``, ``p4_groups.csv``, ``p4_senses.csv``: same-type occurrence separation.
- ``robustness_neighbors.csv``, ``robustness_graphs.csv``, ``core_metrics.csv``.
- ``vocab_summary.csv``, ``vocab_communities.csv``, ``vocab_target_neighbors.csv``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gender_networks.artifacts import RunPaths, ensure_dir, stage_is_fresh, write_manifest
from gender_networks.graphs import directed_edges as graph_directed_edges
from gender_networks.graphs import graph_metrics, symmetrize, to_igraph, undirected_edges
from gender_networks.neighborhood import (
    dominance,
    dominance_ceiling,
    jaccard,
    noise_floor,
    normalized_dominance,
    type_composition_jaccard,
    type_set_jaccard,
)
from gender_networks.partitions import agreement, nmi
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)
LEXICAL = "lex"
PAIRS = [("lex", "L01"), ("L01", "L18"), ("L18", "L36"), ("lex", "L18"), ("lex", "L36")]
CONSECUTIVE = PAIRS[:3]
GROUPINGS: list[tuple[str, list[str]]] = [
    ("all", []),
    ("band_sample", ["band_sample"]),
    ("token_category", ["token_category"]),
    ("stratum", ["stratum"]),
    ("pos_bucket", ["pos_bucket"]),
    ("band_x_category", ["band_sample", "token_category"]),
]
CORE_LABELS = [
    "token_id",
    "theme",
    "pageid",
    "paragraph_id",
    "sentence_id",
    "next_token_id",
    "pred_next",
    "pos_bucket",
]
ALL_LABELS = ["token_id", "theme", "stratum", "token_category"]
P4_MIN_F = 10
OUTPUT_FILES = [
    "vertex_measures.csv",
    "p1_summary.csv",
    "p1_global.csv",
    "p2_transitions.csv",
    "p2_transitions_groups.csv",
    "p3_agreement.csv",
    "p4_types.csv",
    "p4_groups.csv",
    "robustness_neighbors.csv",
]


# --------------------------------------------------------------------------------------------
# loading


def load_occurrences(paths: RunPaths) -> pd.DataFrame:
    frame = pd.read_csv(paths.occurrences, keep_default_na=False, low_memory=False)
    if not np.array_equal(frame["occurrence_id"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"{paths.occurrences}: occurrence_id must equal the row index")
    return frame


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main_reps(settings: Settings, paths: RunPaths) -> list[str]:
    """Main representations whose neighbour file exists, in configuration order."""

    k = settings.networks.k_main
    return [rep for rep in settings.networks.representations if paths.nbr(rep, k).exists()]


def seed_sets(data: Mapping[str, np.ndarray], rep: str) -> list[np.ndarray]:
    """Neighbour sets of the main tie treatment: every seed for ``lex``, seed 0 otherwise."""

    rand = data["rand"]
    return [rand[r] for r in range(rand.shape[0])] if rep == LEXICAL else [rand[0]]


def type_sets(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return data["types_rand"][0] if "types_rand" in data else data["types_pos"]


# --------------------------------------------------------------------------------------------
# P1 / P2: vertex measures


def vertex_measures(
    occurrences: pd.DataFrame, nbrs: Mapping[str, Mapping[str, np.ndarray]], k: int
) -> pd.DataFrame:
    """Per-occurrence dominance, noise floor and change between representations."""

    token_ids = occurrences["token_id"].to_numpy(dtype=np.int64)
    f_sample = occurrences["f_sample"].to_numpy(dtype=np.int64)
    ceiling = dominance_ceiling(token_ids, f_sample, k)
    out = pd.DataFrame({"occurrence_id": occurrences["occurrence_id"].to_numpy()})
    out["ceiling"] = ceiling
    sets = {rep: seed_sets(data, rep) for rep, data in nbrs.items()}
    for rep, rep_sets in sets.items():
        dom = np.mean([dominance(s, token_ids) for s in rep_sets], axis=0)
        out[f"D_{rep}"] = dom
        out[f"Dn_{rep}"] = normalized_dominance(dom, ceiling)
    if LEXICAL in nbrs and nbrs[LEXICAL]["rand"].shape[0] >= 2:
        out["NF_lex"] = noise_floor(nbrs[LEXICAL]["rand"])
    for a, b in PAIRS:
        if a not in sets or b not in sets:
            continue
        pairs = [(sa, sets[b][i % len(sets[b])]) for i, sa in enumerate(sets[a])]
        out[f"J_{a}_{b}"] = np.mean([jaccard(x, y) for x, y in pairs], axis=0)
        out[f"Jw_{a}_{b}"] = type_composition_jaccard(sets[a][0], sets[b][0], token_ids)
        out[f"Jc_{a}_{b}"] = type_set_jaccard(type_sets(nbrs[a]), type_sets(nbrs[b]))
    return out


def measure_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("D_", "Dn_", "NF_", "J_", "Jw_", "Jc_")
    return [c for c in frame.columns if c.startswith(prefixes)]


def summarize(
    frame: pd.DataFrame, measures: Sequence[str], groupings: Sequence[tuple[str, list[str]]]
) -> pd.DataFrame:
    """Long summary ``measure, group, value, count, mean, median, std, q25, q75``."""

    rows: list[dict[str, Any]] = []
    for name, keys in groupings:
        grouped = [((), frame)] if not keys else frame.groupby(keys, sort=True)
        for value, part in grouped:
            label = "all" if not keys else "|".join(str(v) for v in np.atleast_1d(value))
            for measure in measures:
                series = pd.to_numeric(part[measure], errors="coerce").dropna()
                rows.append(
                    {
                        "measure": measure,
                        "group": name,
                        "value": label,
                        "count": int(series.size),
                        "mean": float(series.mean()) if series.size else math.nan,
                        "median": float(series.median()) if series.size else math.nan,
                        "std": float(series.std(ddof=1)) if series.size > 1 else math.nan,
                        "q25": float(series.quantile(0.25)) if series.size else math.nan,
                        "q75": float(series.quantile(0.75)) if series.size else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def transitions_table(
    measures: pd.DataFrame, occurrences: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change per transition overall and by band x category (P2)."""

    available = [(a, b) for a, b in PAIRS if f"J_{a}_{b}" in measures]
    consecutive = [(a, b) for a, b in CONSECUTIVE if (a, b) in available]
    largest = None
    if consecutive:
        stacked = np.vstack([measures[f"J_{a}_{b}"].to_numpy() for a, b in consecutive])
        largest = np.argmin(stacked, axis=0)

    def rows_for(mask: np.ndarray, group: str, value: str) -> list[dict[str, Any]]:
        rows = []
        for a, b in available:
            j = measures.loc[mask, f"J_{a}_{b}"].to_numpy()
            floor = (
                measures.loc[mask, "NF_lex"].to_numpy()
                if a == LEXICAL and "NF_lex" in measures
                else np.ones_like(j)
            )
            da = measures.loc[mask, f"D_{a}"].to_numpy()
            db = measures.loc[mask, f"D_{b}"].to_numpy()
            share = math.nan
            if largest is not None and (a, b) in consecutive:
                index = consecutive.index((a, b))
                share = float(np.mean(largest[mask] == index)) if mask.any() else math.nan
            rows.append(
                {
                    "group": group,
                    "value": value,
                    "transition": f"{a}->{b}",
                    "consecutive": (a, b) in consecutive,
                    "n": int(mask.sum()),
                    "J_mean": float(np.nanmean(j)) if j.size else math.nan,
                    "J_median": float(np.nanmedian(j)) if j.size else math.nan,
                    "noise_floor_mean": float(np.nanmean(floor)) if j.size else math.nan,
                    "excess_change_mean": float(np.nanmean(floor - j)) if j.size else math.nan,
                    "Jw_mean": float(np.nanmean(measures.loc[mask, f"Jw_{a}_{b}"])),
                    "Jc_mean": float(np.nanmean(measures.loc[mask, f"Jc_{a}_{b}"])),
                    "delta_D_mean": float(np.nanmean(db - da)) if j.size else math.nan,
                    "share_largest_change": share,
                }
            )
        return rows

    everything = np.ones(len(measures), dtype=bool)
    overall = pd.DataFrame(rows_for(everything, "all", "all"))
    grouped: list[dict[str, Any]] = []
    for keys in (["band_sample"], ["token_category"], ["band_sample", "token_category"]):
        labels = occurrences[keys].astype(str).agg("|".join, axis=1).to_numpy()
        for value in sorted(set(labels)):
            grouped.extend(rows_for(labels == value, "x".join(keys), value))
    return overall, pd.DataFrame(grouped)


def layer_sweep(
    paths: RunPaths,
    settings: Settings,
    token_ids: np.ndarray,
    categories: np.ndarray,
    lexical_sets: np.ndarray | None,
) -> pd.DataFrame:
    """``J(l-1, l)`` and ``D(l)`` for every block, from ``all_layers.npy`` (position tie rule)."""

    from gender_networks.knn import knn_candidates, load_layer, normalize_rows, select_neighbors

    net = settings.networks
    k = net.k_main
    stack = np.load(paths.all_layers, mmap_mode="r")
    n_layers = stack.shape[0]
    del stack
    ceiling = dominance_ceiling(token_ids, None, k)
    previous = lexical_sets
    rows: list[dict[str, Any]] = []
    groups = ["all", *sorted(set(categories))]
    for layer in range(1, n_layers + 1):
        started = time.perf_counter()
        x = normalize_rows(load_layer(paths.all_layers, layer))
        candidates, _ = knn_candidates(
            x, k, max(2 * k, k + 8), net.eps, net.block_size, label=f"camada {layer}"
        )
        del x
        current = np.asarray(select_neighbors(candidates, k, net.eps, "position"))
        dom = dominance(current, token_ids)
        dn = normalized_dominance(dom, ceiling)
        jac = jaccard(previous, current) if previous is not None else np.full(dom.size, np.nan)
        for group in groups:
            mask = np.ones(dom.size, dtype=bool) if group == "all" else categories == group
            rows.append(
                {
                    "layer": layer,
                    "token_category": group,
                    "n": int(mask.sum()),
                    "J_prev_mean": float(np.nanmean(jac[mask])) if mask.any() else math.nan,
                    "D_mean": float(np.nanmean(dom[mask])) if mask.any() else math.nan,
                    "Dn_mean": float(np.nanmean(dn[mask])) if mask.any() else math.nan,
                }
            )
        previous = current
        LOGGER.info("Camada %d/%d em %.1f s", layer, n_layers, time.perf_counter() - started)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# global metrics, hubs, robustness


def global_table(metrics: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Main graphs (k_main, random tie, union and directed): mean and sd over tie seeds."""

    k = settings.networks.k_main
    reps = settings.networks.representations
    main = metrics[
        (metrics["k"] == k)
        & (metrics["tie"] == "rand")
        & metrics["sym"].isin(["union", "directed"])
        & metrics["rep"].isin(reps)
    ]
    numeric = [
        c
        for c in main.columns
        if c
        not in {"graph_id", "rep", "k", "sym", "tie", "seed", "duplicate_of", "distance_method"}
        and pd.api.types.is_numeric_dtype(main[c])
    ]
    rows = []
    for (rep, sym), part in main.groupby(["rep", "sym"], sort=False):
        row: dict[str, Any] = {"rep": rep, "sym": sym, "seeds": int(part["seed"].nunique())}
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            row[column] = float(values.mean()) if values.size else math.nan
            row[f"{column}_sd"] = float(values.std(ddof=1)) if values.size > 1 else math.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        order = {rep: i for i, rep in enumerate(reps)}
        table = table.sort_values(
            ["sym", "rep"], key=lambda s: s.map(order) if s.name == "rep" else s
        )
    return table


def hubs_table(
    paths: RunPaths, occurrences: pd.DataFrame, settings: Settings, top: int = 10
) -> pd.DataFrame:
    path = paths.metrics_dir / "hubs.csv"
    if not path.exists():
        return pd.DataFrame()
    hubs = pd.read_csv(path, keep_default_na=False)
    k = settings.networks.k_main
    wanted = {f"{rep}|k{k}|directed|rand|s0": rep for rep in settings.networks.representations}
    hubs = hubs[hubs["graph_id"].isin(wanted)].copy()
    hubs["rep"] = hubs["graph_id"].map(wanted)
    hubs = hubs[hubs["rank"].astype(int) <= top]
    vertex = hubs["vertex"].astype(int).to_numpy()
    hubs["token_text"] = occurrences["token_text"].to_numpy()[vertex]
    hubs["token_category"] = occurrences["token_category"].to_numpy()[vertex]
    hubs["stratum"] = occurrences["stratum"].to_numpy()[vertex]
    hubs["f_sample"] = occurrences["f_sample"].to_numpy()[vertex]
    return hubs[
        [
            "rep",
            "rank",
            "vertex",
            "in_degree",
            "token_text",
            "token_category",
            "stratum",
            "f_sample",
        ]
    ]


def robustness_neighbors(
    paths: RunPaths, settings: Settings, occurrences: pd.DataFrame, reps: Iterable[str]
) -> pd.DataFrame:
    """Mean dominance and agreement with the lexical sets for every k, tie rule and variant."""

    token_ids = occurrences["token_id"].to_numpy(dtype=np.int64)
    f_sample = occurrences["f_sample"].to_numpy(dtype=np.int64)
    net = settings.networks
    rows: list[dict[str, Any]] = []
    lexical: dict[tuple[int, str], Any] = {}
    for k in net.k_values:
        if paths.nbr(LEXICAL, k).exists():
            data = load_npz(paths.nbr(LEXICAL, k))
            lexical[(k, "pos")] = data["pos"]
            lexical[(k, "rand")] = data["rand"][0]
            lexical[(k, "all")] = (data["all_indptr"], data["all_idx"])
    for rep in reps:
        for k in net.k_values:
            path = paths.nbr(rep, k)
            if not path.exists():
                continue
            data = load_npz(path)
            ceiling = dominance_ceiling(token_ids, f_sample, k)
            variants = {
                "pos": data["pos"],
                "rand": data["rand"][0],
                "all": (data["all_indptr"], data["all_idx"]),
            }
            for tie, sets in variants.items():
                dom = dominance(sets, token_ids)
                row = {
                    "rep": rep,
                    "k": k,
                    "tie": tie,
                    "eps": net.eps,
                    "D_mean": float(np.nanmean(dom)),
                    "Dn_mean": float(np.nanmean(normalized_dominance(dom, ceiling))),
                    "J_vs_lex_mean": math.nan,
                    "mean_set_size": float(np.mean(np.diff(sets[0])))
                    if isinstance(sets, tuple)
                    else float(k),
                }
                if rep != LEXICAL and (k, tie) in lexical:
                    row["J_vs_lex_mean"] = float(np.nanmean(jaccard(lexical[(k, tie)], sets)))
                rows.append(row)
            eps_path = paths.nbr(rep, k, sensitivity=True)
            if eps_path.exists():
                eps_data = load_npz(eps_path)
                changed = jaccard(data["rand"][0], eps_data["rand"][0]) < 1.0
                dom = dominance(eps_data["rand"][0], token_ids)
                rows.append(
                    {
                        "rep": rep,
                        "k": k,
                        "tie": "rand",
                        "eps": net.eps_sensitivity,
                        "D_mean": float(np.nanmean(dom)),
                        "Dn_mean": float(np.nanmean(normalized_dominance(dom, ceiling))),
                        "J_vs_lex_mean": math.nan,
                        "mean_set_size": float(k),
                        "rows_changed_by_eps": float(np.mean(changed)),
                    }
                )
    return pd.DataFrame(rows)


def robustness_graphs(metrics: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Course metrics of every undirected graph, for the robustness table."""

    columns = [
        "graph_id",
        "rep",
        "k",
        "sym",
        "tie",
        "seed",
        "n_vertices",
        "n_edges",
        "mean_degree",
        "transitivity",
        "avg_local_clustering",
        "n_components",
        "largest_fraction",
        "mean_distance",
        "diameter",
        "distance_method",
        "modularity",
        "n_communities",
    ]
    present = [c for c in columns if c in metrics.columns]
    table = metrics[metrics["sym"].isin(["union", "mutual"])][present]
    return table.reset_index(drop=True)


def core_metrics(
    paths: RunPaths, settings: Settings, occurrences: pd.DataFrame, reps: Iterable[str]
) -> pd.DataFrame:
    """Course metrics of the union graphs induced on the dense core only (robustness)."""

    core = occurrences["stratum"].to_numpy() == "core"
    index = np.flatnonzero(core)
    if index.size < 2:
        return pd.DataFrame()
    remap = np.full(len(occurrences), -1, dtype=np.int64)
    remap[index] = np.arange(index.size)
    rows = []
    k = settings.networks.k_main
    for rep in reps:
        data = load_npz(paths.nbr(rep, k))
        edges = undirected_edges(
            graph_directed_edges(data["rand"][0], len(occurrences)), len(occurrences)
        )
        keep = core[edges[:, 0]] & core[edges[:, 1]]
        sub = remap[edges[keep]]
        graph = to_igraph(symmetrize(sub, index.size, "union"), index.size, directed=False)
        metrics = graph_metrics(graph, "exact", 0, 0)
        for key in ("degree_hist", "component_sizes", "distance_hist"):
            metrics.pop(key, None)
        rows.append({"rep": rep, "subset": "core", **metrics})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# P3: communities


def community_membership(
    paths: RunPaths, graph_id: str, resolution: float, method: str = "leiden"
) -> tuple[np.ndarray, float] | None:
    index_path = paths.metrics_dir / "communities" / "index.csv"
    if not index_path.exists():
        return None
    index = pd.read_csv(index_path, keep_default_na=False)
    rows = index[
        (index["graph_id"] == graph_id)
        & (index["method"] == method)
        & np.isclose(pd.to_numeric(index["resolution"]), resolution)
    ]
    if rows.empty:
        return None
    data = load_npz(paths.metrics_dir / rows.iloc[0]["file"])
    stability = float(data["stability"]) if "stability" in data else math.nan
    return data["membership"].astype(np.int64), stability


def p3_tables(
    paths: RunPaths, settings: Settings, occurrences: pd.DataFrame, reps: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = settings.networks.k_main
    res = float(settings.analysis.resolution)
    labels = occurrences.copy()
    labels["next_token_id"] = (
        labels["next_token_id"].astype(int).where(labels["next_token_id"].astype(int) >= 0)
    )
    if paths.pred_next.exists():
        labels["pred_next"] = np.load(paths.pred_next).astype(np.int64)
    core = labels["stratum"].to_numpy() == "core"
    rows: list[dict[str, Any]] = []
    memberships: dict[str, np.ndarray] = {}
    for rep in reps:
        found = community_membership(paths, f"{rep}|k{k}|union|rand|s0", res)
        if found is None:
            continue
        membership, stability = found
        memberships[rep] = membership
        for subset, mask, names in (("core", core, CORE_LABELS), ("all", None, ALL_LABELS)):
            chosen = mask if mask is not None else np.ones(membership.size, dtype=bool)
            label_map = {
                name: labels[name].to_numpy()[chosen] for name in names if name in labels.columns
            }
            for row in agreement(
                membership[chosen], label_map, settings.analysis.permutations, settings.sample.seed
            ):
                rows.append({"rep": rep, "subset": subset, "stability": stability, **row})
    layer_rows = []
    ordered = [rep for rep in reps if rep in memberships]
    for a, b in zip(ordered, ordered[1:], strict=False):
        layer_rows.append({"from": a, "to": b, "nmi": nmi(memberships[a], memberships[b])})
    seed1 = community_membership(paths, f"{LEXICAL}|k{k}|union|rand|s1", res)
    if seed1 is not None and LEXICAL in memberships:
        layer_rows.append(
            {"from": "lex(s0)", "to": "lex(s1)", "nmi": nmi(memberships[LEXICAL], seed1[0])}
        )
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


# --------------------------------------------------------------------------------------------
# P4: separation of same-type occurrences


def type_groups(occurrences: pd.DataFrame, settings: Settings) -> dict[int, tuple[str, str]]:
    """token_id -> (group, word) for target, control, multitheme and function-word types."""

    groups: dict[int, tuple[str, str]] = {}
    targets, controls = set(settings.sample.targets), set(settings.sample.controls)
    marked = occurrences[occurrences["target_word"] != ""]
    for token_id, word in marked.groupby("token_id")["target_word"].first().items():
        groups[int(token_id)] = (
            "target" if word in targets else "control" if word in controls else "other",
            word,
        )
    multitheme = occurrences.loc[occurrences["stratum"] == "multitheme", "token_id"].unique()
    for token_id in multitheme:
        groups.setdefault(int(token_id), ("multitheme", ""))
    function = occurrences.groupby("token_id")["is_function_word"].agg(
        lambda s: s.astype(str).str.lower().isin(["true", "1"]).mean() > 0.5
    )
    for token_id, is_function in function.items():
        if is_function:
            groups.setdefault(int(token_id), ("function", ""))
    return groups


def _pair_means(x: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Mean cosine over all pairs, within-label pairs and between-label pairs."""

    sim = x @ x.T
    upper = np.triu_indices(len(x), 1)
    values = sim[upper]
    same = labels[upper[0]] == labels[upper[1]]
    within = float(values[same].mean()) if same.any() else math.nan
    between = float(values[~same].mean()) if (~same).any() else math.nan
    return float(values.mean()), within, between


def _effective_communities(membership: np.ndarray) -> tuple[int, float]:
    _, counts = np.unique(membership, return_counts=True)
    p = counts / counts.sum()
    return int(counts.size), float(np.exp(-(p * np.log(p)).sum()))


def p4_tables(
    paths: RunPaths, settings: Settings, occurrences: pd.DataFrame, reps: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from gender_networks.knn import load_embeddings, load_representation, normalize_rows

    k = settings.networks.k_main
    res = float(settings.analysis.resolution)
    token_ids = occurrences["token_id"].to_numpy(dtype=np.int64)
    themes = occurrences["theme"].to_numpy()
    f_sample = occurrences["f_sample"].to_numpy(dtype=np.int64)
    groups = type_groups(occurrences, settings)
    selected = sorted({int(t) for t in token_ids[f_sample >= P4_MIN_F]})
    members = {t: np.flatnonzero(token_ids == t) for t in selected}
    senses = _read_senses(paths)
    rows: list[dict[str, Any]] = []
    sense_rows: list[dict[str, Any]] = []
    for rep in reps:
        found = community_membership(paths, f"{rep}|k{k}|union|rand|s0", res)
        membership = found[0] if found else None
        if rep == LEXICAL:
            weight = load_embeddings(paths.embeddings)
            vectors = None
            lexical_rows = weight
        else:
            vectors = normalize_rows(load_representation(paths, settings, rep))
            lexical_rows = None
        for token_id, index in members.items():
            if vectors is not None:
                x = vectors[index]
            else:
                row = normalize_rows(lexical_rows[[token_id]])
                x = np.repeat(row, index.size, axis=0)
            labels = themes[index]
            mean_all, within, between = _pair_means(x, labels)
            group, word = groups.get(token_id, ("other", ""))
            entry: dict[str, Any] = {
                "rep": rep,
                "token_id": token_id,
                "token_text": occurrences["token_text"].iat[index[0]],
                "group": group,
                "word": word,
                "f_sample": int(index.size),
                "n_themes": int(np.unique(labels).size),
                "self_similarity": mean_all,
                "within_theme": within,
                "between_theme": between,
                "theme_gap": within - between if not math.isnan(between) else math.nan,
                "n_communities": math.nan,
                "effective_communities": math.nan,
                "nmi_community_theme": math.nan,
            }
            if membership is not None:
                comm = membership[index]
                entry["n_communities"], entry["effective_communities"] = _effective_communities(
                    comm
                )
                if entry["n_themes"] > 1:
                    entry["nmi_community_theme"] = nmi(comm, labels)
            rows.append(entry)
        if senses is not None and vectors is not None:
            for word, part in senses.groupby("target_word"):
                index = part["occurrence_id"].to_numpy(dtype=np.int64)
                labels = part["sense"].to_numpy()
                if index.size < 2 or np.unique(labels).size < 2:
                    continue
                mean_all, within, between = _pair_means(vectors[index], labels)
                sense_rows.append(
                    {
                        "rep": rep,
                        "word": word,
                        "n": int(index.size),
                        "n_senses": int(np.unique(labels).size),
                        "within_sense": within,
                        "between_sense": between,
                        "sense_gap": within - between,
                    }
                )
        del vectors
    types = pd.DataFrame(rows)
    measures = ["self_similarity", "theme_gap", "effective_communities", "nmi_community_theme"]
    grouped = (
        types.groupby(["rep", "group"])[measures].agg(["count", "mean", "median"]).reset_index()
        if not types.empty
        else pd.DataFrame()
    )
    if not grouped.empty:
        grouped.columns = ["_".join(c).strip("_") for c in grouped.columns.to_flat_index()]
    return types, grouped, pd.DataFrame(sense_rows)


def _read_senses(paths: RunPaths) -> pd.DataFrame | None:
    if not paths.senses.exists():
        return None
    senses = pd.read_csv(paths.senses, keep_default_na=False)
    senses = senses[senses["sense"].astype(str).str.strip() != ""]
    return senses if not senses.empty else None


# --------------------------------------------------------------------------------------------
# vocabulary network


def vocab_tables(
    paths: RunPaths, settings: Settings, occurrences: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from gender_networks.metrics import vocab_community_k

    if not paths.vocab_types.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    vocab = pd.read_csv(paths.vocab_types, keep_default_na=False)
    decoded = vocab["token_repr"].to_numpy()
    script = vocab["script_class"].to_numpy()
    f_corpus = vocab["f_corpus"].to_numpy(dtype=np.int64)
    k = vocab_community_k(settings)
    summary_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    if k is not None:
        found = community_membership(
            paths, f"vocab|k{k}|union|rand|s0", float(settings.analysis.resolution)
        )
        if found is not None:
            membership, stability = found
            vocab_ids = load_npz(paths.vocab_nbr(k))["vocab_ids"].astype(np.int64)
            in_corpus = f_corpus[vocab_ids] > 0
            for row in agreement(
                membership,
                {"script_class": script[vocab_ids], "in_corpus": in_corpus.astype(int)},
                settings.analysis.permutations,
                settings.sample.seed,
            ):
                summary_rows.append({"k": k, "stability": stability, **row})
            for community in np.unique(membership):
                index = np.flatnonzero(membership == community)
                ids = vocab_ids[index]
                classes, counts = np.unique(script[ids], return_counts=True)
                top = ids[np.argsort(-f_corpus[ids], kind="stable")[:8]]
                community_rows.append(
                    {
                        "community": int(community),
                        "size": int(index.size),
                        "fraction_in_corpus": float(np.mean(f_corpus[ids] > 0)),
                        "main_script_class": classes[np.argmax(counts)],
                        "main_script_share": float(counts.max() / counts.sum()),
                        "examples": " ".join(str(decoded[t]) for t in top),
                    }
                )
        if paths.vocab_nbr(k).exists():
            data = load_npz(paths.vocab_nbr(k))
            vocab_ids = data["vocab_ids"].astype(np.int64)
            row_of = {int(t): i for i, t in enumerate(vocab_ids)}
            words = {**settings.sample.targets, **settings.sample.controls}
            marked = occurrences[occurrences["target_word"] != ""]
            word_ids = marked.groupby("target_word")["token_id"].first().to_dict()
            for word in words:
                token_id = word_ids.get(word)
                if token_id is None or int(token_id) not in row_of:
                    continue
                neighbors = vocab_ids[data["pos"][row_of[int(token_id)]]]
                neighbor_rows.append(
                    {
                        "word": word,
                        "role": "target" if word in settings.sample.targets else "control",
                        "token_id": int(token_id),
                        "neighbors": " ".join(str(decoded[t]) for t in neighbors),
                        "neighbors_in_corpus": int(np.sum(f_corpus[neighbors] > 0)),
                    }
                )
    community_table = pd.DataFrame(community_rows)
    if not community_table.empty:
        community_table = community_table.sort_values("size", ascending=False)
    return pd.DataFrame(summary_rows), community_table, pd.DataFrame(neighbor_rows)


# --------------------------------------------------------------------------------------------
# stage


def analysis_inputs(paths: RunPaths) -> list[Path]:
    return [paths.manifest("sample"), paths.manifest("knn"), paths.manifest("metrics")]


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, na_rep="")


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    out_dir = paths.analysis_dir
    inputs = analysis_inputs(paths)
    outputs = [out_dir / name for name in OUTPUT_FILES]
    if not force and stage_is_fresh(out_dir, settings, inputs, outputs):
        LOGGER.info("Análises já existem em %s; use --force para refazer", out_dir)
        return
    for path in (paths.occurrences, paths.manifest("knn")):
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe; rode as etapas anteriores")
    started = time.time()
    ensure_dir(out_dir)
    (out_dir / "_manifest.json").unlink(missing_ok=True)
    occurrences = load_occurrences(paths)
    k = settings.networks.k_main
    reps = main_reps(settings, paths)
    nbrs = {rep: load_npz(paths.nbr(rep, k)) for rep in reps}
    timings: dict[str, float] = {}

    tick = time.perf_counter()
    measures = vertex_measures(occurrences, nbrs, k)
    labelled = pd.concat(
        [measures, occurrences[["band_sample", "token_category", "stratum", "pos_bucket"]]], axis=1
    )
    _write(measures, out_dir / "vertex_measures.csv")
    _write(summarize(labelled, measure_columns(measures), GROUPINGS), out_dir / "p1_summary.csv")
    overall, grouped = transitions_table(measures, occurrences)
    _write(overall, out_dir / "p2_transitions.csv")
    _write(grouped, out_dir / "p2_transitions_groups.csv")
    timings["p1_p2_s"] = round(time.perf_counter() - tick, 2)

    metrics_path = paths.metrics_dir / "graph_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metrics.empty:
        _write(global_table(metrics, settings), out_dir / "p1_global.csv")
        _write(robustness_graphs(metrics, settings), out_dir / "robustness_graphs.csv")
    else:
        _write(pd.DataFrame(), out_dir / "p1_global.csv")
    _write(hubs_table(paths, occurrences, settings), out_dir / "p1_hubs.csv")

    tick = time.perf_counter()
    from gender_networks.metrics import occurrence_representations

    _write(
        robustness_neighbors(paths, settings, occurrences, occurrence_representations(settings)),
        out_dir / "robustness_neighbors.csv",
    )
    _write(core_metrics(paths, settings, occurrences, reps), out_dir / "core_metrics.csv")
    timings["robustness_s"] = round(time.perf_counter() - tick, 2)

    tick = time.perf_counter()
    agreement_table, layer_nmi = p3_tables(paths, settings, occurrences, reps)
    _write(agreement_table, out_dir / "p3_agreement.csv")
    _write(layer_nmi, out_dir / "p3_layer_nmi.csv")
    timings["p3_s"] = round(time.perf_counter() - tick, 2)

    tick = time.perf_counter()
    types, groups, senses = p4_tables(paths, settings, occurrences, reps)
    _write(types, out_dir / "p4_types.csv")
    _write(groups, out_dir / "p4_groups.csv")
    _write(senses, out_dir / "p4_senses.csv")
    timings["p4_s"] = round(time.perf_counter() - tick, 2)

    tick = time.perf_counter()
    vocab_summary, vocab_communities, vocab_neighbors = vocab_tables(paths, settings, occurrences)
    _write(vocab_summary, out_dir / "vocab_summary.csv")
    _write(vocab_communities, out_dir / "vocab_communities.csv")
    _write(vocab_neighbors, out_dir / "vocab_target_neighbors.csv")
    timings["vocab_s"] = round(time.perf_counter() - tick, 2)

    layer_rows = 0
    if settings.analysis.layer_sweep and paths.all_layers.exists():
        tick = time.perf_counter()
        lexical_pos = nbrs[LEXICAL]["pos"] if LEXICAL in nbrs else None
        sweep = layer_sweep(
            paths,
            settings,
            occurrences["token_id"].to_numpy(dtype=np.int64),
            occurrences["token_category"].to_numpy(),
            lexical_pos,
        )
        _write(sweep, out_dir / "p2_layers.csv")
        layer_rows = len(sweep)
        timings["layer_sweep_s"] = round(time.perf_counter() - tick, 2)

    extra = {
        "n_vertices": len(occurrences),
        "representations": reps,
        "k": k,
        "p4_types": int(types["token_id"].nunique()) if not types.empty else 0,
        "p3_rows": len(agreement_table),
        "layer_rows": layer_rows,
        "timings": timings,
    }
    write_manifest(out_dir, "analyze", settings, started, extra, root=paths.root, inputs=inputs)
    LOGGER.info("Análises gravadas em %s (%s)", out_dir, timings)
