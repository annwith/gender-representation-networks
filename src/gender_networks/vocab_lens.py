"""Optional stage ``lens``: nearest vocabulary rows of each occurrence ("vizinhos no vocabulário").

Every occurrence vector (lexical, blocks 1, 18 and 36, and block 36 after the final RMSNorm) is
compared by cosine with every row of the input embedding matrix. Qwen3-4B-Base ties the input
and output embeddings, so the ranking says which tokens a representation resembles, and the
curve answers the question of the plan: at which layer does an occurrence stop resembling its
own token and start resembling the next one?

Interpretation caveat: only ``L36n`` lives exactly in the space that the output matrix reads.
At intermediate layers the comparison mixes spaces that were never trained to align, so ranks
there are descriptive, not a readout of the model.

Ties get no special treatment: ``own_rank`` counts the rows with a strictly larger cosine, so
tied rows share the best rank (only positions in the ranking are used), and ``own_in_topk`` means
``own_rank < k`` (a row tied at the k-th place counts even if ``topk`` listed another tied row).
Similarities are float32 blocks of ``batch_size x V``; an ``N x V`` matrix is never built.

Besides the five representations of the plan, the stage also writes ``layer_curve.csv`` (own and
next ranks aggregated at every block output of ``all_layers.npy``) when that file exists, because
the plan asks for the curve "camada a camada".
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from gender_networks.artifacts import RunPaths, ensure_dir, write_json, write_manifest
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)

LEXICAL = "lex"
NORMALIZED = "L36n"
PRED_NEXT_FILE = "pred_next.npy"
OCCURRENCE_FIELDS = ["occurrence_id", "token_id", "next_token_id", "token_category", "stratum"]
RANK_COLUMNS = [
    "occurrence_id",
    "rep",
    "own_rank",
    "next_rank",  # -1 when the occurrence has no next token
    "own_in_topk",
    "next_in_topk",
    "own_cos",
    "next_cos",  # empty (NaN) when the occurrence has no next token
]
JACCARD_COLUMNS = ["occurrence_id", "source", "target", "jaccard"]
CURVE_KEYS = ["rep", "layer", "group_by", "group"]
GROUPINGS = ("token_category", "stratum")
NORM_EPS = 1.0e-12
# Largest excess of another row's cosine over the own row that float32 rounding can explain.
LEX_GAP_TOL = 1.0e-5
AGREEMENT_WARN = 0.99
MAX_EXAMPLES = 20
JACCARD_CHUNK = 4096
# vocabulary rows per float32 chunk of the raw-logit check (about 335 MB at d = 2560)
LOGIT_CHUNK_ROWS = 32768


# ---------------------------------------------------------------------------------------------
# Pure computations
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabIndex:
    """Unit-norm embedding rows (float32), their norms and the stored rows, on the device.

    ``weight`` keeps the matrix in its stored dtype (bf16 in a real run, half the size of the
    float32 copy) only for the raw-logit check, which must use ``weight`` itself, not
    ``unit * norms`` (that product rounds differently and could flip near ties).
    """

    unit: torch.Tensor  # [V, d] float32
    norms: torch.Tensor  # [V] float32
    weight: torch.Tensor  # [V, d] stored dtype, never modified

    @property
    def size(self) -> int:
        return int(self.unit.shape[0])

    @property
    def dim(self) -> int:
        return int(self.unit.shape[1])

    @classmethod
    def from_weight(cls, weight: torch.Tensor, device: str | torch.device = "cpu") -> VocabIndex:
        """Normalize a copy of ``weight`` in place (the caller's tensor is never modified).

        The matrix is moved in its stored dtype first, so the float32 copy (about 1.55 GB for
        the full vocabulary) only exists on the target device.
        """

        if weight.ndim != 2:
            raise ValueError(f"Embedding weight must be 2-D, got shape {tuple(weight.shape)}")
        stored = weight.to(device=device)
        rows = stored.to(torch.float32, copy=True)
        norms = torch.linalg.vector_norm(rows, dim=1)
        rows.div_(norms.clamp_min(NORM_EPS).unsqueeze(1))
        return cls(unit=rows, norms=norms, weight=stored)


@dataclass(frozen=True)
class LensResult:
    """Vocabulary ranking of one representation for every occurrence."""

    topk: np.ndarray  # int32 [N, k], nearest rows first
    topk_cos: np.ndarray  # float32 [N, k]
    own_rank: np.ndarray  # int64 [N], rows with a strictly larger cosine than the own token
    next_rank: np.ndarray  # int64 [N], same for the next token; -1 without a next token
    own_cos: np.ndarray  # float32 [N]
    next_cos: np.ndarray  # float32 [N], NaN without a next token
    logit_argmax: np.ndarray | None = None  # int64 [N], argmax of the raw dot product

    @property
    def k(self) -> int:
        return int(self.topk.shape[1])

    def own_in_topk(self) -> np.ndarray:
        return self.own_rank < self.k

    def next_in_topk(self) -> np.ndarray:
        return (self.next_rank >= 0) & (self.next_rank < self.k)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """The GPU when available (the similarity blocks are the bulk of the work), else CPU."""

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def raw_logit_argmax(
    weight: torch.Tensor, rows: torch.Tensor, chunk_rows: int = LOGIT_CHUNK_ROWS
) -> torch.Tensor:
    """Argmax over the vocabulary of ``rows @ weight.T`` in float32 (lowest id on ties).

    This is how the extraction computes ``pred_next`` (tied embeddings: the model's own
    next-token choice). The vocabulary is processed in chunks so that only a slice of the weight
    exists in float32 at a time; the result lives on the device of ``weight``.
    """

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    x = rows.to(device=weight.device, dtype=torch.float32)
    best_value = torch.full((x.shape[0],), -torch.inf, device=weight.device)
    best_index = torch.zeros(x.shape[0], dtype=torch.long, device=weight.device)
    for start in range(0, weight.shape[0], chunk_rows):
        logits = x @ weight[start : start + chunk_rows].to(torch.float32).T
        index = logits.argmax(dim=1)
        value = logits.gather(1, index.unsqueeze(1)).squeeze(1)
        better = value > best_value  # strict: an earlier chunk keeps a tie
        best_value = torch.where(better, value, best_value)
        best_index = torch.where(better, index + start, best_index)
    return best_index


def _check_ids(own: np.ndarray, nxt: np.ndarray, size: int) -> None:
    if own.ndim != 1 or nxt.shape != own.shape:
        raise ValueError("own_ids and next_ids must be 1-D arrays of the same length")
    if own.size and (own.min() < 0 or own.max() >= size):
        raise ValueError(f"own token ids must lie in [0, {size})")
    if nxt.size and (nxt.min() < -1 or nxt.max() >= size):
        raise ValueError(f"next token ids must lie in [0, {size}) or be -1")


@torch.inference_mode()
def lens_ranks(
    index: VocabIndex,
    queries: torch.Tensor | None,
    own_ids: Sequence[int] | np.ndarray,
    next_ids: Sequence[int] | np.ndarray,
    k: int,
    batch_size: int,
    logit_argmax: bool = False,
) -> LensResult:
    """Rank the whole vocabulary by cosine for every query, one batch of rows at a time.

    ``queries`` is ``[N, d]`` in any dtype and on any device, or ``None`` for the lexical
    representation, whose vector is the embedding row of the own token (so it is taken from the
    index instead of being materialized). Only a ``[batch_size, V]`` block exists at a time.

    With ``logit_argmax`` the argmax of the raw float32 dot product ``q @ weight.T`` (unnormalized
    query, stored weight) is returned too, see :func:`raw_logit_argmax`.
    """

    # private writable copies: torch.from_numpy refuses read-only (copy-on-write pandas) views
    own = np.array(own_ids, dtype=np.int64)
    nxt = np.array(next_ids, dtype=np.int64)
    _check_ids(own, nxt, index.size)
    n = own.shape[0]
    if queries is not None and (queries.ndim != 2 or tuple(queries.shape) != (n, index.dim)):
        raise ValueError(f"queries must have shape ({n}, {index.dim}), got {tuple(queries.shape)}")
    if not 1 <= k <= index.size:
        raise ValueError(f"k must lie in [1, {index.size}], got {k}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = index.unit.device
    topk = np.empty((n, k), dtype=np.int32)
    topk_cos = np.empty((n, k), dtype=np.float32)
    own_rank = np.empty(n, dtype=np.int64)
    next_rank = np.empty(n, dtype=np.int64)
    own_cos = np.empty(n, dtype=np.float32)
    next_cos = np.empty(n, dtype=np.float32)
    argmax = np.empty(n, dtype=np.int64) if logit_argmax else None

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        own_t = torch.from_numpy(own[start:stop]).to(device)
        nxt_t = torch.from_numpy(nxt[start:stop]).to(device)
        if queries is None:
            raw = index.weight.index_select(0, own_t) if argmax is not None else None
            q = index.unit.index_select(0, own_t)
        else:
            raw = queries[start:stop].to(device=device, dtype=torch.float32)
            q = F.normalize(raw, dim=1, eps=NORM_EPS)
        sims = q @ index.unit.T  # [b, V]
        best = sims.topk(k, dim=1, largest=True, sorted=True)
        own_sim = sims.gather(1, own_t.unsqueeze(1))
        has_next = nxt_t >= 0
        next_sim = sims.gather(1, nxt_t.clamp_min(0).unsqueeze(1))
        rank_own = (sims > own_sim).sum(dim=1)
        rank_next = torch.where(has_next, (sims > next_sim).sum(dim=1), -1)

        topk[start:stop] = best.indices.to(torch.int32).cpu().numpy()
        topk_cos[start:stop] = best.values.cpu().numpy()
        own_rank[start:stop] = rank_own.cpu().numpy()
        next_rank[start:stop] = rank_next.cpu().numpy()
        own_cos[start:stop] = own_sim.squeeze(1).cpu().numpy()
        next_cos[start:stop] = torch.where(has_next, next_sim.squeeze(1), torch.nan).cpu().numpy()
        if argmax is not None:
            del sims, best  # free the [b, V] block before the logit chunks
            argmax[start:stop] = raw_logit_argmax(index.weight, raw).cpu().numpy()

    return LensResult(
        topk=topk,
        topk_cos=topk_cos,
        own_rank=own_rank,
        next_rank=next_rank,
        own_cos=own_cos,
        next_cos=next_cos,
        logit_argmax=argmax,
    )


def topk_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard of two top-k id lists (each row holds distinct ids, as topk returns)."""

    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("top-k arrays must be 2-D with the same number of rows")
    out = np.empty(a.shape[0], dtype=np.float64)
    for start in range(0, a.shape[0], JACCARD_CHUNK):
        rows = slice(start, start + JACCARD_CHUNK)
        inter = (a[rows, :, None] == b[rows, None, :]).sum(axis=(1, 2))
        out[rows] = inter / (a.shape[1] + b.shape[1] - inter)
    return out


def representations(settings: Settings) -> list[str]:
    """Representations of the stage in depth order: lexical, the chosen blocks, then ``L36n``."""

    layers = sorted(settings.model.layers.items(), key=lambda item: item[1])
    return [LEXICAL, *(name for name, _ in layers), NORMALIZED]


def transitions(reps: Sequence[str]) -> list[tuple[str, str]]:
    """Consecutive pairs along the depth chain, plus the final RMSNorm step (``L36 -> L36n``)."""

    chain = [rep for rep in reps if rep != NORMALIZED]
    pairs = list(zip(chain, chain[1:], strict=False))
    raw_final = NORMALIZED[:-1]
    if NORMALIZED in reps and raw_final in chain:
        pairs.append((raw_final, NORMALIZED))
    return pairs


def lexical_sanity(
    result: LensResult, own_ids: Sequence[int] | np.ndarray, index: VocabIndex
) -> dict[str, Any]:
    """Explain every lexical ``own_rank > 0``: only a row as close as the own row may beat it.

    At ``lex`` the query is the own embedding row, so its cosine is 1 up to rounding. A strictly
    larger cosine is legitimate only for an identical (duplicated) row or a float32 rounding
    excess below :data:`LEX_GAP_TOL`; anything else is counted as ``unexplained`` (a bug).
    """

    own = np.array(own_ids, dtype=np.int64)
    bad = np.flatnonzero(result.own_rank > 0)
    top1 = result.topk[:, 0].astype(np.int64)
    info: dict[str, Any] = {
        "n": int(own.size),
        "own_rank_nonzero": int(bad.size),
        "types_nonzero": int(np.unique(own[bad]).size),
        # tied rows (for example duplicated embeddings) that topk placed before the own row
        "top1_not_own": int((top1 != own).sum()),
        "max_gap": 0.0,
        "identical_rows": 0,
        "unexplained": 0,
        "example_token_ids": [],
    }
    if bad.size == 0:
        return info
    gap = result.topk_cos[bad, 0].astype(np.float64) - result.own_cos[bad].astype(np.float64)
    device = index.unit.device
    own_rows = index.unit.index_select(0, torch.from_numpy(own[bad]).to(device))
    top_rows = index.unit.index_select(0, torch.from_numpy(top1[bad]).to(device))
    identical = (own_rows == top_rows).all(dim=1).cpu().numpy()
    info.update(
        max_gap=float(gap.max()),
        identical_rows=int(identical.sum()),
        unexplained=int(((gap > LEX_GAP_TOL) & ~identical).sum()),
        example_token_ids=[int(t) for t in np.unique(own[bad])[:MAX_EXAMPLES]],
    )
    return info


def pred_next_agreement(
    argmax: np.ndarray, pred_next: np.ndarray, cosine_top1: np.ndarray | None = None
) -> dict[str, Any]:
    """How often our raw-logit argmax at ``L36n`` equals the extraction's ``pred_next``.

    Both come from the same bf16 inputs, but the extraction multiplies matrices of other shapes,
    so float32 rounding can flip near ties; the rate should still be close to 1. With
    ``cosine_top1`` (the nearest row by cosine) the rate at which the cosine lens picks the
    model's prediction is reported too: it differs because the row norms are ignored.
    """

    argmax = np.asarray(argmax, dtype=np.int64)
    pred = np.asarray(pred_next, dtype=np.int64)
    if argmax.shape != pred.shape:
        raise ValueError(f"pred_next has shape {pred.shape}, expected {argmax.shape}")
    agree = argmax == pred
    info: dict[str, Any] = {
        "n": int(agree.size),
        "agree": int(agree.sum()),
        "rate": float(agree.mean()) if agree.size else None,
        "disagreeing_occurrences": [int(i) for i in np.flatnonzero(~agree)[:MAX_EXAMPLES]],
    }
    if cosine_top1 is not None:
        top1 = np.asarray(cosine_top1, dtype=np.int64)
        if top1.shape != pred.shape:
            raise ValueError(f"cosine_top1 has shape {top1.shape}, expected {pred.shape}")
        info["cosine_top1_rate"] = float((top1 == pred).mean()) if top1.size else None
    return info


def _num(value: Any) -> float | None:
    """JSON-safe rounded float (NaN and missing values become None)."""

    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 6)


def rank_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Rank summary of a group of occurrences; next-token figures skip rows without one."""

    valid = frame[frame["next_rank"] >= 0]
    own_rank = frame["own_rank"]
    next_rank = valid["next_rank"]
    return {
        "n": int(len(frame)),
        "own_rank_median": _num(own_rank.median()),
        "own_rank_mean": _num(own_rank.mean()),
        "own_rank0": _num((own_rank == 0).mean()),
        "own_in_topk": _num(frame["own_in_topk"].mean()),
        "own_cos_mean": _num(frame["own_cos"].mean()),
        "next_n": int(len(valid)),
        "next_rank_median": _num(next_rank.median()),
        "next_rank_mean": _num(next_rank.mean()),
        "next_rank0": _num((next_rank == 0).mean()),
        "next_in_topk": _num(valid["next_in_topk"].mean()),
        "next_cos_mean": _num(valid["next_cos"].mean()),
        # the question of the curve: does the occurrence already look more like the next token?
        "next_closer_than_own": _num((next_rank < valid["own_rank"]).mean()),
    }


def _grouped(frame: pd.DataFrame, stats, groupings: Sequence[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"all": stats(frame)}
    for column in groupings:
        entry[f"by_{column}"] = {
            str(key): stats(part) for key, part in frame.groupby(column, sort=True)
        }
    return entry


def summarize_ranks(
    frame: pd.DataFrame, groupings: Sequence[str] = GROUPINGS
) -> dict[str, dict[str, Any]]:
    """Per representation: rank statistics overall and broken down by each grouping column."""

    return {
        str(rep): _grouped(part, rank_stats, groupings)
        for rep, part in frame.groupby("rep", sort=False)
    }


def jaccard_stats(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame["jaccard"]
    return {
        "n": int(len(values)),
        "mean": _num(values.mean()),
        "median": _num(values.median()),
        "zero": _num((values == 0).mean()),
    }


def summarize_jaccard(
    frame: pd.DataFrame, groupings: Sequence[str] = GROUPINGS
) -> dict[str, dict[str, Any]]:
    """Per transition ``source->target``: Jaccard statistics overall and by grouping column."""

    return {
        f"{source}->{target}": _grouped(part, jaccard_stats, groupings)
        for (source, target), part in frame.groupby(["source", "target"], sort=False)
    }


def rank_frame(result: LensResult, rep: str) -> pd.DataFrame:
    """Long-format rows of ``ranks.csv`` for one representation (occurrence_id = row index)."""

    return pd.DataFrame(
        {
            "occurrence_id": np.arange(result.own_rank.shape[0], dtype=np.int64),
            "rep": rep,
            "own_rank": result.own_rank,
            "next_rank": result.next_rank,
            "own_in_topk": result.own_in_topk(),
            "next_in_topk": result.next_in_topk(),
            "own_cos": result.own_cos,
            "next_cos": result.next_cos,
        },
        columns=RANK_COLUMNS,
    )


def jaccard_frame(results: dict[str, LensResult], pairs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Long-format rows of ``topk_jaccard.csv``: one row per occurrence and transition."""

    parts = []
    for source, target in pairs:
        values = topk_jaccard(results[source].topk, results[target].topk)
        parts.append(
            pd.DataFrame(
                {
                    "occurrence_id": np.arange(values.shape[0], dtype=np.int64),
                    "source": source,
                    "target": target,
                    "jaccard": values,
                },
                columns=JACCARD_COLUMNS,
            )
        )
    if not parts:
        return pd.DataFrame(columns=JACCARD_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def with_groups(frame: pd.DataFrame, occurrences: pd.DataFrame) -> pd.DataFrame:
    """Attach the grouping columns of each row's occurrence (occurrence_id = row index)."""

    rows = frame["occurrence_id"].to_numpy()
    out = frame.copy()
    for column in GROUPINGS:
        out[column] = occurrences[column].to_numpy()[rows]
    return out


def curve_rows(rep: str, layer: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows of ``layer_curve.csv`` for one layer: overall and per grouping value."""

    rows = [{"rep": rep, "layer": layer, "group_by": "all", "group": "all", **rank_stats(frame)}]
    for column in GROUPINGS:
        for key, part in frame.groupby(column, sort=True):
            rows.append(
                {"rep": rep, "layer": layer, "group_by": column, "group": str(key)}
                | rank_stats(part)
            )
    return rows


def crossover_layer(curve: pd.DataFrame) -> int | None:
    """First layer where most occurrences are closer to the next token than to their own."""

    overall = curve[curve["group_by"] == "all"].sort_values("layer")
    above = overall[overall["next_closer_than_own"].astype(float) > 0.5]
    return int(above["layer"].iloc[0]) if len(above) else None


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------


def load_occurrences(path: Path) -> pd.DataFrame:
    """The columns of ``occurrences.csv`` this stage needs, checked for row alignment."""

    frame = pd.read_csv(
        path,
        usecols=OCCURRENCE_FIELDS,
        dtype={"token_category": str, "stratum": str},
        keep_default_na=False,
    )
    if not np.array_equal(frame["occurrence_id"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"occurrence_id must equal the row index in {path}")
    return frame


def load_representation(path: Path, n: int, dim: int) -> torch.Tensor:
    """Tensor ``x`` of a representation file (kept in bf16 until a batch is moved)."""

    tensor = load_file(str(path))["x"]
    if tuple(tensor.shape) != (n, dim):
        raise ValueError(f"{path.name} has shape {tuple(tensor.shape)}, expected ({n}, {dim})")
    return tensor


def layer_from_memmap(layers: np.ndarray, layer: int) -> torch.Tensor:
    """Block output ``layer`` (1-based) of ``all_layers.npy`` as a bf16 tensor."""

    raw = np.array(layers[layer - 1]).view(np.int16)  # writable copy of one layer
    return torch.from_numpy(raw).view(torch.bfloat16)


def _expected_outputs(out_dir: Path, reps: Sequence[str], with_curve: bool) -> list[Path]:
    files = ["ranks.csv", "topk_jaccard.csv", "summary.json", "_manifest.json"]
    files += [f"topk_{rep}.npy" for rep in reps]
    if with_curve:
        files.append("layer_curve.csv")
    return [out_dir / name for name in files]


def _layer_curve(
    paths: RunPaths,
    settings: Settings,
    occurrences: pd.DataFrame,
    index: VocabIndex,
    results: dict[str, LensResult],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Own/next ranks at every block output (raw residual stream), aggregated per layer.

    Only aggregates are kept (no per-occurrence files for 36 layers). Layer 0 is the lexical
    ranking: ``hidden_states[0]`` equals the input embedding (checked by ``extract --verify``).
    Layers whose representation file is bitwise equal to the memmap reuse its ranking.
    """

    n = len(occurrences)
    own = occurrences["token_id"].to_numpy(np.int64)
    nxt = occurrences["next_token_id"].to_numpy(np.int64)
    layers = np.load(paths.all_layers, mmap_mode="r")
    if layers.ndim != 3 or layers.shape[1:] != (n, index.dim):
        raise ValueError(f"all_layers.npy has shape {layers.shape}, expected (L, {n}, {index.dim})")
    matches: dict[str, bool] = {}
    for rep, layer in settings.model.layers.items():
        if layer <= layers.shape[0] and paths.rep(rep).exists():
            stored = load_representation(paths.rep(rep), n, index.dim).view(torch.int16).numpy()
            matches[rep] = bool(np.array_equal(stored, layers[layer - 1].view(np.int16)))
    if not all(matches.values()):
        LOGGER.warning("all_layers.npy difere dos arquivos de representação: %s", matches)
    known = {
        settings.model.layers[rep]: results[rep]
        for rep, same in matches.items()
        if same and rep in results
    }

    lexical = results[LEXICAL]
    rows = curve_rows(LEXICAL, 0, with_groups(rank_frame(lexical, LEXICAL), occurrences))
    for layer in range(1, layers.shape[0] + 1):
        result = known.get(layer)
        if result is None:
            result = lens_ranks(
                index,
                layer_from_memmap(layers, layer),
                own,
                nxt,
                settings.lens.k,
                settings.lens.batch_size,
            )
        rep = f"L{layer:02d}"
        rows += curve_rows(rep, layer, with_groups(rank_frame(result, rep), occurrences))
        if layer % 6 == 0:
            LOGGER.info("Curva camada a camada: %d/%d camadas", layer, layers.shape[0])
    curve = pd.DataFrame(rows)
    info = {
        "layers": int(layers.shape[0]),
        "all_layers_match_reps": matches,
        "reused_layers": sorted(known),
        "crossover_layer": crossover_layer(curve),
    }
    return curve, info


def run(
    settings: Settings,
    paths: RunPaths,
    force: bool = False,
    device: str | None = None,
    layer_curve: bool = True,
    **_: object,
) -> None:
    """Rank the vocabulary for every occurrence and representation and write ``lens/``.

    ``device`` overrides the automatic choice (GPU when available); ``layer_curve`` adds the
    per-layer aggregate curve from ``all_layers.npy`` when that file exists.
    """

    out_dir = paths.lens_dir
    reps = representations(settings)
    pairs = transitions(reps)
    with_curve = layer_curve and paths.all_layers.exists()
    if not force and all(p.exists() for p in _expected_outputs(out_dir, reps, with_curve)):
        LOGGER.info("Vizinhos no vocabulário já existem em %s; use --force para refazer", out_dir)
        return

    required = [paths.occurrences, paths.embeddings]
    required += [paths.rep(rep) for rep in reps if rep != LEXICAL]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltam artefatos das etapas sample/extract: " + ", ".join(missing))

    started = time.time()
    occurrences = load_occurrences(paths.occurrences)
    n = len(occurrences)
    own = occurrences["token_id"].to_numpy(np.int64)
    nxt = occurrences["next_token_id"].to_numpy(np.int64)
    compute_device = resolve_device(device)
    weight = load_file(str(paths.embeddings))["weight"]
    index = VocabIndex.from_weight(weight, compute_device)
    del weight
    k, batch_size = settings.lens.k, settings.lens.batch_size
    LOGGER.info(
        "Vizinhos no vocabulário: %d ocorrências, V = %d, d = %d, k = %d, dispositivo %s",
        n,
        index.size,
        index.dim,
        k,
        compute_device,
    )

    ensure_dir(out_dir)
    results: dict[str, LensResult] = {}
    timings: dict[str, float] = {}
    for rep in reps:
        rep_started = time.time()
        queries = None if rep == LEXICAL else load_representation(paths.rep(rep), n, index.dim)
        result = lens_ranks(
            index, queries, own, nxt, k, batch_size, logit_argmax=(rep == NORMALIZED)
        )
        del queries
        np.save(out_dir / f"topk_{rep}.npy", result.topk)
        results[rep] = result
        timings[rep] = round(time.time() - rep_started, 3)
        valid = result.next_rank >= 0
        LOGGER.info(
            "%s: posto mediano do próprio token %.0f, do próximo %.0f (%.1f s)",
            rep,
            float(np.median(result.own_rank)) if n else float("nan"),
            float(np.median(result.next_rank[valid])) if valid.any() else float("nan"),
            timings[rep],
        )

    lex_info = lexical_sanity(results[LEXICAL], own, index)
    if lex_info["unexplained"]:
        LOGGER.warning("Postos lexicais > 0 sem empate que os explique: %s", lex_info)
    elif lex_info["own_rank_nonzero"]:
        LOGGER.info("Postos lexicais > 0 explicados por empates: %s", lex_info)

    agreement: dict[str, Any] | None = None
    pred_path = paths.reps_dir / PRED_NEXT_FILE
    if NORMALIZED in results and pred_path.exists():
        final = results[NORMALIZED]
        agreement = pred_next_agreement(final.logit_argmax, np.load(pred_path), final.topk[:, 0])
        log = LOGGER.warning if (agreement["rate"] or 0.0) < AGREEMENT_WARN else LOGGER.info
        log("Concordância com pred_next: %s de %s", agreement["agree"], agreement["n"])
    else:
        LOGGER.warning("%s ausente; concordância com pred_next não verificada", pred_path)

    ranks = pd.concat([rank_frame(results[rep], rep) for rep in reps], ignore_index=True)
    ranks.to_csv(out_dir / "ranks.csv", index=False, float_format="%.7g")
    jaccard = jaccard_frame(results, pairs)
    jaccard.to_csv(out_dir / "topk_jaccard.csv", index=False, float_format="%.6g")

    curve_info: dict[str, Any] | None = None
    if with_curve:
        curve, curve_info = _layer_curve(paths, settings, occurrences, index, results)
        curve.to_csv(out_dir / "layer_curve.csv", index=False)
        curve_info["file"] = "layer_curve.csv"

    rank_summary = summarize_ranks(with_groups(ranks, occurrences))
    summary = {
        "n_occurrences": n,
        "vocab_size": index.size,
        "dim": index.dim,
        "k": k,
        "representations": reps,
        "transitions": [f"{a}->{b}" for a, b in pairs],
        "ranks": rank_summary,
        "topk_jaccard": summarize_jaccard(with_groups(jaccard, occurrences)),
        "pred_next_agreement": agreement,
        "lex_sanity": lex_info,
        "layer_curve": curve_info,
    }
    write_json(out_dir / "summary.json", summary)

    extra = {
        "n_occurrences": n,
        "vocab_size": index.size,
        "dim": index.dim,
        "k": k,
        "batch_size": batch_size,
        "device": str(compute_device),
        "representations": reps,
        "transitions": summary["transitions"],
        "median_own_rank": {rep: rank_summary[rep]["all"]["own_rank_median"] for rep in reps},
        "median_next_rank": {rep: rank_summary[rep]["all"]["next_rank_median"] for rep in reps},
        "pred_next_agreement_rate": agreement["rate"] if agreement else None,
        "lex_own_rank_nonzero": lex_info["own_rank_nonzero"],
        "lex_unexplained": lex_info["unexplained"],
        "layer_curve": curve_info,
        "rep_seconds": timings,
    }
    write_manifest(out_dir, "lens", settings, started, extra, root=paths.root)
    LOGGER.info("Vizinhos no vocabulário gravados em %s", out_dir)
