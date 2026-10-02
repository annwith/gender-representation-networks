"""Stage ``extract``: occurrence representations from Qwen3-4B-Base (plan decisions 1-3).

Every vertex is a token occurrence. For each sequence of the sample this stage runs one forward
pass and keeps, in ``occurrence_id`` order:

- the **raw** residual stream after the configured decoder blocks (``L01``, ``L18``, ``L36``),
  read by forward hooks on ``model.layers``;
- the output of the final RMSNorm (``L36n``, robustness variant), read by a hook on
  ``model.norm``;
- optionally every block output, written row by row into a uint16 memmap (the layer curve of P2);
- the model's next-token prediction at the vertex (tied embeddings: ``L36n @ E.T``).

It also stores the input embedding matrix (lexical vectors are rows of it and are never
materialized per occurrence), a numerical diagnostic per representation and the result of
:func:`verify`.

Why hooks instead of ``output_hidden_states``: in transformers 5.x ``hidden_states[-1]`` is
overwritten with the post-norm output (``capture_outputs(tie_last_hidden_states=True)``), so the
raw output of the last block only exists inside the forward pass. :func:`verify` checks this and
the other assumptions on the real model before the full extraction.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from safetensors.torch import save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from gender_networks.artifacts import (
    POS_BUCKETS,
    RunPaths,
    ensure_dir,
    library_versions,
    position_bucket,
    read_jsonl,
    stage_is_fresh,
    write_json,
    write_manifest,
)
from gender_networks.settings import ModelSettings, Settings

LOGGER = logging.getLogger(__name__)

NORM_REP = "L36n"
LEX_REP = "lex"
PRED_NEXT_FILE = "pred_next.npy"  # kept for backwards compatibility; use RunPaths.pred_next
DIAGNOSTICS_FILE = "diagnostics.json"
VERIFY_FILE = "verify.json"
OCCURRENCE_INPUT_COLUMNS = [
    "occurrence_id",
    "sequence_id",
    "pos_in_sequence",
    "prefix_group",
    "token_id",
    "pos_bucket",
    "next_token_id",
]
# Checks 1-4 of verify(): a normal run aborts when any of them fails (determinism only warns).
GATING_CHECKS = (
    "hidden_states_count",
    "embedding_input",
    "hooks_match_hidden_states",
    "final_norm",
)
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
DIAGNOSTIC_PAIRS = 100_000
LAYER_CURVE_PAIRS = 20_000
TOP_DIMS = 5
OUTLIER_FACTOR = 10.0
VOCAB_CHUNK_ROWS = 8192
PAIR_CHUNK = 8192
# (rtol, atol) of torch.testing.assert_close: "tight" relative to each dtype's precision.
_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}


# ---------------------------------------------------------------------------------------------
# Artifact locations that RunPaths does not name
# ---------------------------------------------------------------------------------------------


def pred_next_path(paths: RunPaths) -> Path:
    return paths.pred_next


def diagnostics_path(paths: RunPaths) -> Path:
    return paths.diagnostics


def verify_path(paths: RunPaths) -> Path:
    return paths.verify_report


def partial_all_layers_path(paths: RunPaths) -> Path:
    """Where all_layers.npy is filled before it is renamed into place."""

    return paths.all_layers.with_name(paths.all_layers.name + ".partial")


def output_paths(settings: Settings, paths: RunPaths) -> list[Path]:
    """Every file a complete extraction leaves behind (the manifest is written last)."""

    files = [paths.rep(name) for name in settings.model.layers]
    files += [
        paths.rep(NORM_REP),
        paths.embeddings,
        pred_next_path(paths),
        diagnostics_path(paths),
        verify_path(paths),
    ]
    if settings.model.capture_all_layers:
        files.append(paths.all_layers)
    files.append(paths.reps_dir / "_manifest.json")
    return files


# ---------------------------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype '{name}'")
    return dtype


def load_model(model_settings: ModelSettings) -> PreTrainedModel:
    """Load the bare decoder (``AutoModel``: no LM head) in eval mode.

    With ``device_map='auto'`` and ``max_memory`` the blocks that do not fit on the GPU are
    offloaded to the CPU by accelerate and streamed to the GPU during the forward pass.
    """

    kwargs: dict[str, Any] = {
        "revision": model_settings.revision,
        "dtype": _torch_dtype(model_settings.dtype),
    }
    if model_settings.device_map is not None:
        kwargs["device_map"] = model_settings.device_map
    if model_settings.max_memory:
        kwargs["max_memory"] = dict(model_settings.max_memory)
    LOGGER.info(
        "Carregando %s (revisão %s, %s)",
        model_settings.name_or_path,
        model_settings.revision,
        model_settings.dtype,
    )
    model = AutoModel.from_pretrained(model_settings.name_or_path, **kwargs)
    model.eval()
    return model


def load_tokenizer(model_settings: ModelSettings) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        model_settings.name_or_path, revision=model_settings.revision
    )


def decoder_parts(model: nn.Module) -> tuple[nn.ModuleList, nn.Module, nn.Module]:
    """Blocks, final norm and input embedding of a Qwen3-like decoder (bare or ``*ForCausalLM``)."""

    base = model
    if not hasattr(base, "layers") and hasattr(base, "model"):
        base = base.model
    layers = getattr(base, "layers", None)
    norm = getattr(base, "norm", None)
    if layers is None or norm is None:
        raise TypeError(f"{type(model).__name__} has no 'layers'/'norm' decoder attributes")
    return layers, norm, model.get_input_embeddings()


def _accelerate_hooks(module: nn.Module) -> list[Any]:
    """accelerate hooks attached to a module (a ``SequentialHook`` holds several)."""

    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return []
    return [hook, *(getattr(hook, "hooks", None) or [])]


def module_device(module: nn.Module) -> torch.device:
    """Device where a module computes: accelerate's execution device when it is offloaded."""

    for hook in _accelerate_hooks(module):
        device = getattr(hook, "execution_device", None)
        if device is not None and not isinstance(device, Mapping):
            return torch.device(device)
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def embedding_weight(model: nn.Module) -> torch.Tensor:
    """The real input embedding matrix, also when accelerate offloaded it (meta parameter)."""

    module = model.get_input_embeddings()
    weight = module.weight
    if weight.device.type != "meta":
        return weight.detach()
    for hook in _accelerate_hooks(module):
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is not None:
            return weights_map["weight"]
    raise RuntimeError("The embedding weight is on the meta device and has no weights map")


# ---------------------------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------------------------


class HookRecorder:
    """Forward hooks that keep only the vertex rows of block outputs and of the final norm.

    Each hook gathers ``out[0, positions]`` and moves it to the CPU at once, so a forward pass
    never holds more than what the model itself needs plus ``m x d`` rows per captured block.
    Use as a context manager (hooks are removed on exit) and call :meth:`run` per sequence.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: Iterable[int] | None = None,
        store_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        self.model = model
        self.layers, self.norm_module, self.embedding = decoder_parts(model)
        n_layers = len(self.layers)
        indices = range(n_layers) if block_indices is None else block_indices
        self.block_indices = sorted({int(i) for i in indices})
        if any(i < 0 or i >= n_layers for i in self.block_indices):
            raise ValueError(f"Block indices must be in 0..{n_layers - 1}")
        self.store_dtype = store_dtype
        self.norm: torch.Tensor | None = None  # CPU, store_dtype
        self.norm_native: torch.Tensor | None = None  # compute device, model dtype
        self._blocks: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._active = False
        self._positions = torch.zeros(0, dtype=torch.long)
        self._by_device: dict[torch.device, torch.Tensor] = {}

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def __enter__(self) -> HookRecorder:
        if self._handles:
            raise RuntimeError("Hooks are already attached")
        for index in self.block_indices:
            self._handles.append(self.layers[index].register_forward_hook(self._block_hook(index)))
        self._handles.append(self.norm_module.register_forward_hook(self._norm_hook))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False

    def _rows(self, output: Any) -> torch.Tensor:
        hidden = output[0] if isinstance(output, tuple) else output
        index = self._by_device.get(hidden.device)
        if index is None:
            index = self._positions.to(hidden.device)
            self._by_device[hidden.device] = index
        return hidden[0, index]

    def _store(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.to(device="cpu", dtype=self.store_dtype or rows.dtype)

    def _block_hook(self, index: int):
        def hook(module: nn.Module, args: Any, output: Any) -> None:
            if self._active:
                self._blocks[index] = self._store(self._rows(output))

        return hook

    def _norm_hook(self, module: nn.Module, args: Any, output: Any) -> None:
        if self._active:
            rows = self._rows(output)
            self.norm_native = rows
            self.norm = self._store(rows)

    def run(self, input_ids: Sequence[int], positions: Sequence[int], **forward_kwargs: Any) -> Any:
        """One forward pass (batch 1, attention mask of ones, no cache) recording ``positions``."""

        if not self._handles:
            raise RuntimeError("Use HookRecorder as a context manager before calling run()")
        ids = [int(t) for t in input_ids]
        pos = torch.as_tensor([int(p) for p in positions], dtype=torch.long)
        if pos.numel() and (int(pos.min()) < 0 or int(pos.max()) >= len(ids)):
            raise ValueError(f"Positions must be in 0..{len(ids) - 1}")
        device = module_device(self.embedding)
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        self._positions, self._by_device, self._blocks = pos, {}, {}
        self.norm = self.norm_native = None
        forward_kwargs.setdefault("output_hidden_states", False)
        self._active = True
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    use_cache=False,
                    return_dict=True,
                    **forward_kwargs,
                )
        finally:
            self._active = False
        missing = [i for i in self.block_indices if i not in self._blocks]
        if missing or self.norm is None:
            raise RuntimeError(f"Hooks did not fire for blocks {missing} or for the final norm")
        return outputs

    def block(self, index: int) -> torch.Tensor:
        """Recorded rows ``[m, d]`` of block ``index`` (0-based) from the last run."""

        return self._blocks[index]

    @property
    def blocks(self) -> torch.Tensor:
        """Recorded rows of every captured block, ``[len(block_indices), m, d]``."""

        return torch.stack([self._blocks[i] for i in self.block_indices])


def capture_outputs(
    model: nn.Module,
    input_ids: Sequence[int],
    positions: Sequence[int],
    block_indices: Iterable[int] | None = None,
    store_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of the vertex positions: blocks ``[L, m, d]`` and final norm ``[m, d]`` (CPU)."""

    with HookRecorder(model, block_indices, store_dtype) as recorder:
        recorder.run(input_ids, positions)
        assert recorder.norm is not None
        return recorder.blocks, recorder.norm


def predict_next(
    rows: torch.Tensor,
    weight: torch.Tensor,
    vocab_size: int,
    chunk_rows: int = VOCAB_CHUNK_ROWS,
) -> torch.Tensor:
    """Argmax over the first ``vocab_size`` ids of ``rows @ weight.T`` in float32.

    With tied embeddings this is the model's own next-token choice. The vocabulary is processed
    in chunks so the float32 copy of the weight stays small on the GPU; ties keep the lowest id,
    as ``torch.argmax`` does.
    """

    if vocab_size > weight.shape[0]:
        raise ValueError(f"vocab_size {vocab_size} exceeds the {weight.shape[0]} embedding rows")
    with torch.inference_mode():
        x = rows.to(device=weight.device, dtype=torch.float32)
        best_value = torch.full((x.shape[0],), -math.inf, device=weight.device)
        best_index = torch.zeros(x.shape[0], dtype=torch.long, device=weight.device)
        for start in range(0, vocab_size, chunk_rows):
            stop = min(start + chunk_rows, vocab_size)
            logits = x @ weight[start:stop].to(torch.float32).T
            index = logits.argmax(dim=1)
            value = logits.gather(1, index[:, None])[:, 0]
            better = value > best_value
            best_value = torch.where(better, value, best_value)
            best_index = torch.where(better, index + start, best_index)
    return best_index.cpu()


# ---------------------------------------------------------------------------------------------
# bfloat16 <-> numpy (numpy has no bfloat16: raw bits are stored as uint16)
# ---------------------------------------------------------------------------------------------


def bf16_bits(x: torch.Tensor) -> np.ndarray:
    """Raw bfloat16 bits of a tensor as a uint16 array (rounded to nearest even)."""

    bf16 = x.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return bf16.view(torch.int16).numpy().view(np.uint16)


def bits_to_float32(bits: np.ndarray) -> np.ndarray:
    """Exact float32 values of raw bfloat16 bits."""

    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bits_to_tensor(bits: np.ndarray) -> torch.Tensor:
    """bfloat16 tensor sharing nothing with ``bits`` (a contiguous copy)."""

    array = np.array(bits, dtype=np.uint16, copy=True, order="C")
    return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceTask:
    """One forward pass: a sequence and the vertex rows read from it."""

    sequence_id: int
    input_ids: tuple[int, ...]
    positions: np.ndarray  # int64 positions in the sequence (>= 1)
    occurrence_ids: np.ndarray  # int64 rows of every representation file


def build_tasks(
    sequences: Iterable[Mapping[str, Any]],
    occurrences: pd.DataFrame,
    vocab_size: int | None = None,
    prefix_id: int | None = None,
) -> list[SequenceTask]:
    """Group vertices by sequence, checking them against the token ids of the sequences.

    Mismatches between ``occurrences.csv`` and ``sequences.jsonl`` would silently scramble
    vectors, so they fail here, before the model is even loaded.
    """

    required = {"occurrence_id", "sequence_id", "pos_in_sequence", "token_id"}
    missing = required - set(occurrences.columns)
    if missing:
        raise ValueError(f"occurrences lack columns: {', '.join(sorted(missing))}")
    if len(occurrences) == 0:
        raise ValueError("occurrences is empty")
    occurrence_ids = occurrences["occurrence_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(occurrence_ids, np.arange(len(occurrences))):
        raise ValueError("occurrence_id must equal the row index")

    by_id: dict[int, tuple[int, ...]] = {}
    for record in sequences:
        sequence_id = int(record["sequence_id"])
        if sequence_id in by_id:
            raise ValueError(f"Duplicate sequence_id {sequence_id}")
        ids = tuple(int(t) for t in record["input_ids"])
        if not ids:
            raise ValueError(f"Sequence {sequence_id} is empty")
        if prefix_id is not None and ids[0] != prefix_id:
            raise ValueError(f"Sequence {sequence_id} does not start with the prefix {prefix_id}")
        if min(ids) < 0 or (vocab_size is not None and max(ids) >= vocab_size):
            raise ValueError(f"Sequence {sequence_id} has token ids outside the vocabulary")
        by_id[sequence_id] = ids

    sequence_col = occurrences["sequence_id"].to_numpy(dtype=np.int64)
    position_col = occurrences["pos_in_sequence"].to_numpy(dtype=np.int64)
    token_col = occurrences["token_id"].to_numpy(dtype=np.int64)
    rows_by_sequence = pd.Series(sequence_col).groupby(sequence_col, sort=False).indices
    unknown = set(int(s) for s in rows_by_sequence) - set(by_id)
    if unknown:
        raise ValueError(f"Occurrences refer to unknown sequences, e.g. {sorted(unknown)[:5]}")

    tasks: list[SequenceTask] = []
    for sequence_id, ids in by_id.items():
        rows = rows_by_sequence.get(sequence_id)
        if rows is None:
            continue
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        positions = position_col[rows]
        if int(positions.min()) < 1 or int(positions.max()) >= len(ids):
            raise ValueError(
                f"Sequence {sequence_id}: vertex positions must be in 1..{len(ids) - 1}"
            )
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f"Sequence {sequence_id}: repeated vertex positions")
        if not np.array_equal(np.asarray(ids, dtype=np.int64)[positions], token_col[rows]):
            raise ValueError(f"Sequence {sequence_id}: token_id differs from input_ids")
        tasks.append(SequenceTask(sequence_id, ids, positions, rows))
    return tasks


def prefix_group_members(prefix_group: Sequence[int] | np.ndarray) -> list[np.ndarray]:
    """Sorted rows of each prefix group with two or more members (first row = lowest id)."""

    groups: dict[int, list[int]] = defaultdict(list)
    for row, label in enumerate(np.asarray(prefix_group, dtype=np.int64).tolist()):
        if label >= 0:
            groups[label].append(row)
    return [np.asarray(rows, dtype=np.int64) for _, rows in sorted(groups.items()) if len(rows) > 1]


def check_prefix_groups(tasks: Sequence[SequenceTask], prefix_group: np.ndarray) -> None:
    """``prefix_group`` must match the token prefixes: same prefix if and only if same group."""

    prefix_of: dict[int, tuple[int, ...]] = {}
    for task in tasks:
        for row, pos in zip(task.occurrence_ids.tolist(), task.positions.tolist(), strict=True):
            prefix_of[row] = task.input_ids[: pos + 1]
    labels = np.asarray(prefix_group, dtype=np.int64)
    problems: list[str] = []
    rows_by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row, prefix in prefix_of.items():
        rows_by_prefix[prefix].append(row)
    for rows in rows_by_prefix.values():
        found = {int(labels[r]) for r in rows}
        if len(rows) > 1 and (len(found) != 1 or -1 in found):
            problems.append(f"rows {rows[:4]} share a prefix but have groups {sorted(found)}")
    for members in prefix_group_members(labels):
        if len({prefix_of[int(r)] for r in members}) != 1:
            problems.append(f"group {int(labels[members[0]])} mixes different prefixes")
    if problems:
        raise ValueError(
            f"prefix_group is inconsistent in {len(problems)} cases: " + "; ".join(problems[:3])
        )


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Representations in ``occurrence_id`` order, as raw bfloat16 bits."""

    blocks: np.ndarray  # uint16 [len(block_layers), N, d] (memmap or in memory)
    block_layers: list[int]  # 1-based block of each slice of ``blocks``
    norm: np.ndarray  # uint16 [N, d]
    pred_next: np.ndarray  # int32 [N]
    n_sequences: int
    tokens: int
    forward_seconds: float
    seconds: float


def _row_index(rows: np.ndarray) -> slice | np.ndarray:
    """A slice when rows are consecutive (faster memmap writes), else the index array."""

    if len(rows) and int(rows[-1]) - int(rows[0]) + 1 == len(rows):
        if np.all(np.diff(rows) == 1):
            return slice(int(rows[0]), int(rows[-1]) + 1)
    return rows


def extract_representations(
    model: nn.Module,
    tasks: Sequence[SequenceTask],
    n_vertices: int,
    block_layers: Iterable[int],
    vocab_size: int,
    blocks_out: np.ndarray | None = None,
    log_every: int = 100,
) -> ExtractionResult:
    """Run every sequence once and write its vertex rows as they come.

    ``block_layers`` are 1-based block indices (sorted here); ``blocks_out`` may be a
    preallocated ``[len(block_layers), N, d]`` uint16 array such as the ``all_layers.npy``
    memmap.
    """

    layers, _, embedding = decoder_parts(model)
    captured = sorted({int(layer) for layer in block_layers})
    if not captured or captured[0] < 1 or captured[-1] > len(layers):
        raise ValueError(f"block_layers must be in 1..{len(layers)}")
    hidden = int(embedding.weight.shape[1])
    shape = (len(captured), n_vertices, hidden)
    if blocks_out is None:
        blocks_out = np.zeros(shape, dtype=np.uint16)
    elif blocks_out.shape != shape or blocks_out.dtype != np.uint16:
        raise ValueError(f"blocks_out must be uint16 with shape {shape}")
    norm = np.zeros((n_vertices, hidden), dtype=np.uint16)
    pred_next = np.full(n_vertices, -1, dtype=np.int32)
    written = np.zeros(n_vertices, dtype=bool)
    weight = embedding_weight(model)

    started = time.perf_counter()
    forward_seconds = 0.0
    tokens = 0
    with HookRecorder(model, [layer - 1 for layer in captured]) as recorder:
        for done, task in enumerate(tasks, start=1):
            rows = task.occurrence_ids
            if written[rows].any():
                raise ValueError(f"Sequence {task.sequence_id} rewrites rows already written")
            tick = time.perf_counter()
            recorder.run(task.input_ids, task.positions)
            assert recorder.norm is not None and recorder.norm_native is not None
            prediction = predict_next(recorder.norm_native, weight, vocab_size)
            forward_seconds += time.perf_counter() - tick
            index = _row_index(rows)
            blocks_out[:, index, :] = bf16_bits(recorder.blocks)
            norm[index] = bf16_bits(recorder.norm)
            pred_next[index] = prediction.numpy().astype(np.int32)
            written[rows] = True
            tokens += len(task.input_ids)
            if done % log_every == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "Extração: %d/%d sequências, %d tokens, %.0f tokens/s, faltam ~%.0f s",
                    done,
                    len(tasks),
                    tokens,
                    tokens / max(forward_seconds, 1e-9),
                    elapsed / done * (len(tasks) - done),
                )
    if not written.all():
        raise ValueError(f"{int((~written).sum())} vertices are not in any sequence")
    return ExtractionResult(
        blocks=blocks_out,
        block_layers=captured,
        norm=norm,
        pred_next=pred_next,
        n_sequences=len(tasks),
        tokens=tokens,
        forward_seconds=forward_seconds,
        seconds=time.perf_counter() - started,
    )


def copy_prefix_groups(
    prefix_group: Sequence[int] | np.ndarray,
    blocks: np.ndarray,
    block_layers: Sequence[int],
    norm: np.ndarray,
    pred_next: np.ndarray,
) -> dict[str, Any]:
    """Give vertices with identical causal context identical vectors, reporting the noise removed.

    Members of a prefix group are mathematically equal but come from forward passes of different
    lengths, so kernels may round differently. The lowest occurrence_id of each group is copied
    to the others (every captured block, the final norm and the prediction); the largest
    absolute difference seen before copying is the numerical-noise diagnostic.
    """

    groups = prefix_group_members(prefix_group)
    per_layer = np.zeros(len(block_layers), dtype=np.float64)
    norm_max = 0.0
    pred_changed = 0
    copied = 0
    for members in groups:
        leader = int(members[0])
        leader_blocks = bits_to_float32(blocks[:, leader, :])
        leader_norm = bits_to_float32(norm[leader])
        for row in members[1:].tolist():
            diff = np.abs(bits_to_float32(blocks[:, row, :]) - leader_blocks).max(axis=1)
            per_layer = np.maximum(per_layer, diff)
            norm_max = max(norm_max, float(np.abs(bits_to_float32(norm[row]) - leader_norm).max()))
            blocks[:, row, :] = blocks[:, leader, :]
            norm[row] = norm[leader]
            if pred_next[row] != pred_next[leader]:
                pred_changed += 1
                pred_next[row] = pred_next[leader]
            copied += 1
    return {
        "n_groups": len(groups),
        "n_rows_copied": copied,
        "per_layer_max_abs_diff": {
            str(layer): float(value) for layer, value in zip(block_layers, per_layer, strict=True)
        },
        "norm_max_abs_diff": norm_max,
        "pred_next_changed": pred_changed,
    }


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max())


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    return bool(torch.allclose(a.detach().cpu().float(), b.detach().cpu().float(), rtol, atol))


def verify(
    model: nn.Module,
    sequences: Sequence[Sequence[int] | Mapping[str, Any]],
    n: int,
) -> dict[str, Any]:
    """Check on the real model the assumptions the extraction relies on (plan, stage 3).

    Runs the first ``n`` sequences with ``output_hidden_states=True`` and the hooks, comparing
    every position:

    1. ``len(hidden_states) == n_layers + 1``;
    2. ``hidden_states[0]`` equals ``embed_tokens(input_ids)`` exactly (RoPE adds nothing to
       the residual stream);
    3. the hook on ``layers[k - 1]`` equals ``hidden_states[k]`` for k in
       {1, n_layers // 2, n_layers - 1};
    4. the raw hook on the last block differs from ``hidden_states[n_layers]``, while
       ``norm(hook)`` and the norm hook match it;
    5. a second run gives identical outputs (a tiny nonzero difference is only a warning).

    ``passed`` covers checks 1-4, the ones a normal run aborts on.
    """

    layers, norm_module, embedding = decoder_parts(model)
    n_layers = len(layers)
    chosen = [
        [int(t) for t in (s["input_ids"] if isinstance(s, Mapping) else s)]
        for s in list(sequences)[: max(int(n), 0)]
    ]
    if not chosen:
        raise ValueError("verify needs at least one sequence")
    ks = sorted({k for k in (1, n_layers // 2, n_layers - 1) if 1 <= k < n_layers})
    dtype = embedding.weight.dtype
    rtol, atol = _TOLERANCES.get(dtype, (1.3e-6, 1e-5))

    counts: list[int] = []
    embed_equal, embed_diff = True, 0.0
    hook_equal = dict.fromkeys(ks, True)
    hook_diff = dict.fromkeys(ks, 0.0)
    raw_differs_all, raw_diff = True, 0.0
    norm_close_all, norm_diff = True, 0.0
    norm_hook_equal, norm_hook_diff = True, 0.0
    det_diff, det_close = 0.0, True
    started = time.perf_counter()
    last = n_layers - 1
    with HookRecorder(model, [k - 1 for k in ks] + [last], store_dtype=None) as recorder:
        for ids in chosen:
            positions = range(len(ids))
            outputs = recorder.run(ids, positions, output_hidden_states=True)
            hidden = tuple(outputs.hidden_states or ())
            counts.append(len(hidden))
            first = {i: recorder.block(i).clone() for i in recorder.block_indices}
            assert recorder.norm is not None
            first_norm = recorder.norm.clone()
            if len(hidden) == n_layers + 1:
                with torch.inference_mode():
                    tensor = torch.tensor([ids], dtype=torch.long, device=module_device(embedding))
                    embedded = embedding(tensor)[0].cpu()
                    raw = first[last]
                    normed = norm_module(raw.to(module_device(norm_module))).cpu()
                h0 = hidden[0][0].cpu()
                embed_equal &= torch.equal(h0, embedded)
                embed_diff = max(embed_diff, _max_abs_diff(h0, embedded))
                for k in ks:
                    hk = hidden[k][0].cpu()
                    hook_equal[k] &= torch.equal(first[k - 1], hk)
                    hook_diff[k] = max(hook_diff[k], _max_abs_diff(first[k - 1], hk))
                final = hidden[n_layers][0].cpu()
                raw_differs_all &= not _close(raw, final, rtol, atol)
                raw_diff = max(raw_diff, _max_abs_diff(raw, final))
                norm_close_all &= _close(normed, final, rtol, atol)
                norm_diff = max(norm_diff, _max_abs_diff(normed, final))
                norm_hook_equal &= torch.equal(first_norm, final)
                norm_hook_diff = max(norm_hook_diff, _max_abs_diff(first_norm, final))
            recorder.run(ids, positions)
            for i in recorder.block_indices:
                det_diff = max(det_diff, _max_abs_diff(first[i], recorder.block(i)))
                det_close &= _close(first[i], recorder.block(i), rtol, atol)
            det_diff = max(det_diff, _max_abs_diff(first_norm, recorder.norm))
            det_close &= _close(first_norm, recorder.norm, rtol, atol)

    count_ok = all(c == n_layers + 1 for c in counts)
    if det_diff == 0.0:
        det_status = "identical"
    elif det_close:
        det_status = "tiny"
    else:
        det_status = "different"
    checks = {
        "hidden_states_count": {
            "passed": count_ok,
            "expected": n_layers + 1,
            "found": sorted(set(counts)),
        },
        "embedding_input": {
            "passed": count_ok and embed_equal,
            "max_abs_diff": embed_diff,
        },
        "hooks_match_hidden_states": {
            "passed": count_ok and all(hook_equal.values()),
            "layers": {str(k): {"equal": hook_equal[k], "max_abs_diff": hook_diff[k]} for k in ks},
        },
        "final_norm": {
            "passed": count_ok and raw_differs_all and norm_close_all and norm_hook_equal,
            "raw_block_differs_from_hidden": raw_differs_all,
            "raw_block_vs_hidden_max_abs_diff": raw_diff,
            "norm_of_raw_block_close": norm_close_all,
            "norm_of_raw_block_max_abs_diff": norm_diff,
            "norm_hook_equal": norm_hook_equal,
            "norm_hook_max_abs_diff": norm_hook_diff,
        },
        "determinism": {
            "passed": det_status != "different",
            "status": det_status,
            "max_abs_diff": det_diff,
        },
    }
    warnings: list[str] = []
    if det_status != "identical":
        message = f"Two runs differ by up to {det_diff:.3g} ({det_status})"
        warnings.append(message)
        LOGGER.warning("Verificação: %s", message)
    passed = all(bool(checks[name]["passed"]) for name in GATING_CHECKS)
    for name, check in checks.items():
        LOGGER.info("Verificação %s: %s", name, "ok" if check["passed"] else "FALHOU")
    return {
        "passed": passed,
        "n_sequences": len(chosen),
        "tokens": sum(len(ids) for ids in chosen),
        "n_layers": n_layers,
        "dtype": str(dtype).replace("torch.", ""),
        "tolerance": {"rtol": rtol, "atol": atol},
        "checks": checks,
        "warnings": warnings,
        "seconds": round(time.perf_counter() - started, 3),
        "tie_word_embeddings": getattr(getattr(model, "config", None), "tie_word_embeddings", None),
        "attn_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
    }


# ---------------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------------


def random_pairs(n: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform pairs of distinct rows (with replacement across pairs), reproducible by seed."""

    if n < 2:
        raise ValueError("Need at least two rows for pairs")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n, size=n_pairs)
    second = rng.integers(0, n - 1, size=n_pairs)
    second = second + (second >= first)
    return first, second


def mean_pair_cosine(
    x: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], norms: np.ndarray | None = None
) -> float | None:
    """Mean cosine (float32) of the given row pairs, skipping zero-norm rows (anisotropy)."""

    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1) if norms is None else norms
    first, second = pairs
    total, count = 0.0, 0
    for start in range(0, len(first), PAIR_CHUNK):
        i, j = first[start : start + PAIR_CHUNK], second[start : start + PAIR_CHUNK]
        denominator = norms[i] * norms[j]
        valid = denominator > 0
        cosine = np.einsum("ij,ij->i", x[i], x[j]) / np.where(valid, denominator, 1.0)
        total += float(cosine[valid].sum(dtype=np.float64))
        count += int(valid.sum())
    return total / count if count else None


def representation_summary(
    x: np.ndarray,
    pos_bucket: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray] | None,
    top_dims: int = TOP_DIMS,
    outlier_factor: float = OUTLIER_FACTOR,
) -> dict[str, Any]:
    """Norms, anisotropy, dominant dimensions (massive activations) and norms by position."""

    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    median = float(np.median(norms))
    mean_abs = np.abs(x).mean(axis=0, dtype=np.float64)
    top = np.argsort(-mean_abs, kind="stable")[:top_dims]
    squared = norms.astype(np.float64) ** 2
    valid = squared > 0
    shares = x[:, top].astype(np.float64) ** 2 / np.where(valid, squared, 1.0)[:, None]
    shares = shares[valid]
    labels = np.asarray(pos_bucket).astype(str)
    by_bucket = {
        label: float(norms[labels == label].mean())
        for _, _, label in POS_BUCKETS
        if (labels == label).any()
    }
    median_dim = float(np.median(mean_abs))
    return {
        "n": int(x.shape[0]),
        "n_nonfinite_rows": int((~np.isfinite(x).all(axis=1)).sum()),
        "norm_quantiles": {
            str(round(q * 100)): float(v)
            for q, v in zip(QUANTILES, np.quantile(norms, QUANTILES), strict=True)
        },
        "mean_norm": float(norms.mean()),
        "mean_pair_cosine": mean_pair_cosine(x, pairs, norms) if pairs is not None else None,
        "top_dims": [
            {
                "dim": int(dim),
                "mean_abs": float(mean_abs[dim]),
                "mean_share_sq_norm": float(shares[:, rank].mean()) if len(shares) else None,
            }
            for rank, dim in enumerate(top)
        ],
        "top_dims_mean_share_sq_norm": float(shares.sum(axis=1).mean()) if len(shares) else None,
        "top_dim_mean_abs_over_median_dim": (
            float(mean_abs[top[0]] / median_dim) if median_dim > 0 else None
        ),
        "n_norm_above_factor_median": int((norms > outlier_factor * median).sum()),
        "outlier_factor": outlier_factor,
        "mean_norm_by_pos_bucket": by_bucket,
    }


def layer_curve(
    blocks: np.ndarray, block_layers: Sequence[int], n_pairs: int, seed: int
) -> list[dict[str, Any]]:
    """Mean norm and mean random-pair cosine per captured block (layer curve of P2)."""

    n = blocks.shape[1]
    pairs = random_pairs(n, n_pairs, seed) if n >= 2 else None
    curve = []
    for index, layer in enumerate(block_layers):
        x = bits_to_float32(blocks[index])
        norms = np.linalg.norm(x, axis=1)
        curve.append(
            {
                "layer": int(layer),
                "mean_norm": float(norms.mean()),
                "median_norm": float(np.median(norms)),
                "mean_pair_cosine": (
                    mean_pair_cosine(x, pairs, norms) if pairs is not None else None
                ),
            }
        )
    return curve


def describe_devices(model: nn.Module) -> dict[str, Any]:
    """Where the model computes (CUDA is queried only when the model actually uses it)."""

    _, norm_module, embedding = decoder_parts(model)
    device_map = getattr(model, "hf_device_map", None) or {}
    info: dict[str, Any] = {
        "input_device": str(module_device(embedding)),
        "embedding_weight_device": str(embedding_weight(model).device),
        "norm_device": str(module_device(norm_module)),
        "device_map_counts": dict(Counter(str(device) for device in device_map.values())),
        "dtype": str(embedding.weight.dtype).replace("torch.", ""),
        "attn_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
    }
    uses_cuda = any("cuda" in value or value.isdigit() for value in info["device_map_counts"]) or (
        module_device(embedding).type == "cuda"
    )
    if uses_cuda and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return info


def build_diagnostics(
    result: ExtractionResult,
    rep_layers: Mapping[str, int],
    weight: torch.Tensor,
    token_ids: np.ndarray,
    pos_bucket: np.ndarray,
    seed: int,
    n_pairs: int = DIAGNOSTIC_PAIRS,
    curve_pairs: int = LAYER_CURVE_PAIRS,
    next_token_id: np.ndarray | None = None,
) -> dict[str, Any]:
    """Numerical profile of every representation (same random pairs for all of them)."""

    n = result.norm.shape[0]
    pairs = random_pairs(n, n_pairs, seed) if n >= 2 else None
    representations: dict[str, Any] = {}
    for name, layer in rep_layers.items():
        x = bits_to_float32(result.blocks[result.block_layers.index(layer)])
        representations[name] = representation_summary(x, pos_bucket, pairs)
    representations[NORM_REP] = representation_summary(
        bits_to_float32(result.norm), pos_bucket, pairs
    )
    # np.array copies: pandas (copy-on-write) hands out read-only arrays that torch warns about
    lexical = weight[torch.from_numpy(np.array(token_ids, dtype=np.int64))].float().numpy()
    representations[LEX_REP] = representation_summary(lexical, pos_bucket, pairs)
    diagnostics: dict[str, Any] = {
        "n_vertices": int(n),
        "hidden_size": int(result.norm.shape[1]),
        "seed": seed,
        "pairs": n_pairs,
        "layer_curve_pairs": curve_pairs,
        "representations": representations,
        "layer_curve": layer_curve(result.blocks, result.block_layers, curve_pairs, seed),
    }
    if next_token_id is not None:
        target = np.asarray(next_token_id, dtype=np.int64)
        known = target >= 0
        diagnostics["pred_next"] = {
            "n_with_next_token": int(known.sum()),
            "top1_agreement_with_next_token": (
                float((result.pred_next[known] == target[known]).mean()) if known.any() else None
            ),
        }
    return diagnostics


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------


def read_occurrences(path: Path) -> pd.DataFrame:
    """The columns of ``occurrences.csv`` this stage needs."""

    frame = pd.read_csv(
        path,
        usecols=lambda column: column in OCCURRENCE_INPUT_COLUMNS,
        dtype={"pos_bucket": str},
        keep_default_na=False,
    )
    for column in ("occurrence_id", "sequence_id", "pos_in_sequence", "token_id"):
        frame[column] = pd.to_numeric(frame[column]).astype(np.int64)
    for column in ("prefix_group", "next_token_id"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].replace("", -1)).astype(np.int64)
    return frame


def _pos_buckets(occurrences: pd.DataFrame) -> np.ndarray:
    if "pos_bucket" in occurrences and (occurrences["pos_bucket"].astype(str) != "").all():
        return occurrences["pos_bucket"].astype(str).to_numpy()
    return np.array([position_bucket(int(p)) for p in occurrences["pos_in_sequence"]])


def extract_to_directory(
    model: nn.Module,
    paths: RunPaths,
    tasks: Sequence[SequenceTask],
    occurrences: pd.DataFrame,
    *,
    vocab_size: int,
    rep_layers: Mapping[str, int],
    capture_all_layers: bool,
    seed: int,
    timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Extract, align prefix groups and write every representation artifact of ``reps/``.

    Returns the statistics recorded in the stage manifest. ``verify.json`` and the manifest are
    written by :func:`run`.
    """

    layers, _, embedding = decoder_parts(model)
    n_layers = len(layers)
    reserved = {NORM_REP, LEX_REP} & set(rep_layers)
    if reserved:
        raise ValueError(f"Representation names {sorted(reserved)} are reserved")
    if any(not 1 <= layer <= n_layers for layer in rep_layers.values()):
        raise ValueError(f"Representation layers must be in 1..{n_layers}")
    if vocab_size > embedding.weight.shape[0]:
        raise ValueError(f"vocab_size {vocab_size} exceeds the embedding rows")
    ensure_dir(paths.reps_dir)
    n_vertices = len(occurrences)
    hidden = int(embedding.weight.shape[1])
    block_layers = (
        list(range(1, n_layers + 1)) if capture_all_layers else sorted(set(rep_layers.values()))
    )
    blocks_out = None
    partial = partial_all_layers_path(paths)
    partial.unlink(missing_ok=True)
    if capture_all_layers:
        LOGGER.info(
            "Memmap de todas as camadas: %s (%.2f GB)",
            paths.all_layers,
            len(block_layers) * n_vertices * hidden * 2 / 1e9,
        )
        # Filled under a temporary name and renamed only once complete, so a crash never leaves
        # a zero-filled all_layers.npy that looks finished.
        blocks_out = open_memmap(
            partial,
            mode="w+",
            dtype=np.uint16,
            shape=(len(block_layers), n_vertices, hidden),
        )
    elif paths.all_layers.exists():
        LOGGER.warning("Removendo %s de uma execução anterior", paths.all_layers)
        paths.all_layers.unlink()

    try:
        result = extract_representations(
            model, tasks, n_vertices, block_layers, vocab_size, blocks_out=blocks_out
        )
    except BaseException:
        del blocks_out
        partial.unlink(missing_ok=True)
        raise
    tick = time.perf_counter()
    prefix_group = (
        occurrences["prefix_group"].to_numpy(dtype=np.int64)
        if "prefix_group" in occurrences
        else np.full(n_vertices, -1, dtype=np.int64)
    )
    prefix = copy_prefix_groups(
        prefix_group, result.blocks, result.block_layers, result.norm, result.pred_next
    )
    per_layer = prefix["per_layer_max_abs_diff"]
    prefix["max_abs_diff"] = {name: per_layer[str(layer)] for name, layer in rep_layers.items()}
    prefix["max_abs_diff"][NORM_REP] = prefix["norm_max_abs_diff"]
    prefix["max_abs_diff"]["all_layers"] = max(per_layer.values(), default=0.0)
    LOGGER.info(
        "Grupos de prefixo: %d grupos, %d linhas copiadas, maior diferença %s",
        prefix["n_groups"],
        prefix["n_rows_copied"],
        prefix["max_abs_diff"],
    )
    if isinstance(result.blocks, np.memmap):
        result.blocks.flush()
        # The open mapping stays valid after the rename (POSIX), so the arrays remain usable.
        partial.replace(paths.all_layers)
    copy_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    for name, layer in rep_layers.items():
        tensor = bits_to_tensor(result.blocks[result.block_layers.index(layer)])
        save_file(
            {"x": tensor},
            str(paths.rep(name)),
            metadata={"representation": name, "layer": str(layer), "source": "raw block output"},
        )
    save_file(
        {"x": bits_to_tensor(result.norm)},
        str(paths.rep(NORM_REP)),
        metadata={"representation": NORM_REP, "source": "final RMSNorm output"},
    )
    weight = embedding_weight(model)[:vocab_size].detach().to("cpu", torch.bfloat16).contiguous()
    save_file(
        {"weight": weight},
        str(paths.embeddings),
        metadata={"vocab_size": str(vocab_size), "embedding_rows": str(embedding.weight.shape[0])},
    )
    np.save(pred_next_path(paths), result.pred_next.astype(np.int32))
    write_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    diagnostics = build_diagnostics(
        result,
        rep_layers,
        weight,
        occurrences["token_id"].to_numpy(dtype=np.int64),
        _pos_buckets(occurrences),
        seed,
        next_token_id=(
            occurrences["next_token_id"].to_numpy(dtype=np.int64)
            if "next_token_id" in occurrences
            else None
        ),
    )
    diagnostics_seconds = time.perf_counter() - tick
    diagnostics["prefix_groups"] = prefix
    diagnostics["timings"] = {
        **dict(timings or {}),
        "extraction_s": round(result.seconds, 3),
        "forward_s": round(result.forward_seconds, 3),
        "tokens": result.tokens,
        "tokens_per_second": round(result.tokens / max(result.forward_seconds, 1e-9), 1),
        "sequences": result.n_sequences,
        "prefix_copy_s": round(copy_seconds, 3),
        "write_s": round(write_seconds, 3),
        "diagnostics_s": round(diagnostics_seconds, 3),
    }
    diagnostics["devices"] = describe_devices(model)
    diagnostics["vocab_size"] = vocab_size
    diagnostics["embedding_rows"] = int(embedding.weight.shape[0])
    diagnostics["block_layers"] = result.block_layers
    write_json(diagnostics_path(paths), diagnostics)

    return {
        "n_vertices": n_vertices,
        "n_sequences": result.n_sequences,
        "tokens": result.tokens,
        "hidden_size": hidden,
        "vocab_size": vocab_size,
        "representations": [*rep_layers, NORM_REP],
        "all_layers": capture_all_layers,
        "prefix_groups": prefix["n_groups"],
        "prefix_rows_copied": prefix["n_rows_copied"],
        "prefix_max_abs_diff": prefix["max_abs_diff"],
        "timings": diagnostics["timings"],
        "devices": diagnostics["devices"],
    }


def run(
    settings: Settings,
    paths: RunPaths,
    force: bool = False,
    verify_only: bool = False,
    **_: object,
) -> None:
    targets = output_paths(settings, paths)
    inputs = [paths.manifest("sample")]
    if not verify_only and not force and stage_is_fresh(paths.reps_dir, settings, inputs, targets):
        LOGGER.info("Representações já existem em %s; use --force para refazer", paths.reps_dir)
        return
    needed = [paths.sequences] if verify_only else [paths.sequences, paths.occurrences]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Rode a etapa sample antes; faltam: " + ", ".join(missing))
    if not verify_only:
        # Only a run that finishes writes a new manifest; dropping the old one first means a
        # forced run that dies midway is redone by the next normal run instead of skipped.
        (paths.reps_dir / "_manifest.json").unlink(missing_ok=True)
    started = time.time()
    model_settings = settings.model
    sequences = list(read_jsonl(paths.sequences))

    tasks: list[SequenceTask] = []
    occurrences = pd.DataFrame()
    vocab_size = 0
    if not verify_only:
        # Inputs are checked against the tokenizer before the (slow) model load.
        tokenizer = load_tokenizer(model_settings)
        vocab_size = len(tokenizer)
        prefix_id = tokenizer.convert_tokens_to_ids(model_settings.prefix_token)
        if not isinstance(prefix_id, int) or prefix_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(
                f"Prefix token {model_settings.prefix_token!r} is not in the vocabulary"
            )
        occurrences = read_occurrences(paths.occurrences)
        tasks = build_tasks(sequences, occurrences, vocab_size, prefix_id)
        if "prefix_group" in occurrences:
            check_prefix_groups(tasks, occurrences["prefix_group"].to_numpy(dtype=np.int64))
        LOGGER.info(
            "Entrada: %d vértices em %d sequências (vocabulário %d)",
            len(occurrences),
            len(tasks),
            vocab_size,
        )

    tick = time.perf_counter()
    model = load_model(model_settings)
    load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    report = verify(model, [s["input_ids"] for s in sequences], model_settings.verify_sequences)
    verify_seconds = time.perf_counter() - tick
    report["model"] = {
        "name_or_path": model_settings.name_or_path,
        "revision": model_settings.revision,
        "dtype": model_settings.dtype,
    }
    report["devices"] = describe_devices(model)
    report["versions"] = library_versions()
    ensure_dir(paths.reps_dir)
    write_json(verify_path(paths), report)
    if verify_only:
        status = "ok" if report["passed"] else "FALHOU"
        LOGGER.info("Verificação %s; veja %s", status, verify_path(paths))
        return
    if not report["passed"]:
        failed = [name for name in GATING_CHECKS if not report["checks"][name]["passed"]]
        raise RuntimeError(f"Verificação falhou ({', '.join(failed)}); veja {verify_path(paths)}")

    stats = extract_to_directory(
        model,
        paths,
        tasks,
        occurrences,
        vocab_size=vocab_size,
        rep_layers=dict(model_settings.layers),
        capture_all_layers=model_settings.capture_all_layers,
        seed=settings.sample.seed,
        timings={"model_load_s": round(load_seconds, 3), "verify_s": round(verify_seconds, 3)},
    )
    stats["verify_passed"] = report["passed"]
    stats["verify_warnings"] = report["warnings"]
    write_manifest(
        paths.reps_dir, "extract", settings, started, extra=stats, root=paths.root, inputs=inputs
    )
    LOGGER.info("Extração: %d vértices, %d tokens", stats["n_vertices"], stats["tokens"])
