"""Tests of the extraction stage with a tiny random Qwen3 (CPU, no weight downloads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from gender_networks import artifacts
from gender_networks import extract as ex
from gender_networks.artifacts import (
    OCCURRENCE_COLUMNS,
    RunPaths,
    position_bucket,
    read_json,
    write_csv,
    write_jsonl,
)
from gender_networks.settings import ModelSettings, PathsSettings, Settings

N_LAYERS = 4
HIDDEN = 32
EMBEDDING_ROWS = 64
VOCAB = 60  # tokenizer length: the last embedding rows are padding, as in Qwen3
PREFIX = 59
REP_LAYERS = {"L01": 1, "L02": 2, "L04": 4}

# Sequences of the fake sample. Sequences 0 and 1 share the prefix [PREFIX, 5, 6], so their
# vertices at positions 1 and 2 form two prefix groups.
SEQUENCES = [
    {"sequence_id": 0, "paragraph_id": 10, "input_ids": [PREFIX, 5, 6, 7, 8, 9]},
    {"sequence_id": 1, "paragraph_id": 11, "input_ids": [PREFIX, 5, 6, 10, 11]},
    {"sequence_id": 2, "paragraph_id": 12, "input_ids": [PREFIX, 20, 21, 22, 23, 24, 25, 26]},
]
VERTEX_POSITIONS = {0: [1, 2, 4, 5], 1: [1, 2, 3, 4], 2: [2, 5, 7]}


def tiny_model(seed: int = 0, dtype: torch.dtype = torch.float32) -> Qwen3Model:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=EMBEDDING_ROWS,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3Model(config).to(dtype).eval()


@pytest.fixture(scope="module")
def model() -> Qwen3Model:
    return tiny_model()


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifacts, "_git_describe", lambda root: None)


def plain_forward(model: nn.Module, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Independent reference: raw output of the last block (own hook) and the model outputs."""

    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, args: Any, output: Any) -> None:
        captured["raw"] = output[0] if isinstance(output, tuple) else output

    handle = model.layers[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False
            )
    finally:
        handle.remove()
    return captured["raw"][0], outputs.last_hidden_state[0], outputs


# ---------------------------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------------------------


def test_hooks_equal_hidden_states_and_final_norm(model: Qwen3Model) -> None:
    ids = [PREFIX, 3, 14, 15, 9, 2, 6, 5]
    plain_forward(model, ids)  # transformers installs its own capture hooks on first use
    hooks_before = [len(layer._forward_hooks) for layer in [*model.layers, model.norm]]
    with ex.HookRecorder(model, store_dtype=None) as recorder:
        outputs = recorder.run(ids, range(len(ids)), output_hidden_states=True)
    hidden = outputs.hidden_states

    assert len(hidden) == N_LAYERS + 1
    with torch.inference_mode():
        embedded = model.embed_tokens(torch.tensor(ids))
        normed = model.norm(recorder.block(N_LAYERS - 1))
    # RoPE: nothing is added to the residual stream before the first block
    assert torch.equal(hidden[0][0], embedded)
    for k in range(1, N_LAYERS):
        assert torch.equal(recorder.block(k - 1), hidden[k][0])
    # hidden_states[L] is the post-norm output: the raw last block only exists in the hook
    raw = recorder.block(N_LAYERS - 1)
    assert not torch.allclose(raw, hidden[N_LAYERS][0], atol=1e-3)
    torch.testing.assert_close(normed, hidden[N_LAYERS][0], rtol=1e-6, atol=1e-6)
    assert torch.equal(recorder.norm, hidden[N_LAYERS][0])
    assert torch.equal(recorder.norm, outputs.last_hidden_state[0])
    # the context manager removes every hook it added
    assert [len(layer._forward_hooks) for layer in [*model.layers, model.norm]] == hooks_before


def test_capture_outputs_keeps_vertex_rows_in_bfloat16(model: Qwen3Model) -> None:
    ids = [PREFIX, 3, 14, 15, 9, 2, 6]
    positions = [2, 5, 6]
    raw, normed, outputs = plain_forward(model, ids)

    blocks, norm = ex.capture_outputs(model, ids, positions)

    assert blocks.shape == (N_LAYERS, len(positions), HIDDEN) and blocks.dtype == torch.bfloat16
    assert norm.shape == (len(positions), HIDDEN) and norm.dtype == torch.bfloat16
    assert blocks.device.type == "cpu" and norm.device.type == "cpu"
    for layer in range(1, N_LAYERS):
        expected = outputs.hidden_states[layer][0, positions].to(torch.bfloat16)
        assert torch.equal(blocks[layer - 1], expected)
    assert torch.equal(blocks[N_LAYERS - 1], raw[positions].to(torch.bfloat16))
    assert torch.equal(norm, normed[positions].to(torch.bfloat16))

    subset, _ = ex.capture_outputs(model, ids, positions, block_indices=[3, 0])
    assert subset.shape == (2, len(positions), HIDDEN)
    assert torch.equal(subset[0], blocks[0]) and torch.equal(subset[1], blocks[3])


def test_capture_rejects_bad_positions_and_blocks(model: Qwen3Model) -> None:
    with pytest.raises(ValueError, match="Positions"):
        ex.capture_outputs(model, [PREFIX, 1, 2], [3])
    with pytest.raises(ValueError, match="Block indices"):
        ex.capture_outputs(model, [PREFIX, 1, 2], [1], block_indices=[N_LAYERS])
    with pytest.raises(RuntimeError, match="context manager"):
        ex.HookRecorder(model).run([PREFIX, 1], [1])


def test_decoder_parts_accepts_causal_lm() -> None:
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    causal = Qwen3ForCausalLM(tiny_model().config).eval()
    layers, norm, embedding = ex.decoder_parts(causal)
    assert layers is causal.model.layers and norm is causal.model.norm
    assert embedding is causal.model.embed_tokens
    with pytest.raises(TypeError):
        ex.decoder_parts(nn.Linear(2, 2))


def test_offloaded_embedding_is_read_through_the_accelerate_hook() -> None:
    real = torch.arange(12.0).reshape(4, 3)

    class AlignHook:  # the attributes of accelerate's AlignDevicesHook that are used
        execution_device = "cuda:0"
        weights_map = {"weight": real}

    class SequentialHook:
        hooks = [AlignHook()]

    embedding = nn.Embedding(4, 3, device="meta")
    embedding._hf_hook = SequentialHook()
    holder = nn.Module()
    holder.get_input_embeddings = lambda: embedding

    assert ex.module_device(embedding) == torch.device("cuda:0")
    assert ex.embedding_weight(holder) is real
    assert ex.module_device(nn.Linear(2, 2)) == torch.device("cpu")


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def test_verify_passes_on_a_healthy_model(model: Qwen3Model) -> None:
    sequences = [s["input_ids"] for s in SEQUENCES]
    report = ex.verify(model, sequences, 2)

    assert report["passed"] is True
    assert report["n_sequences"] == 2 and report["tokens"] == 6 + 5
    assert report["n_layers"] == N_LAYERS
    checks = report["checks"]
    assert all(check["passed"] for check in checks.values())
    assert checks["hidden_states_count"]["found"] == [N_LAYERS + 1]
    assert checks["embedding_input"]["max_abs_diff"] == 0.0
    assert set(checks["hooks_match_hidden_states"]["layers"]) == {"1", "2", "3"}
    final = checks["final_norm"]
    assert final["raw_block_differs_from_hidden"] and final["norm_hook_equal"]
    assert final["raw_block_vs_hidden_max_abs_diff"] > 0.1
    assert final["norm_of_raw_block_max_abs_diff"] < 1e-5
    assert checks["determinism"]["status"] == "identical"
    assert report["warnings"] == []
    json.dumps(report)  # the report goes to verify.json as is

    # sequences may also be the records of sequences.jsonl
    assert ex.verify(model, SEQUENCES, 1)["passed"] is True
    with pytest.raises(ValueError):
        ex.verify(model, sequences, 0)


def test_verify_detects_a_missing_final_norm() -> None:
    broken = tiny_model(seed=1)
    broken.norm = nn.Identity()  # hidden_states[L] would then be the raw last block

    report = ex.verify(broken, [s["input_ids"] for s in SEQUENCES], 2)

    assert report["passed"] is False
    assert report["checks"]["final_norm"]["passed"] is False
    assert report["checks"]["final_norm"]["raw_block_differs_from_hidden"] is False
    assert report["checks"]["hooks_match_hidden_states"]["passed"] is True


def test_verify_passes_in_bfloat16() -> None:
    report = ex.verify(tiny_model(seed=2, dtype=torch.bfloat16), [SEQUENCES[2]["input_ids"]], 1)
    assert report["passed"] is True and report["dtype"] == "bfloat16"


# ---------------------------------------------------------------------------------------------
# small pure functions
# ---------------------------------------------------------------------------------------------


def test_predict_next_is_the_argmax_over_the_tokenizer_ids() -> None:
    generator = torch.Generator().manual_seed(3)
    rows = torch.randn(6, HIDDEN, generator=generator)
    weight = torch.randn(70, HIDDEN, generator=generator)
    weight[65] = rows.sum(0) * 100  # the global winner is a padding row beyond vocab_size

    expected = (rows.double() @ weight[:64].double().T).argmax(1)
    for chunk in (7, 16, 64, 1000):
        assert torch.equal(ex.predict_next(rows, weight, 64, chunk_rows=chunk), expected)
    assert int((rows @ weight.T).argmax(1)[0]) == 65
    # ties keep the lowest id, as torch.argmax does
    assert ex.predict_next(torch.zeros(2, HIDDEN), weight, 64, chunk_rows=5).tolist() == [0, 0]
    with pytest.raises(ValueError):
        ex.predict_next(rows, weight, 71)


def test_bf16_bits_round_trip() -> None:
    x = torch.randn(5, 7, generator=torch.Generator().manual_seed(4)) * 30
    bits = ex.bf16_bits(x)

    assert bits.dtype == np.uint16 and bits.shape == (5, 7)
    np.testing.assert_array_equal(ex.bits_to_float32(bits), x.to(torch.bfloat16).float().numpy())
    assert torch.equal(ex.bits_to_tensor(bits), x.to(torch.bfloat16))
    # the loading recipe documented for all_layers.npy
    documented = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
    assert torch.equal(documented, x.to(torch.bfloat16))


def fake_occurrences(
    sequences: list[dict[str, Any]], positions: dict[int, list[int]]
) -> pd.DataFrame:
    """Rows ordered by (sequence_id, pos_in_sequence) with prefix groups like sampling's."""

    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        ids = sequence["input_ids"]
        for pos in positions[sequence["sequence_id"]]:
            rows.append(
                {
                    "occurrence_id": len(rows),
                    "sequence_id": sequence["sequence_id"],
                    "pos_in_sequence": pos,
                    "pos_bucket": position_bucket(pos),
                    "token_id": ids[pos],
                    "next_token_id": ids[pos + 1] if pos + 1 < len(ids) else -1,
                    "prefix": tuple(ids[: pos + 1]),
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame["prefix"].value_counts()
    shared = [p for p in frame["prefix"].drop_duplicates() if counts[p] > 1]
    group_of = {prefix: index for index, prefix in enumerate(shared)}
    frame["prefix_group"] = [group_of.get(p, -1) for p in frame["prefix"]]
    return frame.drop(columns="prefix")


def test_build_tasks_groups_vertices_by_sequence() -> None:
    occurrences = fake_occurrences(SEQUENCES, VERTEX_POSITIONS)
    tasks = ex.build_tasks(SEQUENCES, occurrences, VOCAB, PREFIX)

    assert [t.sequence_id for t in tasks] == [0, 1, 2]
    assert tasks[1].positions.tolist() == [1, 2, 3, 4]
    assert tasks[1].occurrence_ids.tolist() == [4, 5, 6, 7]
    assert tasks[2].input_ids == tuple(SEQUENCES[2]["input_ids"])
    assert occurrences["prefix_group"].tolist() == [0, 1, -1, -1, 0, 1, -1, -1, -1, -1, -1]
    ex.check_prefix_groups(tasks, occurrences["prefix_group"].to_numpy())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda o, s: o.loc.__setitem__((0, "token_id"), 1), "token_id differs"),
        (lambda o, s: o.loc.__setitem__((0, "pos_in_sequence"), 0), "positions must be"),
        (lambda o, s: o.loc.__setitem__((1, "pos_in_sequence"), 1), "repeated"),
        (lambda o, s: o.loc.__setitem__((0, "sequence_id"), 9), "unknown sequences"),
        (lambda o, s: o.loc.__setitem__((0, "occurrence_id"), 5), "row index"),
        (lambda o, s: s[0]["input_ids"].__setitem__(0, 1), "prefix"),
        (lambda o, s: s[2]["input_ids"].__setitem__(1, VOCAB), "outside the vocabulary"),
    ],
)
def test_build_tasks_rejects_inconsistent_inputs(change, message: str) -> None:
    sequences = [dict(s, input_ids=list(s["input_ids"])) for s in SEQUENCES]
    occurrences = fake_occurrences(sequences, VERTEX_POSITIONS)
    change(occurrences, sequences)
    with pytest.raises(ValueError, match=message):
        ex.build_tasks(sequences, occurrences, VOCAB, PREFIX)


def test_check_prefix_groups_rejects_wrong_labels() -> None:
    occurrences = fake_occurrences(SEQUENCES, VERTEX_POSITIONS)
    tasks = ex.build_tasks(SEQUENCES, occurrences, VOCAB, PREFIX)
    labels = occurrences["prefix_group"].to_numpy()

    missing = labels.copy()
    missing[4] = -1  # shares its prefix with row 0 but is not grouped
    with pytest.raises(ValueError, match="share a prefix"):
        ex.check_prefix_groups(tasks, missing)
    mixed = labels.copy()
    mixed[2] = 0  # different prefix inside group 0
    with pytest.raises(ValueError, match="mixes different prefixes"):
        ex.check_prefix_groups(tasks, mixed)


def test_copy_prefix_groups_makes_members_identical_and_reports_noise() -> None:
    rng = np.random.default_rng(5)
    values = torch.tensor(rng.standard_normal((3, 6, 4)), dtype=torch.float32)
    values[:, 2] = values[:, 0] + 0.25  # group 0 = rows 0, 2, 5 with noise in row 2
    values[:, 5] = values[:, 0]
    values[1, 5, 3] += 4.0  # a larger difference in the second captured block only
    values[:, 4] = values[:, 3]  # group 1 = rows 3, 4 identical
    blocks = ex.bf16_bits(values)
    norm = ex.bf16_bits(values[2] * 2)
    pred_next = np.array([7, 1, 8, 3, 3, 9], dtype=np.int32)
    groups = np.array([0, -1, 0, 1, 1, 0])
    diff = ex.bits_to_float32(blocks) - ex.bits_to_float32(blocks)[:, [0]]
    norm_diff = ex.bits_to_float32(norm) - ex.bits_to_float32(norm)[[0]]
    expected_layers = np.abs(diff[:, [2, 5]]).max(axis=(1, 2))

    report = ex.copy_prefix_groups(groups, blocks, [1, 5, 9], norm, pred_next)

    assert report["n_groups"] == 2 and report["n_rows_copied"] == 3
    assert report["per_layer_max_abs_diff"] == pytest.approx(
        dict(zip(["1", "5", "9"], expected_layers.tolist(), strict=True))
    )
    assert report["per_layer_max_abs_diff"]["5"] > 3.9
    assert report["norm_max_abs_diff"] == pytest.approx(float(np.abs(norm_diff[[2, 5]]).max()))
    assert report["pred_next_changed"] == 2
    for rows in ([0, 2, 5], [3, 4]):
        assert (blocks[:, rows] == blocks[:, rows[:1]]).all()
        assert (norm[rows] == norm[rows[0]]).all()
        assert (pred_next[rows] == pred_next[rows[0]]).all()
    assert pred_next.tolist() == [7, 1, 7, 3, 3, 7]
    assert ex.prefix_group_members(groups)[0].tolist() == [0, 2, 5]


def test_random_pairs_and_mean_cosine() -> None:
    first, second = ex.random_pairs(5, 2000, seed=1)
    assert (first != second).all() and first.max() == 4 and second.max() == 4
    again = ex.random_pairs(5, 2000, seed=1)
    assert np.array_equal(first, again[0]) and np.array_equal(second, again[1])

    same = np.ones((4, 3), dtype=np.float32)
    assert ex.mean_pair_cosine(same, ex.random_pairs(4, 50, 0)) == pytest.approx(1.0)
    basis = np.eye(3, dtype=np.float32)
    assert ex.mean_pair_cosine(basis, ex.random_pairs(3, 50, 0)) == pytest.approx(0.0)
    with_zero = np.array([[1, 0], [0, 0], [2, 0]], dtype=np.float32)
    pairs = (np.array([0, 0, 1]), np.array([1, 2, 2]))
    assert ex.mean_pair_cosine(with_zero, pairs) == pytest.approx(1.0)  # zero rows skipped


def test_representation_summary_finds_massive_dimensions() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    x[:, 7] += 50.0  # a massive activation shared by every vertex
    x[3] *= 40.0  # one vertex with a huge norm
    buckets = np.array([position_bucket(p) for p in rng.integers(1, 100, 200)])

    summary = ex.representation_summary(x, buckets, ex.random_pairs(200, 1000, 0))

    norms = np.linalg.norm(x, axis=1)
    assert summary["n"] == 200
    assert list(summary["norm_quantiles"]) == ["0", "5", "25", "50", "75", "95", "100"]
    assert summary["norm_quantiles"]["50"] == pytest.approx(float(np.median(norms)), rel=1e-5)
    assert summary["norm_quantiles"]["100"] == pytest.approx(float(norms.max()), rel=1e-5)
    assert summary["top_dims"][0]["dim"] == 7 and len(summary["top_dims"]) == 5
    assert summary["top_dims"][0]["mean_share_sq_norm"] > 0.9
    assert summary["mean_pair_cosine"] > 0.9  # anisotropy caused by the shared dimension
    assert summary["n_norm_above_factor_median"] == 1
    for label, value in summary["mean_norm_by_pos_bucket"].items():
        assert value == pytest.approx(float(norms[buckets == label].mean()), rel=1e-5)


# ---------------------------------------------------------------------------------------------
# the stage end to end
# ---------------------------------------------------------------------------------------------


class FakeTokenizer:
    unk_token_id = None

    def __len__(self) -> int:
        return VOCAB

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return PREFIX if token == "<|endoftext|>" else None


def make_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    capture_all_layers: bool = True,
) -> tuple[Settings, RunPaths, pd.DataFrame]:
    settings = Settings(
        name="tiny",
        paths=PathsSettings(runs_dir="runs", report_dir="relatorio"),
        model=ModelSettings(
            layers=dict(REP_LAYERS), capture_all_layers=capture_all_layers, verify_sequences=2
        ),
    )
    paths = RunPaths.from_settings(settings, tmp_path)
    occurrences = fake_occurrences(SEQUENCES, VERTEX_POSITIONS)
    write_csv(paths.occurrences, occurrences.to_dict("records"), OCCURRENCE_COLUMNS)
    write_jsonl(paths.sequences, SEQUENCES)
    monkeypatch.setattr(ex, "load_model", lambda model_settings: model)
    monkeypatch.setattr(ex, "load_tokenizer", lambda model_settings: FakeTokenizer())
    return settings, paths, occurrences


def reference(model: nn.Module, occurrences: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vertex raw block outputs [L, N, d] and final-norm rows [N, d] (float), unaligned."""

    n = len(occurrences)
    blocks = torch.zeros(N_LAYERS, n, HIDDEN, dtype=torch.float64)
    norm = torch.zeros(n, HIDDEN, dtype=torch.float64)
    for sequence in SEQUENCES:
        chosen = occurrences["sequence_id"] == sequence["sequence_id"]
        rows = occurrences.index[chosen].to_list()
        positions = occurrences.loc[chosen, "pos_in_sequence"].to_list()
        raw, normed, outputs = plain_forward(model, sequence["input_ids"])
        for layer in range(1, N_LAYERS):
            blocks[layer - 1, rows] = outputs.hidden_states[layer][0, positions].double()
        blocks[N_LAYERS - 1, rows] = raw[positions].double()
        norm[rows] = normed[positions].double()
    return blocks, norm


def align_groups(values: torch.Tensor, groups: np.ndarray, axis: int) -> torch.Tensor:
    aligned = values.clone()
    for members in ex.prefix_group_members(groups):
        index = torch.as_tensor(members)
        source = values.index_select(axis, index[:1])
        shape = [-1] * values.dim()
        shape[axis] = len(members)
        aligned.index_copy_(axis, index, source.expand(*shape))
    return aligned


def load_bf16(path: Path, key: str) -> torch.Tensor:
    tensor = load_file(str(path))[key]
    assert tensor.dtype == torch.bfloat16
    return tensor


def test_run_writes_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model, no_git: None
) -> None:
    settings, paths, occurrences = make_run(tmp_path, monkeypatch, model)
    n = len(occurrences)
    groups = occurrences["prefix_group"].to_numpy()

    ex.run(settings, paths)

    for path in ex.output_paths(settings, paths):
        assert path.exists(), path
    raw_blocks, raw_norm = reference(model, occurrences)
    blocks = align_groups(raw_blocks, groups, axis=1)
    norm = align_groups(raw_norm, groups, axis=0)

    stored = np.load(paths.all_layers, mmap_mode="r")
    assert stored.dtype == np.uint16 and stored.shape == (N_LAYERS, n, HIDDEN)
    all_layers = torch.from_numpy(np.array(stored).view(np.int16)).view(torch.bfloat16)
    assert torch.equal(all_layers, blocks.to(torch.bfloat16))
    for name, layer in REP_LAYERS.items():
        x = load_bf16(paths.rep(name), "x")
        assert x.shape == (n, HIDDEN)
        assert torch.equal(x, all_layers[layer - 1])
    l36n = load_bf16(paths.rep(ex.NORM_REP), "x")
    assert torch.equal(l36n, norm.to(torch.bfloat16))
    weight = load_bf16(paths.embeddings, "weight")
    assert weight.shape == (VOCAB, HIDDEN)
    assert torch.equal(weight, model.embed_tokens.weight[:VOCAB].detach().to(torch.bfloat16))

    pred_next = np.load(ex.pred_next_path(paths))
    assert pred_next.dtype == np.int32 and pred_next.shape == (n,)
    logits = norm @ model.embed_tokens.weight[:VOCAB].detach().double().T
    assert pred_next.tolist() == logits.argmax(1).tolist()

    # members of a prefix group are identical in every representation
    for members in ex.prefix_group_members(groups):
        for x in (l36n, *all_layers):
            assert (x[members] == x[members[0]]).all()

    diagnostics = read_json(ex.diagnostics_path(paths))
    assert set(diagnostics["representations"]) == {"L01", "L02", "L04", "L36n", "lex"}
    for summary in diagnostics["representations"].values():
        assert summary["n"] == n
        assert list(summary["norm_quantiles"]) == ["0", "5", "25", "50", "75", "95", "100"]
        assert len(summary["top_dims"]) == 5 and summary["mean_pair_cosine"] is not None
    l01_norms = all_layers[0].float().norm(dim=1)
    assert diagnostics["representations"]["L01"]["norm_quantiles"]["50"] == pytest.approx(
        float(np.median(l01_norms.numpy())), rel=1e-5
    )
    lexical = weight[occurrences["token_id"].to_list()].float().norm(dim=1)
    assert diagnostics["representations"]["lex"]["mean_norm"] == pytest.approx(
        float(lexical.mean()), rel=1e-5
    )
    curve = diagnostics["layer_curve"]
    assert [point["layer"] for point in curve] == [1, 2, 3, 4]
    for point, x in zip(curve, all_layers, strict=True):
        assert point["mean_norm"] == pytest.approx(float(x.float().norm(dim=1).mean()), rel=1e-5)

    prefix = diagnostics["prefix_groups"]
    assert prefix["n_groups"] == 2 and prefix["n_rows_copied"] == 2
    assert set(prefix["max_abs_diff"]) == {"L01", "L02", "L04", "L36n", "all_layers"}
    unaligned = raw_blocks.to(torch.bfloat16).double()
    members = ex.prefix_group_members(groups)
    expected = max(float((unaligned[3, m[1:]] - unaligned[3, m[:1]]).abs().max()) for m in members)
    assert prefix["max_abs_diff"]["L04"] == pytest.approx(expected)

    timings = diagnostics["timings"]
    assert timings["tokens"] == sum(len(s["input_ids"]) for s in SEQUENCES)
    assert timings["sequences"] == 3 and timings["tokens_per_second"] > 0
    assert diagnostics["devices"]["input_device"] == "cpu"
    assert diagnostics["vocab_size"] == VOCAB and diagnostics["embedding_rows"] == EMBEDDING_ROWS

    verify = read_json(ex.verify_path(paths))
    assert verify["passed"] is True and verify["n_sequences"] == 2
    manifest = read_json(paths.reps_dir / "_manifest.json")
    assert manifest["stage"] == "extract" and manifest["n_vertices"] == n
    assert manifest["verify_passed"] is True and manifest["tokens"] == timings["tokens"]


def test_run_skips_existing_outputs_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model, no_git: None
) -> None:
    settings, paths, _ = make_run(tmp_path, monkeypatch, model)
    ex.run(settings, paths)
    stamp = paths.rep("L01").stat().st_mtime_ns

    def refuse(model_settings: ModelSettings) -> nn.Module:
        raise AssertionError("the model must not be loaded when outputs exist")

    monkeypatch.setattr(ex, "load_model", refuse)
    ex.run(settings, paths)
    assert paths.rep("L01").stat().st_mtime_ns == stamp

    monkeypatch.setattr(ex, "load_model", lambda model_settings: model)
    paths.rep("L01").write_bytes(b"stale")
    ex.run(settings, paths, force=True)
    assert load_bf16(paths.rep("L01"), "x").shape == (11, HIDDEN)


def test_failed_forced_run_is_redone_by_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model, no_git: None
) -> None:
    settings, paths, occurrences = make_run(tmp_path, monkeypatch, model)
    ex.run(settings, paths)
    good = np.load(paths.all_layers).copy()
    original = ex.predict_next
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        return original(*args, **kwargs)

    monkeypatch.setattr(ex, "predict_next", flaky)
    with pytest.raises(RuntimeError, match="out of memory"):
        ex.run(settings, paths, force=True)
    # no manifest, no half-filled memmap left behind; the finished all_layers.npy is untouched
    assert not (paths.reps_dir / "_manifest.json").exists()
    assert not ex.partial_all_layers_path(paths).exists()
    assert np.array_equal(np.load(paths.all_layers), good)

    redone = {"n": 0}

    def counting(*args: Any, **kwargs: Any) -> Any:
        redone["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ex, "predict_next", counting)
    ex.run(settings, paths)  # not skipped: the extraction is redone
    assert redone["n"] > 0
    assert (paths.reps_dir / "_manifest.json").exists()
    assert np.array_equal(np.load(paths.all_layers), good)
    assert np.load(paths.all_layers).any(axis=2).all()


def test_first_run_failure_leaves_no_all_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model
) -> None:
    settings, paths, _ = make_run(tmp_path, monkeypatch, model)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(ex, "predict_next", boom)
    with pytest.raises(KeyboardInterrupt):
        ex.run(settings, paths)
    assert not paths.all_layers.exists()
    assert not ex.partial_all_layers_path(paths).exists()
    assert not (paths.reps_dir / "_manifest.json").exists()


def test_failed_forced_verification_drops_the_old_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model, no_git: None
) -> None:
    settings, paths, _ = make_run(tmp_path, monkeypatch, model)
    ex.run(settings, paths)
    broken = tiny_model(seed=1)
    broken.norm = nn.Identity()
    monkeypatch.setattr(ex, "load_model", lambda model_settings: broken)

    with pytest.raises(RuntimeError, match="final_norm"):
        ex.run(settings, paths, force=True)

    assert read_json(ex.verify_path(paths))["passed"] is False
    assert not (paths.reps_dir / "_manifest.json").exists()


def test_run_verify_only_writes_just_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model
) -> None:
    settings, paths, _ = make_run(tmp_path, monkeypatch, model)
    paths.occurrences.unlink()  # verification needs only the sequences

    ex.run(settings, paths, verify_only=True)

    assert read_json(ex.verify_path(paths))["passed"] is True
    assert sorted(p.name for p in paths.reps_dir.iterdir()) == [ex.VERIFY_FILE]


def test_run_aborts_when_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tiny_model(seed=1)
    broken.norm = nn.Identity()
    settings, paths, _ = make_run(tmp_path, monkeypatch, broken)

    with pytest.raises(RuntimeError, match="final_norm"):
        ex.run(settings, paths)

    assert read_json(ex.verify_path(paths))["passed"] is False
    assert not paths.all_layers.exists() and not paths.rep("L01").exists()


def test_run_without_all_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Qwen3Model, no_git: None
) -> None:
    settings, paths, occurrences = make_run(tmp_path, monkeypatch, model, capture_all_layers=False)
    ex.run(settings, paths)

    assert not paths.all_layers.exists()
    blocks, _ = reference(model, occurrences)
    blocks = align_groups(blocks, occurrences["prefix_group"].to_numpy(), axis=1)
    for name, layer in REP_LAYERS.items():
        assert torch.equal(load_bf16(paths.rep(name), "x"), blocks[layer - 1].to(torch.bfloat16))
    curve = read_json(ex.diagnostics_path(paths))["layer_curve"]
    assert [point["layer"] for point in curve] == [1, 2, 4]


def test_run_in_bfloat16_matches_the_saved_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_git: None
) -> None:
    bf16_model = tiny_model(seed=3, dtype=torch.bfloat16)
    settings, paths, occurrences = make_run(tmp_path, monkeypatch, bf16_model)
    ex.run(settings, paths)

    raw_blocks, raw_norm = reference(bf16_model, occurrences)
    groups = occurrences["prefix_group"].to_numpy()
    l36n = load_bf16(paths.rep(ex.NORM_REP), "x")
    assert torch.equal(l36n, align_groups(raw_norm, groups, axis=0).to(torch.bfloat16))
    weight = load_bf16(paths.embeddings, "weight")
    # production dtype: pred_next is reproducible from the stored bf16 artifacts
    expected = (l36n.float() @ weight.float().T).argmax(1)
    assert np.load(ex.pred_next_path(paths)).tolist() == expected.tolist()
    blocks = align_groups(raw_blocks, groups, axis=1).to(torch.bfloat16)
    assert torch.equal(load_bf16(paths.rep("L04"), "x"), blocks[3])


def test_real_tokenizer_matches_the_assumptions() -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Base",
            revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
            local_files_only=True,
        )
    except Exception as error:  # any loading failure means "not cached here"
        pytest.skip(f"Qwen3 tokenizer not available offline: {error}")
    assert len(tokenizer) == 151_669
    prefix_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert prefix_id == 151_643 and prefix_id != tokenizer.unk_token_id
