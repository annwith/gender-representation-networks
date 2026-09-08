"""Serialization of activations and traceable occurrence metadata."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from gender_networks.prompts import PromptPair
from gender_networks.tokenization import TokenDifference, TokenizedPrompt

SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_pair_id(pair_id: str) -> str:
    """Convert a pair ID to a conservative file-name component."""

    safe = SAFE_ID_PATTERN.sub("_", pair_id).strip("._")
    if not safe:
        raise ValueError(f"pair_id {pair_id!r} cannot be converted to a safe file name")
    return safe


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_alignment_report(
    output_dir: str | Path,
    pair: PromptPair,
    tokenized_a: TokenizedPrompt,
    tokenized_b: TokenizedPrompt,
    differences: list[TokenDifference],
) -> Path:
    """Write the explicit positional tokenization comparison for one pair."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_pair_id(pair.pair_id)}__alignment.json"
    _write_json(
        path,
        {
            "pair_id": pair.pair_id,
            "alignment_method": "position_by_position",
            "sequence_length_a": len(tokenized_a.input_ids),
            "sequence_length_b": len(tokenized_b.input_ids),
            "difference_count": len(differences),
            "differences": [asdict(difference) for difference in differences],
        },
    )
    return path


def save_occurrence(
    output_dir: str | Path,
    pair: PromptPair,
    variant: str,
    prompt: str,
    tokenized: TokenizedPrompt,
    hidden_states: dict[int, torch.Tensor],
    model_metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Store tensors and JSON metadata for one member of a prompt pair."""

    if variant not in {"a", "b"}:
        raise ValueError("variant must be 'a' or 'b'")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_pair_id(pair.pair_id)}__{variant}"
    tensor_path = directory / f"{stem}.pt"
    metadata_path = directory / f"{stem}.json"
    torch.save(
        {
            "input_ids": torch.tensor(tokenized.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(tokenized.attention_mask, dtype=torch.long),
            "hidden_states": hidden_states,
        },
        tensor_path,
    )
    _write_json(
        metadata_path,
        {
            "pair_id": pair.pair_id,
            "variant": variant,
            "prompt": prompt,
            "tokens": tokenized.tokens,
            "input_ids": tokenized.input_ids,
            "sequence_length": len(tokenized.input_ids),
            "pair_metadata": pair.metadata,
            "model": model_metadata,
            "tensor_file": tensor_path.name,
            "hidden_state_shapes": {
                str(index): list(tensor.shape) for index, tensor in hidden_states.items()
            },
        },
    )
    return tensor_path, metadata_path

