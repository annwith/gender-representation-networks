"""Typed loading and validation for the experiment YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Hugging Face model loading parameters."""

    name_or_path: str
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    device_map: str | None = "auto"
    max_memory: dict[int | str, str] | None = None


@dataclass(frozen=True)
class TokenizationConfig:
    """Tokenizer behavior shared by both members of every prompt pair."""

    add_special_tokens: bool = True
    truncation: bool = False
    max_length: int | None = None


@dataclass(frozen=True)
class ExtractionConfig:
    """Selection of entries from ``outputs.hidden_states``."""

    hidden_state_indices: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete pilot experiment configuration."""

    model: ModelConfig
    tokenization: TokenizationConfig
    extraction: ExtractionConfig


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping")
    return value


def _boolean(
    mapping: dict[str, Any], key: str, default: bool, qualified_key: str
) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{qualified_key} must be a boolean")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML and validate required fields."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "root")
    model_raw = _mapping(root.get("model"), "model")
    token_raw = _mapping(root.get("tokenization", {}), "tokenization")
    extraction_raw = _mapping(root.get("extraction"), "extraction")

    name = model_raw.get("name_or_path")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.name_or_path must be a non-empty string")
    revision = model_raw.get("revision", "main")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model.revision must be a non-empty string")
    torch_dtype = model_raw.get("torch_dtype", "auto")
    if not isinstance(torch_dtype, str) or not torch_dtype.strip():
        raise ValueError("model.torch_dtype must be a non-empty string")
    trust_remote_code = _boolean(
        model_raw, "trust_remote_code", False, "model.trust_remote_code"
    )

    indices = extraction_raw.get("hidden_state_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
    ):
        raise ValueError("extraction.hidden_state_indices must be a non-empty list of integers")
    if len(indices) != len(set(indices)):
        raise ValueError("extraction.hidden_state_indices must not contain duplicates")

    truncation = _boolean(
        token_raw, "truncation", False, "tokenization.truncation"
    )
    add_special_tokens = _boolean(
        token_raw,
        "add_special_tokens",
        True,
        "tokenization.add_special_tokens",
    )
    max_length = token_raw.get("max_length")
    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        raise ValueError("tokenization.max_length must be null or a positive integer")
    if truncation and max_length is None:
        raise ValueError("tokenization.max_length is required when truncation is true")

    device_map = model_raw.get("device_map", "auto")
    if device_map is not None and not isinstance(device_map, str):
        raise ValueError("model.device_map must be a string or null")
    max_memory = model_raw.get("max_memory")
    if max_memory is not None and (
        not isinstance(max_memory, dict)
        or any(not isinstance(value, str) for value in max_memory.values())
    ):
        raise ValueError("model.max_memory must be null or a mapping of device to size string")

    return ExperimentConfig(
        model=ModelConfig(
            name_or_path=name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        ),
        tokenization=TokenizationConfig(
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        ),
        extraction=ExtractionConfig(hidden_state_indices=tuple(indices)),
    )
