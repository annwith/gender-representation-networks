"""Hugging Face model loading and hidden-state extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from gender_networks.config import ExperimentConfig
from gender_networks.tokenization import TokenizedPrompt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Loaded tokenizer and causal language model."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel


def load_model_and_tokenizer(config: ExperimentConfig) -> ModelBundle:
    """Load the configured tokenizer and causal language model from Hugging Face."""

    model_config = config.model
    LOGGER.info(
        "Loading tokenizer %s at revision %s",
        model_config.name_or_path,
        model_config.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name_or_path,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    LOGGER.info("Loading model %s", model_config.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device_map,
    )
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model)


def _normalize_indices(indices: tuple[int, ...], count: int) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in indices:
        resolved = index if index >= 0 else count + index
        if resolved < 0 or resolved >= count:
            raise IndexError(
                f"hidden-state index {index} is outside the available range "
                f"{-count}..{count - 1}"
            )
        normalized.append(resolved)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Configured hidden-state indices resolve to duplicate entries")
    return tuple(normalized)


def extract_hidden_states(
    model: PreTrainedModel,
    tokenized: TokenizedPrompt,
    indices: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Run one inference pass and select hidden states by tuple index.

    Returned tensors have shape ``[sequence_length, hidden_size]`` and reside on CPU.
    Dictionary keys preserve the indices written in the YAML, including negative indices.
    """

    input_device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.tensor(
        [tokenized.attention_mask], dtype=torch.long, device=input_device
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    resolved_indices = _normalize_indices(indices, len(hidden_states))
    return {
        configured: hidden_states[resolved][0].detach().to(device="cpu")
        for configured, resolved in zip(indices, resolved_indices, strict=True)
    }


def extract_input_embeddings(
    model: PreTrainedModel, tokenized: TokenizedPrompt
) -> torch.Tensor:
    """Return the lexical input embedding for every token occurrence on CPU.

    This intentionally indexes ``model.get_input_embeddings()`` directly instead
    of using ``hidden_states[0]``.  For many causal models the latter already
    includes positional information (and sometimes dropout), while this tensor is
    the shared lexical embedding table specified in the experiment proposal.
    """

    embedding = model.get_input_embeddings()
    input_ids = torch.tensor(tokenized.input_ids, dtype=torch.long, device=embedding.weight.device)
    with torch.inference_mode():
        return embedding(input_ids).detach().to(device="cpu")
