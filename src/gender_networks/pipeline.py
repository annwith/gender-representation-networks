"""Reusable orchestration for the activation-extraction pilot."""

from __future__ import annotations

import logging
from pathlib import Path

from gender_networks.config import ExperimentConfig
from gender_networks.modeling import ModelBundle, extract_hidden_states
from gender_networks.prompts import PromptPair
from gender_networks.storage import safe_pair_id, save_alignment_report, save_occurrence
from gender_networks.tokenization import compare_positionally, tokenize_prompt

LOGGER = logging.getLogger(__name__)


def _ensure_unique_safe_ids(pairs: list[PromptPair]) -> None:
    owners: dict[str, str] = {}
    for pair in pairs:
        safe_id = safe_pair_id(pair.pair_id)
        previous = owners.get(safe_id)
        if previous is not None:
            raise ValueError(
                f"pair_ids {previous!r} and {pair.pair_id!r} map to the same file name {safe_id!r}"
            )
        owners[safe_id] = pair.pair_id


def run_extraction(
    config: ExperimentConfig,
    pairs: list[PromptPair],
    bundle: ModelBundle,
    output_dir: str | Path,
) -> None:
    """Tokenize, compare, extract, and persist all configured prompt pairs."""

    _ensure_unique_safe_ids(pairs)
    model_metadata = {
        "name_or_path": config.model.name_or_path,
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": config.model.torch_dtype,
        "device_map": config.model.device_map,
        "hidden_state_indices": list(config.extraction.hidden_state_indices),
        "tokenization": {
            "add_special_tokens": config.tokenization.add_special_tokens,
            "truncation": config.tokenization.truncation,
            "max_length": config.tokenization.max_length,
            "padding": False,
        },
    }
    for pair_index, pair in enumerate(pairs, start=1):
        LOGGER.info("Processing pair %s (%d/%d)", pair.pair_id, pair_index, len(pairs))
        tokenized_a = tokenize_prompt(bundle.tokenizer, pair.prompt_a, config.tokenization)
        tokenized_b = tokenize_prompt(bundle.tokenizer, pair.prompt_b, config.tokenization)
        differences = compare_positionally(tokenized_a, tokenized_b)
        LOGGER.info(
            "Pair %s: %d positional token difference(s): %s",
            pair.pair_id,
            len(differences),
            [
                {
                    "position": difference.position,
                    "a": difference.token_a,
                    "b": difference.token_b,
                }
                for difference in differences
            ],
        )
        save_alignment_report(
            output_dir, pair, tokenized_a, tokenized_b, differences
        )
        for variant, prompt, tokenized in (
            ("a", pair.prompt_a, tokenized_a),
            ("b", pair.prompt_b, tokenized_b),
        ):
            hidden_states = extract_hidden_states(
                bundle.model, tokenized, config.extraction.hidden_state_indices
            )
            tensor_path, _ = save_occurrence(
                output_dir,
                pair,
                variant,
                prompt,
                tokenized,
                hidden_states,
                model_metadata,
            )
            LOGGER.info("Saved occurrence %s/%s to %s", pair.pair_id, variant, tensor_path)
