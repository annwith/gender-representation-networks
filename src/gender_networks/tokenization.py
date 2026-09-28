"""Tokenization helpers shared by the pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gender_networks.config import TokenizationConfig


class TokenizerLike(Protocol):
    """Minimum tokenizer interface used by this module."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]: ...


@dataclass(frozen=True)
class TokenizedPrompt:
    """IDs, masks and readable tokens for one prompt."""

    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[str]


def tokenize_prompt(
    tokenizer: TokenizerLike, text: str, config: TokenizationConfig
) -> TokenizedPrompt:
    """Tokenize one prompt without padding and return plain Python lists."""

    encoded = tokenizer(
        text,
        add_special_tokens=config.add_special_tokens,
        truncation=config.truncation,
        max_length=config.max_length,
        padding=False,
        return_attention_mask=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(tokens):
        raise ValueError("Tokenizer returned inconsistent sequence lengths")
    return TokenizedPrompt(input_ids, attention_mask, tokens)
