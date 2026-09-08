"""Tokenization and literal position-by-position comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest
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


@dataclass(frozen=True)
class TokenDifference:
    """A position at which two token sequences do not contain the same token ID."""

    position: int
    token_id_a: int | None
    token_a: str | None
    token_id_b: int | None
    token_b: str | None


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


def compare_positionally(
    prompt_a: TokenizedPrompt, prompt_b: TokenizedPrompt
) -> list[TokenDifference]:
    """Report every unequal position, representing absent trailing tokens as ``None``."""

    missing = object()
    differences: list[TokenDifference] = []
    positions = zip_longest(
        zip(prompt_a.input_ids, prompt_a.tokens, strict=True),
        zip(prompt_b.input_ids, prompt_b.tokens, strict=True),
        fillvalue=missing,
    )
    for position, (item_a, item_b) in enumerate(positions):
        id_a, token_a = (None, None) if item_a is missing else item_a
        id_b, token_b = (None, None) if item_b is missing else item_b
        if id_a != id_b:
            differences.append(TokenDifference(position, id_a, token_a, id_b, token_b))
    return differences


def differences_as_dicts(differences: list[TokenDifference]) -> list[dict[str, Any]]:
    """Convert differences into JSON-serializable dictionaries."""

    return [asdict(difference) for difference in differences]

