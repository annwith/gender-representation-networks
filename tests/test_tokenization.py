from typing import Any

import pytest

from gender_networks.config import TokenizationConfig
from gender_networks.tokenization import tokenize_prompt


class FakeTokenizer:
    def __init__(self, drop_mask: bool = False) -> None:
        self.drop_mask = drop_mask
        self.kwargs: dict[str, Any] = {}

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.kwargs = kwargs
        ids = [int(part) for part in text.split()]
        mask = [1] * (len(ids) - 1 if self.drop_mask else len(ids))
        return {"input_ids": ids, "attention_mask": mask}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok-{token_id}" for token_id in token_ids]


def test_tokenize_prompt_returns_plain_lists_without_padding() -> None:
    tokenizer = FakeTokenizer()

    prompt = tokenize_prompt(tokenizer, "10 20 30", TokenizationConfig(add_special_tokens=False))

    assert prompt.input_ids == [10, 20, 30]
    assert prompt.attention_mask == [1, 1, 1]
    assert prompt.tokens == ["tok-10", "tok-20", "tok-30"]
    assert tokenizer.kwargs["padding"] is False
    assert tokenizer.kwargs["add_special_tokens"] is False


def test_tokenize_prompt_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        tokenize_prompt(FakeTokenizer(drop_mask=True), "1 2", TokenizationConfig())
