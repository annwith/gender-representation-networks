from typing import Any

from gender_networks.config import TokenizationConfig
from gender_networks.tokenization import compare_positionally, tokenize_prompt


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        del kwargs
        ids = [int(part) for part in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"tok-{token_id}" for token_id in token_ids]


def test_reports_mismatch_and_missing_position() -> None:
    tokenizer = FakeTokenizer()
    config = TokenizationConfig()
    prompt_a = tokenize_prompt(tokenizer, "10 20 30", config)
    prompt_b = tokenize_prompt(tokenizer, "10 99", config)

    differences = compare_positionally(prompt_a, prompt_b)

    assert [difference.position for difference in differences] == [1, 2]
    assert differences[0].token_a == "tok-20"
    assert differences[0].token_b == "tok-99"
    assert differences[1].token_id_b is None

