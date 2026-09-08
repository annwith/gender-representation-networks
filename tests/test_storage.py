import json
from pathlib import Path

import torch

from gender_networks.prompts import PromptPair
from gender_networks.storage import save_alignment_report, save_occurrence
from gender_networks.tokenization import TokenDifference, TokenizedPrompt


def test_saves_occurrence_and_alignment_metadata(tmp_path: Path) -> None:
    pair = PromptPair("pair/1", "a", "b", {"source": "test"})
    tokenized_a = TokenizedPrompt([1], [1], ["a"])
    tokenized_b = TokenizedPrompt([2], [1], ["b"])
    differences = [TokenDifference(0, 1, "a", 2, "b")]

    report_path = save_alignment_report(
        tmp_path, pair, tokenized_a, tokenized_b, differences
    )
    tensor_path, metadata_path = save_occurrence(
        tmp_path,
        pair,
        "a",
        pair.prompt_a,
        tokenized_a,
        {0: torch.ones(1, 2)},
        {"name_or_path": "test/model"},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = torch.load(tensor_path, weights_only=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert report["differences"][0]["position"] == 0
    assert payload["hidden_states"][0].shape == (1, 2)
    assert metadata["pair_metadata"] == {"source": "test"}

