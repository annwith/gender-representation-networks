from types import SimpleNamespace

import torch
from torch import nn

from gender_networks.modeling import extract_hidden_states
from gender_networks.tokenization import TokenizedPrompt


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 3)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        del kwargs
        embedded = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(embedded, embedded + 1, embedded + 2))


def test_extracts_selected_states_without_batch_dimension() -> None:
    model = TinyModel()
    tokenized = TokenizedPrompt([1, 2], [1, 1], ["one", "two"])

    states = extract_hidden_states(model, tokenized, (0, -1))  # type: ignore[arg-type]

    assert set(states) == {0, -1}
    assert states[0].shape == (2, 3)
    assert torch.equal(states[-1], states[0] + 2)
    assert states[-1].device.type == "cpu"

