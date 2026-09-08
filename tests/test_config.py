from pathlib import Path

import pytest

from gender_networks.config import load_config


def test_load_config() -> None:
    config = load_config(Path("configs/model.yaml"))

    assert config.model.name_or_path == "Qwen/Qwen3-4B"
    assert config.extraction.hidden_state_indices == (0, -1)
    assert config.tokenization.truncation is False


def test_rejects_truncation_without_max_length(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
model:
  name_or_path: test/model
tokenization:
  truncation: true
extraction:
  hidden_state_indices: [0]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_length"):
        load_config(path)

