from pathlib import Path

import pytest

from gender_networks.prompts import read_prompt_pairs


def test_reads_prompts_and_extra_metadata(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "pair_id,prompt_a,prompt_b,source\np1,ela chegou,ele chegou,synthetic\n",
        encoding="utf-8",
    )

    pairs = read_prompt_pairs(path)

    assert pairs[0].prompt_a == "ela chegou"
    assert pairs[0].metadata == {"source": "synthetic"}


def test_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "pair_id,prompt_a,prompt_b\np1,a,b\np1,c,d\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate pair_id"):
        read_prompt_pairs(path)

