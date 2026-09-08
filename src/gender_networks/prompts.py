"""CSV input handling for counterfactual prompt pairs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptPair:
    """One counterfactual pair and any user-provided metadata columns."""

    pair_id: str
    prompt_a: str
    prompt_b: str
    metadata: dict[str, str]


REQUIRED_COLUMNS = {"pair_id", "prompt_a", "prompt_b"}


def read_prompt_pairs(path: str | Path) -> list[PromptPair]:
    """Read and validate counterfactual prompt pairs from a UTF-8 CSV file."""

    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        pairs: list[PromptPair] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            pair_id = (row.get("pair_id") or "").strip()
            prompt_a = row.get("prompt_a") or ""
            prompt_b = row.get("prompt_b") or ""
            if not pair_id:
                raise ValueError(f"CSV line {line_number}: pair_id is empty")
            if pair_id in seen_ids:
                raise ValueError(f"CSV line {line_number}: duplicate pair_id '{pair_id}'")
            if not prompt_a or not prompt_b:
                raise ValueError(f"CSV line {line_number}: prompts must not be empty")
            seen_ids.add(pair_id)
            metadata = {
                key: value or ""
                for key, value in row.items()
                if key not in REQUIRED_COLUMNS and key is not None
            }
            pairs.append(PromptPair(pair_id, prompt_a, prompt_b, metadata))

    if not pairs:
        raise ValueError("CSV contains no prompt pairs")
    return pairs

