"""Command-line interface for the pilot extraction pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from gender_networks.config import load_config
from gender_networks.modeling import load_model_and_tokenizer
from gender_networks.pipeline import run_extraction
from gender_networks.prompts import read_prompt_pairs


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Extract selected hidden states for counterfactual prompt pairs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/activations"))
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line extraction workflow."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(args.config)
    pairs = read_prompt_pairs(args.prompts)
    bundle = load_model_and_tokenizer(config)
    run_extraction(config, pairs, bundle, args.output_dir)
    return 0

