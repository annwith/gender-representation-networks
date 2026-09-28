"""Command-line interface: one subcommand per stage of the experiment.

Usage::

    uv run gender-networks <stage> [--config configs/experiment.yaml] [--force]

Each stage reads the artifacts of the previous one from disk (see :mod:`artifacts`), so stages
can be rerun independently. ``all`` runs every stage in order.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from gender_networks.artifacts import RunPaths
from gender_networks.settings import load_settings

LOGGER = logging.getLogger(__name__)

# stage name -> (module, help). Modules are imported lazily so that, for example, the corpus
# stage does not import torch.
STAGES: dict[str, tuple[str, str]] = {
    "corpus": ("gender_networks.corpus", "Baixa e limpa os artigos da Wikipédia"),
    "sample": ("gender_networks.sampling", "Monta a amostra híbrida de ocorrências"),
    "extract": ("gender_networks.extract", "Extrai as representações do modelo"),
    "knn": ("gender_networks.knn", "Calcula candidatos e vizinhos k-NN"),
    "metrics": ("gender_networks.metrics", "Métricas de rede e comunidades"),
    "analyze": ("gender_networks.analysis", "Análises das perguntas P1–P4"),
    "report": ("gender_networks.report", "Gera figuras, tabelas e números do relatório"),
    "lens": ("gender_networks.vocab_lens", "Extensão: vizinhos no vocabulário"),
}
PIPELINE = ["corpus", "sample", "extract", "knn", "metrics", "analyze", "report"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gender-networks",
        description="Da rede lexical à rede contextual: pipeline do experimento.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."), help="Raiz do repositório")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, (_, help_text) in STAGES.items():
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--force", action="store_true", help="Refaz a etapa mesmo com cache")
        if name == "extract":
            stage.add_argument(
                "--verify", action="store_true", help="Só roda as checagens no modelo real"
            )
    everything = sub.add_parser("all", help="Roda todas as etapas em ordem")
    everything.add_argument("--force", action="store_true")
    return parser


def run_stage(name: str, config: Path, root: Path, force: bool, **options: object) -> None:
    settings = load_settings(config)
    paths = RunPaths.from_settings(settings, root)
    module = importlib.import_module(STAGES[name][0])
    started = time.time()
    LOGGER.info("Etapa %s (execução %s)", name, settings.name)
    module.run(settings, paths, force=force, **options)
    LOGGER.info("Etapa %s concluída em %.1f s", name, time.time() - started)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.stage == "all":
        for name in PIPELINE:
            run_stage(name, args.config, args.root, args.force)
        return 0
    options = {"verify_only": True} if getattr(args, "verify", False) else {}
    run_stage(args.stage, args.config, args.root, args.force, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
