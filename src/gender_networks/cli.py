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
    "corpus-rebuild": (
        "gender_networks.corpus_rebuild",
        "Reconstrói o corpus a partir das revisões de data/corpus/manifest.csv",
    ),
    "sample": ("gender_networks.sampling", "Monta a amostra híbrida de ocorrências"),
    "extract": ("gender_networks.extract", "Extrai as representações do modelo"),
    "knn": ("gender_networks.knn", "Calcula candidatos e vizinhos k-NN"),
    "metrics": ("gender_networks.metrics", "Métricas de rede e comunidades"),
    "analyze": ("gender_networks.analysis", "Análises das perguntas P1–P4"),
    "report": ("gender_networks.report", "Gera figuras, tabelas e números do relatório"),
    "lens": ("gender_networks.vocab_lens", "Extensão: vizinhos no vocabulário"),
}
# Stages the ``all`` command runs, in order. ``lens`` (optional extension) and ``corpus-rebuild``
# are not part of it and have to be run by name.
PIPELINE = ["corpus", "sample", "extract", "knn", "metrics", "analyze", "report"]


def _common_options(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Options accepted both before and after the stage name.

    Sub-parsers use SUPPRESS defaults so that a value given before the stage is not reset.

    Args:
        parser: Main parser or a stage sub-parser; ``--config``, ``--root`` and ``--log-level``
            are added to it.
        suppress: True for sub-parsers: an option that is not given leaves no attribute, so
            the value parsed by the main parser survives.
    """

    def default(value: object) -> object:
        """The default of an option: ``value``, or ``argparse.SUPPRESS`` in a sub-parser.

        Args:
            value: Default used by the main parser.

        Returns:
            ``value`` when ``suppress`` is false, ``argparse.SUPPRESS`` otherwise.
        """

        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--config", type=Path, default=default(Path("configs/experiment.yaml")))
    parser.add_argument("--root", type=Path, default=default(Path(".")), help="Raiz do repositório")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default=default("INFO")
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``gender-networks`` parser: one sub-command per stage, plus ``all``.

    Every stage accepts the common options and ``--force``; ``extract`` adds ``--verify``,
    ``lens`` adds ``--device``, and ``all`` adds ``--force-corpus``.

    Returns:
        The parser; the chosen sub-command is stored in ``args.stage``.
    """

    parser = argparse.ArgumentParser(
        prog="gender-networks",
        description="Da rede lexical à rede contextual: pipeline do experimento.",
    )
    _common_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, (_, help_text) in STAGES.items():
        stage = sub.add_parser(name, help=help_text)
        _common_options(stage, suppress=True)
        stage.add_argument("--force", action="store_true", help="Refaz a etapa mesmo com cache")
        if name == "extract":
            stage.add_argument(
                "--verify", action="store_true", help="Só roda as checagens no modelo real"
            )
        if name == "lens":
            stage.add_argument(
                "--device", default=None, help="cpu ou cuda (padrão: GPU quando houver)"
            )
    everything = sub.add_parser("all", help="Roda todas as etapas em ordem")
    _common_options(everything, suppress=True)
    everything.add_argument(
        "--force", action="store_true", help="Refaz as etapas da execução (não o corpus)"
    )
    everything.add_argument(
        "--force-corpus", action="store_true", help="Também baixa e limpa o corpus de novo"
    )
    return parser


def run_stage(name: str, config: Path, root: Path, force: bool, **options: object) -> None:
    """Load the configuration and run one stage.

    Args:
        name: Stage name, a key of :data:`STAGES`.
        config: YAML configuration file.
        root: Repository root every artifact path is resolved against.
        force: Redo the stage even when it is up to date.
        **options: Extra keyword arguments passed to the stage's ``run`` (``verify_only``,
            ``device``).

    Side effects:
        Imports the stage module, runs it (which reads and writes artifacts) and logs the
        elapsed time.

    Raises:
        ValueError: If the configuration is invalid (see ``settings.load_settings``).
    """

    settings = load_settings(config)
    paths = RunPaths.from_settings(settings, root)
    module = importlib.import_module(STAGES[name][0])
    started = time.time()
    LOGGER.info("Etapa %s (execução %s)", name, settings.name)
    module.run(settings, paths, force=force, **options)
    LOGGER.info("Etapa %s concluída em %.1f s", name, time.time() - started)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``gender-networks`` command.

    ``all`` runs the stages of :data:`PIPELINE` in order, each with a freshly loaded
    configuration; ``lens`` is not one of them. ``--force`` applies to every stage but the
    corpus, which is only rebuilt with ``--force-corpus``.

    Args:
        argv: Command-line arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        0 on success. Errors propagate as exceptions (argparse exits with status 2).

    Side effects:
        Configures the root logger with ``logging.basicConfig``.
    """

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.stage == "all":
        # The corpus is shared by every run and costs network requests: --force alone does not
        # rebuild it.
        for name in PIPELINE:
            force = args.force_corpus if name == "corpus" else args.force
            run_stage(name, args.config, args.root, force)
        return 0
    options: dict[str, object] = {}
    if getattr(args, "verify", False):
        options["verify_only"] = True
    if getattr(args, "device", None):
        options["device"] = args.device
    run_stage(args.stage, args.config, args.root, args.force, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
