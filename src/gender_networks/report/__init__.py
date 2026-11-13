"""Stage ``report``: figures, tables and numbers of the technical report (plan stage 6).

Reads whatever the earlier stages left on disk and writes, under ``paths.report_dir``:

- ``figuras/<name>.pdf``: vector figures in the single style of :mod:`gender_networks.plots`;
- ``tabelas/<name>.tex``: only the ``tabular`` environment (booktabs, Portuguese headers);
  caption and label live in the report (``\\tabelaopcional``);
- ``tabelas/numeros.tex``: ``\\newcommand`` macros with the numbers cited in the text. The file is
  ``\\input`` *before* the ``\\providecommand`` fallbacks of ``relatorio.tex``, so a macro that
  is not written here prints *[pendente]*.

The names and the content of each item follow the captions in ``report/relatorio/secoes``
(see :data:`ITEMS`; each item is built by a function of :mod:`gender_networks.report.sections`,
one module per section of the report). Every item is optional: when its inputs are missing it
is skipped (and a stale file of the same name from an older run is removed, so the report never
mixes runs); the reason is logged and recorded in ``<run>/report/_manifest.json``. An item that
fails with an error does not stop the others; the stage raises at the end, listing them.

Numbers are plain pt-BR text (``15.104``, ``0,83``), never ``\\num``: the report does not load
siunitx. Characters that pdflatex cannot typeset with T1 fonts (CJK, emoji, replacement chars of
byte-fragment tokens) are written as ``[U+XXXX]``.

Freshness: the stage always regenerates everything. It only reads small tables and a few
neighbour files (seconds at the size of the main run), and checking every input of every item
would cost more code than it saves. When ``latexmk`` exists and ``report_dir`` holds
``relatorio.tex`` (the real report, not the mini rehearsal), the document is compiled at the end;
a compilation failure is logged and recorded but does not fail the stage.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gender_networks import plots
from gender_networks.artifacts import (
    MANIFEST,
    RunPaths,
    ensure_dir,
    write_manifest,
)
from gender_networks.report.context import STAGE, Context, Skip
from gender_networks.report.numbers import MACROS, STAGE_MANIFESTS, write_numbers
from gender_networks.report.sections.corpus import (
    figure_corpus_descartes,
    figure_corpus_paragrafos,
    table_corpus_temas,
)
from gender_networks.report.sections.drawings import figure_p4_ego_banco, figure_subgrafo_camadas
from gender_networks.report.sections.extraction import (
    figure_extracao_camadas,
    figure_extracao_dimensoes,
    table_extracao_diagnosticos,
    table_extracao_verificacao,
)
from gender_networks.report.sections.knn import table_knn_empates
from gender_networks.report.sections.lens import figure_lente_ranking
from gender_networks.report.sections.metrics import figure_graus_camadas, table_metricas_globais
from gender_networks.report.sections.p1 import (
    figure_p1_dominancia,
    figure_p1_jaccard_faixas,
    table_p1_hubs,
    table_p1_resumo,
)
from gender_networks.report.sections.p2 import (
    figure_p2_curvas_camadas,
    figure_p2_transicoes,
    table_p2_transicoes,
)
from gender_networks.report.sections.p3 import figure_p3_nmi_camadas, table_p3_concordancia
from gender_networks.report.sections.p4 import (
    figure_p4_comunidades_sentidos,
    figure_p4_separacao,
    table_p4_alvos,
)
from gender_networks.report.sections.robustness import figure_robustez_empates, table_robustez
from gender_networks.report.sections.sample import (
    figure_amostra_composicao,
    table_amostra_composicao,
    table_amostra_cotas,
    table_amostra_faixas,
)
from gender_networks.report.sections.versions import table_versoes
from gender_networks.report.sections.vocab import (
    figure_vocab_comunidades,
    figure_vocab_graus,
    table_vocab_comunidades,
    table_vocab_vizinhos_alvos,
)
from gender_networks.settings import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Item:
    """One generated figure or table of the report.

    Attributes:
        kind: ``figure`` (written to ``figuras/<name>.pdf``) or ``table`` (written to
            ``tabelas/<name>.tex``).
        name: File name without extension; the same name the report passes to
            ``\\figuraopcional`` or ``\\tabelaopcional``.
        build: Function that receives the :class:`Context` and returns a matplotlib figure
            or a :class:`~gender_networks.report.latex.Tabular`; it raises :class:`Skip`
            when its inputs are missing.
    """

    kind: str  # figure | table
    name: str
    build: Callable[[Context], Any]


ITEMS = [
    Item("table", "corpus_temas", table_corpus_temas),
    Item("figure", "corpus_paragrafos", figure_corpus_paragrafos),
    Item("figure", "corpus_descartes", figure_corpus_descartes),
    Item("table", "amostra_composicao", table_amostra_composicao),
    Item("figure", "amostra_composicao", figure_amostra_composicao),
    Item("table", "amostra_faixas", table_amostra_faixas),
    Item("table", "amostra_cotas", table_amostra_cotas),
    Item("table", "extracao_diagnosticos", table_extracao_diagnosticos),
    Item("table", "extracao_verificacao", table_extracao_verificacao),
    Item("figure", "extracao_camadas", figure_extracao_camadas),
    Item("figure", "extracao_dimensoes", figure_extracao_dimensoes),
    Item("table", "knn_empates", table_knn_empates),
    Item("table", "metricas_globais", table_metricas_globais),
    Item("figure", "graus_camadas", figure_graus_camadas),
    Item("figure", "subgrafo_camadas", figure_subgrafo_camadas),
    Item("figure", "p1_jaccard_faixas", figure_p1_jaccard_faixas),
    Item("figure", "p1_dominancia", figure_p1_dominancia),
    Item("table", "p1_resumo", table_p1_resumo),
    Item("table", "p1_hubs", table_p1_hubs),
    Item("figure", "p2_transicoes", figure_p2_transicoes),
    Item("table", "p2_transicoes", table_p2_transicoes),
    Item("figure", "p2_curvas_camadas", figure_p2_curvas_camadas),
    Item("figure", "p3_nmi_camadas", figure_p3_nmi_camadas),
    Item("table", "p3_concordancia", table_p3_concordancia),
    Item("figure", "p4_separacao", figure_p4_separacao),
    Item("table", "p4_alvos", table_p4_alvos),
    Item("figure", "p4_ego_banco", figure_p4_ego_banco),
    Item("figure", "p4_comunidades_sentidos", figure_p4_comunidades_sentidos),
    Item("table", "robustez", table_robustez),
    Item("figure", "robustez_empates", figure_robustez_empates),
    Item("figure", "vocab_graus", figure_vocab_graus),
    Item("figure", "vocab_comunidades", figure_vocab_comunidades),
    Item("table", "vocab_comunidades", table_vocab_comunidades),
    Item("table", "vocab_vizinhos_alvos", table_vocab_vizinhos_alvos),
    Item("figure", "lente_ranking", figure_lente_ranking),
    Item("table", "versoes", table_versoes),
]
# Every file the report expects (\figuraopcional / \tabelaopcional in report/relatorio/secoes).
FIGURES = [item.name for item in ITEMS if item.kind == "figure"]
TABLES = [item.name for item in ITEMS if item.kind == "table"]


def output_path(paths: RunPaths, item: Item) -> Path:
    """File an item is written to.

    Args:
        paths: Paths of the run.
        item: The item.

    Returns:
        ``figuras/<name>.pdf`` for a figure, ``tabelas/<name>.tex`` for a table.
    """

    if item.kind == "figure":
        return paths.figures_dir / f"{item.name}.pdf"
    return paths.tables_dir / f"{item.name}.tex"


def produce(ctx: Context, item: Item) -> Path:
    """Build one item and write it; raises :class:`Skip` when its inputs are missing.

    Args:
        ctx: Context of the run.
        item: The item to build.

    Returns:
        Path of the written file.

    Side effects:
        Writes the PDF (through ``plots.save``, which closes the figure) or the ``.tex``
        fragment.
    """

    path = output_path(ctx.paths, item)
    result = item.build(ctx)
    ensure_dir(path.parent)
    if item.kind == "figure":
        plots.save(result, path)
    else:
        path.write_text(result.render(), encoding="utf-8")
    return path


def compile_report(report_dir: Path) -> dict[str, Any]:
    """``latexmk -pdf relatorio.tex`` when both exist; never raises.

    Args:
        report_dir: Directory that holds ``relatorio.tex``.

    Returns:
        ``{"compiled": bool, ...}`` with the return code and the time, or the reason the
        compilation was not attempted.

    Side effects:
        Runs ``latexmk`` in ``report_dir`` (10-minute limit), which writes the PDF and the
        auxiliary files there.
    """

    latexmk = shutil.which("latexmk")
    if latexmk is None or not (report_dir / "relatorio.tex").exists():
        return {"compiled": False, "reason": "latexmk ou relatorio.tex ausente"}
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "relatorio.tex"],
            cwd=report_dir,
            capture_output=True,
            text=True,
            # pdfTeX writes 8-bit (Latin-1) bytes for accented words in its messages.
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"compiled": False, "reason": str(error)}
    ok = result.returncode == 0
    if not ok:
        LOGGER.warning(
            "latexmk falhou (código %d); veja %s", result.returncode, report_dir / "relatorio.log"
        )
    return {
        "compiled": ok,
        "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 1),
    }


def run(settings: Settings, paths: RunPaths, force: bool = False, **_: object) -> None:
    """Write every figure and table whose inputs exist, plus numeros.tex (always regenerated).

    Args:
        settings: Run configuration.
        paths: Paths of the run.
        force: Ignored: the stage always regenerates everything.
        **_: Ignored; accepted so every stage has the same call.

    Raises:
        RuntimeError: At the end, when one or more items failed with an error (the other
            items and the manifest are still written).

    Side effects:
        Sets the global matplotlib style; writes the items of :data:`ITEMS` under
        ``report_dir``, removes the file of every skipped or failed item, writes
        ``tabelas/numeros.tex``, compiles the report when ``latexmk`` and ``relatorio.tex``
        exist, and writes ``<run>/report/_manifest.json`` with the status of every item.
    """

    del force  # the stage is cheap and always regenerates everything
    started = time.time()
    plots.apply_style()
    ctx = Context(settings, paths)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.tables_dir)
    status: dict[str, dict[str, str]] = {"figure": {}, "table": {}}
    failed: list[str] = []
    for item in ITEMS:
        key = f"{item.kind} {item.name}"
        try:
            produce(ctx, item)
        except Skip as reason:
            status[item.kind][item.name] = f"pulada: {reason}"
            stale = output_path(paths, item)
            if stale.exists():
                stale.unlink()
                LOGGER.info("%s antiga removida (%s)", stale, reason)
            LOGGER.info("Pulada %s: %s", key, reason)
        except Exception as error:  # noqa: BLE001 - one broken item must not stop the others
            LOGGER.exception("Falhou %s", key)
            status[item.kind][item.name] = f"erro: {type(error).__name__}: {error}"
            failed.append(key)
            output_path(paths, item).unlink(missing_ok=True)
        else:
            status[item.kind][item.name] = "gerada"
    numbers = write_numbers(ctx, paths.tables_dir / "numeros.tex")
    compiled = compile_report(paths.report_dir)
    generated = sum(v == "gerada" for kind in status.values() for v in kind.values())
    LOGGER.info(
        "Relatório: %d de %d itens gerados, %d macros em numeros.tex (%s)",
        generated,
        len(ITEMS),
        len(numbers),
        paths.report_dir,
    )
    manifest_inputs = sorted(
        {path for path in ctx.read if path.name == MANIFEST}
        | {p for p in (paths.manifest(d) for _, d in STAGE_MANIFESTS if d) if p.exists()}
    )
    write_manifest(
        paths.stage_dir(STAGE),
        STAGE,
        settings,
        started,
        extra={
            "report_dir": str(paths.report_dir),
            "figures": status["figure"],
            "tables": status["table"],
            "macros": sorted(numbers),
            "missing_macros": [m for m in MACROS if m not in numbers],
            "latex": compiled,
            "failed": failed,
            "files_read": sorted(str(p) for p in ctx.read),
        },
        root=paths.root,
        inputs=manifest_inputs,
    )
    if failed:
        raise RuntimeError(f"Itens do relatório com erro: {', '.join(failed)}")
