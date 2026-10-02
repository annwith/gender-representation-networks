"""Figures of the feasibility study for the technical report (vector PDF, print friendly).

The numbers are copied by hand from the outputs kept next to this script, so the figures depend
neither on the pipeline nor on the 7 MB article dump (``wiki_articles.jsonl``, git-ignored):

- ``sim_<model>.txt`` (``simulate_sample.py``): the ``pool:`` line gives tokens per word, and
  the block ``=== f_max=50, amostragem natural`` gives the share of whole-word vertices, the
  content words with f >= 10 and >= 3 occurrences in >= 3 themes, and the target words with
  >= 10 occurrences in >= 2 themes.
- ``hyb_<model>.txt`` (``sim_hybrid.py``): the same two counts for the hybrid design.

The Qwen3 rows come from the ``Qwen/Qwen3-4B`` tokenizer, whose vocabulary is identical to the
one of ``Qwen/Qwen3-4B-Base`` (151,669 ids), the model used in the experiment.

Usage::

    .venv/bin/python scripts/feasibility/plot_feasibility.py [--out report/relatorio/figuras]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "report" / "relatorio" / "figuras"


@dataclass(frozen=True)
class TokenizerResult:
    """Feasibility numbers of one tokenizer (see the module docstring for the sources)."""

    label: str
    source: str
    tokens_per_word: float
    whole_word_pct: int  # share of vertices that are whole words, natural sample, f_max = 50
    content_natural: int  # content words, f >= 10, >= 3 occurrences in >= 3 themes
    content_hybrid: int
    targets_natural: int  # target words with >= 10 occurrences in >= 2 themes
    targets_hybrid: int
    targets_single_token: int  # candidate target words that are a single token


RESULTS = [
    TokenizerResult("Qwen3-4B", "Qwen3-4B", 1.78, 28, 4, 169, 0, 3, 12),
    TokenizerResult("Tucano-2b4", "Tucano-2b4", 1.33, 54, 5, 172, 0, 7, 16),
    TokenizerResult("GPT-2 small PT", "gpt2-small-portuguese", 1.31, 72, 9, 173, 0, 7, 16),
]

# Colors from the reference palette of the dataviz guide: one blue hue in two steps (validated as
# an ordinal pair on white: monotone lightness, light end at 2.11:1), plus the muted gray used
# for de-emphasis. Dark versus light also survives grayscale printing; the light bars carry a
# 45-degree hatch as a second channel.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_DARK = "#184f95"  # blue step 600
BLUE_LIGHT = "#86b6ef"  # blue step 250
BLUE_HATCH = "#5598e7"  # blue step 350, tone-on-tone hatch over the light step
GRAY_BAR = MUTED

WIDTH_IN = 6.3  # a bit less than the text width of the report (about 6.8 in)


def decimal_br(value: float, digits: int = 2) -> str:
    """Portuguese decimal comma (1.78 -> '1,78')."""

    return f"{value:.{digits}f}".replace(".", ",")


def apply_style() -> None:
    """One quiet style for every figure: sans text, hairline axes, embedded TrueType fonts."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": INK_SECONDARY,
            "xtick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 3,
            "ytick.major.pad": 4,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def prepare_axis(ax: Axes, xmax: float, title: str, formatter: FuncFormatter) -> None:
    """Horizontal-bar axis: vertical hairline grid behind the bars, only the baseline spine."""

    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.set_title(title, loc="left", color=INK, pad=6)


def label_bar(ax: Axes, x: float, y: float, text: str, xmax: float) -> None:
    """Value label just past the bar tip, in text ink (never in the series color)."""

    ax.text(x + xmax * 0.012, y, text, va="center", ha="left", color=INK, fontsize=8)


def tokenizer_figure(results: list[TokenizerResult]) -> Figure:
    """Tokens per word and whole-word share: Qwen3 (the chosen model) against the others.

    Emphasis form: the chosen model in the accent blue, the alternatives in the muted gray.
    """

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.8), sharey=True, layout="constrained")
    labels = [r.label for r in results]
    ys = list(range(len(results)))[::-1]
    colors = [BLUE_DARK if r.label.startswith("Qwen3") else GRAY_BAR for r in results]
    height = 0.55

    ax = axes[0]
    xmax = 2.0
    prepare_axis(
        ax,
        xmax,
        "Tokens por palavra",
        FuncFormatter(lambda v, _: decimal_br(v, 1) if v else "0"),
    )
    ax.barh(ys, [r.tokens_per_word for r in results], height=height, color=colors, linewidth=0)
    for y, r in zip(ys, results, strict=True):
        label_bar(ax, r.tokens_per_word, y, decimal_br(r.tokens_per_word), xmax)
    ax.set_yticks(ys, labels)

    ax = axes[1]
    xmax = 100.0
    prepare_axis(
        ax,
        xmax,
        "Vértices que são palavras inteiras (%)",
        FuncFormatter(lambda v, _: f"{v:.0f}"),
    )
    ax.barh(ys, [r.whole_word_pct for r in results], height=height, color=colors, linewidth=0)
    for y, r in zip(ys, results, strict=True):
        label_bar(ax, r.whole_word_pct, y, f"{r.whole_word_pct}%", xmax)

    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06)
    return fig


def sampling_figure(results: list[TokenizerResult]) -> Figure:
    """Natural sampling versus the hybrid design, for the two counts that decide P4.

    One hue in two ordinal steps (natural = light, hybrid = dark); the light bars also carry a
    45-degree hatch, so the pair survives grayscale printing without relying on color.
    """

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), sharey=True, layout="constrained")
    labels = [r.label for r in results]
    centers = [float(y) for y in range(len(results))][::-1]
    height = 0.34
    offset = height / 2 + 0.02  # a small white gap between the two bars of a group

    def pair(
        ax: Axes,
        natural: list[int],
        hybrid: list[int],
        xmax: float,
        texts: tuple[list[str], list[str]],
    ) -> None:
        y_nat = [c + offset for c in centers]
        y_hyb = [c - offset for c in centers]
        ax.barh(
            y_nat,
            natural,
            height=height,
            color=BLUE_LIGHT,
            hatch="////",
            hatchcolor=BLUE_HATCH,
            linewidth=0,
        )
        ax.barh(y_hyb, hybrid, height=height, color=BLUE_DARK, linewidth=0)
        for y, value, text in zip(y_nat, natural, texts[0], strict=True):
            label_bar(ax, value, y, text, xmax)
        for y, value, text in zip(y_hyb, hybrid, texts[1], strict=True):
            label_bar(ax, value, y, text, xmax)

    ax = axes[0]
    xmax = 200.0
    prepare_axis(
        ax,
        xmax,
        "Palavras de conteúdo com f ≥ 10\ne ≥ 3 ocorrências em ≥ 3 temas",
        FuncFormatter(lambda v, _: f"{v:.0f}"),
    )
    natural = [r.content_natural for r in results]
    hybrid = [r.content_hybrid for r in results]
    pair(ax, natural, hybrid, xmax, ([str(v) for v in natural], [str(v) for v in hybrid]))
    ax.set_yticks(centers, labels)

    ax = axes[1]
    xmax = 16.0
    prepare_axis(
        ax,
        xmax,
        "Palavras-alvo com ≥ 10 ocorrências\nem ≥ 2 temas (de quantas são 1 token)",
        FuncFormatter(lambda v, _: f"{v:.0f}"),
    )
    natural = [r.targets_natural for r in results]
    hybrid = [r.targets_hybrid for r in results]
    texts = (
        [f"{r.targets_natural} de {r.targets_single_token}" for r in results],
        [f"{r.targets_hybrid} de {r.targets_single_token}" for r in results],
    )
    pair(ax, natural, hybrid, xmax, texts)

    handles = [
        Patch(
            facecolor=BLUE_LIGHT,
            hatch="////",
            edgecolor=BLUE_HATCH,
            linewidth=0,
            label="Amostragem natural (parágrafos inteiros)",
        ),
        Patch(facecolor=BLUE_DARK, linewidth=0, label="Desenho híbrido (núcleo + estratos)"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handleheight=0.9,
        columnspacing=1.8,
        labelcolor=INK_SECONDARY,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06)
    return fig


def save(fig: Figure, path: Path) -> None:
    """Vector PDF without a creation date, so reruns give byte-identical files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"figura gravada: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Pasta das figuras")
    args = parser.parse_args(argv)
    apply_style()
    save(tokenizer_figure(RESULTS), args.out / "viabilidade_tokenizers.pdf")
    save(sampling_figure(RESULTS), args.out / "viabilidade_amostragem.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
