"""Figure style and figure builders of the ``report`` stage (vector PDF, print friendly).

One style for every generated figure, the same one as
``scripts/feasibility/plot_feasibility.py``: DejaVu Sans at 8-8.5 pt, hairline axes and grid,
embedded TrueType fonts, figures drawn at their final width (at most 6.3 in, a bit less than the
text width of the report) so that ``\\figuraopcional`` never rescales the fonts.

Colors come from the reference palette of the dataviz guide and every encoding has a second,
color-free channel, so the figures stay readable in black and white:

- representations (``lex``, ``L01``, ``L18``, ``L36``) are ordered: the lexical network is the
  muted gray and the blocks are an ordinal blue ramp (light = shallow, dark = deep), each with its
  own marker;
- nominal groups (token categories, strata, P4 groups, script classes, themes) take the
  categorical slots in a fixed order (the color follows the entity, never its rank), each with
  its own marker or hatch;
- text (values, labels, legends) is always in the ink colors, never in a series color.

The builders take tidy data already read by :mod:`gender_networks.report` and return a
:class:`~matplotlib.figure.Figure`; :func:`save` writes it without a creation date, so reruns give
byte-identical files. Every label is in Portuguese and numbers use the decimal comma.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter  # noqa: E402

# --------------------------------------------------------------------------------------------
# palette (dataviz reference palette, light mode: the report is printed on white)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FAINT = "#d6d5ce"  # inactive vertices of the fixed-layout drawings
SURFACE = "white"

# Categorical slots in their validated order (blue, orange, aqua, yellow, magenta, green,
# violet, red). Slots 3-5 sit below 3:1 on white, so they always come with a marker or hatch
# and a legend (the relief rule of the guide).
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCHES = ["", "////", "....", "\\\\\\\\", "xxxx", "----", "++++", "oo"]
# Blue ramp, steps 250..700: the ordinal range that keeps the lightest step above 2:1 on white.
BLUE_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
WIDTH_IN = 6.3

# --------------------------------------------------------------------------------------------
# Portuguese labels shared with the tables

REP_LABELS = {
    "lex": "lex",
    "L01": "L01",
    "L18": "L18",
    "L36": "L36",
    "L36n": "L36n",
    "vocab": "vocabulário",
}
CATEGORY_ORDER = ["whole_word", "word_start", "continuation", "punctuation", "number"]
CATEGORY_LABELS = {
    "whole_word": "palavra inteira",
    "word_start": "início de palavra",
    "continuation": "continuação",
    "punctuation": "pontuação",
    "number": "número",
    "whitespace": "espaço",
    "all": "todos",
}
STRATUM_ORDER = ["core", "target", "control", "multitheme"]
STRATUM_LABELS = {
    "core": "núcleo",
    "target": "alvo",
    "control": "controle",
    "multitheme": "multitema",
    "all": "todos",
}
GROUP_ORDER = ["target", "control", "multitheme", "function"]
GROUP_LABELS = {
    "target": "alvos",
    "control": "controles",
    "multitheme": "multitema",
    "function": "funcionais",
    "other": "outros",
}
SCRIPT_ORDER = [
    "latin",
    "cjk",
    "mixed",
    "punctuation",
    "other_script",
    "byte_fragment",
    "whitespace",
    "digits",
    "special",
]
SCRIPT_LABELS = {
    "latin": "latina",
    "cjk": "chinês/japonês/coreano",
    "mixed": "código/misto",
    "punctuation": "pontuação",
    "other_script": "outras escritas",
    "byte_fragment": "pedaços de bytes",
    "whitespace": "espaços",
    "digits": "dígitos",
    "special": "especiais",
    "other": "outras",
}
THEME_LABELS = {
    "fisica": "Física",
    "biologia": "Biologia",
    "economia": "Economia",
    "politica": "Política",
    "computacao": "Computação",
    "musica": "Música",
    "esporte": "Esporte",
    "geografia": "Geografia",
}
REASON_ORDER = ["multi_theme", "biography", "disambiguation", "no_paragraphs", "duplicate"]
REASON_LABELS = {
    "multi_theme": "multitema",
    "biography": "biografia",
    "disambiguation": "desambiguação",
    "no_paragraphs": "sem parágrafos",
    "duplicate": "duplicada",
}
LABEL_LABELS = {
    "token_id": "token",
    "theme": "tema",
    "pageid": "artigo",
    "paragraph_id": "parágrafo",
    "sentence_id": "sentença",
    "next_token_id": "próximo token",
    "pred_next": "token previsto",
    "pos_bucket": "faixa de posição",
    "stratum": "estrato",
    "token_category": "categoria",
}
BAND_ORDER = ["1-2", "3-9", "10-49", "50", "50+"]


def label(mapping: Mapping[str, str], key: Any) -> str:
    """Portuguese label of a code, falling back to the code itself."""

    return mapping.get(str(key), str(key))


def theme_label(theme: str) -> str:
    return THEME_LABELS.get(theme, theme.capitalize())


def band_label(band: str) -> str:
    """``10-49`` -> ``10–49`` (en dash), ``50`` -> ``50``."""

    return str(band).replace("-", "–")


def ordered(values: Sequence[Any], order: Sequence[str]) -> list[str]:
    """Distinct values, known ones in ``order`` first, the others sorted after them."""

    present = [str(v) for v in dict.fromkeys(values)]
    known = [v for v in order if v in present]
    return known + sorted(v for v in present if v not in known)


# --------------------------------------------------------------------------------------------
# number formatting (pt-BR)


def fmt_int(value: float | int) -> str:
    """``15104`` -> ``15.104``."""

    return f"{int(round(float(value))):,}".replace(",", ".")


def fmt_dec(value: float, digits: int = 2) -> str:
    """``1234.5`` -> ``1.234,50`` (thousands with a dot, decimal comma)."""

    text = f"{float(value):,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


class BrFormatter(ScalarFormatter):
    """Tick labels with the decimal comma (and a thousands dot for large integers)."""

    def __init__(self) -> None:
        super().__init__(useOffset=False, useMathText=False)
        self.set_scientific(False)

    def __call__(self, x: float, pos: int | None = None) -> str:
        if abs(x) >= 10_000 and float(x).is_integer():
            return ("−" if x < 0 else "") + fmt_int(abs(x))
        return super().__call__(x, pos).replace(".", ",")


def percent_formatter(digits: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{fmt_dec(v * 100, digits)}%")


# --------------------------------------------------------------------------------------------
# style and figure helpers


def apply_style() -> None:
    """One quiet style for every figure: sans text, hairline axes, embedded TrueType fonts."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.titlelocation": "left",
            "axes.titlepad": 6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.major.pad": 3,
            "ytick.major.pad": 4,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "hatch.linewidth": 0.6,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.labelcolor": INK_SECONDARY,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.unicode_minus": True,
        }
    )


def figure(
    nrows: int = 1, ncols: int = 1, height: float = 2.4, width: float = WIDTH_IN, **kwargs: Any
) -> tuple[Figure, np.ndarray]:
    """Constrained-layout figure at the final width; ``axes`` is always a 2-D array."""

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), layout="constrained", squeeze=False, **kwargs
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.08)
    return fig, axes


def prepare(ax: Axes, title: str | None = None, grid: str = "y") -> None:
    """Hairline grid behind the marks, only the baseline spines."""

    if grid in ("x", "y", "both"):
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid == "x":
        ax.spines["bottom"].set_visible(False)
    if title:
        ax.set_title(title, color=INK)


def decimal_axis(ax: Axes, which: str = "y") -> None:
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(BrFormatter())


def save(fig: Figure, path: Path) -> Path:
    """Vector PDF without a creation date, so reruns give byte-identical files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def empty_note(ax: Axes, text: str = "sem dados") -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


@dataclass(frozen=True)
class Style:
    """Color, marker and hatch of one series."""

    color: str
    marker: str = "o"
    hatch: str = ""
    label: str = ""


def rep_styles(reps: Sequence[str]) -> dict[str, Style]:
    """lex in gray, the blocks on the ordinal blue ramp (by depth), robust variants after."""

    blocks = sorted(
        (r for r in reps if r.startswith("L") and r[1:3].isdigit() and not r[3:]),
        key=lambda r: int(r[1:3]),
    )
    if len(blocks) == 1:
        ramp = [BLUE_RAMP[6]]
    else:
        ramp = [BLUE_RAMP[round(i)] for i in np.linspace(1, 9, len(blocks))]
    block_markers = ["s", "^", "D", "v", "P", "X"]
    styles: dict[str, Style] = {}
    for i, rep in enumerate(blocks):
        styles[rep] = Style(ramp[i], block_markers[i % len(block_markers)], label=rep)
    extra = iter([CATEGORICAL[1], CATEGORICAL[6], CATEGORICAL[5], CATEGORICAL[7]])
    for rep in reps:
        if rep in styles:
            continue
        if rep == "lex":
            styles[rep] = Style(MUTED, "o", label="lex")
        elif rep == "vocab":
            styles[rep] = Style(INK_SECONDARY, "o", label=REP_LABELS["vocab"])
        else:
            styles[rep] = Style(next(extra, INK_SECONDARY), "*", label=rep)
    return styles


def categorical_styles(keys: Sequence[str], labels: Mapping[str, str] | None = None) -> dict:
    """Fixed-order categorical slots; keys beyond the eighth fold into the gray."""

    styles = {}
    for i, key in enumerate(keys):
        if i < len(CATEGORICAL):
            styles[key] = Style(CATEGORICAL[i], MARKERS[i], HATCHES[i], label(labels or {}, key))
        else:
            styles[key] = Style(MUTED, "o", "", label(labels or {}, key))
    return styles


def ordinal_styles(keys: Sequence[str], labels: Mapping[str, str] | None = None) -> dict:
    """Ordered keys (frequency bands, transitions) on the blue ramp, light to dark."""

    if len(keys) == 1:
        ramp = [BLUE_RAMP[6]]
    else:
        ramp = [BLUE_RAMP[round(i)] for i in np.linspace(0, 9, len(keys))]
    return {
        key: Style(
            ramp[i], MARKERS[i % len(MARKERS)], HATCHES[i % len(HATCHES)], label(labels or {}, key)
        )
        for i, key in enumerate(keys)
    }


def plot_series(ax: Axes, x: Sequence[float], y: Sequence[float], style: Style, **kw: Any) -> None:
    ax.plot(
        x,
        y,
        color=style.color,
        marker=style.marker,
        markersize=kw.pop("markersize", 4.5),
        markeredgecolor=SURFACE,
        markeredgewidth=0.6,
        label=style.label,
        clip_on=False,
        **kw,
    )


def legend_top(fig: Figure, handles: Sequence[Any], ncol: int | None = None) -> None:
    fig.legend(
        handles=list(handles),
        loc="outside upper left",
        ncol=ncol or min(len(handles), 5),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )


def line_handle(style: Style) -> Line2D:
    return Line2D(
        [],
        [],
        color=style.color,
        marker=style.marker,
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=0.6,
        label=style.label,
    )


def patch_handle(style: Style) -> Patch:
    return Patch(
        facecolor=style.color,
        hatch=style.hatch,
        edgecolor=SURFACE,
        linewidth=0,
        label=style.label,
    )


def _hatched_bar(ax: Axes, horizontal: bool, *args: Any, style: Style, **kw: Any) -> Any:
    draw = ax.barh if horizontal else ax.bar
    bars = draw(*args, color=style.color, linewidth=0, **kw)
    if style.hatch:
        for bar in bars:
            bar.set_hatch(style.hatch)
            bar.set_edgecolor(SURFACE)  # hatch drawn in the surface color: tone-on-tone
            bar.set_linewidth(0)
    return bars


def _label_bars(ax: Axes, values: Sequence[float], ys: Sequence[float], xmax: float) -> None:
    for y, value in zip(ys, values, strict=True):
        ax.text(
            value + xmax * 0.015,
            y,
            fmt_int(value),
            va="center",
            ha="left",
            color=INK,
            fontsize=7.5,
        )


def hbar_panel(
    ax: Axes, labels: Sequence[str], values: Sequence[float], title: str, color: str = BLUE_RAMP[7]
) -> None:
    """Single-series horizontal bars (one color), value labels past the tips."""

    prepare(ax, title, grid="x")
    ys = np.arange(len(labels))[::-1]
    values = [float(v) for v in values]
    xmax = max(values, default=1.0) * 1.18 or 1.0
    ax.barh(ys, values, height=0.62, color=color, linewidth=0)
    _label_bars(ax, values, ys, xmax)
    ax.set_yticks(ys, labels)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    decimal_axis(ax, "x")


# --------------------------------------------------------------------------------------------
# corpus and sample


def corpus_paragraphs_figure(counts: pd.Series, lengths: np.ndarray) -> Figure:
    """Paragraphs per theme (bars) and the distribution of paragraph lengths in words."""

    fig, axes = figure(1, 2, height=2.5, width_ratios=[1, 1.15])
    order = counts.sort_values(ascending=False)
    hbar_panel(
        axes[0, 0], [theme_label(t) for t in order.index], order.to_numpy(), "Parágrafos por tema"
    )
    ax = axes[0, 1]
    prepare(ax, "Tamanho dos parágrafos (palavras)", grid="y")
    lengths = np.asarray(lengths, dtype=float)
    if lengths.size:
        high = float(np.quantile(lengths, 0.995))
        bins = np.linspace(lengths.min(), max(high, lengths.min() + 1), 40)
        ax.hist(
            np.clip(lengths, None, high), bins=bins, color=BLUE_RAMP[7], linewidth=0, rwidth=0.88
        )
        median = float(np.median(lengths))
        ax.axvline(median, color=INK, linewidth=0.8)
        ax.text(
            median,
            1.0,
            f" mediana {fmt_int(median)}",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            color=INK,
            fontsize=7.5,
        )
        ax.set_xlabel("palavras por parágrafo (cauda acima do quantil 99,5% agrupada)")
        ax.set_ylabel("parágrafos")
        decimal_axis(ax, "y")
        decimal_axis(ax, "x")
    else:
        empty_note(ax)
    return fig


def corpus_discards_figure(table: pd.DataFrame) -> Figure:
    """Small multiples: discarded candidate pages per theme, one panel per reason."""

    reasons = [r for r in REASON_ORDER if r in table.columns] + [
        r for r in table.columns if r not in REASON_ORDER
    ]
    themes = list(table.index)
    fig, axes = figure(1, len(reasons), height=0.35 + 0.22 * len(themes) + 0.5, sharey=True)
    ys = np.arange(len(themes))[::-1]
    xmax = max(float(table.to_numpy().max()) * 1.3, 1.0)
    for ax, reason in zip(axes[0], reasons, strict=True):
        prepare(ax, label(REASON_LABELS, reason), grid="x")
        values = table[reason].to_numpy(dtype=float)
        ax.barh(ys, values, height=0.62, color=BLUE_RAMP[7], linewidth=0)
        _label_bars(ax, values, ys, xmax)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        decimal_axis(ax, "x")
        ax.set_title(f"{label(REASON_LABELS, reason)}\n({fmt_int(values.sum())} no total)")
    axes[0, 0].set_yticks(ys, [theme_label(t) for t in themes])
    return fig


def sample_composition_figure(panels: Sequence[tuple[str, Sequence[str], Sequence[int]]]) -> Figure:
    """Vertices by stratum, theme, frequency band and token category (one panel each)."""

    n = len(panels)
    ncols = 2
    nrows = math.ceil(n / ncols)
    rows = max(len(p[1]) for p in panels)
    fig, axes = figure(nrows, ncols, height=nrows * (0.45 + 0.17 * rows) + 0.2)
    for ax, (title, labels, values) in zip(axes.flat, panels, strict=False):
        hbar_panel(ax, labels, values, title)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    return fig


# --------------------------------------------------------------------------------------------
# extraction diagnostics


def extraction_layers_figure(
    curve: pd.DataFrame, lexical: Mapping[str, float] | None, marked: Mapping[str, int]
) -> Figure:
    """Mean norm (log scale) and mean random-pair cosine per block; layer 0 = embedding."""

    fig, axes = figure(1, 2, height=2.3)
    curve = curve.sort_values("layer")
    for ax, column, title in (
        (axes[0, 0], "mean_norm", "Norma média"),
        (axes[0, 1], "mean_pair_cosine", "Cosseno médio entre pares aleatórios"),
    ):
        prepare(ax, title, grid="both")
        x = curve["layer"].to_numpy(dtype=float)
        y = pd.to_numeric(curve[column], errors="coerce").to_numpy(dtype=float)
        if lexical is not None and lexical.get(column) is not None:
            x = np.concatenate([[0.0], x])
            y = np.concatenate([[float(lexical[column])], y])
        ax.plot(
            x, y, color=BLUE_RAMP[7], linewidth=1.4, marker="o", markersize=2.6, markeredgewidth=0
        )
        for name, layer in marked.items():
            hit = np.flatnonzero(x == layer)
            if hit.size:
                ax.plot(
                    x[hit],
                    y[hit],
                    linestyle="none",
                    marker="D",
                    markersize=5.5,
                    color=INK,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.6,
                )
                ax.annotate(
                    name,
                    (x[hit[0]], y[hit[0]]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=INK,
                )
        ax.set_xlabel("bloco (0 = embedding de entrada)")
        ax.set_xlim(-0.5, max(x.max(initial=1.0), 1.0) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        if column == "mean_norm":
            ax.set_yscale("log")
        else:
            decimal_axis(ax, "y")
    return fig


def extraction_dimensions_figure(summaries: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Per representation: top coordinates over the median one, their share of the squared
    norm, and the mean norm by position bucket (relative to the representation's mean)."""

    reps = list(summaries)
    styles = rep_styles(reps)
    fig, axes = figure(1, 3, height=2.4, width_ratios=[1, 1, 1.25])
    ys = np.arange(len(reps))[::-1]

    ax = axes[0, 0]
    prepare(ax, "Maior |coordenada| média\nsobre a mediana das dimensões", grid="x")
    values = [summaries[r].get("top_dim_mean_abs_over_median_dim") or np.nan for r in reps]
    ax.barh(ys, values, height=0.62, color=[styles[r].color for r in reps], linewidth=0)
    for y, v in zip(ys, values, strict=True):
        if np.isfinite(v):
            ax.text(v * 1.08, y, fmt_dec(v, 0 if v >= 100 else 1), va="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yticks(ys, [label(REP_LABELS, r) for r in reps])
    finite = [v for v in values if np.isfinite(v) and v > 0]
    if finite:
        ax.set_xlim(min(1.0, min(finite)) * 0.8, max(finite) * 6)

    ax = axes[0, 1]
    prepare(ax, "Fração da norma² nas\ndimensões dominantes", grid="x")
    values = [summaries[r].get("top_dims_mean_share_sq_norm") or np.nan for r in reps]
    n_dims = max((len(summaries[r].get("top_dims") or []) for r in reps), default=0)
    ax.barh(ys, values, height=0.62, color=[styles[r].color for r in reps], linewidth=0)
    for y, v in zip(ys, values, strict=True):
        if np.isfinite(v):
            ax.text(v + 0.02, y, f"{fmt_dec(v * 100, 0)}%", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_yticks(ys, [])
    if n_dims:
        ax.set_xlabel(f"{n_dims} dimensões de maior |x| médio")

    ax = axes[0, 2]
    prepare(
        ax, "Norma média por faixa de posição\n(relativa à média da representação)", grid="both"
    )
    buckets = ordered(
        [b for r in reps for b in (summaries[r].get("mean_norm_by_pos_bucket") or {})],
        ["1-4", "5-16", "17-64", "65+"],
    )
    xs = np.arange(len(buckets))
    for rep in reps:
        by = summaries[rep].get("mean_norm_by_pos_bucket") or {}
        mean = summaries[rep].get("mean_norm") or np.nan
        y = [by.get(b, np.nan) / mean if mean else np.nan for b in buckets]
        plot_series(ax, xs, y, styles[rep])
    ax.set_xticks(xs, [band_label(b) for b in buckets])
    ax.set_xlabel("posição na sequência")
    decimal_axis(ax, "y")
    ax.legend(loc="best", fontsize=7, handlelength=1.4)
    return fig


# --------------------------------------------------------------------------------------------
# networks


def ccdf(degrees: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``P(K >= d)`` at each observed degree ``d > 0``, from a degree histogram."""

    order = np.argsort(degrees)
    degrees = np.asarray(degrees, dtype=float)[order]
    counts = np.asarray(counts, dtype=float)[order]
    total = counts.sum()
    if total <= 0:
        return np.zeros(0), np.zeros(0)
    tail = counts[::-1].cumsum()[::-1] / total
    keep = degrees > 0
    return degrees[keep], tail[keep]


def degree_ccdf_figure(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    k: int | None,
    styles: Mapping[str, Style],
    title: str,
    height: float = 2.6,
) -> Figure:
    """CCDF of the in-degree, log-log, one line per series; vertical line at ``k``."""

    fig, axes = figure(1, 1, height=height, width=4.4)
    ax = axes[0, 0]
    prepare(ax, title, grid="both")
    for key, (degrees, counts) in series.items():
        x, y = ccdf(degrees, counts)
        if x.size == 0:
            continue
        style = styles[key]
        ax.step(x, y, where="post", color=style.color, linewidth=1.4, label=style.label)
        every = max(1, x.size // 8)
        ax.plot(
            x[::every],
            y[::every],
            linestyle="none",
            marker=style.marker,
            color=style.color,
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
        )
    if k:
        ax.axvline(k, color=INK, linewidth=0.8)
        ax.text(
            k,
            1.0,
            f" k = {k}",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            fontsize=7.5,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grau de entrada d")
    ax.set_ylabel("P(K ≥ d)")
    handles = [line_handle(styles[key]) for key in series]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5)
    return fig


@dataclass
class GraphPanel:
    """One panel of a fixed-layout drawing: the edges of one network and its active vertices."""

    title: str
    edges: np.ndarray  # [m, 2] indices into the shared positions
    active: np.ndarray  # bool [n]: vertices touched by the panel's edges


def fixed_layout_figure(
    positions: np.ndarray,
    panels: Sequence[GraphPanel],
    focal: np.ndarray,
    focal_styles: Sequence[Style],
    focal_keys: np.ndarray,
    neighbor_keys: np.ndarray | None,
    legend_title: str,
    height: float = 4.9,
) -> Figure:
    """The same vertex set drawn with fixed positions in several networks (2 x 2 grid).

    ``focal`` (bool [n]) marks the vertices whose out-edges are drawn; ``focal_keys`` /
    ``neighbor_keys`` index into ``focal_styles`` (``-1`` = gray "outros"). Vertices that are
    not touched by a panel's edges stay as faint dots, so positions are comparable.
    """

    ncols = 2
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = figure(nrows, ncols, height=height)
    pos = np.asarray(positions, dtype=float)
    span = np.ptp(pos, axis=0)
    pad = 0.04 * np.where(span > 0, span, 1.0)
    lo, hi = pos.min(axis=0) - pad, pos.max(axis=0) + pad

    def colors_of(keys: np.ndarray) -> list[str]:
        return [focal_styles[k].color if k >= 0 else MUTED for k in keys]

    for ax, panel in zip(axes.flat, panels, strict=False):
        ax.set_title(panel.title, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        if panel.edges.size:
            segments = np.stack([pos[panel.edges[:, 0]], pos[panel.edges[:, 1]]], axis=1)
            ax.add_collection(LineCollection(segments, colors=AXIS, linewidths=0.35, zorder=1))
        inactive = ~panel.active & ~focal
        ax.scatter(pos[inactive, 0], pos[inactive, 1], s=3, color=FAINT, linewidths=0, zorder=2)
        neighbors = panel.active & ~focal
        if neighbor_keys is not None:
            idx = np.flatnonzero(neighbors)
            for key in np.unique(neighbor_keys[idx]):
                part = idx[neighbor_keys[idx] == key]
                style = focal_styles[key] if key >= 0 else Style(MUTED, "o")
                ax.scatter(
                    pos[part, 0],
                    pos[part, 1],
                    s=9,
                    color=style.color,
                    marker=style.marker,
                    linewidths=0,
                    zorder=3,
                )
        else:
            ax.scatter(
                pos[neighbors, 0],
                pos[neighbors, 1],
                s=7,
                color=INK_SECONDARY,
                linewidths=0,
                zorder=3,
            )
        idx = np.flatnonzero(focal)
        for key in np.unique(focal_keys[idx]):
            part = idx[focal_keys[idx] == key]
            style = focal_styles[key] if key >= 0 else Style(MUTED, "o")
            ax.scatter(
                pos[part, 0],
                pos[part, 1],
                s=30,
                color=style.color,
                marker=style.marker,
                edgecolors=INK,
                linewidths=0.6,
                zorder=4,
            )
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=s.marker,
            markersize=5.5,
            color=s.color,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            label=s.label,
        )
        for s in focal_styles
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5.5,
            color=MUTED,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            label="outros",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=3,
            color=FAINT,
            label="fora da vizinhança nesta rede",
        )
    )
    fig.legend(
        handles=handles,
        loc="outside upper left",
        ncol=min(len(handles), 4),
        title=legend_title,
        title_fontsize=8,
        alignment="left",
        columnspacing=1.2,
    )
    return fig


# --------------------------------------------------------------------------------------------
# P1 / P2


def jaccard_bands_figure(
    values: Mapping[str, Mapping[str, np.ndarray]],
    floors: Mapping[str, float],
    bands: Sequence[str],
) -> Figure:
    """Box plots of ``J(lex, l)`` per frequency band and layer, with the noise floor."""

    reps = list(values)
    styles = rep_styles(reps)
    fig, axes = figure(1, 1, height=2.6)
    ax = axes[0, 0]
    prepare(ax, "J(lex, ℓ) por faixa de frequência na amostra", grid="y")
    width = 0.8 / max(len(reps), 1)
    for b, band in enumerate(bands):
        for r, rep in enumerate(reps):
            data = np.asarray(values[rep].get(band, []), dtype=float)
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            x = b - 0.4 + width * (r + 0.5)
            box = ax.boxplot(
                [data],
                positions=[x],
                widths=width * 0.78,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                medianprops={"color": INK, "linewidth": 1.0},
                whiskerprops={"color": INK_SECONDARY, "linewidth": 0.6},
                capprops={"color": INK_SECONDARY, "linewidth": 0.6},
                boxprops={"linewidth": 0},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(styles[rep].color)
            ax.plot(
                [x],
                [float(np.mean(data))],
                marker=styles[rep].marker,
                color=SURFACE,
                markeredgecolor=INK,
                markeredgewidth=0.6,
                markersize=3.6,
                zorder=5,
            )
        floor = floors.get(band)
        if floor is not None and np.isfinite(floor):
            ax.plot([b - 0.45, b + 0.45], [floor, floor], color=INK, linewidth=1.1, zorder=6)
    ax.set_xticks(np.arange(len(bands)), [band_label(b) for b in bands])
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("faixa de frequência do tipo na amostra (f_t)")
    ax.set_ylabel("J")
    decimal_axis(ax, "y")
    handles = [patch_handle(styles[r]) for r in reps]
    handles.append(
        Line2D([], [], color=INK, linewidth=1.1, label="piso de ruído (J médio entre sementes lex)")
    )
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=SURFACE,
            markeredgecolor=INK,
            markersize=3.6,
            label="média",
        )
    )
    legend_top(fig, handles, ncol=len(handles))
    return fig


def dominance_figure(
    data: pd.DataFrame, reps: Sequence[str], panels: Sequence[tuple[str, str, Sequence[str]]]
) -> Figure:
    """``D`` (row 1) and normalized ``D`` (row 2) per representation, one line per group.

    ``data`` columns: ``measure`` (D | Dn), ``grouping``, ``group``, ``rep``, ``value``;
    ``panels`` lists ``(grouping, title, groups)``.
    """

    fig, axes = figure(2, len(panels), height=4.4, sharex=True, sharey="row")
    xs = np.arange(len(reps))
    for col, (grouping, title, groups) in enumerate(panels):
        if grouping == "band_sample":
            styles = ordinal_styles(groups, {g: f"f_t {band_label(g)}" for g in groups})
        else:
            styles = categorical_styles(groups, CATEGORY_LABELS)
        for row, measure in enumerate(("D", "Dn")):
            ax = axes[row, col]
            symbol = "D" if measure == "D" else "D̃ = D / teto"
            prepare(ax, f"{symbol}, por {title}", grid="both")
            for group in groups:
                part = data[
                    (data["measure"] == measure)
                    & (data["grouping"] == grouping)
                    & (data["group"] == group)
                ].set_index("rep")
                y = [float(part["value"].get(rep, np.nan)) for rep in reps]
                plot_series(ax, xs, y, styles[group])
            ax.set_ylim(-0.02, 1.05)
            decimal_axis(ax, "y")
            ax.set_xticks(xs, [label(REP_LABELS, r) for r in reps])
            if row == 0:
                ax.legend(loc="upper right", fontsize=7, handlelength=1.4)
    return fig


def transition_share_figure(
    blocks: Sequence[tuple[str, pd.DataFrame]], transitions: Sequence[str]
) -> Figure:
    """100% stacked bars: share of vertices whose largest change falls in each transition.

    Each block is ``(title, frame)`` with one row per group (index = label) and one column per
    transition. The transitions are ordered, so the segments use the ordinal ramp plus hatches.
    """

    rows = sum(len(frame) for _, frame in blocks)
    fig, axes = figure(1, len(blocks), height=0.75 + 0.2 * max(len(f) for _, f in blocks))
    styles = ordinal_styles(list(transitions), {t: t.replace("->", " → ") for t in transitions})
    for ax, (title, frame) in zip(axes[0], blocks, strict=True):
        prepare(ax, title, grid="x")
        ys = np.arange(len(frame))[::-1]
        left = np.zeros(len(frame))
        for transition in transitions:
            values = frame[transition].to_numpy(dtype=float) if transition in frame else 0 * left
            values = np.nan_to_num(values)
            _hatched_bar(ax, True, ys, values, height=0.64, left=left, style=styles[transition])
            for y, v, l0 in zip(ys, values, left, strict=True):
                if v >= 0.12:
                    ax.text(
                        l0 + v / 2,
                        y,
                        f"{fmt_dec(v * 100, 0)}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=SURFACE if transition != transitions[0] else INK,
                        bbox={"boxstyle": "round,pad=0.12", "fc": "none", "ec": "none"},
                    )
            left = left + values
        ax.set_yticks(ys, list(frame.index))
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(percent_formatter())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    del rows
    legend_top(fig, [patch_handle(styles[t]) for t in transitions])
    return fig


def layer_curves_figure(layers: pd.DataFrame, categories: Sequence[str]) -> Figure:
    """``J(l-1, l)`` and ``D(l)`` over every block, one line per token category."""

    fig, axes = figure(1, 2, height=2.6, sharex=True)
    keys = ["all", *categories]
    styles = {
        "all": Style(INK, "", label="todos"),
        **categorical_styles(categories, CATEGORY_LABELS),
    }
    for ax, column, title in (
        (axes[0, 0], "J_prev_mean", "J(ℓ−1, ℓ) entre blocos consecutivos"),
        (axes[0, 1], "D_mean", "Dominância lexical D(ℓ)"),
    ):
        prepare(ax, title, grid="both")
        for key in keys:
            part = layers[layers["token_category"] == key].sort_values("layer")
            if part.empty:
                continue
            style = styles[key]
            x = part["layer"].to_numpy(dtype=float)
            y = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=style.color,
                linewidth=1.8 if key == "all" else 1.1,
                label=style.label,
                zorder=4 if key == "all" else 3,
            )
            if style.marker:
                every = max(1, x.size // 7)
                ax.plot(
                    x[::every],
                    y[::every],
                    linestyle="none",
                    marker=style.marker,
                    color=style.color,
                    markersize=4,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.5,
                )
        ax.set_xlabel("bloco ℓ (em ℓ = 1, comparação com a rede lex por posição)")
        ax.set_ylim(-0.02, 1.02)
        decimal_axis(ax, "y")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    handles = [Line2D([], [], color=INK, linewidth=1.8, label="todos")]
    handles += [line_handle(styles[c]) for c in categories]
    legend_top(fig, handles, ncol=len(handles))
    return fig


# --------------------------------------------------------------------------------------------
# P3 / P4


def nmi_layers_figure(
    frame: pd.DataFrame, reps: Sequence[str], labels: Sequence[str], stability: Mapping[str, float]
) -> Figure:
    """Small multiples of NMI minus the permutation baseline per label; Leiden stability."""

    n = len(labels) + 1
    ncols = 3 if n <= 9 else 4
    nrows = math.ceil(n / ncols)
    fig, axes = figure(nrows, ncols, height=1.35 * nrows + 0.3, sharex=True)
    xs = np.arange(len(reps))
    values = pd.to_numeric(frame["nmi_minus_baseline"], errors="coerce")
    top = max(float(values.max()) if values.notna().any() else 0.1, 0.05) * 1.15
    bottom = min(float(values.min()) if values.notna().any() else 0.0, 0.0)
    style = Style(BLUE_RAMP[7], "o")
    flat = list(axes.flat)
    for ax, name in zip(flat, labels, strict=False):
        prepare(ax, label(LABEL_LABELS, name), grid="y")
        part = frame[frame["label"] == name].set_index("rep")
        y = [
            float(pd.to_numeric(part["nmi_minus_baseline"], errors="coerce").get(r, np.nan))
            for r in reps
        ]
        plot_series(ax, xs, y, style)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        ax.set_ylim(bottom - 0.02, top)
        decimal_axis(ax, "y")
        ax.set_xticks(xs, [label(REP_LABELS, r) for r in reps])
    ax = flat[len(labels)]
    prepare(ax, "estabilidade do Leiden", grid="y")
    y = [stability.get(r, np.nan) for r in reps]
    ax.bar(xs, y, width=0.6, color=MUTED, linewidth=0)
    for x, v in zip(xs, y, strict=True):
        if np.isfinite(v):
            ax.text(x, v + 0.02, fmt_dec(v, 2), ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.15)
    decimal_axis(ax, "y")
    ax.set_xticks(xs, [label(REP_LABELS, r) for r in reps])
    for extra in flat[len(labels) + 1 :]:
        extra.set_visible(False)
    fig.supylabel("NMI − linha de base", fontsize=8, color=INK_SECONDARY)
    return fig


def separation_figure(
    groups: pd.DataFrame,
    reps: Sequence[str],
    measures: Sequence[tuple[str, str]],
    keys: Sequence[str],
) -> Figure:
    """Same-type separation per layer: one panel per measure, one line per group."""

    ncols = 2
    nrows = math.ceil(len(measures) / ncols)
    fig, axes = figure(nrows, ncols, height=2.0 * nrows + 0.35, sharex=True)
    styles = categorical_styles(list(keys), GROUP_LABELS)
    xs = np.arange(len(reps))
    for ax, (column, title) in zip(axes.flat, measures, strict=False):
        prepare(ax, title, grid="both")
        for key in keys:
            part = groups[groups["group"] == key].set_index("rep")
            if column not in part:
                continue
            y = [float(pd.to_numeric(part[column], errors="coerce").get(r, np.nan)) for r in reps]
            plot_series(ax, xs, y, styles[key])
        decimal_axis(ax, "y")
        ax.set_xticks(xs, [label(REP_LABELS, r) for r in reps])
    for ax in list(axes.flat)[len(measures) :]:
        ax.set_visible(False)
    legend_top(fig, [line_handle(styles[k]) for k in keys], ncol=len(keys))
    return fig


# --------------------------------------------------------------------------------------------
# robustness, vocabulary, lens


def tie_variants_figure(
    frame: pd.DataFrame,
    reps: Sequence[str],
    bands: Sequence[str],
    variants: Sequence[str],
    variant_labels: Mapping[str, str],
) -> Figure:
    """Mean ``J(lex, l)`` per frequency band for each tie variant, one panel per layer."""

    fig, axes = figure(1, len(reps), height=2.6, sharey=True)
    styles = categorical_styles(list(variants), variant_labels)
    xs = np.arange(len(bands))
    for ax, rep in zip(axes[0], reps, strict=True):
        prepare(ax, f"J(lex, {label(REP_LABELS, rep)})", grid="both")
        for variant in variants:
            part = frame[(frame["rep"] == rep) & (frame["variant"] == variant)].set_index("band")
            y = [float(part["value"].get(b, np.nan)) for b in bands]
            plot_series(ax, xs, y, styles[variant])
        ax.set_xticks(xs, [band_label(b) for b in bands])
        ax.set_xlabel("faixa de f_t na amostra")
        ax.set_ylim(-0.02, 1.02)
        decimal_axis(ax, "y")
    legend_top(fig, [line_handle(styles[v]) for v in variants], ncol=min(len(variants), 5))
    return fig


def community_composition_figure(
    shares: pd.DataFrame, sizes: Sequence[int], classes: Sequence[str]
) -> Figure:
    """100% stacked bars: script-class composition of the largest communities."""

    n = len(shares)
    fig, axes = figure(1, 1, height=0.9 + 0.2 * n)
    ax = axes[0, 0]
    prepare(ax, "Composição por classe de escrita (tamanho da comunidade à direita)", grid="x")
    keys = [c for c in classes if c != "other"]
    styles = categorical_styles(keys, SCRIPT_LABELS)
    if "other" in classes:
        styles["other"] = Style(MUTED, "o", "", SCRIPT_LABELS["other"])
    ys = np.arange(n)[::-1]
    left = np.zeros(n)
    for key in classes:
        values = np.nan_to_num(shares[key].to_numpy(dtype=float))
        _hatched_bar(ax, True, ys, values, height=0.66, left=left, style=styles[key])
        left = left + values
    for y, size in zip(ys, sizes, strict=True):
        ax.text(1.015, y, fmt_int(size), va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_yticks(ys, [str(i) for i in shares.index])
    ax.set_ylabel("comunidade (por tamanho)")
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(percent_formatter())
    legend_top(fig, [patch_handle(styles[c]) for c in classes], ncol=min(len(classes), 4))
    return fig


def lens_ranks_figure(curve: pd.DataFrame, categories: Sequence[str], discrete: bool) -> Figure:
    """Median rank (log scale, rank + 1) of the own and the next token per layer and category.

    ``curve`` columns: ``x`` (layer number or position), ``xlabel``, ``group``,
    ``own_rank_median``, ``next_rank_median``.
    """

    fig, axes = figure(1, 2, height=2.6, sharey=True)
    keys = ["all", *categories]
    styles = {
        "all": Style(INK, "o", label="todos"),
        **categorical_styles(categories, CATEGORY_LABELS),
    }
    ticks = curve.drop_duplicates("x").sort_values("x")
    for ax, column, title in (
        (axes[0, 0], "own_rank_median", "Posto mediano do próprio token"),
        (axes[0, 1], "next_rank_median", "Posto mediano do próximo token"),
    ):
        prepare(ax, title, grid="both")
        for key in keys:
            part = curve[curve["group"] == key].sort_values("x")
            if part.empty:
                continue
            style = styles[key]
            x = part["x"].to_numpy(dtype=float)
            y = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=float) + 1.0
            if discrete:
                plot_series(ax, x, y, style, linewidth=1.8 if key == "all" else 1.1)
            else:
                ax.plot(x, y, color=style.color, linewidth=1.8 if key == "all" else 1.1)
                every = max(1, x.size // 7)
                ax.plot(
                    x[::every],
                    y[::every],
                    linestyle="none",
                    marker=style.marker,
                    color=style.color,
                    markersize=4,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.5,
                )
        ax.set_yscale("log")
        if discrete:
            ax.set_xticks(ticks["x"].to_numpy(), ticks["xlabel"].tolist())
        else:
            ax.set_xlabel("bloco (0 = embedding de entrada)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axes[0, 0].set_ylabel("posto + 1 (1 = vizinho mais próximo)")
    handles = [line_handle(styles[k]) for k in keys]
    legend_top(fig, handles, ncol=len(handles))
    return fig
