"""LaTeX text and pt-BR numbers for the generated tables (``Tabular``)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from gender_networks.labels import (
    fmt_dec,
    fmt_int,
)

HEADER = "% Gerado pela etapa `report` (gender_networks.report). Não editar à mão.\n"


# Code points pdflatex typesets with inputenc (utf8) + T1 + lmodern, probed one by one.
_TEX_RANGES = [
    (0xA0, 0x125), (0x128, 0x137), (0x139, 0x13E), (0x141, 0x148), (0x14A, 0x165),
    (0x168, 0x17E), (0x192, 0x192), (0x1C4, 0x1D4), (0x1E2, 0x1E3), (0x1E6, 0x1EB),
    (0x1F0, 0x1F0), (0x1F4, 0x1F5), (0x218, 0x21B), (0x232, 0x233), (0x237, 0x237),
    (0x2010, 0x2016), (0x2018, 0x201A), (0x201C, 0x201E), (0x2020, 0x2022), (0x2026, 0x2026),
    (0x2030, 0x2031), (0x2039, 0x203B), (0x203D, 0x203D), (0x20AC, 0x20AC), (0x2122, 0x2122),
]  # fmt: skip


_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "%": r"\%",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    '"': r"\textquotedbl{}",
    "`": r"\textasciigrave{}",
    "\u00ad": "-{}",
}


_LIGATURES = set("-,'!?")  # "--", ",,", "''", "!`" and "?`" would become other glyphs


def _typesettable(char: str) -> bool:
    """Whether pdflatex can typeset a character with the fonts of the report.

    Args:
        char: One character.

    Returns:
        True for printable ASCII and for the code points of ``_TEX_RANGES``.
    """

    code = ord(char)
    if 0x20 <= code < 0x7F:
        return True
    return any(low <= code <= high for low, high in _TEX_RANGES)


def tex_text(text: Any, visible_space: bool = False) -> str:
    """Escape arbitrary text for LaTeX; unsupported characters become ``[U+XXXX]``.

    Args:
        text: Value to escape; converted with ``str``.
        visible_space: Write spaces as ``\\textvisiblespace{}``.

    Returns:
        LaTeX source: special characters are escaped, newlines and tabs are shown as
        backslash escapes, characters that would form ligatures get ``{}`` after them, and a
        run of characters the fonts lack becomes one bracket with its first two code points.
    """

    out: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        """Write the pending run of unsupported characters as one ``[U+XXXX ...]`` bracket."""

        if pending:
            shown = " ".join(f"U+{ord(c):04X}" for c in pending[:2])
            more = r"\ldots{}" if len(pending) > 2 else ""
            out.append(f"[{shown}{more}]")
            pending.clear()

    for char in str(text):
        if char in _TEX_SPECIAL:
            flush()
            out.append(_TEX_SPECIAL[char])
        elif char == " ":
            flush()
            out.append(r"\textvisiblespace{}" if visible_space else " ")
        elif char in "\n\r\t":
            flush()
            out.append(
                {
                    "\n": r"\textbackslash{}n",
                    "\r": r"\textbackslash{}r",
                    "\t": r"\textbackslash{}t",
                }[char]
            )
        elif _typesettable(char):
            flush()
            out.append(char + "{}" if char in _LIGATURES else char)
        else:
            pending.append(char)
    flush()
    return "".join(out)


def tex_token(text: Any) -> str:
    """A token as it appears in the text: typewriter, leading/inner spaces made visible.

    Args:
        text: Token text; ``None`` and NaN count as empty.

    Returns:
        ``\\texttt{...}``, or an italic ``(vazio)`` for an empty token.
    """

    value = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    return r"\texttt{" + (tex_text(value, visible_space=True) or r"\textit{(vazio)}") + "}"


def finite_or_none(value: Any) -> float | None:
    """``value`` as a float, or None when it is missing, not a number or infinite.

    Args:
        value: Any value.

    Returns:
        The float, or ``None``.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _signed(text: str) -> str:
    """Replace a leading hyphen by a math-mode minus sign.

    Args:
        text: Formatted number.

    Returns:
        The text with ``$-$`` in place of a leading ``-``.
    """

    return "$-$" + text[1:] if text.startswith("-") else text


DASH = "--"


def num(value: Any, digits: int = 2) -> str:
    """Decimal with comma (``0,83``); ``--`` when missing.

    Args:
        value: Number, or anything :func:`finite_or_none` rejects.
        digits: Decimal places. Defaults to 2.

    Returns:
        The pt-BR text; a value that rounds to zero loses its minus sign.
    """

    number = finite_or_none(value)
    if number is None:
        return DASH
    text = fmt_dec(number, digits)
    if re.fullmatch(r"-0(,0*)?", text):
        text = text[1:]
    return _signed(text)


def integer(value: Any) -> str:
    """Integer with a thousands dot (``15.104``); ``--`` when missing.

    Args:
        value: Number; rounded to the nearest integer.

    Returns:
        The pt-BR text.
    """

    number = finite_or_none(value)
    return DASH if number is None else _signed(fmt_int(number))


def pct(value: Any, digits: int = 1) -> str:
    """A fraction as a percentage (``0.283`` -> ``28,3\\,\\%``); ``--`` when missing.

    Args:
        value: Fraction between 0 and 1.
        digits: Decimal places of the percentage. Defaults to 1.

    Returns:
        The LaTeX text.
    """

    number = finite_or_none(value)
    return DASH if number is None else num(number * 100, digits) + r"\,\%"


def sci(value: Any, digits: int = 1) -> str:
    """Scientific notation in math mode for small or large values (``$2{,}4\\times10^{-6}$``).

    Args:
        value: Number.
        digits: Decimal places of the mantissa. Defaults to 1.

    Returns:
        ``--`` when missing, ``0`` for zero, a plain decimal for magnitudes between
        ``10^-3`` and ``10^3`` (with enough places to show the leading digits), and
        ``mantissa x 10^exponent`` otherwise.
    """

    number = finite_or_none(value)
    if number is None:
        return DASH
    if number == 0:
        return "0"
    exponent = math.floor(math.log10(abs(number)))
    if -3 <= exponent <= 3:
        return num(number, digits - exponent if exponent < 0 else digits)
    mantissa = number / 10**exponent
    text = fmt_dec(mantissa, digits).replace(",", "{,}")
    return f"${text}\\times10^{{{exponent}}}$"


def pm(mean: Any, sd: Any, digits: int = 2, kind: str = "num") -> str:
    """``mean ± sd`` (the sd only when finite and positive).

    Args:
        mean: Mean value.
        sd: Standard deviation.
        digits: Decimal places.
        kind: ``num`` for decimals, ``int`` to show the mean as an integer.

    Returns:
        The LaTeX text.
    """

    fmt = integer if kind == "int" else (lambda v: num(v, digits))
    text = fmt(mean)
    sd_value = finite_or_none(sd)
    if sd_value is not None and sd_value > 0 and text != DASH:
        shown = num(sd_value, digits if kind != "int" else 1)
        text += r"\,$\pm$\," + shown
    return text


def duration(seconds: Any) -> str:
    """A time span in seconds as ``12,3 s``, ``4 min 05 s`` or ``1 h 02 min``.

    Args:
        seconds: Number of seconds.

    Returns:
        The text, or ``--`` when missing.
    """

    value = finite_or_none(seconds)
    if value is None:
        return DASH
    if value < 60:
        return num(value, 1) + " s"
    minutes, secs = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def bold(text: str) -> str:
    """Wrap LaTeX text in ``\\textbf``.

    Args:
        text: LaTeX source, already escaped.

    Returns:
        The bold text.
    """

    return r"\textbf{" + text + "}"


@dataclass
class Tabular:
    """A booktabs ``tabular`` (or ``tabularx``) fragment built row by row.

    Attributes:
        spec: Column specification.
        header: Header lines, already complete LaTeX rows or rules.
        rows: Body rows: a list of cells, or a string written as is (a rule or a
            ``\\multicolumn`` line).
        size: Font size command written before the table (``\\footnotesize``).
        notes: Sentences written under the table.
        width: Total width; when given the environment is ``tabularx``.
        colsep: Value for ``\\tabcolsep``.
    """

    spec: str
    header: list[str]
    rows: list[Any]
    size: str = ""  # e.g. \footnotesize
    notes: list[str] | None = None
    width: str | None = None  # tabularx width (e.g. \linewidth)
    colsep: str | None = None

    def render(self) -> str:
        """The LaTeX source of the fragment, starting with the "generated, do not edit" comment.

        Returns:
            Text of the ``.tex`` file: optional size and column separation, the table between
            ``\\toprule`` and ``\\bottomrule``, and the notes.
        """

        env = "tabularx" if self.width else "tabular"
        head = [HEADER]
        if self.size:
            head.append(self.size + "\n")
        if self.colsep:
            head.append(f"\\setlength{{\\tabcolsep}}{{{self.colsep}}}\n")
        width = f"{{{self.width}}}" if self.width else ""
        lines = [f"\\begin{{{env}}}{width}{{{self.spec}}}", r"\toprule"]
        lines += self.header
        lines.append(r"\midrule")
        for row in self.rows:
            if isinstance(row, str):
                lines.append(row)
            else:
                lines.append(" & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(f"\\end{{{env}}}")
        body = "\n".join(lines) + "\n"
        if self.notes:
            notes = " ".join(self.notes)
            body += "\n\\par\\smallskip{\\footnotesize\\raggedright " + notes + "\\par}\n"
        return "".join(head) + body


def header_row(cells: Sequence[str]) -> str:
    """One LaTeX table row from its cells.

    Args:
        cells: Cell sources.

    Returns:
        The cells joined with ``&`` and ended with ``\\\\``.
    """

    return " & ".join(cells) + r" \\"
