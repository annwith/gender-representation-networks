"""Occurrence representations: what each name (``lex``, ``L01``, ``L36n``, ``L01c``) means.

Every stage refers to a representation by its name, which is also part of the artifact file
names (``reps/L01.safetensors``, ``knn/nbr_L01c_k10.npz``). This module is the only place that
reads meaning into a name; the rest of the code asks a :class:`Representation`.

Names follow one grammar:

- ``lex``: the input embedding row of the occurrence's token (all occurrences of a type share
  it);
- ``L{block:02d}``: the raw output of a Transformer block (1-based), e.g. ``L01``, ``L18``;
- ``L{block:02d}n``: the output of the model's final RMSNorm, which reads the last block
  (``L36n`` for the 36 blocks of Qwen3-4B-Base);
- a trailing ``c``: the same vectors centered (the mean over the occurrences is subtracted)
  before the cosine, e.g. ``L01c``.

The number in a name orders representations by depth. The block a configured name is read
from comes from ``model.layers`` (``L18: 18`` in the main run; the tests name the three blocks of
a tiny model ``L01``, ``L18`` and ``L36``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gender_networks.settings import Settings

Kind = Literal["lex", "layer", "final_norm"]

LEX = "lex"
# Blocks of the model the experiment uses (Qwen3-4B-Base): the final RMSNorm reads block 36.
FINAL_NORM_BLOCK = 36
_NAME = re.compile(r"^L(\d{2,})(n?)(c?)$")


def layer_name(block: int) -> str:
    """Name of the raw output of ``block`` (1-based): ``L01``, ``L18``..."""

    return f"L{block:02d}"


@dataclass(frozen=True)
class Representation:
    """One way of turning an occurrence into a vector."""

    kind: Kind
    block: int | None = None  # depth in the name; for final_norm, the block the norm reads
    centered: bool = False

    def __post_init__(self) -> None:
        if (self.kind == LEX) != (self.block is None):
            raise ValueError("only the lexical representation has no block")
        if self.block is not None and self.block < 1:
            raise ValueError(f"blocks are 1-based, got {self.block}")

    @property
    def name(self) -> str:
        if self.kind == LEX:
            base = LEX
        else:
            assert self.block is not None
            base = layer_name(self.block) + ("n" if self.kind == "final_norm" else "")
        return base + ("c" if self.centered else "")

    @property
    def is_lexical(self) -> bool:
        return self.kind == LEX

    @property
    def is_block(self) -> bool:
        """A raw, uncentered block output (``L01``, ``L18``...)."""

        return self.kind == "layer" and not self.centered

    @property
    def source(self) -> Representation:
        """The stored vectors this one is computed from (itself unless it is centered)."""

        return Representation(self.kind, self.block)

    def with_centering(self) -> Representation:
        return Representation(self.kind, self.block, centered=True)

    def raw(self) -> Representation:
        """For the final RMSNorm, the raw output of the block it reads; otherwise itself."""

        if self.kind != "final_norm":
            return self
        return Representation("layer", self.block, self.centered)

    def sort_key(self) -> tuple[bool, int, int, bool]:
        """Uncentered first; within each group ``lex``, then blocks by depth, each block's final
        norm after it (``lex, L01, L18, L36, L36n, L01c, L18c, L36c``)."""

        depth = 0 if self.is_lexical else 1
        return (self.centered, depth, self.block or 0, self.kind == "final_norm")

    def __str__(self) -> str:
        return self.name


FINAL_NORM = Representation("final_norm", FINAL_NORM_BLOCK)


def parse(name: str) -> Representation:
    """The representation a name stands for; ``ValueError`` when the name is not one."""

    if name == LEX:
        return Representation(LEX)
    match = _NAME.match(name)
    if match is None:
        raise ValueError(f"'{name}' is not a representation name (lex, L01, L36n, L01c...)")
    block, norm, centered = match.groups()
    kind: Kind = "final_norm" if norm else "layer"
    return Representation(kind, int(block), centered=bool(centered))


def try_parse(name: str) -> Representation | None:
    """Like :func:`parse`, but ``None`` for names that are not representations (``vocab``)."""

    try:
        return parse(name)
    except ValueError:
        return None


def is_lexical(name: str) -> bool:
    return name == LEX


def sort_names(names: Iterable[str]) -> list[str]:
    """Representation names in depth order; other names keep their order at the end."""

    names = list(names)
    known = sorted((n for n in names if try_parse(n)), key=lambda n: parse(n).sort_key())
    return known + [n for n in names if not try_parse(n)]


def occurrence_representations(settings: Settings) -> list[Representation]:
    """Every occurrence-level representation of a run, in the order the stages process them.

    The main ones (``networks.representations``), then the robust ones
    (``networks.robust_representations``), then, with ``networks.centered``, the centered
    variant of each main block (``L01c``...).
    """

    net = settings.networks
    reps: list[Representation] = []
    for name in [*net.representations, *net.robust_representations]:
        rep = parse(name)
        if rep not in reps:
            reps.append(rep)
    if net.centered:
        for name in net.representations:
            rep = parse(name)
            if rep.is_block and rep.with_centering() not in reps:
                reps.append(rep.with_centering())
    return reps


def occurrence_names(settings: Settings) -> list[str]:
    return [rep.name for rep in occurrence_representations(settings)]


def centered_names(settings: Settings) -> list[str]:
    """Names of the centered variants of the main blocks (``L01c``...)."""

    return [rep.name for rep in occurrence_representations(settings) if rep.centered]


def transition_pairs(names: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Comparisons between representations in depth order: ``(consecutive, all)``.

    ``consecutive`` joins each representation to the next (``lex→L01``, ``L01→L18``...);
    ``all`` adds the cumulative change from ``lex`` to every block after the first
    (``lex→L18``, ``lex→L36``).
    """

    names = list(names)
    consecutive = list(zip(names, names[1:], strict=False))
    cumulative = [(names[0], b) for b in names[2:]] if names and is_lexical(names[0]) else []
    return consecutive, consecutive + cumulative
