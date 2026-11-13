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
    """Name of the raw output of ``block`` (1-based): ``L01``, ``L18``...

    Args:
        block: 1-based index of the Transformer block (``1`` is the first block, whose module
            is ``layers[0]``).

    Returns:
        ``"L"`` followed by the block number zero-padded to at least two digits.
    """

    return f"L{block:02d}"


@dataclass(frozen=True)
class Representation:
    """One way of turning an occurrence into a vector.

    The value object behind a representation name. It is frozen and hashable, so two names that
    mean the same thing compare equal and can key dictionaries.

    Attributes:
        kind: ``"lex"`` (input embedding row), ``"layer"`` (raw output of a block) or
            ``"final_norm"`` (output of the model's final RMSNorm).
        block: Depth in the name, 1-based; ``None`` only for ``lex``. For ``final_norm`` it is
            the block the norm reads (36 in Qwen3-4B-Base), not a block of its own.
        centered: True when the mean vector over the occurrences is subtracted before the
            cosine (the trailing ``c`` of the name).
    """

    kind: Kind
    block: int | None = None  # depth in the name; for final_norm, the block the norm reads
    centered: bool = False

    def __post_init__(self) -> None:
        """Reject combinations that no name can spell.

        Raises:
            ValueError: If ``lex`` has a block, if another kind has none, or if the block is
                smaller than 1.
        """

        if (self.kind == LEX) != (self.block is None):
            raise ValueError("only the lexical representation has no block")
        if self.block is not None and self.block < 1:
            raise ValueError(f"blocks are 1-based, got {self.block}")

    @property
    def name(self) -> str:
        """The name used in configurations, file names and graph ids.

        Returns:
            ``lex`` or ``L{block:02d}``, then ``n`` for the final norm, then ``c`` when centered
            (``lex``, ``L18``, ``L36n``, ``L01c``). :func:`parse` is the inverse.
        """

        if self.kind == LEX:
            base = LEX
        else:
            assert self.block is not None
            base = layer_name(self.block) + ("n" if self.kind == "final_norm" else "")
        return base + ("c" if self.centered else "")

    @property
    def is_lexical(self) -> bool:
        """True for the lexical representation (centered or not)."""

        return self.kind == LEX

    @property
    def is_block(self) -> bool:
        """A raw, uncentered block output (``L01``, ``L18``...).

        Returns:
            True when ``kind`` is ``"layer"`` and the vectors are not centered; False for ``lex``,
            the final norm and every centered variant.
        """

        return self.kind == "layer" and not self.centered

    @property
    def source(self) -> Representation:
        """The stored vectors this one is computed from (itself unless it is centered).

        Centered vectors are never written to disk: the ``knn`` stage subtracts the mean from the
        vectors of ``source`` when it needs them.

        Returns:
            The same kind and block with ``centered=False``.
        """

        return Representation(self.kind, self.block)

    def with_centering(self) -> Representation:
        """The centered variant of this representation (same kind and block, ``centered=True``)."""

        return Representation(self.kind, self.block, centered=True)

    def raw(self) -> Representation:
        """For the final RMSNorm, the raw output of the block it reads; otherwise itself.

        Returns:
            ``self`` unless ``kind`` is ``"final_norm"``; then the ``"layer"`` representation of the
            same block, keeping ``centered`` (``L36n`` gives ``L36``).
        """

        if self.kind != "final_norm":
            return self
        return Representation("layer", self.block, self.centered)

    def sort_key(self) -> tuple[bool, int, int, bool]:
        """Uncentered first; within each group ``lex``, then blocks by depth, each block's final
        norm after it (``lex, L01, L18, L36, L36n, L01c, L18c, L36c``).

        Returns:
            ``(centered, 0 for lex or 1 otherwise, block or 0, is final norm)``, to be compared as
            a tuple.
        """

        depth = 0 if self.is_lexical else 1
        return (self.centered, depth, self.block or 0, self.kind == "final_norm")

    def __str__(self) -> str:
        """The representation name, so the object can be formatted straight into messages."""

        return self.name


FINAL_NORM = Representation("final_norm", FINAL_NORM_BLOCK)


def parse(name: str) -> Representation:
    """The representation a name stands for; ``ValueError`` when the name is not one.

    Args:
        name: ``lex`` or ``L`` + at least two digits + optional ``n`` + optional ``c``
            (``L01``, ``L36n``, ``L18c``). Matching is exact and case-sensitive; ``lexc`` is not
            a name.

    Returns:
        The matching :class:`Representation`.

    Raises:
        ValueError: If the name does not follow the grammar, or names block 0 (``L00``).

    Notes:
        Any block number parses, so ``L12n`` is a valid name here; whether a run may use it is
        decided by ``settings._known_representation``.
    """

    if name == LEX:
        return Representation(LEX)
    match = _NAME.match(name)
    if match is None:
        raise ValueError(f"'{name}' is not a representation name (lex, L01, L36n, L01c...)")
    block, norm, centered = match.groups()
    kind: Kind = "final_norm" if norm else "layer"
    return Representation(kind, int(block), centered=bool(centered))


def try_parse(name: str) -> Representation | None:
    """Like :func:`parse`, but ``None`` for names that are not representations (``vocab``).

    Args:
        name: Candidate name.

    Returns:
        The :class:`Representation`, or ``None`` when :func:`parse` raises ``ValueError``.
    """

    try:
        return parse(name)
    except ValueError:
        return None


def is_lexical(name: str) -> bool:
    """True when ``name`` is exactly ``lex`` (the centered ``lexc`` does not exist).

    Args:
        name: Representation name.

    Returns:
        ``name == "lex"``.
    """

    return name == LEX


def sort_names(names: Iterable[str]) -> list[str]:
    """Representation names in depth order; other names keep their order at the end.

    Args:
        names: Names to sort; may mix representation names and others (``vocab``).

    Returns:
        New list: the names :func:`try_parse` accepts, sorted by
        :meth:`Representation.sort_key`, followed by the remaining names in input order.
        Duplicates are kept.
    """

    names = list(names)
    known = sorted((n for n in names if try_parse(n)), key=lambda n: parse(n).sort_key())
    return known + [n for n in names if not try_parse(n)]


def occurrence_representations(settings: Settings) -> list[Representation]:
    """Every occurrence-level representation of a run, in the order the stages process them.

    The main ones (``networks.representations``), then the robust ones
    (``networks.robust_representations``), then, with ``networks.centered``, the centered
    variant of each main block (``L01c``...).

    Args:
        settings: Run configuration; only the ``networks`` section is read.

    Returns:
        Representations without repeats. ``lex`` and the robust representations (``L36n``) get
        no centered variant: only names for which :attr:`Representation.is_block` holds do.

    Raises:
        ValueError: If a configured name is not a representation name (see :func:`parse`).
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
    """Names of :func:`occurrence_representations`, in the same order.

    Args:
        settings: Run configuration.

    Returns:
        For the main run: ``lex, L01, L18, L36, L36n, L01c, L18c, L36c``.
    """

    return [rep.name for rep in occurrence_representations(settings)]


def centered_names(settings: Settings) -> list[str]:
    """Names of the centered variants of the main blocks (``L01c``...).

    Args:
        settings: Run configuration.

    Returns:
        The centered entries of :func:`occurrence_names`; empty when ``networks.centered`` is
        false.
    """

    return [rep.name for rep in occurrence_representations(settings) if rep.centered]


def transition_pairs(names: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Comparisons between representations in depth order: ``(consecutive, all)``.

    ``consecutive`` joins each representation to the next (``lex→L01``, ``L01→L18``...);
    ``all`` adds the cumulative change from ``lex`` to every block after the first
    (``lex→L18``, ``lex→L36``).

    Args:
        names: Representation names already in depth order (the function does not sort).

    Returns:
        Two lists of ``(from, to)`` name pairs. The cumulative pairs exist only when the first
        name is ``lex``; with fewer than two names both lists are empty.
    """

    names = list(names)
    consecutive = list(zip(names, names[1:], strict=False))
    cumulative = [(names[0], b) for b in names[2:]] if names and is_lexical(names[0]) else []
    return consecutive, consecutive + cumulative
