from __future__ import annotations

import pytest

from gender_networks.representations import (
    FINAL_NORM,
    Representation,
    centered_names,
    is_lexical,
    layer_name,
    occurrence_names,
    parse,
    sort_names,
    transition_pairs,
    try_parse,
)
from gender_networks.settings import ModelSettings, NetworkSettings, Settings


@pytest.mark.parametrize("name", ["lex", "L01", "L18", "L36", "L36n", "L01c", "L36nc", "L120"])
def test_names_round_trip(name: str) -> None:
    assert parse(name).name == name


def test_parse_meaning() -> None:
    assert parse("lex") == Representation("lex")
    assert parse("L18") == Representation("layer", 18)
    assert parse("L36n") == FINAL_NORM == Representation("final_norm", 36)
    assert parse("L01c") == Representation("layer", 1, centered=True)
    assert parse("L01c").source == parse("L01")
    assert parse("L36n").raw() == parse("L36")
    assert parse("L18").raw() == parse("L18")
    assert parse("lex").is_lexical and is_lexical("lex") and not is_lexical("L01")
    assert parse("L01").is_block
    assert not parse("L01c").is_block and not parse("L36n").is_block
    assert layer_name(1) == "L01" and layer_name(36) == "L36"


@pytest.mark.parametrize("name", ["vocab", "L1", "L01x", "l01", "", "lexc"])
def test_other_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        parse(name)
    assert try_parse(name) is None


def test_sort_names_follows_depth() -> None:
    names = ["L36n", "L01c", "vocab", "L36", "lex", "L18", "L01"]
    assert sort_names(names) == ["lex", "L01", "L18", "L36", "L36n", "L01c", "vocab"]


def test_occurrence_representations_of_a_run() -> None:
    settings = Settings(
        name="t",
        model=ModelSettings(layers={"L01": 1, "L18": 18, "L36": 36}),
        networks=NetworkSettings(
            representations=["lex", "L01", "L18", "L36"],
            robust_representations=["L36n"],
            centered=True,
        ),
    )
    assert occurrence_names(settings) == [
        "lex",
        "L01",
        "L18",
        "L36",
        "L36n",
        "L01c",
        "L18c",
        "L36c",
    ]
    assert centered_names(settings) == ["L01c", "L18c", "L36c"]
    uncentered = Settings(
        name="t",
        model=settings.model,
        networks=NetworkSettings(
            representations=["lex", "L01"], robust_representations=[], centered=False
        ),
    )
    assert occurrence_names(uncentered) == ["lex", "L01"]


def test_transition_pairs() -> None:
    consecutive, pairs = transition_pairs(["lex", "L01", "L18", "L36"])
    assert consecutive == [("lex", "L01"), ("L01", "L18"), ("L18", "L36")]
    assert pairs == [*consecutive, ("lex", "L18"), ("lex", "L36")]
    assert transition_pairs(["L01", "L18"]) == ([("L01", "L18")], [("L01", "L18")])
    assert transition_pairs([]) == ([], [])
