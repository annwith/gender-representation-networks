"""Figures over every block switch to a layer axis; with few representations nothing changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gender_networks import plots

BLOCKS = [f"L{b:02d}" for b in range(1, 11)]
REPS = ["lex", *BLOCKS]


def test_layer_axis_only_past_the_threshold() -> None:
    assert not plots.layer_axis(["lex", "L01", "L18", "L36"])
    assert plots.layer_axis(REPS)
    assert list(plots.rep_positions(["lex", "L01", "L18", "L36"])) == [0, 1, 2, 3]
    assert list(plots.rep_positions(["lex", *BLOCKS])) == list(range(11))


def test_figures_draw_on_a_layer_axis() -> None:
    rng = np.random.default_rng(0)
    bands, groups = ["1-2", "3-9"], ["whole_word", "continuation"]

    rows = [
        {"measure": m, "grouping": "token_category", "group": g, "rep": r, "value": rng.random()}
        for m in ("D", "Dn")
        for g in groups
        for r in REPS
    ]
    plots.dominance_figure(pd.DataFrame(rows), REPS, [("token_category", "categoria", groups)])

    nmi = pd.DataFrame(
        [{"label": "theme", "rep": r, "nmi_minus_baseline": rng.random()} for r in REPS]
    )
    plots.nmi_layers_figure(nmi, REPS, ["theme"], {r: rng.random() for r in REPS})

    sep = pd.DataFrame([{"group": "target", "rep": r, "self_sim": rng.random()} for r in REPS])
    plots.separation_figure(sep, REPS, [("self_sim", "autossimilaridade")], ["target"])

    values = {r: {b: rng.random(30) for b in bands} for r in BLOCKS}
    fig = plots.jaccard_bands_figure(values, {"1-2": 0.4, "3-9": 0.5}, bands)
    assert fig.axes[0].get_xlabel() == plots.LAYER_AXIS_LABEL

    transitions = [f"{a}->{b}" for a, b in zip(REPS, REPS[1:], strict=False)]
    frame = pd.DataFrame(rng.random((3, len(transitions))), columns=transitions)
    frame.index = ["todos", "palavra inteira", "continuação"]
    plots.transition_share_figure([("Por categoria", frame)], transitions)
