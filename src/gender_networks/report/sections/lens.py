"""Extension: nearest vocabulary rows (Section 8)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.labels import (
    CATEGORY_ORDER,
    ordered,
)
from gender_networks.report.context import Context, Skip


def figure_lente_ranking(ctx: Context) -> Any:
    curve_path = ctx.paths.lens_file("layer_curve.csv")
    if curve_path.exists() and not ctx.table(curve_path).empty:
        curve = ctx.table(curve_path).copy()
        curve = curve[curve["group_by"].isin(["all", "token_category"])]
        curve = curve.assign(
            x=curve["layer"].astype(float),
            xlabel=curve["rep"].astype(str),
            group=curve["group"].astype(str),
        )
        discrete = False
    else:
        summary = ctx.json(ctx.paths.lens_file("summary.json"))
        ranks = summary.get("ranks") or {}
        if not ranks:
            raise Skip("lens/summary.json não tem postos")
        rows = []
        for x, rep in enumerate(summary.get("representations") or list(ranks)):
            entry = ranks.get(rep) or {}
            groups = {"all": entry.get("all") or {}, **(entry.get("by_token_category") or {})}
            for group, stats in groups.items():
                rows.append(
                    {
                        "x": float(x),
                        "xlabel": rep,
                        "group": group,
                        "own_rank_median": stats.get("own_rank_median"),
                        "next_rank_median": stats.get("next_rank_median"),
                    }
                )
        curve = pd.DataFrame(rows)
        discrete = True
    categories = ordered([g for g in curve["group"] if g != "all"], CATEGORY_ORDER)
    return plots.lens_ranks_figure(curve, categories, discrete)
