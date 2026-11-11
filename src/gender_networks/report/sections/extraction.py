"""Report items of the ``extract`` stage (Section 5)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gender_networks import plots
from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import (
    DASH,
    Tabular,
    bold,
    header_row,
    integer,
    num,
    pct,
    sci,
    tex_text,
)
from gender_networks.representations import LEX, sort_names


def _diagnostics(ctx: Context) -> dict[str, Any]:
    return ctx.json(ctx.paths.diagnostics)


def table_extracao_diagnosticos(ctx: Context) -> Tabular:
    reps = _diagnostics(ctx).get("representations") or {}
    if not reps:
        raise Skip("diagnostics.json não tem representações")
    rows = []
    factor = None
    n_top = 0
    for rep in sort_names(r for r in reps if ctx.shows(r)):
        info = reps[rep]
        q = info.get("norm_quantiles") or {}
        top = info.get("top_dims") or []
        n_top = max(n_top, len(top))
        factor = info.get("outlier_factor", factor)
        rows.append(
            [
                r"\texttt{" + tex_text(rep) + "}",
                num(q.get("5"), 1),
                num(q.get("50"), 1),
                num(q.get("95"), 1),
                num(q.get("100"), 1),
                num(info.get("mean_pair_cosine"), 3),
                integer(top[0]["dim"]) if top else DASH,
                num(info.get("top_dim_mean_abs_over_median_dim"), 1),
                pct(info.get("top_dims_mean_share_sq_norm"), 1),
                integer(info.get("n_norm_above_factor_median")),
            ]
        )
    header = [
        header_row(
            [
                "",
                r"\multicolumn{4}{c}{Norma (quantis)}",
                "",
                r"\multicolumn{3}{c}"
                r"{Dimensões dominantes}",
                "",
            ]
        ),
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-9}",
        header_row(
            [
                "Repr.",
                "5\\%",
                "50\\%",
                "95\\%",
                "máx.",
                "Cos. médio",
                "Maior",
                "$|x|$/mediana",
                f"norma² ({n_top} dim.)",
                f"$>{num(factor, 0)}\\times$med." if factor else "Atípicas",
            ]
        ),
    ]
    notes = [
        "Cos. médio: cosseno médio entre pares aleatórios de vértices (anisotropia). Maior: "
        "índice da dimensão de maior $|x|$ médio; $|x|$/mediana: esse valor sobre a mediana das "
        "dimensões; norma²: fração média da norma ao quadrado nas dimensões de maior $|x|$; "
        "a última coluna conta vértices com norma acima desse múltiplo da mediana."
    ]
    return Tabular("l" + "r" * 9, header, rows, size=r"\footnotesize", notes=notes, colsep="4pt")


def figure_extracao_camadas(ctx: Context) -> Any:
    diagnostics = _diagnostics(ctx)
    curve = pd.DataFrame(diagnostics.get("layer_curve") or [])
    if curve.empty:
        raise Skip("diagnostics.json não tem a curva por camada (capture_all_layers desligado?)")
    lexical = (diagnostics.get("representations") or {}).get(LEX)
    marked = {name: layer for name, layer in ctx.settings.model.layers.items() if ctx.shows(name)}
    return plots.extraction_layers_figure(curve, lexical, marked)


def figure_extracao_dimensoes(ctx: Context) -> Any:
    reps = _diagnostics(ctx).get("representations") or {}
    if not reps:
        raise Skip("diagnostics.json não tem representações")
    shown = sort_names(r for r in reps if ctx.shows(r))
    return plots.extraction_dimensions_figure({rep: reps[rep] for rep in shown})


def _yes(flag: Any) -> str:
    return "ok" if flag else bold("falhou")


def table_extracao_verificacao(ctx: Context) -> Tabular:
    report = ctx.json(ctx.paths.verify_report)
    checks = report.get("checks") or {}
    tol = report.get("tolerance") or {}
    tol_text = f"rtol {sci(tol.get('rtol'))}, atol {sci(tol.get('atol'))}" if tol else DASH
    rows: list[Any] = []
    if "hidden_states_count" in checks:
        c = checks["hidden_states_count"]
        found = ", ".join(str(v) for v in c.get("found", []))
        rows.append(
            [
                "\\texttt{len(hidden\\_states)} $=$ blocos $+1$",
                _yes(c.get("passed")),
                f"{found} (esperado {c.get('expected')})",
                "igualdade",
            ]
        )
    if "embedding_input" in checks:
        c = checks["embedding_input"]
        rows.append(
            [
                "\\texttt{hidden\\_states[0]} $=$ \\texttt{embed\\_tokens(ids)}",
                _yes(c.get("passed")),
                sci(c.get("max_abs_diff")),
                "bit a bit",
            ]
        )
    if "hooks_match_hidden_states" in checks:
        c = checks["hooks_match_hidden_states"]
        for layer, info in (c.get("layers") or {}).items():
            rows.append(
                [
                    f"\\textit{{hook}} em \\texttt{{layers[{int(layer) - 1}]}} $=$ "
                    f"\\texttt{{hidden\\_states[{layer}]}}",
                    _yes(info.get("equal")),
                    sci(info.get("max_abs_diff")),
                    "bit a bit",
                ]
            )
    if "final_norm" in checks:
        c = checks["final_norm"]
        rows.append(
            [
                "bloco final bruto $\\neq$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("raw_block_differs_from_hidden")),
                sci(c.get("raw_block_vs_hidden_max_abs_diff")),
                "fora da tolerância",
            ]
        )
        rows.append(
            [
                "\\texttt{norm(bloco final)} $\\approx$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("norm_of_raw_block_close")),
                sci(c.get("norm_of_raw_block_max_abs_diff")),
                tol_text,
            ]
        )
        rows.append(
            [
                "\\textit{hook} em \\texttt{norm} $=$ \\texttt{hidden\\_states[L]}",
                _yes(c.get("norm_hook_equal")),
                sci(c.get("norm_hook_max_abs_diff")),
                "bit a bit",
            ]
        )
    if "determinism" in checks:
        c = checks["determinism"]
        status = {
            "identical": "idênticas",
            "tiny": "quase idênticas",
            "different": "diferentes",
        }.get(c.get("status"), str(c.get("status")))
        rows.append(
            [
                f"duas execuções ({status})",
                _yes(c.get("passed")),
                sci(c.get("max_abs_diff")),
                tol_text + " (só aviso)",
            ]
        )
    if not rows:
        raise Skip("verify.json não tem checagens")
    model = report.get("model") or {}
    notes = [
        f"{integer(report.get('n_sequences'))} sequências, {integer(report.get('tokens'))} tokens, "
        f"{integer(report.get('n_layers'))} blocos, "
        f"\\texttt{{{tex_text(report.get('dtype', ''))}}}; "
        f"resultado geral: {_yes(report.get('passed'))}."
    ]
    if model.get("revision"):
        notes.append(
            f"Modelo \\texttt{{{tex_text(model.get('name_or_path', ''))}}}, revisão "
            f"\\texttt{{{tex_text(str(model['revision'])[:12])}}}."
        )
    header = [header_row(["Checagem", "Resultado", "Maior dif. absoluta", "Critério"])]
    return Tabular(
        r">{\raggedright\arraybackslash}X l l >{\raggedright\arraybackslash}p{0.25\linewidth}",
        header,
        rows,
        size=r"\footnotesize",
        notes=notes,
        width=r"\linewidth",
    )
