"""The report stage on a tiny fake run (knn -> metrics -> analyze -> report, offline)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_analysis import tiny_settings, write_fake_run

from gender_networks import analysis, knn, metrics, report
from gender_networks.artifacts import (
    RunPaths,
    read_json,
    write_json,
    write_jsonl,
    write_manifest,
)
from gender_networks.settings import Settings

REPO = Path(__file__).resolve().parents[1]
RELATORIO = REPO / "report" / "relatorio"


def write_corpus(paths: RunPaths, settings: Settings) -> None:
    articles, paragraphs = [], []
    reasons = [None, None, None, "multi_theme", "biography", None, "no_paragraphs"]
    for i in range(14):
        theme = "a" if i % 2 else "b"
        reason = reasons[i % len(reasons)]
        articles.append(
            {
                "pageid": 100 + i,
                "revid": 9000 + i,
                "title": f"Artigo {i}",
                "theme": theme,
                "themes_reached": [theme],
                "source_category": "C",
                "depth": 1,
                "origin": "root",
                "is_biography": reason == "biography",
                "dropped_reason": reason,
            }
        )
        if reason is None:
            for j in range(3):
                words = " ".join(["palavra"] * (40 + 7 * i + j))
                paragraphs.append(
                    {
                        "paragraph_id": f"{100 + i}-{j}",
                        "pageid": 100 + i,
                        "theme": theme,
                        "text": words + ".",
                    }
                )
    write_jsonl(paths.articles, articles)
    write_jsonl(paths.paragraphs, paragraphs)
    write_manifest(
        paths.corpus_dir,
        "corpus",
        settings,
        time.time(),
        extra={
            "articles_considered": len(articles),
            "articles_kept": sum(a["dropped_reason"] is None for a in articles),
            "paragraphs": len(paragraphs),
            "words": 123_456,
        },
        root=paths.root,
    )


def write_sample_manifest(paths: RunPaths, settings: Settings, occ: pd.DataFrame) -> None:
    counts = occ["stratum"].value_counts().to_dict()
    write_manifest(
        paths.sample_dir,
        "sample",
        settings,
        time.time(),
        extra={
            "n_vertices": len(occ),
            "n_types": int(occ["token_id"].nunique()),
            "n_sequences": int(occ["sequence_id"].nunique()),
            "tokens_to_process": 15_104,
            "corpus_tokens": 2_657_592,
            "core_paragraphs": 18,
            "by_stratum": {k: int(v) for k, v in counts.items()},
            "prefix_groups": 0,
            "targets": {
                "alvo": {
                    "token_id": 3,
                    "kept": True,
                    "themes": {
                        "a": {
                            "requested": 6,
                            "core": 2,
                            "drawn": 4,
                            "obtained": 6,
                            "in_sample": True,
                            "final": 6,
                        },
                        "b": {
                            "requested": 6,
                            "core": 1,
                            "drawn": 5,
                            "obtained": 6,
                            "in_sample": True,
                            "final": 6,
                        },
                    },
                }
            },
            "controls": {},
            "kept_targets": ["alvo"],
            "dropped_targets": [
                {"word": "planta", "themes_reaching_min": ["a"], "removed_occurrences": 4}
            ],
            "dropped_controls": [],
            "not_single_token": {"órgão": [1, 2]},
            "vocab": {"size": 60, "in_corpus": 10, "in_sample": 10},
        },
        root=paths.root,
    )


def write_extract_outputs(paths: RunPaths, settings: Settings, n: int) -> None:
    def summary(scale: float) -> dict:
        return {
            "n": n,
            "n_nonfinite_rows": 0,
            "norm_quantiles": {
                "0": scale * 0.5,
                "5": scale * 0.8,
                "25": scale * 0.9,
                "50": scale,
                "75": scale * 1.1,
                "95": scale * 1.4,
                "100": scale * 3.0,
            },
            "mean_norm": scale,
            "mean_pair_cosine": 0.1 + scale / 1000,
            "top_dims": [
                {"dim": 4, "mean_abs": scale / 3, "mean_share_sq_norm": 0.2},
                {"dim": 9, "mean_abs": scale / 9, "mean_share_sq_norm": 0.05},
            ],
            "top_dims_mean_share_sq_norm": 0.25,
            "top_dim_mean_abs_over_median_dim": scale / 2,
            "n_norm_above_factor_median": 3,
            "outlier_factor": 5.0,
            "mean_norm_by_pos_bucket": {"1-4": scale * 2, "5-16": scale},
        }

    reps = {
        "lex": summary(1.0),
        "L01": summary(10.0),
        "L18": summary(60.0),
        "L36": summary(300.0),
        "L36n": summary(80.0),
    }
    curve = [
        {
            "layer": layer,
            "mean_norm": 10.0 * layer,
            "median_norm": 9.0 * layer,
            "mean_pair_cosine": 0.1 + 0.05 * layer,
        }
        for layer in (1, 2, 3)
    ]
    write_json(
        paths.diagnostics,
        {"n_vertices": n, "hidden_size": 16, "representations": reps, "layer_curve": curve},
    )
    write_json(
        paths.verify_report,
        {
            "passed": True,
            "n_sequences": 8,
            "tokens": 1149,
            "n_layers": 36,
            "dtype": "bfloat16",
            "tolerance": {"rtol": 0.016, "atol": 1e-05},
            "checks": {
                "hidden_states_count": {"passed": True, "expected": 37, "found": [37]},
                "embedding_input": {"passed": True, "max_abs_diff": 0.0},
                "hooks_match_hidden_states": {
                    "passed": True,
                    "layers": {
                        "1": {"equal": True, "max_abs_diff": 0.0},
                        "18": {"equal": True, "max_abs_diff": 0.0},
                    },
                },
                "final_norm": {
                    "passed": True,
                    "raw_block_differs_from_hidden": True,
                    "raw_block_vs_hidden_max_abs_diff": 1136.77,
                    "norm_of_raw_block_close": True,
                    "norm_of_raw_block_max_abs_diff": 0.0,
                    "norm_hook_equal": True,
                    "norm_hook_max_abs_diff": 0.0,
                },
                "determinism": {"passed": True, "status": "identical", "max_abs_diff": 0.0},
            },
            "warnings": [],
            "model": {
                "name_or_path": "Qwen/Qwen3-4B-Base",
                "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
                "dtype": "bfloat16",
            },
            "devices": {"cuda_device": "GPU de teste"},
        },
    )
    write_manifest(
        paths.reps_dir,
        "extract",
        settings,
        time.time(),
        extra={
            "prefix_groups": 2,
            "prefix_max_abs_diff": {"L01": 0.0, "L36": 0.0039, "all_layers": 0.0039},
        },
        root=paths.root,
    )


def write_lens_outputs(paths: RunPaths, occ: pd.DataFrame) -> None:
    paths.lens_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    categories = sorted(set(occ["token_category"]))
    for layer in range(0, 4):
        for group_by, group in [("all", "all")] + [("token_category", c) for c in categories]:
            rows.append(
                {
                    "rep": "lex" if layer == 0 else f"L{layer:02d}",
                    "layer": layer,
                    "group_by": group_by,
                    "group": group,
                    "n": 10,
                    "own_rank_median": layer * 40.0,
                    "next_rank_median": 900 / (layer + 1),
                }
            )
    pd.DataFrame(rows).to_csv(paths.lens_dir / "layer_curve.csv", index=False)
    write_json(
        paths.lens_dir / "summary.json",
        {
            "representations": ["lex", "L01", "L18", "L36", "L36n"],
            "ranks": {"lex": {"all": {"own_rank_median": 0, "next_rank_median": 800}}},
            "pred_next_agreement": {"n": 200, "agree": 199, "rate": 0.995},
        },
    )


@pytest.fixture(scope="module")
def reported(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("report")
    settings = tiny_settings()
    paths, occurrences = write_fake_run(root, settings)
    write_corpus(paths, settings)
    write_sample_manifest(paths, settings, occurrences)
    write_extract_outputs(paths, settings, len(occurrences))
    knn.run(settings, paths)
    metrics.run(settings, paths)
    analysis.run(settings, paths)
    write_lens_outputs(paths, occurrences)
    report.run(settings, paths)
    return settings, paths


def test_every_item_is_generated(reported) -> None:
    _, paths = reported
    manifest = read_json(paths.manifest("report"))

    assert manifest["failed"] == []
    for name in report.FIGURES:
        assert manifest["figures"][name] == "gerada", (name, manifest["figures"][name])
        pdf = paths.figures_dir / f"{name}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF"), name
    for name in report.TABLES:
        assert manifest["tables"][name] == "gerada", (name, manifest["tables"][name])
        text = (paths.tables_dir / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in text and "\\bottomrule" in text, name
        assert "\\num{" not in text and not re.search(r"\bnan\b", text.lower()), name


def test_item_lists_match_the_report_sections() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in (RELATORIO / "secoes").glob("*.tex"))
    figures = set(re.findall(r"\\figuraopcional(?:\[[^\]]*\])?\s*\{([^}]*)\}", text))
    tables = set(
        re.findall(r"\\tabelaopcional(?:\[[^\]]*(?:\[[^\]]*\][^\]]*)*\])?\s*\{([^}]*)\}", text)
    )
    figures -= {"viabilidade_tokenizers", "viabilidade_amostragem"}
    assert figures == set(report.FIGURES)
    assert tables == set(report.TABLES)
    provided = set(
        re.findall(
            r"\\providecommand\{\\(\w+)\}",
            (RELATORIO / "relatorio.tex").read_text(encoding="utf-8"),
        )
    )
    assert set(report.MACROS) == provided - {"pendente"}


def test_numbers_use_brazilian_formatting(reported) -> None:
    _, paths = reported
    text = (paths.tables_dir / "numeros.tex").read_text(encoding="utf-8")
    values = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))

    assert values["nPalavrasCorpus"] == "123.456"
    assert values["nTokensCorpus"] == "2.657.592"
    assert values["nTokensProcessados"] == "15.104"
    assert values["nArtigosCandidatos"] == "14"
    assert values["nAlvosMantidos"] == "1"
    assert values["nGruposPrefixo"] == "2"
    assert values["difMaxPrefixo"] == "0,0039"
    assert values["concordanciaPredicao"] == "99,5\\,\\%"
    assert values["nTiposVocab"] == "58"
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", values["dataExecucao"])
    assert set(values) <= set(report.MACROS)
    assert "nVertices" in values and "nTiposVocabCorpus" in values


def test_formatting_helpers() -> None:
    assert report.num(1234.5, 1) == "1.234,5"
    assert report.num(-0.5) == "$-$0,50"
    assert report.num(-0.001) == "0,00"
    assert report.num(float("nan")) == "--"
    assert report.integer(151643) == "151.643"
    assert report.pct(0.9951) == "99,5\\,\\%"
    assert report.sci(2.4e-6) == "$2{,}4\\times10^{-6}$"
    assert report.sci(0.0039) == "0,0039"
    assert report.tex_text("a_b & 中文 c") == "a\\_b \\& [U+4E2D U+6587] c"
    assert report.tex_token(" banco") == "\\texttt{\\textvisiblespace{}banco}"
    assert report.tex_text("--") == "-{}-{}"
    assert report.split_reprs("' de' 'a b' \"it's\"") == [" de", "a b", "it's"]


def test_missing_inputs_are_skipped(tmp_path: Path) -> None:
    settings = tiny_settings()
    paths = RunPaths.from_settings(settings, tmp_path)
    # a stale table from an older run must not survive when its inputs are gone
    paths.tables_dir.mkdir(parents=True)
    (paths.tables_dir / "metricas_globais.tex").write_text("velha", encoding="utf-8")
    report.run(settings, paths)
    manifest = read_json(paths.manifest("report"))

    assert manifest["failed"] == []
    assert all(v.startswith("pulada") for v in manifest["figures"].values())
    assert all(v.startswith("pulada") for v in manifest["tables"].values())
    assert not (paths.tables_dir / "metricas_globais.tex").exists()
    numbers = (paths.tables_dir / "numeros.tex").read_text(encoding="utf-8")
    assert "\\newcommand" not in numbers
    assert manifest["missing_macros"] == report.MACROS


def test_partial_run_writes_what_it_can(tmp_path: Path, reported) -> None:
    settings, source = reported
    paths = RunPaths.from_settings(settings, tmp_path)
    shutil.copytree(source.sample_dir, paths.sample_dir)
    report.run(settings, paths)
    manifest = read_json(paths.manifest("report"))

    assert manifest["tables"]["amostra_composicao"] == "gerada"
    assert manifest["tables"]["amostra_cotas"] == "gerada"
    assert manifest["figures"]["amostra_composicao"] == "gerada"
    assert manifest["figures"]["graus_camadas"].startswith("pulada")
    assert "nVertices" in manifest["macros"]


def _preamble() -> str:
    tex = (RELATORIO / "relatorio.tex").read_text(encoding="utf-8")
    keep = [
        line
        for line in tex.splitlines()
        if re.match(r"\s*\\(usepackage|usetikzlibrary|geometry|captionsetup)", line)
        and "hyperref" not in line
    ]
    return "\\documentclass[11pt,a4paper]{article}\n" + "\n".join(keep) + "\n"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex não instalado")
def test_generated_material_compiles(reported, tmp_path: Path) -> None:
    _, paths = reported
    body = [
        "\\begin{document}",
        "\\input{tabelas/numeros.tex}",
        "\\nVertices; \\nPalavrasCorpus; \\difMaxPrefixo; \\concordanciaPredicao.",
    ]
    for name in report.TABLES:
        body += [
            "\\begin{table}[htbp]\\centering\\caption{x}",
            "\\small",
            f"\\input{{tabelas/{name}.tex}}",
            "\\end{table}",
            "\\clearpage",
        ]
    for name in report.FIGURES:
        body.append(
            f"\\begin{{figure}}\\includegraphics[width=\\linewidth]{{figuras/{name}.pdf}}"
            "\\end{figure}\\clearpage"
        )
    body.append("\\end{document}")
    document = tmp_path / "teste.tex"
    shutil.copytree(paths.tables_dir, tmp_path / "tabelas")
    shutil.copytree(paths.figures_dir, tmp_path / "figuras")
    document.write_text(_preamble() + "\n".join(body) + "\n", encoding="utf-8")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", document.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    log = (tmp_path / "teste.log").read_text(encoding="latin-1")
    assert result.returncode == 0, log[-3000:]
    assert "Missing character" not in log
    overfull = re.findall(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)", log)
    assert all(float(v) < 20 for v in overfull), overfull


def test_manifest_lists_inputs(reported) -> None:
    _, paths = reported
    manifest = read_json(paths.manifest("report"))
    assert any(path.endswith("knn/_manifest.json") for path in manifest["inputs"])
    assert manifest["latex"]["compiled"] is False
    assert json.dumps(manifest)  # serializable
    assert np.isfinite(manifest["elapsed_s"])


def test_compile_report_tolerates_latin1_output(tmp_path: Path, monkeypatch) -> None:
    # pdfTeX prints accented words as Latin-1 bytes; decoding them must not crash the stage.
    fake = tmp_path / "latexmk"
    fake.write_bytes(b"#!/bin/sh\nprintf 'Se\\363es\\n'\nexit 0\n")
    fake.chmod(0o755)
    (tmp_path / "relatorio.tex").write_text("", encoding="utf-8")
    monkeypatch.setattr(report.shutil, "which", lambda name: str(fake))
    assert report.compile_report(tmp_path)["compiled"] is True
