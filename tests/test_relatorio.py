"""Consistency checks between the technical report, the README files and the code they describe.

The report is prose, so these tests only pin the factual claims that a review found wrong once
and that are cheap to check mechanically.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from gender_networks import artifacts, cli

REPO = Path(__file__).resolve().parents[1]
RELATORIO = REPO / "report" / "relatorio"
SECOES = RELATORIO / "secoes"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Collapse line breaks so that phrases split across lines still match."""

    return re.sub(r"\s+", " ", text)


def test_readme_lists_every_loaded_tex_package() -> None:
    tex = read(RELATORIO / "relatorio.tex")
    readme = read(RELATORIO / "README.md")
    packages: set[str] = set()
    for group in re.findall(r"^\s*\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", tex, re.MULTILINE):
        packages.update(name.strip() for name in group.split(","))
    packages -= {"inputenc", "fontenc"}  # part of the LaTeX kernel distribution
    missing = sorted(name for name in packages if f"`{name}`" not in readme)
    assert not missing, f"packages loaded but not listed in report/relatorio/README.md: {missing}"


def test_report_does_not_use_siunitx_num() -> None:
    """Generated numbers are pt-BR strings; \\num would read '15.104' as a decimal."""

    tex = read(RELATORIO / "relatorio.tex")
    code = "\n".join(line.split("%", 1)[0] for line in tex.splitlines())
    assert "siunitx" not in code
    assert "\\num" not in code
    for section in SECOES.glob("*.tex"):
        assert "\\num{" not in read(section), section.name
    assert "sem `\\num`" in read(RELATORIO / "README.md")


def test_manifest_library_list_matches_artifacts() -> None:
    text = flat(read(SECOES / "13-apendices.tex"))
    match = re.search(r"efetivamente usadas em cada etapa \(([^)]*)\)", text)
    assert match, "library list not found in the reproducibility section"
    listed = {name.strip().lower() for name in match.group(1).split(",")}
    assert listed == set(artifacts.library_versions()) | {"python"}


def test_union_graph_is_not_claimed_connected() -> None:
    metodos = flat(read(SECOES / "07-metodos.tex"))
    redes = flat(read(SECOES / "06-redes.tex"))
    assert "costuma ser conexa" not in metodos
    assert "\\emph{não} é necessariamente conexa" in metodos
    assert "fração de vértices no maior componente" in metodos
    assert "as duas versões podem se fragmentar" in redes


def test_noise_floor_equation_uses_disjoint_pairs() -> None:
    metodos = flat(read(SECOES / "07-metodos.tex"))
    assert r"\mathrm{lex}_{2q},\mathrm{lex}_{2q+1}" in metodos
    assert r"m=\lfloor R/2\rfloor" in metodos
    assert r"\operatorname*{\text{média}}_{r\neq r'}" not in metodos


def test_code_facts_in_sections() -> None:
    redes = flat(read(SECOES / "06-redes.tex"))
    dados = flat(read(SECOES / "04-dados.tex"))
    assert "64 candidatos por tipo" not in redes
    assert "blocos de até 1.024 linhas" in redes
    assert "cortado na primeira seção" not in dados
    assert "seções de conteúdo que vêm depois delas são mantidas" in dados
    assert "fórmulas triviais" in dados


def stage_lines(readme: str, name: str) -> list[str]:
    return [ln for ln in readme.splitlines() if ln.startswith(f"uv run gender-networks {name} ")]


def test_readme_flags_stages_without_a_module() -> None:
    """Missing stages are flagged, and the flags must go once the modules exist."""

    readme = read(REPO / "README.md")
    modules = {name: module for name, (module, _) in cli.STAGES.items()}
    missing = {name for name, module in modules.items() if importlib.util.find_spec(module) is None}
    for name in modules:
        lines = stage_lines(readme, name)
        assert lines, name
        for line in lines:
            if name in missing:
                assert "ainda não implementada" in line, line
            else:
                assert "ainda não implementad" not in line, f"stale marker: {line}"
    missing_files = {modules[name].rsplit(".", 1)[-1] + ".py" for name in missing}
    for line in readme.splitlines():
        if "ainda não implementad" in line:
            for filename in re.findall(r"\b(\w+\.py)\b", line):
                assert filename in missing_files, f"stale marker for {filename}: {line}"
    all_line = stage_lines(readme, "all")[0]
    if missing:
        assert "depende de" in all_line
    else:
        assert "depende de" not in all_line
        assert "ModuleNotFoundError" not in readme


def test_seed_claims_match_metrics_plan() -> None:
    """Only the lexical network is measured over every tie seed; the text must say so."""

    from gender_networks.metrics import LEXICAL, plan_communities, plan_graphs
    from gender_networks.settings import load_settings

    settings = load_settings(REPO / "configs" / "experiment.yaml")
    specs = plan_graphs(settings)
    k_main = settings.networks.k_main
    main_rand = [s for s in specs if s.group == "main" and s.tie == "rand" and s.k == k_main]
    lex_seeds = {s.seed for s in main_rand if s.rep == LEXICAL}
    other_seeds = {s.seed for s in main_rand if s.rep != LEXICAL}
    assert lex_seeds == set(range(settings.networks.seeds))
    assert other_seeds == {0}
    requests = plan_communities(settings, specs)
    community_seeds = {int(r.graph_id.rsplit("|s", 1)[1]) for r in requests}
    assert community_seeds <= {0, 1}

    redes = flat(read(SECOES / "06-redes.tex"))
    resultados = flat(read(SECOES / "08-resultados.tex"))
    assert "cada métrica é reportada como média $\\pm$ desvio entre elas" not in redes
    assert "usam a semente 0" in redes
    assert "Barras de erro: desvio entre sementes" not in resultados
    assert "semente 0 nas camadas contextuais" in resultados
    assert "média $\\pm$ desvio entre as 20 sementes nas redes por ocorrência" not in resultados


def test_tie_claims_allow_duplicate_and_near_tied_embeddings() -> None:
    redes = flat(read(SECOES / "06-redes.tex"))
    metodos = flat(read(SECOES / "07-metodos.tex"))
    assert "Empates só acontecem entre tipos com" not in redes
    assert "sem empates (salvo tipos" not in redes
    assert "58 grupos" in redes and "58 grupos" in metodos
    assert "é invariante à semente na etapa lexical" not in metodos
    assert "sempre que o bloco da fronteira tem um só tipo" in metodos


def test_quota_text_counts_core_occurrences() -> None:
    dados = flat(read(SECOES / "04-dados.tex"))
    assert "As ocorrências vêm de parágrafos fora do núcleo" not in dados
    assert "A cota conta as ocorrências do alvo que já estão no núcleo" in dados
    assert "completadas até 20 ocorrências no total" in dados


def test_core_only_networks_marked_as_planned() -> None:
    robustez = flat(read(SECOES / "09-robustez.tex"))
    redes = flat(read(SECOES / "06-redes.tex"))
    assert "Planejado, ainda sem etapa" in robustez
    assert "não é o subgrafo induzido" in robustez
    assert "exige um $k$-NN próprio" in redes


def test_feasibility_figures_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    script = REPO / "scripts" / "feasibility" / "plot_feasibility.py"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for name in ("viabilidade_tokenizers.pdf", "viabilidade_amostragem.pdf"):
        path = tmp_path / name
        assert path.exists() and path.read_bytes().startswith(b"%PDF")
