from pathlib import Path

import pytest

from gender_networks.artifacts import RunPaths, position_bucket
from gender_networks.settings import load_settings


def test_main_config_loads_and_validates() -> None:
    settings = load_settings(Path("configs/experiment.yaml"))

    assert settings.name == "main"
    assert settings.model.name_or_path == "Qwen/Qwen3-4B-Base"
    assert settings.model.layers == {"L01": 1, "L18": 18, "L36": 36}
    assert len(settings.corpus.themes) == 8
    assert settings.sample.targets["banco"] == {"economia": 17, "computacao": 17, "geografia": 16}
    assert settings.networks.k_main == 10
    assert settings.corpus.paragraph.min_words == 40


def test_mini_extends_main_replacing_keys_within_sections() -> None:
    mini = load_settings(Path("configs/mini.yaml"))
    main = load_settings(Path("configs/experiment.yaml"))

    assert mini.name == "mini"
    assert mini.corpus == main.corpus  # same shared corpus
    assert mini.sample.targets == {"carga": {"fisica": 12, "economia": 12}}
    assert mini.sample.f_max == main.sample.f_max  # inherited key
    assert mini.paths.corpus_dir == main.paths.corpus_dir
    assert mini.paths.report_dir != main.paths.report_dir


def test_unknown_keys_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\ncorpus:\n  themes: {a: [Categoria:A]}\n  tema: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tema"):
        load_settings(path)


def test_quota_themes_must_exist(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\ncorpus:\n  themes: {a: [Categoria:A]}\nsample:\n  targets: {w: {b: 3}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown theme"):
        load_settings(path)


def test_run_paths_and_position_buckets(tmp_path: Path) -> None:
    paths = RunPaths.from_settings(load_settings(Path("configs/mini.yaml")), tmp_path)

    assert paths.occurrences == tmp_path / "outputs/experiment/mini/sample/occurrences.csv"
    assert paths.rep("L18").name == "L18.safetensors"
    assert [position_bucket(p) for p in (1, 4, 5, 16, 17, 64, 65, 400)] == [
        "1-4", "1-4", "5-16", "5-16", "17-64", "17-64", "65+", "65+",
    ]
    with pytest.raises(ValueError):
        position_bucket(0)
