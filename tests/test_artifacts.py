import os
import time
from dataclasses import replace
from pathlib import Path

from gender_networks.artifacts import (
    file_fingerprint,
    settings_digest,
    stage_is_fresh,
    token_text,
    write_manifest,
)
from gender_networks.settings import load_settings


def setup(tmp_path: Path):
    settings = load_settings(Path("configs/mini.yaml"))
    upstream = tmp_path / "upstream.json"
    upstream.write_text("{}", encoding="utf-8")
    stage = tmp_path / "stage"
    output = stage / "out.csv"
    stage.mkdir()
    output.write_text("x", encoding="utf-8")
    write_manifest(stage, "test", settings, time.time(), inputs=[upstream], root=tmp_path)
    return settings, upstream, stage, output


def test_stage_is_fresh_when_inputs_and_settings_are_unchanged(tmp_path: Path) -> None:
    settings, upstream, stage, output = setup(tmp_path)

    assert stage_is_fresh(stage, settings, [upstream], [output])


def test_rerunning_an_upstream_stage_makes_the_stage_stale(tmp_path: Path) -> None:
    settings, upstream, stage, output = setup(tmp_path)

    upstream.write_text('{"rerun": true}', encoding="utf-8")
    os.utime(upstream, ns=(time.time_ns(), time.time_ns() + 10**9))

    assert not stage_is_fresh(stage, settings, [upstream], [output])


def test_changed_settings_or_missing_outputs_make_the_stage_stale(tmp_path: Path) -> None:
    settings, upstream, stage, output = setup(tmp_path)
    other = replace(settings, sample=replace(settings.sample, f_max=7))

    assert settings_digest(other) != settings_digest(settings)
    assert not stage_is_fresh(stage, other, [upstream], [output])
    output.unlink()
    assert not stage_is_fresh(stage, settings, [upstream], [output])


def test_settings_digest_ignores_the_config_path_and_missing_inputs_are_recorded(
    tmp_path: Path,
) -> None:
    settings = load_settings(Path("configs/mini.yaml"))

    assert settings_digest(replace(settings, source="elsewhere.yaml")) == settings_digest(settings)
    assert file_fingerprint([tmp_path / "nope"]) == {str(tmp_path / "nope"): None}


def test_a_stage_is_stale_only_for_the_settings_it_reads(tmp_path: Path) -> None:
    settings, upstream, _, _ = setup(tmp_path)
    stage = tmp_path / "knn"
    output = stage / "out.npz"
    stage.mkdir()
    output.write_text("x", encoding="utf-8")
    write_manifest(stage, "knn", settings, time.time(), inputs=[upstream], root=tmp_path)
    later = replace(settings, analysis=replace(settings.analysis, permutations=99))
    assert stage_is_fresh(stage, later, [upstream], [output])
    changed = replace(settings, networks=replace(settings.networks, k_main=5))
    assert not stage_is_fresh(stage, changed, [upstream], [output])
    assert settings_digest(later, "knn") == settings_digest(settings, "knn")
    assert settings_digest(later, "analyze") != settings_digest(settings, "analyze")


def test_token_text_reads_the_vocab_repr_column() -> None:
    assert token_text("' banco'") == " banco"
    assert token_text('"it\'s"') == "it's"
    assert token_text("'\\tload'") == "\tload"
    assert token_text("not a repr") == "not a repr"
