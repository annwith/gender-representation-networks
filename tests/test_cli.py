from pathlib import Path

from gender_networks.cli import STAGES, build_parser


def test_config_is_accepted_before_or_after_the_stage() -> None:
    parser = build_parser()

    before = parser.parse_args(["--config", "configs/mini.yaml", "sample"])
    after = parser.parse_args(["sample", "--config", "configs/mini.yaml", "--force"])
    default = parser.parse_args(["knn"])

    assert before.config == after.config == Path("configs/mini.yaml")
    assert after.force is True
    assert default.config == Path("configs/experiment.yaml")
    assert default.log_level == "INFO"


def test_every_stage_has_a_subcommand_and_extract_has_verify() -> None:
    parser = build_parser()

    for stage in STAGES:
        assert parser.parse_args([stage]).stage == stage
    assert parser.parse_args(["extract", "--verify"]).verify is True


def test_all_force_does_not_force_the_corpus(monkeypatch) -> None:
    from gender_networks import cli

    calls: list[tuple[str, bool]] = []
    def record(name, config, root, force, **options):
        calls.append((name, force))

    monkeypatch.setattr(cli, "run_stage", record)

    cli.main(["all", "--force"])
    assert calls[0] == ("corpus", False) and all(force for _, force in calls[1:])
    calls.clear()
    cli.main(["all", "--force-corpus"])
    assert calls[0] == ("corpus", True) and not any(force for _, force in calls[1:])


def test_lens_device_option_reaches_the_stage(monkeypatch) -> None:
    from gender_networks import cli

    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_stage", lambda name, config, root, force, **o: seen.update(o))

    cli.main(["lens", "--device", "cpu"])
    assert seen == {"device": "cpu"}
