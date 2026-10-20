"""Versions and timings of every stage (appendix)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from gender_networks.report.context import Context, Skip
from gender_networks.report.latex import DASH, Tabular, _finite, duration, header_row, tex_text
from gender_networks.report.numbers import stage_manifests


def _when(manifest: Mapping[str, Any]) -> str:
    stamp = _finite(manifest.get("started_unix"))
    if stamp is None:
        return DASH
    finished = stamp + (_finite(manifest.get("elapsed_s")) or 0.0)
    return datetime.fromtimestamp(finished).strftime("%d/%m/%Y %H:%M")


def table_versoes(ctx: Context) -> Tabular:
    manifests = stage_manifests(ctx)
    if not manifests:
        raise Skip("nenhum _manifest.json de etapa")
    rows: list[Any] = []
    for stage, manifest in manifests.items():
        rows.append(
            [
                r"\texttt{" + stage + "}",
                _when(manifest),
                duration(manifest.get("elapsed_s")),
                r"\texttt{" + tex_text(manifest.get("git") or DASH) + "}",
                r"\texttt{" + tex_text(Path(str(manifest.get("config_source") or DASH)).name) + "}",
            ]
        )
    versions: dict[str, list[str]] = {}
    for manifest in manifests.values():
        for name, version in (manifest.get("versions") or {}).items():
            versions.setdefault(name, [])
            if str(version) not in versions[name]:
                versions[name].append(str(version))
    width = 5
    rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{Bibliotecas e modelo}}}} \\")
    for name, values in versions.items():
        rows.append(
            [tex_text(name), r"\multicolumn{4}{l}{\texttt{" + tex_text(" / ".join(values)) + "}}"]
        )
    model = ctx.settings.model
    revision = model.revision
    verify = ctx.paths.verify_report
    devices: dict[str, Any] = {}
    if verify.exists():
        report = ctx.json(verify)
        revision = (report.get("model") or {}).get("revision") or revision
        devices = report.get("devices") or {}
    rows.append(
        [
            "modelo",
            rf"\multicolumn{{4}}{{l}}{{\texttt{{{tex_text(model.name_or_path)}}} "
            rf"(\texttt{{{tex_text(model.dtype)}}})}}",
        ]
    )
    rows.append(["revisão", rf"\multicolumn{{4}}{{l}}{{\texttt{{{tex_text(revision)}}}}}"])
    if devices.get("cuda_device"):
        rows.append(["GPU", rf"\multicolumn{{4}}{{l}}{{{tex_text(devices['cuda_device'])}}}"])
    header = [
        header_row(["Etapa", "Concluída em", "Tempo", "Revisão \\texttt{git}", "Configuração"])
    ]
    return Tabular("lllll", header, rows, size=r"\footnotesize")
