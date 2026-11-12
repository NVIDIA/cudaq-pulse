# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release tag checks, the workflow pin check, the release hygiene check,
the GPU CI matrix, the classical remnant check and the license header check
(source tree only)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
pytestmark = pytest.mark.skipif(
    not (_TOOLS / "release-meta.py").exists(), reason="needs the source tree's tools/"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tag, version, prerelease",
    [
        ("v0.1.0", "0.1.0", False),
        ("v0.1.0rc1", "0.1.0rc1", True),
        ("v2.0.3b4", "2.0.3b4", True),
        ("v0.1.0.post1", "0.1.0.post1", False),
    ],
)
def test_release_tags(tag, version, prerelease):
    assert _load("release-meta").release_version(tag) == (version, prerelease)


@pytest.mark.parametrize(
    "tag",
    ["0.1.0", "v0.1", "v0.1.0rc", "v0.1.0.dev1", "v0.1.0+local", "v0.1.0rc1.post1", "v0.1.0post1"],
)
def test_invalid_release_tags(tag):
    with pytest.raises(ValueError):
        _load("release-meta").release_version(tag)


def test_a_release_is_blocked_while_the_perf_gate_is_bootstrapping(tmp_path):
    meta = _load("release-meta")
    baseline = tmp_path / "perf_gate.json"
    baseline.write_text(json.dumps({"format": 1, "bootstrap": True, "cases": {}}))
    assert "bootstrap" in meta.release_blockers(baseline)[0]
    baseline.write_text(json.dumps({"format": 1, "cases": {"c": {"instructions": 1}}}))
    assert meta.release_blockers(baseline) == []
    assert meta.release_blockers(tmp_path / "absent.json") == []


_CHANGELOG = """# Changelog

## [Unreleased]

- Next thing.

## [0.2.0] - Unreleased

### Added

- Second thing.

## [0.1.0] - 2026-11-02

- First thing.

[Unreleased]: https://example.invalid/compare
[0.1.0]: https://example.invalid/v0.1.0
"""


def test_release_notes_come_from_the_version_section():
    notes = _load("release-notes").release_notes
    assert notes("0.1.0", _CHANGELOG) == "- First thing.\n"
    assert notes("0.2.0rc1", _CHANGELOG) == "### Added\n\n- Second thing.\n"


@pytest.mark.parametrize(
    "version, message",
    [("0.2.0", "must be dated"), ("0.3.0rc1", "no '## \\[0.3.0\\]'"), ("0.1.0.post1", "no")],
)
def test_release_notes_fail_closed(version, message):
    with pytest.raises(ValueError, match=message):
        _load("release-notes").release_notes(version, _CHANGELOG)


def test_release_notes_reject_an_empty_section():
    with pytest.raises(ValueError, match="empty"):
        _load("release-notes").release_notes("0.1.0rc1", "## [0.1.0] - Unreleased\n\n## [0.0.1]\n")


def _next_release(changelog):
    """The version of the changelog's undated section and the section's lines."""
    sections = re.split(r"^## ", changelog, flags=re.M)[1:]
    undated = [s for s in sections if not re.match(r"\[[^\]]+\] - \d{4}-\d{2}-\d{2}\s", s)]
    assert len(undated) == 1, "CHANGELOG.md needs exactly one undated section"
    heading, *lines = undated[0].splitlines()
    match = re.fullmatch(r"\[(\d+\.\d+\.\d+)\] - Unreleased", heading)
    assert match, f"the undated section is '## {heading}', not '## [X.Y.Z] - Unreleased'"
    return match.group(1), lines


def test_the_release_notes_hold_every_unreleased_entry():
    changelog = (_TOOLS.parent / "CHANGELOG.md").read_text()
    version, lines = _next_release(changelog)
    notes = _load("release-notes").release_notes(f"{version}rc1", changelog)
    entries = [line for line in lines if line.startswith("- ")]
    assert entries and "### Added" in notes
    assert [line for line in entries if line not in notes.splitlines()] == []


@pytest.mark.parametrize(
    "page", ["README.md", "docs/getting_started.rst", "docs/user_guide/gpu_execution.rst"]
)
def test_the_install_commands_install_from_pypi(page):
    text = (_TOOLS.parent / page).read_text()
    commands = re.findall(r"pip install (\"?cudaq-pulse\S*)", text)
    assert commands
    assert all(re.fullmatch(r'cudaq-pulse|"cudaq-pulse\[\w+\]"', c) for c in commands), commands
    assert "expanded_assets" not in text and "--find-links" not in text


@pytest.mark.parametrize(
    "workflow",
    sorted((_TOOLS.parent / ".github" / "workflows").glob("*.yml")),
    ids=lambda p: p.name,
)
def test_every_workflow_job_has_a_timeout(workflow):
    jobs = workflow.read_text().split("\njobs:\n", 1)[1]
    blocks = re.split(r"^  (?=[\w-]+:\s*$)", jobs, flags=re.M)[1:]
    assert blocks
    # A job that calls a reusable workflow cannot set one; the called jobs do.
    missing = [
        b.split(":", 1)[0]
        for b in blocks
        if not re.search(r"^    (timeout-minutes|uses):", b, re.M)
    ]
    assert not missing, f"{workflow.name}: jobs without timeout-minutes: {missing}"


def _gpu_matrix():
    return _load("ci/gpu-matrix")


def test_gpu_runners_variable():
    parse = _gpu_matrix().parse_runners
    assert parse("") == parse("  ") == {}
    assert parse("gpu-label") == {"x86_64": "gpu-label"}
    assert parse('["a", "b"]') == {"x86_64": ["a", "b"]}
    both = '{"x86_64": "a", "aarch64": ["b", "c"]}'
    assert parse(both) == {"x86_64": "a", "aarch64": ["b", "c"]}
    for bad in ('{"x86": "a"}', '{"x86_64": ""}', '{"x86_64": []}', '{"x86_64": [1]}'):
        with pytest.raises(ValueError):
            parse(bad)


def test_gpu_matrix_scopes_and_roles():
    gpu = _gpu_matrix()
    every = {role: f"label-{role}" for role in gpu.ROLES}
    quick, skipped = gpu.plan("wheels", "quick", every)
    assert [(r["extra"], r["runner"]) for r in quick] == [
        ("cu12", "label-x86_64"),
        ("cu13", "label-x86_64"),
    ]
    assert not skipped
    full, _ = gpu.plan("wheels", "full", every)
    assert len(full) == len(gpu.ROWS["wheels"])
    assert {r["platform"] for r in full} == {"linux-x86_64", "linux-aarch64"}
    old = [r for r in full if r["runner"] == "label-x86_64-min-driver"]
    assert {r["extra"] for r in old} == {"cu12", "cu13"}
    assert all(r["flags"] == "--allow-old-driver" for r in old)
    assert all(r["flags"] == "" for r in full if r not in old)
    partial, skipped = gpu.plan("wheels", "full", {"x86_64": "a"})
    assert {r["runner"] for r in partial} == {"a"} and len(skipped) == 4
    build, _ = gpu.plan("build", "full", every)
    assert [(r["extra"], r["image"]) for r in build] == [("cu12", gpu.CUDA_DEVEL)]


def test_gpu_matrix_images_are_pinned():
    gpu = _gpu_matrix()
    for rows in gpu.ROWS.values():
        for row in rows:
            assert re.search(r"@sha256:[0-9a-f]{64}$", row.image), row.image


def test_gpu_matrix_output(capsys, monkeypatch, tmp_path):
    main = _gpu_matrix().main
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("GPU_RUNNERS", "")
    assert main(["--source", "wheels"]) == 1
    assert "matrix=" not in capsys.readouterr().out
    monkeypatch.setenv("GPU_RUNNERS", '{"aarch64": "a"}')
    assert main(["--source", "build", "--scope", "full"]) == 1
    monkeypatch.setenv("GPU_RUNNERS", '{"x86_64": ["a", "b"]}')
    assert main(["--source", "wheels", "--scope", "full"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("matrix=") and out.count("\n") == 1
    matrix = json.loads(out.removeprefix("matrix="))
    assert [r["runner"] for r in matrix["include"]] == [["a", "b"]] * 3
    assert "skipped: no aarch64 runner" in (tmp_path / "summary.md").read_text()


def test_workflow_pins(tmp_path):
    check = _load("check-workflow-pins").check
    assert check(_TOOLS.parent / ".github" / "workflows" / "wheels.yml") == []
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "jobs:\n  a:\n    container: quay.io/x/y\n    steps:\n"
        "      - uses: actions/checkout@v7\n"
        "      - uses: actions/checkout@0123456789abcdef0123456789abcdef01234567\n"
        "      - uses: ./.github/actions/local\n"
    )
    assert len(check(bad)) == 3


def test_release_hygiene(tmp_path):
    check = _load("check-release-hygiene").check
    assert check(_TOOLS.parent / "SECURITY.md") == []
    # Assembled at run time so this file passes the check itself.
    bad = [
        "/" + "Users/someone/src",
        "https://build." + "nvi" + "dia.com/x",
        "10." + "1.2.3",
        "-----BEGIN " + "RSA PRIVATE KEY-----",
        "gh" + "p_" + "a" * 36,
    ]
    version = "10." + "4.4.72"
    good = [f'version = "{version}"', f"nvidia_curand-{version}-py3", "www.nvidia.com/en-us"]
    path = tmp_path / "f.txt"
    path.write_text("\n".join(bad + good) + "\n")
    assert [error.split(":")[1] for error in check(path)] == ["1", "2", "3", "4", "5"]


def test_classical_remnants(tmp_path):
    check = _load("check-classical-remnants").check
    assert check(_TOOLS.parent / "docs" / "user_guide" / "classical_calls.rst") == []
    # Assembled at run time so this file passes the check itself.
    old = "Fl" + "oats"
    bad = [
        f"from cudaq_pulse.classical import {old}",
        f"x = classical.{old}",
        "@pulse.classical(ar" + "gs=(Bi" + "ts(1),), max_latency_ns=1)",
        f"A ``{old}(7)`` argument.",
        "def f(b: Bi" + "ts[3]) -> None: ...",
        "@pulse.classical(max_latency_ns=1, len" + "gths={'bits': 3})",
        "A ``len" + "gths['return']`` entry.",
        "@pulse.classical.external(max_latency_ns=1)",
        "def table(index: pulse.Bits) -> float:",
        "    raise Not" + "ImplementedError",
    ]
    good = [
        "@pulse.classical(ar" + "gs=()) is rejected  # remnant-ok",
        "class Bits(list):",
        "def _schema_function(name, args=(), results=()):",
        "_Floats(2)",
        "n = lengths[r.source]",
        "@pulse.classical.external(max_latency_ns=1)",
        "def table(index: pulse.Length[pulse.Bits, 3]) -> float:",
        "    return float()",
        "def other() -> None:",
        "    raise Not" + "ImplementedError",
    ]
    path = tmp_path / "f.py"
    path.write_text("\n".join(bad + good) + "\n")
    assert [int(error.split(":")[1]) for error in check(path)] == [1, 2, 3, 3, 4, 5, 6, 7, 10]
    notebook = tmp_path / "n.ipynb"
    notebook.write_text(json.dumps({"cells": [{"source": ["x = 1\n", bad[1]]}]}))
    assert [error.split(":")[1] for error in check(notebook)] == ["2"]


def test_removed_names(tmp_path):
    tool = _load("check-removed-names")
    for page in ("user_guide/quantum_systems.rst", "user_guide/classical_calls.rst"):
        assert tool.check(_TOOLS.parent / "docs" / page) == []
    # Assembled at run time, as the docs must not spell them.
    bad = [
        "qs = pulse." + "device('transmon_" + "17q')",
        "from cudaq_pulse." + "targets import Qubit",
        "h = qs.qudit_" + "handles[0]",
        "program = scheduled.export_" + "controller()",
        "result = pulse.simulate(k, " + "shots=100)",
    ]
    good = [
        "qs = pulse.quantum_system('transmon_surface17')",
        "h = qs.qudits[0]",
        "result = pulse.simulate(k, shots_count=100)",
        "a transmon_" + "17q-style name  # rename-ok",
    ]
    path = tmp_path / "f.rst"
    path.write_text("\n".join(bad + good) + "\n")
    errors = tool.check(path)
    assert [int(error.split(":")[1]) for error in errors] == [1, 1, 2, 3, 4, 5]
    assert any("use transmon_surface17" in error for error in errors)
    assert tool.covered(Path("docs/index.rst")) and tool.covered(Path("README.md"))
    assert not tool.covered(Path("tests/test_x.py"))


_SPDX = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.",
    "SPDX-License-Identifier: Apache-2.0",
)


@pytest.mark.parametrize(
    "header",
    [
        [f"# {line}" for line in _SPDX],
        ["#!/usr/bin/env python3", *(f"# {line}" for line in _SPDX)],
        [f"// {line}" for line in _SPDX],
        ["//===- a.cpp - A file -----===//", "//", *(f"// {line}" for line in _SPDX), "//"],
        ["/*", *(f" * {line}" for line in _SPDX), " */"],
        ["..", *(f"   {line}" for line in _SPDX)],
    ],
)
def test_license_header_accepts_the_spdx_pair(tmp_path, header):
    path = tmp_path / "f"
    path.write_text("\n".join([*header, "", "code"]) + "\n")
    assert _load("check-license-headers").problem(path) is None


@pytest.mark.parametrize(
    "header, message",
    [
        (
            [
                "# Copyright (c) 2026 NVIDIA Corporation & Affiliates.",
                "# All rights reserved.",
                "# This source code and the accompanying materials are made available under",
                "# the terms of the Apache License 2.0 which accompanies this distribution.",
            ],
            "old prose",
        ),
        ([f"# {line.replace('CORPORATION', 'Corporation')}" for line in _SPDX], "old prose"),
        ([f"# {_SPDX[0]}"], "missing"),
        ([f"# {_SPDX[0]}", f"// {_SPDX[1]}"], "missing"),
        ([f"# {_SPDX[0].replace('2026', '26')}", f"# {_SPDX[1]}"], "missing"),
        (["", *[f"#{i}" for i in range(12)], *(f"# {line}" for line in _SPDX)], "missing"),
    ],
)
def test_license_header_refuses_anything_else(tmp_path, capsys, header, message):
    check = _load("check-license-headers")
    path = tmp_path / "f"
    path.write_text("\n".join([*header, "code"]) + "\n")
    assert message in check.problem(path)
    assert check.main([str(path)]) == 1
    assert capsys.readouterr().out.startswith(f"{path}: ")


def _locked(package):
    lock = (_TOOLS.parent / "uv.lock").read_text()
    match = re.search(rf'^name = "{package}"\nversion = "([^"]+)"', lock, re.M)
    assert match, package
    return match.group(1)


def test_tool_pins_agree():
    precommit = (_TOOLS.parent / ".pre-commit-config.yaml").read_text()
    ci = (_TOOLS.parent / ".github" / "workflows" / "ci.yml").read_text()
    revs = dict(re.findall(r"repo: https://github.com/\S+/(\S+)\n\s+rev: v(\S+)", precommit))
    assert revs["ruff-pre-commit"] == _locked("ruff")
    assert set(re.findall(r"zizmor==(\S+)", ci)) == {revs["zizmor-pre-commit"]}
    assert set(re.findall(r"pyright==(\S+)", ci)) == {_locked("pyright")}
