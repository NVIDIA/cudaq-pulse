# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release tag checks, the release bundle check, the workflow pin check,
the release hygiene check, the GPU CI matrix, the classical remnant check and
the license header check, and the MLIR toolchain fetcher (source tree only)."""

import argparse
import hashlib
import importlib.util
import json
import re
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError

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


@pytest.mark.parametrize(
    "version, release, message",
    [
        ("0.1.0", True, None),
        ("0.1.0rc1", True, None),
        ("0.1.0.post1", True, None),
        ("0.1.dev5", False, None),
        ("0.1.0rc2.dev1", False, None),
        ("0.1.dev5+gc807058fb", False, "local segment"),
        ("0.1.0+local", True, "local segment"),
        ("0.1.dev5", True, "not a release version"),
        ("0.1", True, "not a release version"),
        ("0.1.0", False, "not a dev version"),
        ("v0.1.0", True, "not a normalized"),
        ("0.1.0-rc1", True, "not a normalized"),
    ],
)
def test_dist_versions(version, release, message):
    problems = _load("check-dist").check_version(version, release)
    assert problems == [] if message is None else message in problems[0]


_DIST_PLATFORMS = [
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_11_0_arm64",
]


def _dist_file(dist, name, version, tags=None, *, meta_version=None, size=0):
    import io
    import zipfile

    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {meta_version or version}\n\n"
    stem = f"{name.replace('-', '_')}-{version}"
    if tags is None:
        path = dist / f"{stem}.tar.gz"
        with tarfile.open(path, "w:gz") as sdist:
            info = tarfile.TarInfo(f"{stem}/PKG-INFO")
            info.size = len(metadata)
            sdist.addfile(info, io.BytesIO(metadata.encode()))
    else:
        path = dist / f"{stem}-{tags}.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(f"{stem}.dist-info/METADATA", metadata)
            wheel.writestr("padding", b"\0" * size)
    return path


def _bundle(tmp_path, version="0.1.0rc1"):
    dist = tmp_path / "bundle" / "dist"
    dist.mkdir(parents=True)
    _dist_file(dist, "cudaq-pulse", version)
    for platform in _DIST_PLATFORMS:
        _dist_file(dist, "cudaq-pulse", version, f"cp312-abi3-{platform}")
        _dist_file(dist, "cudaq-pulse", version, f"cp314-cp314t-{platform}")
        _dist_file(dist, "cudaq-pulse-tools", version, f"py3-none-{platform}")
    return dist


def test_a_complete_bundle_passes(tmp_path):
    check = _load("check-dist")
    dist = _bundle(tmp_path)
    assert check.check_files(sorted(dist.iterdir()), "0.1.0rc1", 90) == []


def test_a_bundle_needs_every_file_once_at_its_version(tmp_path):
    check = _load("check-dist")
    dist = _bundle(tmp_path)
    (dist / "cudaq_pulse_tools-0.1.0rc1-py3-none-macosx_11_0_arm64.whl").unlink()
    _dist_file(dist, "cudaq-pulse", "0.1.0rc1", "cp313-cp313-manylinux_2_28_x86_64")
    _dist_file(dist, "cudaq-pulse", "0.1.0rc2", "cp312-abi3-win_amd64")
    _dist_file(dist, "cudaq-pulse", "0.1.0rc0")
    (dist / "SHA256SUMS").write_text("")
    problems = check.check_files(sorted(dist.iterdir()), "0.1.0rc1", 90)
    assert "missing cudaq_pulse_tools py3-none wheel for macos-arm64" in problems
    assert any("cp313-cp313" in p and "not one of the expected" in p for p in problems)
    assert any("win_amd64" in p and "file name version 0.1.0rc2" in p for p in problems)
    assert any(p.startswith("more than one cudaq_pulse sdist") for p in problems)
    assert "SHA256SUMS: not a wheel or sdist" in problems


def test_a_bundle_wheel_has_one_platform_tag(tmp_path):
    check = _load("check-dist")
    dist = _bundle(tmp_path)
    (dist / "cudaq_pulse-0.1.0rc1-cp312-abi3-manylinux_2_28_x86_64.whl").unlink()
    double = "manylinux_2_27_x86_64.manylinux_2_28_x86_64"
    _dist_file(dist, "cudaq-pulse", "0.1.0rc1", f"cp312-abi3-{double}")
    problems = check.check_files(sorted(dist.iterdir()), "0.1.0rc1", 90)
    name = f"cudaq_pulse-0.1.0rc1-cp312-abi3-{double}.whl"
    assert f"{name}: platform tags {double}, expected exactly one" in problems
    assert "missing cudaq_pulse cp312-abi3 wheel for linux-x86_64" in problems


def test_a_bundle_file_must_carry_its_version_and_fit(tmp_path):
    check = _load("check-dist")
    dist = _bundle(tmp_path)
    wheel = "cudaq_pulse-0.1.0rc1-cp312-abi3-macosx_11_0_arm64.whl"
    (dist / wheel).unlink()
    _dist_file(
        dist, "cudaq-pulse", "0.1.0rc1", "cp312-abi3-macosx_11_0_arm64", meta_version="0.1.0"
    )
    _dist_file(
        dist, "cudaq-pulse-tools", "0.1.0rc1", "py3-none-manylinux_2_28_aarch64", size=1 << 20
    )
    problems = check.check_files(sorted(dist.iterdir()), "0.1.0rc1", 0.5)
    assert f"{wheel}: metadata Version 0.1.0, expected 0.1.0rc1" in problems
    assert any("aarch64.whl: 1.0 MiB, over the 0.5 MiB limit" in p for p in problems)


def test_bundle_digests(tmp_path):
    check = _load("check-dist")
    dist = _bundle(tmp_path)
    files = sorted(dist.iterdir())
    (dist.parent / check.SUMS).write_text(check.render_sums(files))
    assert check.verify_sums(dist.parent, files) == []
    (dist / "cudaq_pulse-0.1.0rc1.tar.gz").write_bytes(b"changed")
    _dist_file(dist, "cudaq-pulse", "0.1.0rc1", "cp312-abi3-linux_x86_64")
    problems = check.verify_sums(dist.parent, sorted(dist.iterdir()))
    assert problems == [
        "SHA256SUMS does not list cudaq_pulse-0.1.0rc1-cp312-abi3-linux_x86_64.whl",
        "cudaq_pulse-0.1.0rc1.tar.gz: SHA-256 differs from SHA256SUMS",
    ]
    manifest = json.loads(check.manifest(files[:1], "0.1.0rc1"))
    assert manifest["files"][0]["distribution"] == "cudaq-pulse"
    assert manifest["files"][0]["file"] == f"dist/{files[0].name}"


def _top_release(changelog):
    """The version the changelog's top section releases (its own version once
    dated, else its first release candidate) and the section's lines. Only
    the top section may be undated, as ``## [X.Y.Z] - Unreleased``; it is
    dated when X.Y.Z is released."""
    sections = re.split(r"^## ", changelog, flags=re.M)[1:]
    undated = [s for s in sections if not re.match(r"\[[^\]]+\] - \d{4}-\d{2}-\d{2}\s", s)]
    assert undated in ([], sections[:1]), "only the top section of CHANGELOG.md may be undated"
    heading, *lines = sections[0].splitlines()
    match = re.fullmatch(r"\[(\d+\.\d+\.\d+)\] - (Unreleased|\d{4}-\d{2}-\d{2})", heading)
    assert match, f"the top section is '## {heading}', not '## [X.Y.Z] - Unreleased' or dated"
    version = match.group(1)
    return f"{version}rc1" if match.group(2) == "Unreleased" else version, lines


def test_the_release_notes_hold_every_entry_of_the_top_section():
    changelog = (_TOOLS.parent / "CHANGELOG.md").read_text()
    release, lines = _top_release(changelog)
    notes = _load("release-notes").release_notes(release, changelog)
    entries = [line for line in lines if line.startswith("- ")]
    assert entries and re.search(r"^### \w", notes, flags=re.M)
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


@pytest.fixture
def fetch_mlir():
    return _load("fetch-mlir-toolchain")


@pytest.mark.parametrize(
    "system, machine, expected",
    [("Darwin", "arm64", "macos-aarch64"), ("Windows", "AMD64", None)],
)
def test_mlir_host_platform(fetch_mlir, monkeypatch, system, machine, expected):
    monkeypatch.setattr(fetch_mlir.platform, "system", lambda: system)
    monkeypatch.setattr(fetch_mlir.platform, "machine", lambda: machine)
    if expected is None:
        with pytest.raises(fetch_mlir.FetchError, match=r"build-mlir-toolchain\.sh"):
            fetch_mlir.host_platform()
    else:
        assert fetch_mlir.host_platform() == expected


@pytest.mark.parametrize(
    "name, explicit, expected, error",
    [
        ("pinned.tar.xz", None, "a" * 64, None),
        ("local.tar.xz", "B" * 64, "b" * 64, None),
        ("pinned.tar.xz", "b" * 64, None, "contradicts the pinned checksum"),
        ("local.tar.xz", None, None, "pins no checksum"),
    ],
)
def test_mlir_expected_sha256(fetch_mlir, name, explicit, expected, error):
    lock = {"assets": {"pinned.tar.xz": "a" * 64}}
    if error is None:
        assert fetch_mlir.expected_sha256(lock, name, explicit) == expected
    else:
        with pytest.raises(fetch_mlir.FetchError, match=error):
            fetch_mlir.expected_sha256(lock, name, explicit)


_MLIR_CONFIG = "lib/cmake/mlir/MLIRConfig.cmake"


def _mlir_archive(tmp_path, name, member=_MLIR_CONFIG):
    archive = tmp_path / "local.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.addfile(tarfile.TarInfo(f"{name}/{member}"))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_mlir_unpack_rejects_checksum_mismatch(fetch_mlir, tmp_path):
    archive, _ = _mlir_archive(tmp_path, "toolchain")
    cache = tmp_path / "cache"
    with pytest.raises(fetch_mlir.FetchError, match="checksum mismatch"):
        fetch_mlir.unpack(archive, "0" * 64, "toolchain", cache)
    assert not cache.exists()


def test_mlir_unpack_requires_mlir_config(fetch_mlir, tmp_path):
    archive, digest = _mlir_archive(tmp_path, "toolchain", "README")
    cache = tmp_path / "cache"
    with pytest.raises(fetch_mlir.FetchError, match=r"does not contain .*MLIRConfig\.cmake"):
        fetch_mlir.unpack(archive, digest, "toolchain", cache)
    assert not (cache / "toolchain").exists()


def test_mlir_unpack_reuses_the_cached_tree(fetch_mlir, tmp_path):
    archive, digest = _mlir_archive(tmp_path, "toolchain")
    cache = tmp_path / "cache"
    target = fetch_mlir.unpack(archive, digest, "toolchain", cache)
    assert target == cache / "toolchain"
    assert (target / fetch_mlir.MARKER).read_text().strip() == digest
    config = target / _MLIR_CONFIG
    config.write_text("# keep this local change\n")
    assert fetch_mlir.unpack(archive, digest, "toolchain", cache) == target
    assert config.read_text() == "# keep this local change\n"


@pytest.mark.parametrize(
    "error, message",
    [
        (HTTPError("https://example.invalid", 404, "Not Found", {}, None), "HTTP 404"),
        (TimeoutError(), "timed out: no data for 2.5 s"),
        (URLError(ConnectionRefusedError()), "cannot connect"),
    ],
)
def test_mlir_download_error(fetch_mlir, error, message):
    result = fetch_mlir.download_error("https://example.invalid", error, 2.5)
    assert isinstance(result, fetch_mlir.FetchError)
    assert message in str(result)
    assert "--from-file" in str(result)


@pytest.mark.parametrize("value", ["0", "-1", "inf", "abc"])
def test_mlir_timeout_rejects_invalid_values(fetch_mlir, value):
    with pytest.raises(argparse.ArgumentTypeError, match="expected a positive number"):
        fetch_mlir.positive_seconds(value)


def test_mlir_fetch_from_file(fetch_mlir, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CUDAQ_PULSE_MLIR_FETCH_TIMEOUT", raising=False)
    monkeypatch.setattr(fetch_mlir, "download", lambda *_: pytest.fail("unexpected download"))
    platform = "linux-x86_64"
    name = fetch_mlir.toolchain_name(fetch_mlir.load_lock(), platform)
    archive, digest = _mlir_archive(tmp_path, name)
    cache = tmp_path / "cache"
    args = [
        "--from-file",
        str(archive),
        "--sha256",
        digest,
        "--platform",
        platform,
        "--cache-dir",
        str(cache),
    ]
    assert fetch_mlir.main(args) == 0
    target = cache / name
    assert capsys.readouterr().out.strip() == str(target)
    assert (target / _MLIR_CONFIG).is_file()
