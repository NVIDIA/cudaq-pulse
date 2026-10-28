# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The release tag checks, the workflow pin check, the release hygiene check
and the GPU CI matrix (source tree only)."""

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


@pytest.mark.parametrize("page", ["README.md", "docs/getting_started.rst"])
def test_the_install_commands_name_the_next_release(page):
    version, _ = _next_release((_TOOLS.parent / "CHANGELOG.md").read_text())
    text = (_TOOLS.parent / page).read_text()
    assert f"releases/expanded_assets/v{version}" in text
    pins = re.findall(r'pip install "cudaq-pulse(?:\[\w+\])?==([^"]+)"', text)
    assert pins and set(pins) == {version}
    assert not re.search(r"pip install (?:\"?)cudaq-pulse(?:\[\w+\])?(?:\"?)\s*$", text, re.M)


@pytest.mark.parametrize(
    "workflow",
    sorted((_TOOLS.parent / ".github" / "workflows").glob("*.yml")),
    ids=lambda p: p.name,
)
def test_every_workflow_job_has_a_timeout(workflow):
    jobs = workflow.read_text().split("\njobs:\n", 1)[1]
    blocks = re.split(r"^  (?=[\w-]+:\s*$)", jobs, flags=re.M)[1:]
    assert blocks
    missing = [
        b.split(":", 1)[0] for b in blocks if not re.search(r"^    timeout-minutes:", b, re.M)
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
