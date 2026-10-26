# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The release tag checks, the workflow pin check and the release hygiene check
(source tree only)."""

import importlib.util
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


def test_the_changelog_has_notes_for_the_first_release_candidate():
    changelog = (_TOOLS.parent / "CHANGELOG.md").read_text()
    assert "### Added" in _load("release-notes").release_notes("0.1.0rc1", changelog)


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
