# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The release tag checks and the workflow pin check (source tree only)."""

import importlib.util
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
    [("v0.1.0", "0.1.0", False), ("v0.1.0rc1", "0.1.0rc1", True), ("v2.0.3b4", "2.0.3b4", True)],
)
def test_release_tags(tag, version, prerelease):
    assert _load("release-meta").release_version(tag) == (version, prerelease)


@pytest.mark.parametrize(
    "tag", ["0.1.0", "v0.1", "v0.1.0rc", "v0.1.0.dev1", "v0.1.0+local", "v0.1.0.post1"]
)
def test_invalid_release_tags(tag):
    with pytest.raises(ValueError):
        _load("release-meta").release_version(tag)


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
