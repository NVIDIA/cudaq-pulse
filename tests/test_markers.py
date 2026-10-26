# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The pytest markers the workflows select on are registered, used and applied."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def test_every_registered_marker_is_used():
    config = tomllib.loads((_TESTS.parent / "pyproject.toml").read_text())
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    registered = {line.split(":", 1)[0].strip() for line in markers}
    used = set()
    for path in _TESTS.rglob("*.py"):
        used.update(re.findall(r"pytest\.mark\.(\w+)", path.read_text()))
    assert registered <= used


def test_the_nightly_physics_selection_runs_every_physics_test():
    def collected(*args):
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
            + list(args)
            + [str(_TESTS / "physics")],
            cwd=_TESTS.parent,
            capture_output=True,
            text=True,
            timeout=300,
        )
        match = re.search(r"(\d+)(?:/\d+)? tests? collected", out.stdout)
        assert match, out.stdout + out.stderr
        return int(match.group(1))

    everything = collected()
    assert everything > 0
    assert collected("-m", "physics and not gpu", "--run-nightly") == everything
