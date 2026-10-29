#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check a release tag and print the version it releases.

Usage:
    python tools/release-meta.py TAG               # prints e.g. 0.1.0rc1
    python tools/release-meta.py --prerelease TAG  # prints true or false

The tag must match ``[tool.setuptools_scm] tag_regex`` in pyproject.toml,
so the version setuptools-scm builds from it is exactly the tag's version:
final releases ``vX.Y.Z``, post releases ``vX.Y.Z.postN`` (packaging fixes;
see the versioning policy in CONTRIBUTING.md) and pre-releases
``vX.Y.Z{a,b,rc}N``; no dev or local versions. The tree must also be
releasable: the perf gate's baseline must not be in bootstrap mode, so a
release gates instruction counts. Exits 1 with a message otherwise.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_PERF_GATE = _ROOT / "benchmarks" / "baselines" / "perf_gate.json"


def release_blockers(perf_gate: Path = _PERF_GATE) -> list[str]:
    """What in the tree stops a release; empty if nothing."""
    if perf_gate.exists() and "bootstrap" in json.loads(perf_gate.read_text()):
        return [
            f'{perf_gate.name} has "bootstrap": the perf gate does not gate instruction '
            "counts yet; adopt a measured baseline first (CONTRIBUTING.md, The perf gate)"
        ]
    return []


def release_version(tag: str) -> tuple[str, bool]:
    pattern = tomllib.loads(_PYPROJECT.read_text())["tool"]["setuptools_scm"]["tag_regex"]
    match = re.fullmatch(pattern, tag)
    if not match:
        raise ValueError(f"tag {tag!r} does not match tag_regex {pattern!r} in pyproject.toml")
    version = match.group("version")
    if not re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+|\.post\d+)?", version):
        raise ValueError(f"tag {tag!r} must name a version X.Y.Z, X.Y.Z.postN or X.Y.Z{{a,b,rc}}N")
    return version, bool(re.search(r"(a|b|rc)\d+$", version))


def main() -> int:
    args = sys.argv[1:]
    prerelease_only = "--prerelease" in args
    tags = [a for a in args if a != "--prerelease"]
    if len(tags) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        version, prerelease = release_version(tags[0])
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    blockers = release_blockers()
    for blocker in blockers:
        print(f"error: {blocker}", file=sys.stderr)
    if blockers:
        return 1
    print(("true" if prerelease else "false") if prerelease_only else version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
