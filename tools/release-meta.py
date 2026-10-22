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
final releases ``vX.Y.Z`` and pre-releases ``vX.Y.Z{a,b,rc}N``; no dev,
post or local versions. Exits 1 with a message otherwise.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def release_version(tag: str) -> tuple[str, bool]:
    pattern = tomllib.loads(_PYPROJECT.read_text())["tool"]["setuptools_scm"]["tag_regex"]
    match = re.fullmatch(pattern, tag)
    if not match:
        raise ValueError(f"tag {tag!r} does not match tag_regex {pattern!r} in pyproject.toml")
    version = match.group("version")
    if not re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", version):
        raise ValueError(f"tag {tag!r} must name a version X.Y.Z or X.Y.Z{{a,b,rc}}N")
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
    print(("true" if prerelease else "false") if prerelease_only else version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
