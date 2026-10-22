#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fail if a GitHub workflow or action uses a third-party action by a tag.

Every ``uses: owner/repo@ref`` must pin a full 40-character commit SHA,
followed by a ``# vX.Y.Z`` comment naming the release; local actions
(``./...``) are exempt. Container images given as ``container:`` must be
pinned by ``@sha256:`` digest, except the install-test images, which test
against current distribution images on purpose.

Usage: python tools/check-workflow-pins.py [FILES...]   (default: .github)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_USES = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>[^\s#]+)\s*(?P<comment>#.*)?$")
_PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
_CONTAINER = re.compile(r"^\s*container:\s*(?P<image>[^\s#$][^\s#]*)")


def check(path: Path) -> list[str]:
    errors = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _USES.match(line)
        if match:
            ref = match.group("ref").strip("'\"")
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            comment = match.group("comment") or ""
            if not _PINNED.match(ref) or not re.search(r"#\s*v\d", comment):
                errors.append(
                    f"{path}:{number}: pin {ref} to a commit SHA with a '# vX.Y.Z' comment"
                )
        match = _CONTAINER.match(line)
        if match and "@sha256:" not in match.group("image") and match.group("image") != '""':
            errors.append(f"{path}:{number}: pin container {match.group('image')} by digest")
    return errors


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path(".github").rglob("*.yml"))
    errors = [error for path in paths if path.suffix in (".yml", ".yaml") for error in check(path)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
