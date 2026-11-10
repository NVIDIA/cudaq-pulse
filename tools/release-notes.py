#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the release notes for a version from CHANGELOG.md.

Usage:
    python tools/release-notes.py VERSION    # e.g. 0.1.0, 0.1.0rc1, 0.1.0.post1

A final or post release takes the section ``## [VERSION] - YYYY-MM-DD``,
which must be dated. A pre-release (``X.Y.Z{a,b,rc}N``) takes the section of
the release it precedes, ``## [X.Y.Z]``, dated or ``- Unreleased``. The
section must not be empty. Exits 1 with a message otherwise, so a release is
never published with empty notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?:\s+-\s+(?P<date>.+?))?\s*$")
_LINK_REFERENCE = re.compile(r"^\[[^\]]+\]:\s")


def release_notes(version: str, changelog: str) -> str:
    base = re.sub(r"(a|b|rc)\d+$", "", version)
    prerelease = base != version
    sections: dict[str, tuple[str | None, list[str]]] = {}
    current: list[str] | None = None
    for line in changelog.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = []
            sections[heading.group("version")] = (heading.group("date"), current)
        elif line.startswith("## "):
            current = None
        elif current is not None and not _LINK_REFERENCE.match(line):
            current.append(line)
    if base not in sections:
        raise ValueError(f"CHANGELOG.md has no '## [{base}]' section")
    date, lines = sections[base]
    if not prerelease and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        raise ValueError(
            f"the CHANGELOG.md section for {base} must be dated '- YYYY-MM-DD' "
            f"before {version} is released (it says {date or 'nothing'!r})"
        )
    notes = "\n".join(lines).strip()
    if not notes:
        raise ValueError(f"the CHANGELOG.md section for {base} is empty")
    return notes + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        notes = release_notes(sys.argv[1], _CHANGELOG.read_text())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
