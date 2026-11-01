#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fail if a tracked file still spells a classical function the old way.

A classical function's type hints say what it takes and returns, so the
declaration types (``Floats(2)``, ``classical.Levels``, ``Bits(3)`` and the
rest) and the ``args=``/``results=`` options are gone from the API, and so
are ``lengths=`` and ``id=``: ``pulse.Length`` states a fixed length, a
positive whole number, and a function is dispatched by the hash of its
compiled name or the key its controller fixes. An external's stub
is a declaration whose body is a placeholder for its result, not ``raise
NotImplementedError``. This finds them in docs, examples, notebooks,
docstrings, messages and tests. A
line that must name an old spelling (the error that says it is gone, the
test of that error, the changelog entry) carries ``remnant-ok``.

Usage: python tools/check-classical-remnants.py [FILES...]   (default: git ls-files)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_OLD = r"(?:Levels|IQ|Floats|Integers|Bytes|Float|Integer|Byte)"
_PATTERNS = (
    ("an old declaration type", re.compile(rf"\bclassical\.{_OLD}\b")),
    ("an old declaration type", re.compile(rf"(?<![\w.]){_OLD}\(")),
    ("an old declaration type", re.compile(rf"``{_OLD}(?:\([^`]*\))?``")),
    (
        "an old declaration type",
        re.compile(rf"from\s+cudaq_pulse\.classical\s+import\s+[^\n]*\b{_OLD}\b"),
    ),
    ("a sized Bits", re.compile(r"(?<![\w.])(?<!class )(?:classical\.|pulse\.)?Bits\(\s*[\w*]")),
    ("a sized Bits", re.compile(r"(?<![\w.])Bits\[")),
    (
        "args= or results= on a declaration",
        re.compile(
            r"(?<!\w)(?:classical|external|function|ClassicalFunction)\((?:[^()]|\([^()]*\))*?"
            r"\b(?:args|results)\s*=",
            re.S,
        ),
    ),
    (
        "lengths= on a declaration",
        re.compile(
            r"(?<!\w)(?:classical|external|function|ClassicalFunction)\((?:[^()]|\([^()]*\))*?"
            r"\blengths\s*=",
            re.S,
        ),
    ),
    (
        "id= on a declaration",
        re.compile(
            r"(?<!\w)(?:classical|external|function|ClassicalFunction)\((?:[^()]|\([^()]*\))*?"
            r"\bid\s*=",
            re.S,
        ),
    ),
    (
        "a length that isn't a whole number",
        re.compile(r"\bLength\[(?:[^\[\]\n]|\[[^\]\n]*\])*,\s*['\"]"),
    ),
    ("a lengths= entry", re.compile(r"\blengths\[\s*['\"]")),
    ("a lengths= entry", re.compile(r"\{\s*['\"]return['\"]\s*:")),
)
_EXTERNAL = re.compile(r"^([ \t]*)@(?:pulse\.)?(?:classical\.)?external\(", re.M)
_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".ipynb", ".td", ".h"}


def _text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    # The cells' sources only, so a hit's line number is a line of a cell.
    try:
        cells = json.loads(raw).get("cells", [])
    except ValueError:
        return raw
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


def check(path: Path) -> list[str]:
    if path.suffix not in _SUFFIXES:
        return []
    try:
        text = _text(path)
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        return []
    lines = text.splitlines()
    found = set()
    for what, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if "remnant-ok" not in lines[number - 1]:
                found.add((number, what))
    for match in _EXTERNAL.finditer(text):
        number = _stub_raises(lines, text.count("\n", 0, match.start()), len(match.group(1)))
        if number and "remnant-ok" not in lines[number - 1]:
            found.add((number, "raise NotImplementedError as an external's body"))
    return [f"{path}:{number}: {what}" for number, what in sorted(found)]


def _stub_raises(lines: list[str], start: int, indent: int) -> int:
    """The line number of ``raise NotImplementedError`` in the body of the
    stub decorated at line *start* (0-based), or 0. An external's body is a
    placeholder for its result, such as ``return float()``."""
    i = start
    while i < len(lines) and not lines[i].lstrip().startswith("def "):
        i += 1
    while i < len(lines) and not lines[i].rstrip().endswith(":"):
        i += 1
    for j in range(i + 1, len(lines)):
        line = lines[j]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            return 0
        if line.strip() == "raise NotImplementedError":
            return j + 1
    return 0


def main() -> int:
    if sys.argv[1:]:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, check=True, text=True
        )
        paths = [Path(p) for p in listed.stdout.split("\0") if p]
    me = Path(__file__).resolve()
    errors = [error for path in paths if path.resolve() != me for error in check(path)]
    for error in errors:
        print(error)
    if errors:
        print(
            f"{len(errors)} old classical spellings: declare with type hints"
            " (def f(bits: pulse.Bits) -> float), a fixed length with pulse.Length"
            " (pulse.Length[pulse.Bits, 3]), no id=, and an external's result with a placeholder"
            " (return float()), or mark a line that must name one with remnant-ok"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
