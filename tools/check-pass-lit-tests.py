#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check that every cudaq-pulse pass is run by a lit test.

Usage:
    python tools/check-pass-lit-tests.py CUDAQ_PULSE_OPT [--tests DIR]

Reads the passes ``CUDAQ_PULSE_OPT --help`` lists, keeps this project's (the
``pulse-``, ``qop-`` and ``cudm-`` ones; the rest are MLIR's), and exits 1
naming each one that no ``RUN:`` line under DIR (default: ``test/``) runs,
as ``--name`` or inside a ``--pass-pipeline``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PREFIXES = ("pulse-", "qop-", "cudm-")
# A pass's own line in --help; its options are indented further.
_PASS_LINE = re.compile(r"^ {6}--([a-z0-9][a-z0-9-]*)\b")
_RUN_LINE = re.compile(r"^\s*//\s*RUN:(.*)$")


def project_passes(help_text: str) -> list[str]:
    """The passes in the ``Passes:`` section of ``--help`` with a project prefix."""
    lines = help_text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Passes:")
    except StopIteration:
        return []
    names = []
    for line in lines[start + 1 :]:
        if line.strip().endswith(":") and not line.lstrip().startswith("-"):
            break
        match = _PASS_LINE.match(line)
        if match and match[1].startswith(PREFIXES):
            names.append(match[1])
    return sorted(set(names))


def run_text(path: Path) -> str:
    """Every ``RUN:`` line of a lit test, joined into one string."""
    return " ".join(
        match[1] for line in path.read_text().splitlines() if (match := _RUN_LINE.match(line))
    )


def untested(passes: list[str], tests: Path) -> list[str]:
    commands = [run_text(path) for path in sorted(tests.rglob("*.mlir"))]
    return [
        name
        for name in passes
        if not any(
            re.search(rf"(?<![\w-])(?:--?)?{re.escape(name)}(?![\w-])", run) for run in commands
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("opt", type=Path, help="the built cudaq-pulse-opt")
    parser.add_argument("--tests", type=Path, default=Path(__file__).resolve().parents[1] / "test")
    args = parser.parse_args()
    help_text = subprocess.run(
        [str(args.opt), "--help"], capture_output=True, text=True, check=True
    ).stdout
    passes = project_passes(help_text)
    if not passes:
        print(f"error: {args.opt} --help lists no {', '.join(PREFIXES)} passes", file=sys.stderr)
        return 1
    missing = untested(passes, args.tests)
    for name in missing:
        print(f"error: no lit test under {args.tests} runs --{name}", file=sys.stderr)
    if missing:
        return 1
    print(f"{len(passes)} passes, each run by a lit test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
