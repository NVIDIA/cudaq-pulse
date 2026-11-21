#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Say which CI jobs a change needs, from the paths it touches.

Reads the changed paths, one per line, from standard input and prints three
lines, ``native=``, ``python=`` and ``docs=``, each ``true`` or ``false``:

* ``native``: the change can reach the compiler, the runtime, the build or
  the workflows. Every job runs.
* ``python``: it can reach the Python package, its tests, the examples, the challenges, the
  benchmarks or the tools. Every job but the old-compiler builds runs.
* ``docs``: it can reach the built pages, or a file that tests read (the
  README, the changelog, the contributing guide, the license files). The
  docs build runs, and one test job limited to the tests that read them.

Each class includes the ones after it, so ``native=true`` comes with
``python=true`` and ``docs=true``. A change of none of the three (a file of
``META``, which no test reads) needs lint alone.

A path this file does not name is ``native``: a new directory runs every job
until someone places it. No paths at all, which is what a failed ``git diff``
gives, is ``native`` too. ``--all`` asks for that outright.
"""

from __future__ import annotations

import sys

# Prefixes end in "/"; anything else is a whole path.
META = (
    ".github/ISSUE_TEMPLATE/",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".gitignore",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "REUSE.toml",
    "SUPPORT.md",
)
DOCS = (
    "docs/",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    "LICENSES/",
    "NOTICE",
    "THIRD_PARTY_NOTICES",
)
PYTHON = (
    "core/frontend/",
    "tests/",
    "examples/",
    "challenges/",
    "benchmarks/",
    "tools/",
    "pyproject.toml",
    "pyrightconfig.json",
    "uv.lock",
    ".pre-commit-config.yaml",
)


def _in(path: str, names: tuple[str, ...]) -> bool:
    return any(path.startswith(name) if name.endswith("/") else path == name for name in names)


def classify(paths: list[str]) -> dict[str, bool]:
    """The three answers for *paths*; every path counts, the widest class wins."""
    paths = [path.strip() for path in paths if path.strip()]
    native = not paths or any(not _in(path, META + DOCS + PYTHON) for path in paths)
    python = native or any(_in(path, PYTHON) for path in paths)
    docs = python or any(_in(path, DOCS) for path in paths)
    return {"native": native, "python": python, "docs": docs}


def main(argv: list[str]) -> int:
    answers = classify([] if "--all" in argv else sys.stdin.read().splitlines())
    for name, value in answers.items():
        print(f"{name}={'true' if value else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
