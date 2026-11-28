#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The test files a pull request runs above its test level: the ones it adds
or changes, and the ones that read a file it changes.

A pull request runs test level 0 (CONTRIBUTING.md, Test levels), so a test of
level 1 or 2 that it adds or edits would first run after the merge. With
``pytest --level 0 --changed-paths FILE`` (tests/_levels.py), where FILE lists
the paths the pull request changes, one per line, the same run also takes:

* each changed path that is a test file (``tests/**/test_*.py``) and exists.
  A file the change adds or edits is one; a rename is listed under both
  names, so the new name is one and the old name, like any removed file, is
  not;
* each test file of ``WATCHES`` when a path it watches changed: the tests
  of the workflows and of the scripts in ``tools/`` run when those change.

These files run up to level ``MAX_LEVEL``; level 3 stays with the nightly
run. They add at most ``MAX_CASES`` cases to the run: ``plan`` gives the
files their level 1 cases, the smallest file first, then their level 2
cases, and a file whose cases no longer fit keeps the level it reached. What
is left runs on the push to ``main``, and the run's log lists every file with
what ran and what was left. Cases are counted as collected, before ``-m``
and ``-k`` deselect any: a GPU case counts although ``-m "not gpu"`` then
leaves it out.

Any other changed path under ``tests/`` (a helper, a ``conftest.py``, a data
file, the level table) does not select the tests that use it: they are most
of the suite, which runs at level 0 here and at levels 0 to 2 on the push to
``main``. It selects only the test files that watch it, which for
``tests/conftest.py``, ``tests/_levels.py`` and the level table are the
tests of the levels themselves. The log names these paths too.

The paths come from ``git diff --name-only`` and are read as data: a path is
selected only if it matches ``TEST_FILE`` in full, and is never passed to a
shell.

``ci-changed-tests.py [FILE]`` prints the selection for the paths in FILE or
on standard input::

    git diff --name-only --no-renames origin/main... | python tools/ci-changed-tests.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: A test file, as pytest collects one under tests/.
TEST_FILE = re.compile(r"tests/(?:[A-Za-z0-9_]+/)*test_[A-Za-z0-9_]+\.py")
#: The highest level a selected file runs at.
MAX_LEVEL = 2
#: The most cases the selected files add to a run.
MAX_CASES = 1000
#: The paths a test file reads, beside itself. Prefixes end in "/"; anything
#: else is a whole path.
WATCHES = {
    "tests/test_ci_changes.py": (".github/", "tools/ci-changes.py", "tools/ci-ok.py"),
    "tests/test_ci_changed_tests.py": (
        ".github/",
        "tools/ci-changed-tests.py",
        "tests/_levels.py",
        "tests/conftest.py",
    ),
    "tests/test_levels.py": ("tests/_levels.py", "tests/levels.toml", "tests/conftest.py"),
    "tests/test_packed_schema.py": ("tools/check-packed-schema.py",),
    "tests/test_pass_lit_tests.py": ("tools/check-pass-lit-tests.py",),
    "tests/test_perf_gate.py": (".github/", "tools/perf-gate.py", "benchmarks/perf_gate/"),
    "tests/test_release_tools.py": (".github/", "tools/"),
    "tests/test_third_party_notices.py": (
        "tools/third-party-notices.py",
        "tools/check-wheel-contents.py",
        "tools/backend/",
    ),
    "tests/test_tutorial_scripts.py": ("tools/tutorial-scripts.py",),
}


@dataclass
class Selection:
    """What the changed paths select."""

    #: Each selected test file, with why: ``changed``, or the path it watches.
    tests: dict[str, str] = field(default_factory=dict)
    #: Test files the change removed (or moved away).
    removed: list[str] = field(default_factory=list)
    #: Other changed paths under tests/, which select nothing.
    shared: list[str] = field(default_factory=list)


def _watched(path: str, names: Iterable[str]) -> bool:
    return any(path.startswith(name) if name.endswith("/") else path == name for name in names)


def select(paths: Iterable[str], exists: Callable[[str], bool]) -> Selection:
    """The selection for the changed *paths*; *exists* says whether a path
    is a file of the tree under test."""
    paths = sorted({path.strip() for path in paths if path.strip()})
    selection = Selection()
    for path in paths:
        if TEST_FILE.fullmatch(path):
            if exists(path):
                selection.tests[path] = "changed"
            else:
                selection.removed.append(path)
        elif path.startswith("tests/"):
            selection.shared.append(path)
    for test, names in WATCHES.items():
        if test in selection.tests or not exists(test):
            continue
        read = next((path for path in paths if _watched(path, names)), None)
        if read is not None:
            selection.tests[test] = f"reads {read}"
    selection.tests = dict(sorted(selection.tests.items()))
    return selection


def plan(
    counts: Mapping[str, Mapping[int, int]],
    level: int,
    *,
    max_level: int = MAX_LEVEL,
    max_cases: int = MAX_CASES,
) -> dict[str, int]:
    """The level each selected file runs up to in a run at *level*.

    *counts* gives, for each selected file, its number of cases per level.
    """
    reached = dict.fromkeys(counts, level)
    left = max_cases
    for step in range(level + 1, max_level + 1):
        for file in sorted(counts, key=lambda file: (counts[file].get(step, 0), file)):
            cases = counts[file].get(step, 0)
            # A file takes a level only with the levels below it.
            if reached[file] == step - 1 and cases <= left:
                reached[file] = step
                left -= cases
    return reached


def report(
    selection: Selection,
    counts: Mapping[str, Mapping[int, int]],
    reached: Mapping[str, int],
    level: int,
    *,
    max_level: int = MAX_LEVEL,
    max_cases: int = MAX_CASES,
) -> list[str]:
    """The lines a run prints: each selected file with the cases that run
    above *level* and the ones left, then the paths that select nothing."""

    def cases(file: str, levels: Iterable[int]) -> str:
        found = [(n, counts[file].get(n, 0)) for n in levels]
        return ", ".join(f"{count} of level {n}" for n, count in found if count)

    lines = [
        f"test files of this change, run above level {level}: to level {max_level}, "
        f"at most {max_cases} cases (tools/ci-changed-tests.py)"
    ]
    if not selection.tests:
        lines.append("  none")
    width = max(map(len, selection.tests), default=0)
    highest = max((n for file in counts for n in counts[file]), default=level)
    for file, why in selection.tests.items():
        ran = cases(file, range(level + 1, reached[file] + 1))
        over = cases(file, range(reached[file] + 1, max_level + 1))
        nightly = cases(file, range(max_level + 1, highest + 1))
        parts = [f"runs {ran}" if ran else f"runs nothing above level {level}"]
        if over:
            parts.append(f"over {max_cases} cases, left for the push to main: {over}")
        if nightly:
            parts.append(f"left for the nightly run: {nightly}")
        lines.append(f"  {file:<{width}}  ({why}) " + "; ".join(parts))
    if selection.removed:
        lines.append("  removed: " + ", ".join(selection.removed))
    if selection.shared:
        lines.append(
            "  changed, and shared by many tests, which run at their own level: "
            + ", ".join(selection.shared)
        )
    return lines


def main(argv: list[str]) -> int:
    if len(argv) > 1 or (argv and argv[0].startswith("-")):
        print("usage: ci-changed-tests.py [FILE]", file=sys.stderr)
        return 2
    text = Path(argv[0]).read_text(errors="replace") if argv else sys.stdin.read()
    root = Path(__file__).resolve().parents[1]
    selection = select(text.splitlines(), lambda path: (root / path).is_file())
    for file, why in selection.tests.items():
        print(f"{file}  ({why})")
    if selection.removed:
        print("removed: " + ", ".join(selection.removed))
    if selection.shared:
        print("shared, selects nothing: " + ", ".join(selection.shared))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
