# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The files of the source tree that the tests of an installed wheel read.

A wheel's tests run from a copy of ``tests/`` outside the checkout, so that
the installed package is the one imported. The copy holds what
``test-sources`` of ``[tool.cibuildwheel]`` in ``pyproject.toml`` lists, and
nothing else of the source tree: cibuildwheel copies that list, and
``tools/ci/install-test.sh`` copies it with ``python tests/_sources.py copy``.

A test that reads a file outside ``tests/`` wherever it runs takes its path
from ``source_file``, which refuses a file the list leaves out. A test that
needs the rest of the tree (``docs/``, ``tools/``, every example) skips
itself where the file is absent instead.

``python tests/_sources.py check`` (and ``tests/test_sources.py``, at level
0) fails in a second, before a build or a long test run, when the layout
lacks a listed file or a test asks for a file that is not listed.

Only the standard library is imported, and nothing of ``tests``: this file
also runs as a script, before anything is built or installed.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_LIST = "test-sources of [tool.cibuildwheel] in pyproject.toml"
#: The first command of a wheel's tests.
CHECK = "python tests/_sources.py check"
_PLATFORMS = ("linux", "macos", "windows", "pyodide", "android", "ios")


def _cibuildwheel(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "cibuildwheel"
    ]


def listed(root: Path = ROOT) -> list[str]:
    """The files and directories the wheel's tests are given."""
    return _cibuildwheel(root)["test-sources"]


def _command_problems(root: Path) -> list[str]:
    """The tables of ``[tool.cibuildwheel]`` that give a platform a list of
    its own, or test commands that do not begin with the check."""
    config = _cibuildwheel(root)
    tables = {"[tool.cibuildwheel]": config}
    tables |= {f"[tool.cibuildwheel.{name}]": config[name] for name in _PLATFORMS if name in config}
    tables |= {
        f"[[tool.cibuildwheel.overrides]] {number}": value
        for number, value in enumerate(config.get("overrides", []), 1)
    }
    found = []
    for name, table in tables.items():
        if name != "[tool.cibuildwheel]" and "test-sources" in table:
            found.append(f"{name} has test-sources of its own; {_LIST} is the one list")
        command = table.get("test-command")
        first = command if isinstance(command, str) else "".join((command or [])[:1])
        if command is not None and first != CHECK:
            found.append(f"test-command of {name} does not begin with {CHECK!r}")
    if "test-command" not in config:
        found.append("[tool.cibuildwheel] has no test-command")
    return found


def _covered(relative: str, entries: list[str]) -> bool:
    if ".." in relative.split("/"):
        return False
    return any(relative == entry or relative.startswith(f"{entry}/") for entry in entries)


def source_file(relative: str) -> Path:
    """The file *relative* of the source tree (``benchmarks/...``), for a
    test that reads it in a checkout and beside an installed wheel alike."""
    if not _covered(relative, listed()):
        raise LookupError(
            f"{relative} is not in {_LIST}, so a wheel's tests do not have it: "
            "list it there, or skip the test where the file is absent"
        )
    return ROOT / relative


def whole_tree(root: Path = ROOT) -> bool:
    """Whether *root* is the source tree, not the copy a wheel's tests get."""
    return (root / "CMakeLists.txt").is_file()


def _strings(path: Path) -> list[tuple[ast.Call | None, str]]:
    """Every string of *path*'s code, and the ``source_file`` call it is the
    argument of, if any; a call without a literal argument gives ``""``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[ast.Call | None, str]] = []
    arguments = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "source_file":
                literal = (
                    len(node.args) == 1
                    and not node.keywords
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                )
                found.append((node, node.args[0].value if literal else ""))
                arguments.update(id(arg) for arg in node.args)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in arguments:
                found.append((None, node.value))
    return found


def problems(root: Path = ROOT) -> list[str]:
    """What is wrong with the layout at *root*, one line each.

    Everywhere: a listed file that is absent, a ``source_file`` call in
    ``tests/`` for a file that is not listed, and a test command of
    ``[tool.cibuildwheel]`` that does not begin with this check or a
    platform with a list of its own. In the source tree also: a
    listed Python file outside ``tests/`` that names a file of the tree that
    is not listed, as a benchmark that loads another benchmark does.
    """
    entries = listed(root)
    found = [
        f"{entry} is in {_LIST} and is not in {root}"
        for entry in entries
        if not (root / entry).exists()
    ]
    for path in sorted((root / "tests").rglob("*.py")):
        if "source_file" not in path.read_text(encoding="utf-8"):
            continue
        where = path.relative_to(root).as_posix()
        for call, text in _strings(path):
            if call is None:
                continue
            if not text:
                found.append(f"{where}:{call.lineno}: source_file takes one literal path")
            elif not _covered(text, entries):
                found.append(f"{where}:{call.lineno}: {text} is not in {_LIST}")
    found += _command_problems(root)
    if whole_tree(root):
        for entry in entries:
            path = root / entry
            if path.suffix != ".py" or not path.is_file():
                continue
            for _, text in _strings(path):
                if _names_a_file(root, text) and not _covered(text, entries):
                    found.append(f"{entry} names {text}, which is not in {_LIST}")
    return found


def _names_a_file(root: Path, text: str) -> bool:
    """Whether *text* is the path of a file of the tree, as ``a/b.py``."""
    if "/" not in text or text != text.strip() or len(text) > 200:
        return False
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (root / path).is_file()


def copy(destination: Path, root: Path = ROOT) -> None:
    """Copy the listed files of *root* into *destination*, each at its path,
    as cibuildwheel does."""
    for entry in listed(root):
        source, target = root / entry, destination / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True
            )
        else:
            shutil.copy2(source, target)


def main(argv: list[str], root: Path = ROOT) -> int:
    if argv == ["check"]:
        found = problems(root)
        for line in found:
            print(f"error: {line}", file=sys.stderr)
        if not found:
            print(f"the {len(listed(root))} test sources are in {root}")
        return 1 if found else 0
    if len(argv) == 2 and argv[0] == "copy":
        copy(Path(argv[1]), root)
        return 0
    print("usage: python tests/_sources.py check | copy DESTINATION", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
