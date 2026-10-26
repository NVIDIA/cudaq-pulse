# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``python -m cudaq_pulse.doctor --setup-editor``: configure mypy and ruff.

Pyright-based editors need nothing (the wheel's ``__builtins__.pyi``). mypy
and ruff take no names from an installed package, so for a project that
configures them this adds the mypy plugin and the ruff builtins for the
kernel language, to the file each tool reads, shown as a diff and written
only when confirmed. Projects without a mypy or ruff configuration are left
alone.
"""

from __future__ import annotations

import difflib
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import ops

OPS = tuple(ops.__all__)
MYPY_PLUGIN = "cudaq_pulse.mypy_plugin"


@dataclass
class Change:
    tool: str
    path: Path
    old: str
    new: str

    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.old.splitlines(keepends=True),
                self.new.splitlines(keepends=True),
                f"a/{self.path.name}",
                f"b/{self.path.name}",
            )
        )


def _toml_list(key: str, values: list[str]) -> str:
    return f"{key} = [\n" + "".join(f'    "{v}",\n' for v in values) + "]\n"


def _merge(current: Any, wanted: list[str]) -> list[str] | None:
    """The merged list, or None when *current* already has every value."""
    if isinstance(current, str):
        current = [v.strip() for v in current.split(",") if v.strip()]
    have = list(current or [])
    missing = [v for v in wanted if v not in have]
    return [*have, *missing] if missing else None


def _toml_section_span(text: str, header: str | None) -> tuple[int, int] | None:
    """Character span of a TOML table's body (or of the root table)."""
    tables = list(re.finditer(r"^\s*\[[^\]\n]+\]\s*$", text, re.MULTILINE))
    if header is None:
        return (0, tables[0].start() if tables else len(text))
    for i, m in enumerate(tables):
        if m.group().strip() == f"[{header}]":
            end = tables[i + 1].start() if i + 1 < len(tables) else len(text)
            return (min(m.end() + 1, len(text)), end)
    return None


def _edit_toml(text: str, header: str | None, key: str, wanted: list[str]) -> str | None:
    if text and not text.endswith("\n"):
        text += "\n"
    data = tomllib.loads(text)
    table = data
    for part in (header or "").split(".") if header else []:
        table = table.get(part, {})
    merged = _merge(table.get(key), wanted)
    if merged is None:
        return None
    span = _toml_section_span(text, header)
    assert span is not None
    start, end = span
    body = text[start:end]
    assignment = re.search(
        rf'^{re.escape(key)}\s*=\s*(\[.*?\]|"[^"\n]*"|\'[^\'\n]*\')[^\n]*\n?',
        body,
        re.MULTILINE | re.DOTALL,
    )
    if assignment is not None:
        body = body[: assignment.start()] + _toml_list(key, merged) + body[assignment.end() :]
    else:
        body = _toml_list(key, merged) + body
    new = text[:start] + body + text[end:]
    tomllib.loads(new)
    return new


def _edit_ini(text: str, section: str, key: str, wanted: list[str]) -> str | None:
    header = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.MULTILINE)
    assert header is not None
    nxt = re.search(r"^\[[^\]\n]+\]\s*$", text[header.end() :], re.MULTILINE)
    end = header.end() + nxt.start() if nxt else len(text)
    body = text[header.end() : end]
    line = re.search(rf"^{re.escape(key)}\s*[=:]\s*(.*)$", body, re.MULTILINE)
    merged = _merge(line.group(1) if line else None, wanted)
    if merged is None:
        return None
    entry = f"{key} = {', '.join(merged)}"
    if line is not None:
        body = body[: line.start()] + entry + body[line.end() :]
    else:
        body = "\n" + entry + body
    return text[: header.end()] + body + text[end:]


def _has_ini_section(path: Path, section: str) -> bool:
    return (
        path.is_file()
        and re.search(rf"^\[{re.escape(section)}\]\s*$", path.read_text(), re.MULTILINE) is not None
    )


def _has_toml_table(path: Path, dotted: str) -> bool:
    if not path.is_file():
        return False
    table: Any = tomllib.loads(path.read_text())
    for part in dotted.split("."):
        if not isinstance(table, dict) or part not in table:
            return False
        table = table[part]
    return True


def _mypy_change(root: Path) -> Change | None | str:
    """The change for the mypy configuration mypy reads, in its own order."""
    for name in ("mypy.ini", ".mypy.ini"):
        path = root / name
        if _has_ini_section(path, "mypy"):
            old = path.read_text()
            new = _edit_ini(old, "mypy", "plugins", [MYPY_PLUGIN])
            return Change("mypy", path, old, new) if new else str(path)
    path = root / "pyproject.toml"
    if _has_toml_table(path, "tool.mypy"):
        old = path.read_text()
        new = _edit_toml(old, "tool.mypy", "plugins", [MYPY_PLUGIN])
        return Change("mypy", path, old, new) if new else str(path)
    path = root / "setup.cfg"
    if _has_ini_section(path, "mypy"):
        old = path.read_text()
        new = _edit_ini(old, "mypy", "plugins", [MYPY_PLUGIN])
        return Change("mypy", path, old, new) if new else str(path)
    return None


def _ruff_change(root: Path) -> Change | None | str:
    """The change for the ruff configuration ruff reads, in its own order."""
    for name in (".ruff.toml", "ruff.toml"):
        path = root / name
        if path.is_file():
            old = path.read_text()
            new = _edit_toml(old, None, "builtins", list(OPS))
            return Change("ruff", path, old, new) if new else str(path)
    path = root / "pyproject.toml"
    if _has_toml_table(path, "tool.ruff"):
        old = path.read_text()
        new = _edit_toml(old, "tool.ruff", "builtins", list(OPS))
        return Change("ruff", path, old, new) if new else str(path)
    return None


def setup_editor(
    root: Path,
    *,
    assume_yes: bool,
    out: TextIO,
    confirm: Callable[[str], bool] | None,
) -> int:
    """Show, and on confirmation write, the mypy and ruff configuration.

    *confirm* asks the user; None means nobody can answer, so nothing is
    written without *assume_yes*. Returns the process exit code: 1 when
    changes are needed but were not written.
    """

    def write(line: str) -> None:
        print(line, file=out)

    write(
        "Pylance, Cursor, pyright and basedpyright need no setup: cudaq-pulse "
        "installs __builtins__.pyi for them."
    )
    changes: list[Change] = []
    configured = False
    for tool, found in (("mypy", _mypy_change(root)), ("ruff", _ruff_change(root))):
        if found is None:
            write(f"{tool}: no configuration in {root}; nothing to change.")
        elif isinstance(found, str):
            configured = True
            write(f"{tool}: {found} is already set up.")
        else:
            configured = True
            changes.append(found)
    if not changes:
        if not configured:
            write("Run this in the project directory that holds your mypy or ruff configuration.")
        return 0
    # Merge edits to the same file (pyproject.toml configuring both tools).
    by_path: dict[Path, Change] = {}
    for change in changes:
        if change.path in by_path:
            first = by_path[change.path]
            if change.tool == "ruff":
                new = _edit_toml(first.new, "tool.ruff", "builtins", list(OPS))
            else:
                new = _edit_toml(first.new, "tool.mypy", "plugins", [MYPY_PLUGIN])
            by_path[change.path] = Change(
                f"{first.tool} and {change.tool}", change.path, first.old, new or first.new
            )
        else:
            by_path[change.path] = change
    for change in by_path.values():
        write(f"\n{change.tool}: {change.path}")
        write(change.diff().rstrip("\n"))
    if not assume_yes:
        if confirm is None:
            write("\nNothing written. Run again with --yes to write these changes.")
            return 1
        if not confirm("\nWrite these changes? [y/N] "):
            write("Nothing written.")
            return 1
    for change in by_path.values():
        change.path.write_text(change.new)
        write(f"Wrote {change.path}")
    return 0
