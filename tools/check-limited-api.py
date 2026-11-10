#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the native extension compiles against the limited Python API.

Usage:
    python tools/check-limited-api.py BUILD_DIR

The abi3 wheels build the extension with ``Py_LIMITED_API`` set to the
version of pyproject.toml's ``wheel.py-api`` (``cp312`` is ``0x030C0000``),
which hides the C API's macros and internals, such as ``PyList_SET_ITEM``.
A development build finds no stable ABI to build against and never sets it,
so code that only a wheel build rejects passes every other job. This
compiles the extension's own sources (``core/mlir/bindings``) with the
commands BUILD_DIR's ninja build runs, plus ``-fsyntax-only`` and that
define, and exits 1 if one fails. BUILD_DIR must be a GIL build: a
free-threaded CPython has no limited API before 3.15.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "core" / "mlir" / "bindings"
TARGET = "_cudaq_pulse_native.dir"
# Flags that write files: a syntax check writes none.
_WITH_VALUE = {"-o", "-MF", "-MT", "-MQ"}
_ALONE = {"-c", "-MD", "-MMD"}


def limited_api() -> str:
    """``Py_LIMITED_API`` for pyproject.toml's ``wheel.py-api``."""
    with (ROOT / "pyproject.toml").open("rb") as f:
        api = tomllib.load(f)["tool"]["scikit-build"]["wheel"]["py-api"]
    match = re.fullmatch(r"cp3(\d+)", api)
    if not match:
        raise SystemExit(f"error: wheel.py-api {api!r} is not cp3XX")
    return f"0x03{int(match[1]):02X}0000"


def syntax_check(command: list[str], define: str) -> list[str]:
    """*command*, which compiles one source, as a syntax check with *define*."""
    out: list[str] = []
    skip = False
    for arg in command:
        if skip:
            skip = False
        elif arg in _WITH_VALUE:
            skip = True
        elif arg not in _ALONE and not arg.startswith(("-o", "-MF", "-MT", "-MQ")):
            out.append(arg)
    return [*out, "-fsyntax-only", f"-DPy_LIMITED_API={define}"]


def commands(build: Path) -> list[tuple[str, list[str]]]:
    """(source, command) for each of the extension's own sources."""
    compdb = subprocess.run(
        ["ninja", "-C", str(build), "-t", "compdb"], capture_output=True, text=True, check=True
    ).stdout
    found = []
    for entry in json.loads(compdb):
        source = Path(entry["directory"], entry["file"]).resolve()
        if source.parent != SOURCES or TARGET not in entry.get("output", entry["command"]):
            continue
        command = entry.get("arguments") or shlex.split(entry["command"])
        found.append((source.name, command))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build", type=Path, help="a configured CMake build directory (ninja)")
    args = parser.parse_args()
    cache = (args.build / "CMakeCache.txt").read_text()
    if re.search(r"^CUDAQ_PULSE_FREE_THREADED:BOOL=ON$", cache, re.MULTILINE):
        print(f"error: {args.build} is a free-threaded build", file=sys.stderr)
        return 1
    define = limited_api()
    found = commands(args.build)
    if not found:
        print(f"error: {args.build} compiles no source of {SOURCES}", file=sys.stderr)
        return 1

    def run(item: tuple[str, list[str]]) -> tuple[str, subprocess.CompletedProcess[str]]:
        name, command = item
        return name, subprocess.run(
            syntax_check(command, define), capture_output=True, text=True, cwd=args.build
        )

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(run, found))
    failed = [name for name, result in results if result.returncode != 0]
    for name, result in results:
        if result.returncode != 0:
            print(f"error: {name} does not compile with Py_LIMITED_API={define}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
    if failed:
        return 1
    print(f"{len(found)} sources compile with Py_LIMITED_API={define}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
