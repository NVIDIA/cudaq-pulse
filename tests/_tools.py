# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools that some tests need: skipped without them locally, required in CI.

When ``CI`` is set (GitHub Actions sets it), a missing tool fails the test
instead of skipping it, so a job that lost a tool can't pass by skipping
the tests that use it. A job that leaves a tool out on purpose names it in
``CUDAQ_PULSE_OPTIONAL_TOOLS``, comma-separated.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
from collections.abc import Mapping
from types import ModuleType
from typing import NoReturn

import pytest

OPTIONAL_TOOLS_VAR = "CUDAQ_PULSE_OPTIONAL_TOOLS"
_PRELOAD_VARS = ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")
# GCC's libasan.so.8, clang's libclang_rt.asan_osx_dynamic.dylib and
# libclang_rt.asan-x86_64.so, and the other sanitizers' runtimes.
_SANITIZER_RUNTIME = re.compile(r"lib(?:clang_rt\.)?(?:a|hwa|l|m|t|ub)san[._-]")


def in_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false", "no")


def _optional() -> set[str]:
    return {name.strip() for name in os.environ.get(OPTIONAL_TOOLS_VAR, "").split(",")}


def missing(tool: str, reason: str) -> NoReturn:
    """Skip the test, or fail it in CI unless *tool* is optional there."""
    if in_ci() and tool not in _optional():
        pytest.fail(
            f"{reason}, and CI requires it (a job that leaves it out on purpose "
            f"sets {OPTIONAL_TOOLS_VAR}={tool})",
            pytrace=False,
        )
    pytest.skip(reason, allow_module_level=True)


def import_module(name: str, tool: str | None = None) -> ModuleType:
    """Import *name*, or report the tool that provides it as missing."""
    try:
        return importlib.import_module(name)
    except ImportError:
        missing(tool or name.partition(".")[0], f"{name} is not installed")


def c_compiler() -> str:
    """The first C compiler on PATH."""
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        missing("cc", "no C compiler (cc, clang or gcc) on PATH")
    return compiler


def under_sanitizers(env: Mapping[str, str] | None = None) -> bool:
    """Whether *env* (``os.environ`` by default) preloads a sanitizer runtime,
    as the sanitizer jobs do for the uninstrumented interpreter."""
    env = os.environ if env is None else env
    return any(
        _SANITIZER_RUNTIME.match(os.path.basename(lib))
        for var in _PRELOAD_VARS
        for lib in re.split(r"[:\s]+", env.get(var, ""))
        if lib
    )


def skip_raw_limit(what: str) -> None:
    """Skip the test under a sanitizer runtime: *what* is a ceiling on raw
    time or memory, which the instrumentation multiplies, so it says nothing
    about the code there. The unsanitized jobs check it."""
    if under_sanitizers():
        pytest.skip(f"{what} is not checked under a sanitizer runtime")


def tool_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """*env* (``os.environ`` by default) without preloaded sanitizer runtimes.

    For a child that runs none of our code, such as pyright's node, git or a
    C compiler. The sanitizer job preloads the runtimes into the
    uninstrumented interpreter for the extension module, and every child
    inherits them; node fails under them. Child interpreters and
    cudaq-pulse-opt keep them. Other preloaded libraries, and the
    ``*SAN_OPTIONS`` variables, which do nothing without a runtime, are kept.
    """
    result = dict(os.environ if env is None else env)
    for var in _PRELOAD_VARS:
        if var in result:
            kept = [
                lib
                for lib in re.split(r"[:\s]+", result[var])
                if lib and not _SANITIZER_RUNTIME.match(os.path.basename(lib))
            ]
            if kept:
                result[var] = ":".join(kept)
            else:
                del result[var]
    return result
