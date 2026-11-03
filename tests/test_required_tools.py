# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A missing tool skips its tests locally and fails them in CI; a tool runs
without preloaded sanitizer runtimes."""

from __future__ import annotations

import os

import pytest

from tests import _tools


@pytest.fixture
def ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv(_tools.OPTIONAL_TOOLS_VAR, raising=False)
    return monkeypatch


@pytest.mark.parametrize("value", ["", "0", "false", "False", "no"])
def test_a_missing_tool_skips_outside_ci(monkeypatch, value):
    monkeypatch.setenv("CI", value)
    with pytest.raises(pytest.skip.Exception, match="pyright is not installed"):
        _tools.missing("pyright", "pyright is not installed")
    monkeypatch.delenv("CI")
    with pytest.raises(pytest.skip.Exception):
        _tools.missing("pyright", "pyright is not installed")


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_a_missing_tool_fails_in_ci(ci, value):
    ci.setenv("CI", value)
    with pytest.raises(pytest.fail.Exception, match="mypy is not installed, and CI requires it"):
        _tools.missing("mypy", "mypy is not installed")


def test_a_job_can_leave_named_tools_out(ci):
    ci.setenv(_tools.OPTIONAL_TOOLS_VAR, "pyright, mypy")
    with pytest.raises(pytest.skip.Exception):
        _tools.missing("mypy", "mypy is not installed")
    with pytest.raises(pytest.fail.Exception, match="CUDAQ_PULSE_OPTIONAL_TOOLS=cc"):
        _tools.missing("cc", "no C compiler")


def test_a_missing_module_names_its_tool(ci):
    with pytest.raises(pytest.fail.Exception, match="no_such_tool.api is not installed"):
        _tools.import_module("no_such_tool.api")
    ci.setenv(_tools.OPTIONAL_TOOLS_VAR, "no_such_tool")
    with pytest.raises(pytest.skip.Exception):
        _tools.import_module("no_such_tool.api")
    assert _tools.import_module("json").dumps([]) == "[]"


def test_a_missing_c_compiler_fails_in_ci(ci):
    ci.setattr(_tools.shutil, "which", lambda name: None)
    with pytest.raises(pytest.fail.Exception, match="no C compiler"):
        _tools.c_compiler()


def test_a_tool_runs_without_the_sanitizer_runtimes_and_keeps_the_rest():
    env = {
        "LD_PRELOAD": "/gcc/libasan.so.8:/opt/libjemalloc.so.2 /gcc/libubsan.so.1:/gcc/libtsan.so",
        "DYLD_INSERT_LIBRARIES": "/clang/libclang_rt.asan_osx_dynamic.dylib",
        "ASAN_OPTIONS": "detect_leaks=0",
        "PATH": "/bin",
    }
    assert _tools.tool_env(env) == {
        "LD_PRELOAD": "/opt/libjemalloc.so.2",
        "ASAN_OPTIONS": "detect_leaks=0",
        "PATH": "/bin",
    }
    assert env["DYLD_INSERT_LIBRARIES"].endswith("asan_osx_dynamic.dylib")


def test_a_tool_env_defaults_to_this_process_environment(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/gcc/libasan.so.8")
    env = _tools.tool_env()
    assert "LD_PRELOAD" not in env and env["PATH"] == os.environ["PATH"]
