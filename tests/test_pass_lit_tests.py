# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""tools/check-pass-lit-tests.py finds the passes no lit test runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "check-pass-lit-tests.py"
if not _TOOL.exists():
    pytest.skip("tools/check-pass-lit-tests.py is not in this tree", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("check_pass_lit_tests", _TOOL)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

_HELP = """\
OVERVIEW: cudaq-pulse optimizer driver

  Compiler passes to run
    Passes:
      --canonicalize                         -   Canonicalize operations
        --max-iterations=<long>              - Seed the worklist in general order
      --cudm-to-llvm                         -   Lower cudm to LLVM
      --pulse-schedule-rcp                   -   Resource-constrained schedule
        --pulse-verify-like-option           - An option, not a pass
      --pulse-verify                         -   Verify
      --qop-to-cudm                          -   Lower qop to cudm
    Pass Pipelines:
      --pulse-pipeline-not-a-pass            -   A pipeline
"""


def test_only_the_projects_passes_are_listed():
    assert check.project_passes(_HELP) == [
        "cudm-to-llvm",
        "pulse-schedule-rcp",
        "pulse-verify",
        "qop-to-cudm",
    ]
    assert check.project_passes("no passes here") == []


def test_a_pass_counts_when_a_run_line_runs_it(tmp_path):
    (tmp_path / "Dialect").mkdir()
    (tmp_path / "Dialect" / "a.mlir").write_text(
        "// RUN: cudaq-pulse-opt --pulse-verify %s \\\n"
        "// RUN:   --pass-pipeline='builtin.module(func.func(pulse-schedule-rcp))' | FileCheck %s\n"
        "// CHECK: qop-to-cudm\n"
    )
    (tmp_path / "b.mlir").write_text("// RUN: cudaq-pulse-opt --cudm-to-llvm-extra %s\n")
    passes = ["cudm-to-llvm", "pulse-schedule-rcp", "pulse-verify", "qop-to-cudm"]
    # A CHECK line and a longer pass name do not count.
    assert check.untested(passes, tmp_path) == ["cudm-to-llvm", "qop-to-cudm"]
