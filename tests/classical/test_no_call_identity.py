# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programs without classical calls compile as they did before calls existed:
the same op counts, and no classical ops, types or attributes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

_ROOT = Path(__file__).resolve().parents[2]


def _workloads():
    spec = importlib.util.spec_from_file_location(
        "compile_time", _ROOT / "benchmarks" / "compiler" / "compile_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return {name: (kernel, n) for name, kernel, n, _ in module.WORKLOADS}


WORKLOADS = _workloads()
BASELINE = json.loads((_ROOT / "benchmarks" / "baselines" / "compile_time.json").read_text())


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_a_program_without_calls_has_no_classical_ir(name):
    kernel, qudit_count = WORKLOADS[name]
    lowered = pulse.compile(kernel).lower(*[Q[i] for i in range(qudit_count)])
    text = lowered.mlir()
    assert "classical" not in text
    assert lowered.classical_report() == []
    assert [e.opened_by for e in lowered.segments()] == ["start"]
    if name in BASELINE:
        assert lowered.metrics.op_count == BASELINE[name]["op_count"]
