# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Op-count checks for the frontend features the compiler relies on.

Loop retention, exact unrolling of index-dependent loops, closures and
helpers, symbolic parameters, and measurement-dependent branches kept as
conditional regions in the packed compile() path. Expected counts come from the
kernel structure, not from a previous run, so a frontend that drops or
duplicates ops fails here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest

from tests._handles import Q
from tests._ir import python_ir
from tests.kernel.ir_corpus import bicycle_syndrome, dd_loop, surface_syndrome

_ROOT = Path(__file__).resolve().parents[1]


def _compile_time_workloads():
    spec = importlib.util.spec_from_file_location(
        "compile_time", _ROOT / "benchmarks" / "compiler" / "compile_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return {name: (k, n, f) for name, k, n, f in module.WORKLOADS}


# ── Loop retention ───────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 8, 32, 128, 512])
def test_dd_loop_is_retained_with_constant_op_count(n):
    ir = python_ir(dd_loop(n), Q[0], unroll_threshold=0)
    # get_drive_line, square, scf.for, drive, scf.for_end
    assert ir.op_count() == 5
    (loop,) = [op for op in ir.ops if op.kind == "scf.for"]
    assert loop.attrs["ub"] == n


def test_compile_time_workload_op_counts():
    expected = {
        "bell": 15,
        "cnot_cr": 12,
        "qaoa4": 52,
        "syndrome": 53,
        "dd_cpmg8": 7,
        "vqe_hea": 58,
    }
    workloads = _compile_time_workloads()
    for name, count in expected.items():
        kernel, qudit_count, freq = workloads[name]
        ir = python_ir(kernel, *[Q[i] for i in range(qudit_count)])
        assert ir.op_count() == count, name


@pytest.mark.parametrize("n", [1, 8, 512])
def test_compile_retains_loops_in_the_packed_program(n):
    ck = pulse.compile(dd_loop(n), passes=(), unroll_threshold=0).lower(Q[0])
    # get_drive_line, square, scf.for, drive, scf.for_end
    assert ck.metrics.op_count == 5
    assert ck.mlir().count("= pulse.drive ") == 1
    assert len(ck.schedule) == n


# ── Index-dependent loops and conditions are traced exactly ──────────


def _surface_ops(d: int) -> int:
    data_count = d * d
    ancilla_count = (d - 1) * (d - 1) + 2 * (d - 1)
    ops = (data_count + ancilla_count) + 3 + ancilla_count  # lines, waveforms, one H per ancilla
    for a in range(ancilla_count):
        bulk = a < (d - 1) * (d - 1)
        row = a // (d - 1) if bulk else -1
        col = a % (d - 1) if bulk else a - (d - 1) * (d - 1)
        for dr in range(2):
            for dc in range(2):
                if 0 <= row + dr < d and 0 <= col + dc < d:
                    ops += 3  # sync and two CR drives
    return ops


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_code_op_count(d):
    kernel, n = surface_syndrome(d)
    ck = pulse.compile(kernel, passes=()).lower(Q[0:n])
    assert ck.metrics.op_count == _surface_ops(d)


@pytest.mark.parametrize("lm", [(3, 3), (6, 6)])
def test_bicycle_code_op_count(lm):
    l, m = lm  # noqa: E741
    kernel, n = bicycle_syndrome(l, m)
    ck = pulse.compile(kernel, passes=()).lower(Q[0:n])
    # lines, two waveforms, and per check six (sync + two drives)
    assert ck.metrics.op_count == n + 2 + 2 * l * m * 6 * 3


# ── Lines and tones held in lists ────────────────────────────────────


def test_list_held_lines_are_rebound():

    @pulse.kernel
    def k(q0):
        line = get_drive_line(q0)
        lines, tones = [line], [line.tone]
        x = square(40, 0.1)
        drive(lines[0], x, tone=tones[0])
        drive(lines[0], x, tone=tones[0])

    first, second = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.drive"]
    assert second.operands[0] is first.results[0]
    assert second.operands[2] is first.results[1]


def test_loop_over_a_list_held_line_stays_rolled():
    @pulse.kernel
    def k(q0):
        line = get_drive_line(q0)
        lines, tones = [line], [line.tone]
        x = square(40, 0.1)
        for _ in range(6):
            drive(lines[0], x, tone=tones[0])
        wait(lines[0], 10)

    ir = python_ir(k, Q[0], unroll_threshold=0)
    assert [op.kind for op in ir.ops].count("scf.for") == 1
    compiled = pulse.compile(k, schedule="asap", unroll_threshold=0).lower(Q[0])
    assert max(e.start_cycles + e.duration_cycles for e in compiled.schedule) == 6 * 40 + 10


def test_compile_with_default_passes_accepts_list_indexed_kernels():
    workloads = _compile_time_workloads()
    for name in ("qaoa4", "syndrome", "vqe_hea"):
        kernel, qudit_count, freq = workloads[name]
        kernel.lower(*[Q[i] for i in range(qudit_count)])
    for kernel, n in (surface_syndrome(3), bicycle_syndrome(3, 3)):
        kernel.lower(Q[0:n])


# ── Closures, helpers, parameters ────────────────────────────────────


def test_closure_values_and_inlined_helpers():
    shift = 0.25

    def play(line, tone, amp=0.3):
        drive(line, square(10, amp), tone=tone)

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        shift_phase(d0.tone, shift)
        play(d0, d0.tone)
        play(d0, d0.tone, amp=shift)

    ir = python_ir(k, Q[0])
    amps = [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"]
    assert amps == [0.3, 0.25]
    phase = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
    assert phase.attrs["phase_rad"] == 0.25


def test_symbolic_parameters_compile_once():

    @pulse.kernel
    def k(q0, amp: float):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(40, amp, 10.0))

    ck = pulse.compile(k)
    q = Q[0]
    assert ck.parameters == ("amp",)
    assert ck.lower(q, amp=0.5).mlir() != ck.lower(q, amp=0.25).mlir()
    assert ck.cache_info().compiled == 1


def test_compile_retains_a_measurement_branch():

    @pulse.kernel
    def k(q0):
        r0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2))
        if bit:
            wait(r0, 10)

    ir = python_ir(k, Q[0])
    compiled = k.lower(Q[0])
    assert compiled.metrics.op_count == ir.op_count()
    assert compiled.mlir().count("scf.if") == 1
