# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Op-count checks for the frontend features the compiler relies on.

Loop retention, exact unrolling of index-dependent loops, closures and
helpers, symbolic parameters, and rejection of measurement-dependent
branches in the packed compile() path. Expected counts come from the
kernel structure, not from a previous run, so a frontend that drops or
duplicates ops fails here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.lower import _to_program

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
    ir = dd_loop(n)(pulse.qudit_ref())
    program = _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    # alloc_drive_line, square, for_loop, drive, end_for
    assert program.op_count() == 5
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
        kernel, n_qubits, freq = workloads[name]
        ir = kernel(*[pulse.qudit_ref() for _ in range(n_qubits)])
        program = _to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq)
        assert program.op_count() == count, name


def test_compile_unrolls_loops_into_the_packed_program():
    ck = pulse.compile(dd_loop(8), [pulse.qudit_ref()], qubit_freq_hz={0: 5e9}, passes=())
    # alloc_drive_line, square, and one drive per iteration
    assert ck.metrics.op_count == 10
    assert ck.mlir.count("= pulse.drive ") == 8


# ── Index-dependent loops and conditions are traced exactly ──────────


def _surface_ops(d: int) -> int:
    n_data = d * d
    n_anc = (d - 1) * (d - 1) + 2 * (d - 1)
    ops = (n_data + n_anc) + 3 + n_anc  # lines, waveforms, one H per ancilla
    for a in range(n_anc):
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
    ck = pulse.compile(
        kernel, [pulse.qvec_ref(n)], qubit_freq_hz={i: 5e9 for i in range(n)}, passes=()
    )
    assert ck.metrics.op_count == _surface_ops(d)


@pytest.mark.parametrize("lm", [(3, 3), (6, 6)])
def test_bicycle_code_op_count(lm):
    l, m = lm  # noqa: E741
    kernel, n = bicycle_syndrome(l, m)
    ck = pulse.compile(
        kernel, [pulse.qvec_ref(n)], qubit_freq_hz={i: 5e9 for i in range(n)}, passes=()
    )
    # lines, two waveforms, and per check six (sync + two drives)
    assert ck.metrics.op_count == n + 2 + 2 * l * m * 6 * 3


# ── Lines and tones held in lists ────────────────────────────────────


def test_list_held_lines_are_rebound():

    @pulse.kernel
    def k(q0):
        line, tone = get_drive_line(q0)
        lines, tones = [line], [tone]
        x = square(40, 0.1)
        drive(lines[0], x, tones[0])
        drive(lines[0], x, tones[0])

    first, second = [op for op in k(pulse.qudit_ref()).ops if op.kind == "pulse.drive"]
    assert second.operands[0] is first.results[0]
    assert second.operands[2] is first.results[1]


def test_loop_over_a_list_held_line_stays_rolled():
    from cudaq_pulse.passes.scheduling import schedule_asap

    @pulse.kernel
    def k(q0):
        line, tone = get_drive_line(q0)
        lines, tones = [line], [tone]
        x = square(40, 0.1)
        for _ in range(6):
            drive(lines[0], x, tones[0])
        wait(lines[0], 10)

    ir = k(pulse.qudit_ref())
    assert [op.kind for op in ir.ops].count("scf.for") == 1
    program = _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    assert schedule_asap(program)[1].total_length_vtu == 6 * 40 + 10


def test_compile_with_default_passes_accepts_list_indexed_kernels():
    workloads = _compile_time_workloads()
    for name in ("qaoa4", "syndrome", "vqe_hea"):
        kernel, n_qubits, freq = workloads[name]
        pulse.compile(kernel, [pulse.qudit_ref() for _ in range(n_qubits)], qubit_freq_hz=freq)
    for kernel, n in (surface_syndrome(3), bicycle_syndrome(3, 3)):
        pulse.compile(kernel, [pulse.qvec_ref(n)], qubit_freq_hz={i: 5e9 for i in range(n)})


# ── Closures, helpers, parameters ────────────────────────────────────


def test_closure_values_and_inlined_helpers():
    shift = 0.25

    def play(line, tone, amp=0.3):
        drive(line, square(10, amp), tone)

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        t0 = shift_phase(t0, shift)
        play(d0, t0)
        play(d0, t0, amp=shift)

    ir = k(pulse.qudit_ref())
    amps = [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"]
    assert amps == [0.3, 0.25]
    phase = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
    assert phase.attrs["phase_rad"] == 0.25


def test_symbolic_parameters_compile_once():

    @pulse.kernel
    def k(q0, amp: float):
        d0, t0 = get_drive_line(q0)
        drive(d0, gaussian(40, amp, 10.0), t0)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9})
    assert ck.parameters == ["amp"]
    assert ck(amp=0.5).mlir != ck(amp=0.25).mlir


def test_compile_rejects_measurement_branch_at_the_condition():

    @pulse.kernel
    def k(q0):
        r0, t0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2), t0)
        if bit:
            wait(r0, 10)

    with pytest.raises(CompilationError, match="measurement-conditioned branches") as info:
        pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9})
    decoded, index = info.value.pulse_source
    _, line, *_ = decoded.location(index)
    assert line == k.__wrapped__.__code__.co_firstlineno + 4
