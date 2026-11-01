# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pytest harness for the workload programs (W1–W6) and bug corpus."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest

from tests._handles import Q
from tests._ir import python_ir
from tests.workloads.bug_corpus import CORRECT, BugCase


def _make_qubits(n):
    return [Q[i] for i in range(n)]


_WORKLOAD_MODULES = [
    "w1_bell",
    "w2_cnot_cr",
    "w3_qaoa4",
    "w4_syndrome",
    "w5_dd_cpmg8",
    "w6_vqe_hea",
]


@pytest.mark.parametrize("name", _WORKLOAD_MODULES)
def test_workload_compiles_and_schedules(name):
    import importlib

    module = importlib.import_module(f"tests.workloads.{name}")
    n = module.NUM_QUBITS
    compiled = pulse.compile(module.build, schedule="asap").lower(*_make_qubits(n))
    events = compiled.schedule
    assert compiled.metrics.op_count > 0
    assert max(e.start_cycles + e.duration_cycles for e in events) > 0
    for line in {e.line for e in events}:
        timeline = [e for e in events if e.line == line]
        for before, after in zip(timeline, timeline[1:]):
            assert after.start_cycles >= before.start_cycles + before.duration_cycles


@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_cross_resonance_pulses_wait_for_the_target_tone(policy):
    from tests.workloads import w1_bell

    compiled = pulse.compile(w1_bell.build, schedule=policy).lower(*_make_qubits(2))
    drives = [e for e in compiled.schedule if e.kind == "drive"]
    # Program order: sx on q0, then sx(q1), CR, x(q0), CR-, sx(q1). The CR
    # pulses play on line 0 at q1's tone, between the two pulses on line 1.
    _, sx_first, cr, _, cr_neg, sx_last = drives
    assert cr.start_cycles >= sx_first.start_cycles + sx_first.duration_cycles
    assert sx_last.start_cycles >= cr_neg.start_cycles + cr_neg.duration_cycles


# ── Bug corpus: module imports and stats are sane ─────────────────────


def test_bug_corpus_loads():
    from tests.workloads.bug_corpus import ALL_CASES, CORPUS_STATS

    assert len(ALL_CASES) >= 80
    assert CORPUS_STATS["total"] == len(ALL_CASES)
    for cat in (
        "correct",
        "unintentional_overlap",
        "backward_time_travel",
        "phase_bookkeeping",
        "cross_resonance_miscalibration",
    ):
        assert CORPUS_STATS[cat] > 0, f"missing category: {cat}"


@pytest.mark.parametrize("case", CORRECT, ids=[c.name for c in CORRECT])
def test_correct_corpus_compiles(case: BugCase):
    """Every correct case builds a kernel, which compiles."""
    kernel = case.build_fn()
    assert isinstance(kernel, pulse.Kernel), f"{case.name}: build_fn() returned {kernel!r}"

    nargs = kernel.__wrapped__.__code__.co_argcount
    compiled = pulse.compile(kernel, unroll_threshold=0).lower(*_make_qubits(nargs))
    assert compiled.metrics.op_count > 0


# ── Every workload through both paths ────────────────────────────────


@pytest.mark.parametrize("name", _WORKLOAD_MODULES)
def test_workload_compiles_on_the_packed_path(name):
    import importlib

    module = importlib.import_module(f"tests.workloads.{name}")
    n = module.NUM_QUBITS
    ir = python_ir(module.build, *_make_qubits(n))
    compiled = module.build.lower(*_make_qubits(n))
    assert compiled.metrics.op_count == ir.op_count()


def test_list_amplitude_packs_like_complex():
    from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    def words(amplitude):
        @pulse.kernel
        def k(q0):
            d0 = get_drive_line(q0)
            drive(d0, square(40, amplitude))

        builder = PackedIRBuilder(name="k", qubit_freq_hz={0: 5e9})
        _trace_kernel_with_builder(k.__wrapped__, builder, [Q[0]])
        return builder.get_buffer().tolist()

    assert words([0.047, 0.01]) == words(0.047 + 0.01j)
    assert words((0.047,)) == words(0.047)
    with pytest.raises(pulse.CompilationError, match=r"\[re\] or \[re, im\]"):
        words([0.1, 0.2, 0.3])
