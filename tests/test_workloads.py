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

from tests.workloads.bug_corpus import CORRECT, BugCase


def _make_qubits(n):
    return [pulse.qudit_ref() for _ in range(n)]


def _freq(n):
    return {i: 5.0e9 + i * 0.1e9 for i in range(n)}


def _ir_op_count(ir):
    """Pulse-program ops in kernel IR, not counting qudit binding."""
    return sum(op.kind != "pulse.qudit_arg" for op in ir.ops)


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
    compiled = pulse.compile(module.build, _make_qubits(n), qubit_freq_hz=_freq(n), schedule="asap")
    events = compiled.events()
    assert compiled.metrics.op_count > 0
    assert max(e.start_vtu + e.duration_vtu for e in events) > 0
    for line in {e.line for e in events}:
        timeline = [e for e in events if e.line == line]
        for before, after in zip(timeline, timeline[1:]):
            assert after.start_vtu >= before.start_vtu + before.duration_vtu


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
    """Correct cases should compile without crash (kernel may be a factory)."""
    fn_or_kern = case.build_fn()
    if not callable(fn_or_kern):
        pytest.skip("build_fn returned non-callable")
    if not hasattr(fn_or_kern, "__wrapped__"):
        pytest.skip("not a kernel")

    nargs = fn_or_kern.__wrapped__.__code__.co_argcount
    compiled = pulse.compile(fn_or_kern, _make_qubits(nargs), qubit_freq_hz=_freq(nargs))
    assert compiled.metrics.op_count > 0


# ── Every workload through both paths ────────────────────────────────


@pytest.mark.parametrize("name", _WORKLOAD_MODULES)
def test_workload_compiles_on_the_packed_path(name):
    import importlib

    module = importlib.import_module(f"tests.workloads.{name}")
    n = module.NUM_QUBITS
    ir = module.build(*_make_qubits(n))
    compiled = pulse.compile(module.build, _make_qubits(n), qubit_freq_hz=_freq(n))
    if not any(op.kind == "scf.for" for op in ir.ops):
        assert compiled.metrics.op_count == _ir_op_count(ir)


def test_list_amplitude_packs_like_complex():
    from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    def words(amplitude):
        @pulse.kernel
        def k(q0):
            d0, t0 = get_drive_line(q0)
            drive(d0, square(40, amplitude), t0)

        builder = PackedIRBuilder(name="k", qubit_freq_hz={0: 5e9})
        _trace_kernel_with_builder(k.__wrapped__, builder, [pulse.qudit_ref()])
        return builder.get_buffer().tolist()

    assert words([0.047, 0.01]) == words(0.047 + 0.01j)
    assert words((0.047,)) == words(0.047)
    with pytest.raises(pulse.CompilationError, match=r"\[re\] or \[re, im\]"):
        words([0.1, 0.2, 0.3])
