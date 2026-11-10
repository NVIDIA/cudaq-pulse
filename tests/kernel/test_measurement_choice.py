# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numbers and waveforms chosen by a measurement-conditioned if.

A name the arms of an if on a measurement leave different numbers on holds,
after the if, the value of the arm that ran: one select on the measurement.
Where the target takes that parameter at run time every use reads the
select; elsewhere an op given it runs in each arm of an if on the same
measurement. Each chosen form must compile, schedule and simulate exactly
like the form with the op written in the arms, with either lowering.
"""

from __future__ import annotations

import inspect
import math

import cudaq_pulse as pulse
import cudaq_pulse.compiler
import cudaq_pulse.ops
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets._registry import current_target

from tests._handles import Q
from tests._ir import python_ir
from tests.runtime._lowering import EVOLVE_PASSES, _handles
from tests.runtime.test_simulate import _HALF_PI, _PI, _run, _target, conditional_flip


def _lower(kernel, n=1, **kwargs):
    return pulse.compile(kernel, **kwargs).lower(*[Q[i] for i in range(n)])


def _events(kernel, n=1, **kwargs):
    return [
        (e.kind, e.line, e.start_cycles, e.duration_cycles)
        for e in _lower(kernel, n, **kwargs).schedule
    ]


def _line_of(kernel, text):
    """The source line of *kernel* that contains *text*."""
    lines, first = inspect.getsourcelines(kernel.__wrapped__)
    return first + next(i for i, line in enumerate(lines) if text in line)


def _kinds(kernel, *args):
    return [op.kind for op in python_ir(kernel, *args).ops]


# ── Chosen in both arms ──────────────────────────────────────────────


@pulse.kernel
def chosen(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    if m:
        amp = 0.3
    else:
        amp = 0.1
    drive(d, square(40, amp))
    wait(d, 10)


@pulse.kernel
def in_arms(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    if m:
        drive(d, square(40, 0.3))
    else:
        drive(d, square(40, 0.1))
    wait(d, 10)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_number_chosen_in_both_arms_schedules_like_the_op_in_the_arms(schedule):
    events = _events(chosen, schedule=schedule)
    assert events == _events(in_arms, schedule=schedule)
    # The drive waits for the measurement that chose its amplitude.
    assert [e[2] for e in events if e[0] == "drive"] == [100, 100]


def test_the_op_given_a_choice_is_emitted_in_each_arm():
    kinds = _kinds(chosen, Q[0])
    # The if that chose amp has empty arms; the drive is in the next one.
    start = kinds.index("scf.if")
    assert kinds[start : start + 5] == [
        "scf.if",
        "scf.yield",
        "scf.else",
        "scf.yield",
        "scf.if_end",
    ]
    # One waveform for each amplitude, then the drive in each arm.
    assert kinds[start + 5 :] == [
        "pulse.square",
        "pulse.square",
        "scf.if",
        "pulse.drive",
        "scf.yield",
        "scf.else",
        "pulse.drive",
        "scf.yield",
        "scf.if_end",
        "pulse.wait",
    ]
    assert _amplitudes(chosen) == [0.3, 0.1]
    compiled = _lower(chosen)
    compiled._module.run_passes(["pulse-verify"])
    for export in (compiled.export_controller, compiled.export_emulation):
        assert export()["events"]


def test_a_prior_value_and_one_arm_choose_too():
    @pulse.kernel
    def prior(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        amp = 0.1
        if m:
            amp = 0.3
        drive(d, square(40, amp))
        wait(d, 10)

    assert _events(prior) == _events(in_arms)


def test_a_wait_chosen_by_a_count():
    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05))
        n = 1
        if bit:
            n = 2
        wait(d0, 10 * n)
        drive(d0, square(40, 0.1))

    ir = python_ir(k, Q[0])
    waits = [op.attrs["duration"] for op in ir.ops if op.kind == "pulse.wait"]
    assert waits == [20, 10]
    # The drive after the if starts when the longer arm ends.
    assert _events(k)[-1] == ("drive", 0, 300 + 20, 40)


# ── elif chains and nested ifs ───────────────────────────────────────


@pulse.kernel
def elif_chain(q0, q1):
    d = get_drive_line(q0)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    a = readout(r0, square(100, 0.05))
    b = readout(r1, square(60, 0.05))
    if a:
        amp = 0.3
    elif b:
        amp = 0.2
    else:
        amp = 0.1
    drive(d, square(40, amp))


@pulse.kernel
def elif_in_arms(q0, q1):
    d = get_drive_line(q0)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    a = readout(r0, square(100, 0.05))
    b = readout(r1, square(60, 0.05))
    if a:
        drive(d, square(40, 0.3))
    elif b:
        drive(d, square(40, 0.2))
    else:
        drive(d, square(40, 0.1))


@pulse.kernel
def nested(q0, q1):
    d = get_drive_line(q0)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    a = readout(r0, square(100, 0.05))
    b = readout(r1, square(60, 0.05))
    if a:
        if b:
            amp = 0.4
        else:
            amp = 0.3
    else:
        amp = 0.1
    drive(d, square(40, amp))


@pulse.kernel
def nested_in_arms(q0, q1):
    d = get_drive_line(q0)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    a = readout(r0, square(100, 0.05))
    b = readout(r1, square(60, 0.05))
    if a:
        if b:
            drive(d, square(40, 0.4))
        else:
            drive(d, square(40, 0.3))
    else:
        drive(d, square(40, 0.1))


@pytest.mark.parametrize(
    ("kernel", "reference"), [(elif_chain, elif_in_arms), (nested, nested_in_arms)]
)
def test_elif_chains_and_nested_ifs_choose_like_the_op_in_the_arms(kernel, reference):
    assert _events(kernel, 2) == _events(reference, 2)
    kinds = _kinds(kernel, Q[0], Q[1])
    # The two ifs that chose amp, then the drive's two.
    assert kinds.count("scf.if") == 2 + 2 and kinds.count("pulse.drive") == 3
    assert _amplitudes(kernel, 2) == _amplitudes(reference, 2)


def _amplitudes(kernel, n=1):
    """The amplitude of each drive's waveform, in program order."""
    ops = python_ir(kernel, *[Q[i] for i in range(n)]).ops
    wf = {op.results[0].vid: op.attrs.get("amplitude") for op in ops if op.results}
    return [wf[op.operands[1].vid] for op in ops if op.kind == "pulse.drive"]


def test_choices_on_two_measurements_combine():
    @pulse.kernel
    def k(q0, q1):
        d = get_drive_line(q0)
        r0 = get_readout_line(q0)
        r1 = get_readout_line(q1)
        a = readout(r0, square(100, 0.05))
        b = readout(r1, square(60, 0.05))
        scale = 1.0
        if a:
            scale = 2.0
        amp = 0.1
        if b:
            amp = 0.2
        drive(d, square(40, amp * scale))

    # b decides the outer if of the drive, a the inner ones.
    assert _amplitudes(k, 2) == pytest.approx([0.4, 0.2, 0.2, 0.1])
    assert _kinds(k, Q[0], Q[1]).count("scf.if") == 2 + 3


# ── Using a choice ───────────────────────────────────────────────────


def test_arithmetic_casts_and_pure_calls_apply_to_each_value():
    def halve(x):
        return x / 2

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            x = 0.6
        else:
            x = 0.2
        drive(d, square(40, -halve(x)))
        drive(d, square(40, min(x, 0.5)))
        drive(d, square(40, math.sin(x)))
        wait(d, int(x * 100))

    assert _amplitudes(k) == pytest.approx([-0.3, -0.1, 0.5, 0.2, math.sin(0.6), math.sin(0.2)])
    ops = python_ir(k, Q[0]).ops
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.wait"] == [60, 20]


def test_a_choice_as_a_condition_branches_on_its_measurement():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            amp = 0.3
        else:
            amp = 0.0
        if amp > 0.2:
            drive(d, square(40, 0.1))
        if not amp:
            wait(d, 30)
        if amp < 1.0:  # true in both arms: an if known while tracing
            wait(d, 5)

    ir = python_ir(k, Q[0])
    ifs = [op for op in ir.ops if op.kind == "scf.if"]
    assert len(ifs) == 3 and all(op.operands[0].vtype == "measurement" for op in ifs)
    kinds = [op.kind for op in ir.ops]
    arms = []
    for start in [i for i, kind in enumerate(kinds) if kind == "scf.if"]:
        middle = kinds.index("scf.else", start)
        end = kinds.index("scf.if_end", middle)
        arms.append((kinds[start + 1 : middle - 1], kinds[middle + 1 : end - 1]))
    # amp > 0.2 holds where m does, and `not amp` where m does not.
    assert arms == [([], []), (["pulse.square", "pulse.drive"], []), ([], ["pulse.wait"])]
    assert kinds[-1] == "pulse.wait"


def test_waveforms_made_from_a_choice_are_chosen_too():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            n = 40
        else:
            n = 60
        wf = wf_add(square(n, 0.1), square(n, 0.2))
        drive(d, wf_scale(wf, 0.5))
        wait(d, 10)

    @pulse.kernel
    def reference(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            drive(d, wf_scale(wf_add(square(40, 0.1), square(40, 0.2)), 0.5))
        else:
            drive(d, wf_scale(wf_add(square(60, 0.1), square(60, 0.2)), 0.5))
        wait(d, 10)

    assert _events(k) == _events(reference)
    assert [e[3] for e in _events(k) if e[0] == "drive"] == [40, 60]


def test_a_choice_in_a_helper():
    def pulse_with(d, amp):
        drive(d, square(40, amp))

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            amp = 0.3
        else:
            amp = 0.1
        pulse_with(d, amp)
        wait(d, 10)

    assert _events(k) == _events(in_arms)


# ── Rolled loops ─────────────────────────────────────────────────────


@pulse.kernel
def chosen_in_loop(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(4):
        m = readout(r, square(100, 0.05))
        if m:
            amp = 0.3
        else:
            amp = 0.1
        drive(d, square(40, amp))


@pulse.kernel
def in_arms_in_loop(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(4):
        m = readout(r, square(100, 0.05))
        if m:
            drive(d, square(40, 0.3))
        else:
            drive(d, square(40, 0.1))


def test_a_choice_made_in_a_rolled_body_stays_rolled():
    compiled = _lower(chosen_in_loop, unroll_threshold=0)
    assert compiled.mlir().count("scf.for") == 1
    reference = _lower(in_arms_in_loop, unroll_threshold=0)
    assert [(e.kind, e.start_cycles) for e in compiled.schedule] == [
        (e.kind, e.start_cycles) for e in reference.schedule
    ]


def test_a_choice_carried_by_a_loop_unrolls_it():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        amp = 0.0
        for _ in range(3):
            m = readout(r, square(100, 0.05))
            if m:
                amp = 0.3
            drive(d, square(40, amp))

    compiled = _lower(k, unroll_threshold=0)
    assert "scf.for" not in compiled.mlir()
    # Each drive takes one of the values the earlier ifs may have chosen.
    drives = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.drive"]
    assert len(drives) == 2 + 3 + 4


def test_a_choice_with_too_many_values_is_rejected():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        amp = 0.0
        for i in range(70):
            if readout(r, square(10, 0.05)):
                amp = 0.001 * i
            drive(d, square(10, amp))

    with pytest.raises(CompilationError, match="would have more than 64 values"):
        _lower(k, unroll_threshold=1000)


def test_a_choice_made_in_a_rolled_body_is_rejected_after_it():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(4):
            m = readout(r, square(100, 0.05))
            if m:
                amp = 0.3
            else:
                amp = 0.1
            drive(d, square(40, amp))
        drive(d, square(40, amp))

    with pytest.raises(CompilationError, match="outside the loop or branch"):
        _lower(k, unroll_threshold=0)
    # Unrolled, the last iteration's choice is defined after the loop.
    assert len(_lower(k, unroll_threshold=100).schedule) == 4 + 4 * 2 + 2


# ── Simulation ───────────────────────────────────────────────────────


def test_a_chosen_reset_simulates_like_the_op_in_the_arms():
    @pulse.kernel
    def chosen_flip(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        r0 = get_readout_line(q0)
        drive(d0, square(40, _HALF_PI))
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        amp = 0.0
        if bit:
            amp = _PI
        drive(d1, square(40, amp))

    expected = _run(conditional_flip, 2, shots=200, seed=3).density_matrix
    actual = _run(chosen_flip, 2, shots=200, seed=3).density_matrix
    np.testing.assert_allclose(actual, expected, atol=1e-12)

    @pulse.kernel
    def reset(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        drive(d0, square(40, _HALF_PI))
        sync(d0, r0)
        if readout(r0, square(200, 0.05)):
            amp = _PI
        else:
            amp = 0.0
        drive(d0, square(40, amp))

    rho = _run(reset, shots=200, seed=2).density_matrix
    np.testing.assert_allclose(rho, [[1, 0], [0, 0]], atol=1e-6)


# ── Rejected ─────────────────────────────────────────────────────────


def test_a_name_assigned_in_one_arm_only_is_rejected_with_its_line():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if m:
            amp = 0.3
        drive(d, square(40, amp))

    line = _line_of(k, "if m:")
    with pytest.raises(
        CompilationError,
        match=rf"'amp' is assigned in only one arm of the measurement-conditioned if at "
        rf"line {line} and has no value before it",
    ):
        _lower(k)


def test_a_name_assigned_in_one_arm_of_an_elif_is_rejected():
    @pulse.kernel
    def k(q0, q1):
        d = get_drive_line(q0)
        r0 = get_readout_line(q0)
        r1 = get_readout_line(q1)
        a = readout(r0, square(100, 0.05))
        b = readout(r1, square(60, 0.05))
        if a:
            amp = 0.3
        elif b:
            amp = 0.2
        drive(d, square(40, amp))

    with pytest.raises(CompilationError, match="'amp' is assigned in only one arm"):
        _lower(k, 2)


@pytest.mark.parametrize(
    ("then", "other", "what"),
    [
        ("d", "r", "a drive line"),
        ("0.3", "square(40, 0.3)", "a waveform"),
        ("0.3", "'x'", "str 'x'"),
    ],
)
def test_values_an_if_cannot_choose_are_rejected(then, other, what):
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    m = readout(r, square(100, 0.05))\n"
        "    if m:\n"
        f"        x = {then}\n"
        "    else:\n"
        f"        x = {other}\n"
        "    wait(d, 10)\n"
    )
    namespace: dict = {}
    exec(compile(source, "<choice>", "exec"), namespace)
    with pytest.raises(CompilationError, match="'x' would depend on a measurement result") as e:
        _lower(pulse.kernel(namespace["k"]))
    assert what in str(e.value) and "line 5" in str(e.value)


@pytest.mark.parametrize(
    ("use", "message"),
    [
        ("for _ in range(n): wait(d, 10)", "a range.. bound is the value the measurement"),
        ("[0, 1, 2][n]", "cannot be used for an index"),
    ],
)
def test_python_cannot_use_a_choice_while_tracing(use, message):
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    m = readout(r, square(100, 0.05))\n"
        "    if m:\n"
        "        n = 2\n"
        "    else:\n"
        "        n = 1\n"
        f"    {use}\n"
    )
    namespace: dict = {}
    exec(compile(source, "<choice>", "exec"), namespace)
    with pytest.raises(CompilationError, match=message):
        _lower(pulse.kernel(namespace["k"]))


# ── Lowering: one run-time value, or the op in each arm ─────────────


_KINDS = ("amplitude", "phase", "frequency")
_UNFUSED = tuple(p for p in EVOLVE_PASSES if p != "fuse_pulses")


@pytest.fixture
def at_runtime(monkeypatch):
    """Lower with a target that takes the kinds given at run time."""

    def use(*kinds):
        monkeypatch.setattr(
            pulse.compiler, "_can_select_at_runtime", lambda target, kind: kind in kinds
        )

    return use


def _fresh(kernel, n=1, **options):
    kernel.cache_clear()
    dev = pulse.device(_target(n))
    lowered = pulse.compile(kernel, **options).lower(*_handles(dev, [Q[i] for i in range(n)]))
    kernel.cache_clear()
    return lowered


def _both(kernel, at_runtime, n=1, **kwargs):
    """``kernel`` lowered with each op in the arms, then with run-time values."""
    arms = _fresh(kernel, n, **kwargs)
    at_runtime(*_KINDS)
    return arms, _fresh(kernel, n, **kwargs)


@pulse.kernel
def chosen_phase(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        ph = math.pi
    else:
        ph = 0.0
    shift_phase(d0.tone, ph)
    drive(d0, square(40, _HALF_PI))


@pulse.kernel
def chosen_frequency(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        df = 2.0e6
    else:
        df = -1.0e6
    shift_frequency(d0.tone, df)
    drive(d0, square(400, _HALF_PI))


@pulse.kernel
def chosen_amplitudes(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        amp = _HALF_PI
    else:
        amp = 0.5 * _HALF_PI
    drive(d0, square(40, amp))
    wait(d0, 8)
    drive(d0, square(24, amp))
    drive(d0, square(16, amp))


@pulse.kernel
def chosen_on_two(q0, q1):
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    drive(d1, square(40, _HALF_PI))
    sync(d1, r1)
    a = readout(r0, square(100, 0.05))
    b = readout(r1, square(60, 0.05))
    if a:
        amp = _PI
    elif b:
        amp = _HALF_PI
    else:
        amp = 0.0
    drive(d1, square(40, amp))


def test_without_a_target_that_takes_values_at_run_time_each_op_runs_in_the_arms():
    for target in (current_target(), pulse.device(_target())):
        assert not any(pulse.compiler._can_select_at_runtime(target, kind) for kind in _KINDS)
    lowered = _fresh(chosen_amplitudes)
    assert not lowered.runtime_selects
    assert "arith.select" not in lowered.mlir()
    assert lowered.mlir().count("scf.if") == 3


def test_a_value_taken_at_run_time_is_selected_once_and_read_by_every_use(at_runtime):
    arms, kept = _both(chosen_amplitudes, at_runtime, passes=_UNFUSED)
    assert kept.runtime_selects
    text = kept.mlir()
    assert text.count("arith.select") == 1
    assert "scf.if" not in text
    # One drive per use rather than one per use and arm.
    assert [e.kind for e in kept.schedule].count("drive") == 4
    assert [e.kind for e in arms.schedule].count("drive") == 7


@pytest.mark.parametrize("uses", [1, 5, 20])
def test_run_time_values_do_not_grow_with_the_uses(at_runtime, uses):
    namespace = {}
    body = "\n".join(f"    drive(d, square({40 + 8 * i}, amp))" for i in range(uses))
    exec(
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    if readout(r, square(100, 0.05)):\n"
        "        amp = 0.3\n"
        "    else:\n"
        "        amp = 0.1\n" + body + "\n",
        vars(pulse.ops),
        namespace,
    )
    kernel = pulse.kernel(namespace["k"])
    arms, kept = _both(kernel, at_runtime, passes=_UNFUSED)
    assert arms.mlir().count("scf.if") == uses
    assert kept.mlir().count("scf.if") == 0
    assert len(kept.export_controller()["events"]) == uses + 1
    assert len(arms.export_controller()["events"]) == 2 * uses + 1


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize(
    ("kernel", "n"),
    [(chosen_amplitudes, 1), (chosen_phase, 1), (chosen_frequency, 1), (chosen_on_two, 2)],
)
def test_run_time_values_schedule_and_simulate_like_the_op_in_the_arms(
    monkeypatch, at_runtime, kernel, n, schedule
):
    # Fusion can merge uses of one run-time value that the arms keep apart,
    # so timing is compared without it.
    for passes in (_UNFUSED, EVOLVE_PASSES):
        monkeypatch.undo()
        arms, kept = _both(kernel, at_runtime, n, passes=passes, schedule=schedule)
        assert kept.runtime_selects and not arms.runtime_selects
        if "fuse_pulses" not in passes:
            # Each use plays when it does in either arm.
            assert _timing(kept) == _timing(arms)
        expected = pulse.simulate(arms, shots=200, seed=4, levels=2).density_matrix
        actual = pulse.simulate(kept, shots=200, seed=4, levels=2).density_matrix
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def _timing(lowered):
    return sorted({(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule})


def test_values_that_move_ops_in_time_always_run_in_the_arms(at_runtime):
    @pulse.kernel
    def chosen_duration(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        n = 40
        if readout(r, square(100, 0.05)):
            n = 80
        drive(d, square(n, 0.1))
        wait(d, n)

    arms, kept = _both(chosen_duration, at_runtime)
    assert not kept.runtime_selects
    assert kept.mlir() == arms.mlir()
    assert "arith.select" not in kept.mlir()


def test_each_kind_is_lowered_as_the_target_takes_it(at_runtime):
    @pulse.kernel
    def both_kinds(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        if readout(r0, square(200, 0.05)):
            amp, ph = _PI, math.pi
        else:
            amp, ph = 0.0, 0.0
        shift_phase(d0.tone, ph)
        drive(d0, square(40, amp))

    at_runtime("amplitude")
    text = _fresh(both_kinds).mlir()
    # The amplitude is selected once; the phase shift runs in the arms.
    assert text.count("arith.select") == 1
    assert text.count("scf.if") == 1
    assert "pulse.shift_phase" in text.split("scf.if")[1]
