# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A retained loop schedules exactly like its unrolled form."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.lower import _to_program
from cudaq_pulse.passes.ir_types import OpKind
from cudaq_pulse.passes.scheduling import (
    MachineModel,
    ScheduleError,
    schedule_alap,
    schedule_asap,
    schedule_rcp,
)

_FREQ = {0: 5.0e9, 1: 5.1e9, 2: 5.2e9}


def _touch(index):
    """Reading the induction variable makes the frontend unroll the loop."""


# Loop bodies as source templates: {loop} opens the loop under test and {inner}
# a nested one; the rolled variant never reads their induction variables.
_BODIES = {
    "single_line": (
        1,
        """
    d0, t0 = get_drive_line(q0)
    x = square(40, 0.1)
    wait(d0, 30)
    {loop}
        drive(d0, x, t0)
        wait(d0, 200)
""",
    ),
    "synced_lines": (
        2,
        """
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    x = square(40, 0.1)
    cr = gaussian(160, 0.2, 40.0)
    drive(d1, x, t1)
    {loop}
        sync(d0, d1)
        drive(d0, cr, t1)
        drive(d1, x, t1)
        wait(d1, 20)
    drive(d0, x, t0)
""",
    ),
    "independent_rates": (
        2,
        """
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    x = square(40, 0.1)
    {loop}
        drive(d0, x, t0)
        wait(d1, 200)
    drive(d1, x, t1)
""",
    ),
    "tone_coupled": (
        2,
        """
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    x = square(40, 0.1)
    y = square(120, 0.1)
    {loop}
        drive(d0, x, t1)
        t1 = shift_phase(t1, 0.25)
        drive(d1, y, t1)
""",
    ),
    "readout_loop": (
        1,
        """
    d0, t0 = get_drive_line(q0)
    r0, rt = get_readout_line(q0)
    x = square(40, 0.1)
    m = square(300, 0.05)
    {loop}
        drive(d0, x, t0)
        sync(d0, r0)
        readout(r0, m, rt)
""",
    ),
    "feedback": (
        1,
        """
    d0, t0 = get_drive_line(q0)
    r0, rt = get_readout_line(q0)
    x = square(40, 0.1)
    m = square(300, 0.05)
    {loop}
        bit = readout(r0, m, rt)
        sync(d0, r0)
        if bit:
            drive(d0, x, t0)
        else:
            wait(d0, 70)
""",
    ),
    "nested": (
        2,
        """
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    x = square(40, 0.1)
    {loop}
        {inner}
            drive(d0, x, t0)
        sync(d0, d1)
        wait(d1, 70)
""",
    ),
}


def _kernel(name, n, unroll):
    n_qubits, body = _BODIES[name]
    if unroll:
        loop = "for i in range(n):\n        _touch(i)"
        inner = "for j in range(3):\n            _touch(j)"
    else:
        loop = "for _ in range(n):"
        inner = "for _ in range(3):"
    args = ", ".join(f"q{k}" for k in range(n_qubits))
    source = f"def k({args}):" + body.format(loop=loop, inner=inner)
    namespace = {"n": n, "_touch": _touch}
    exec(compile(source, f"<loop body {name}>", "exec"), namespace)
    return pulse.kernel(namespace["k"]), n_qubits


BODIES = list(_BODIES)
TRIP_COUNTS = [0, 1, 2, 3, 4, 7, 50]
SCHEDULERS = {
    "asap": schedule_asap,
    "alap": schedule_alap,
    "rcp": lambda p: schedule_rcp(p, MachineModel(max_concurrent_drives=1)),
}


def _program(body, n, unroll):
    kernel, n_qubits = _kernel(body, n, unroll)
    ir = kernel(*[pulse.qudit_ref() for _ in range(n_qubits)])
    assert ("scf.for" in [op.kind for op in ir.ops]) == (not unroll)
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz=_FREQ)


def _summary(metrics):
    return (
        metrics.total_length_vtu,
        sorted(metrics.per_line_length_vtu.values()),
        metrics.idle_total_vtu,
    )


@pytest.mark.parametrize("scheduler", SCHEDULERS, ids=str)
@pytest.mark.parametrize("n", TRIP_COUNTS)
@pytest.mark.parametrize("body", BODIES)
def test_rolled_loop_schedules_like_unrolled(body, n, scheduler):
    run = SCHEDULERS[scheduler]
    rolled_events, rolled = run(_program(body, n, unroll=False))
    unrolled_events, unrolled = run(_program(body, n, unroll=True))
    assert _summary(rolled) == _summary(unrolled)
    if body == "nested":
        return  # events are per op, not per (outer, inner) iteration pair

    # Rolled loop-body events are the first iteration. Ops before the loop,
    # that first iteration, and ops after the loop line up with the
    # unrolled schedule.
    def timed(events):
        return [
            (e.kind, e.start_vtu, e.duration_vtu, e.iterations)
            for e in events
            if e.kind in (OpKind.DRIVE, OpKind.WAIT, OpKind.READOUT)
        ]

    rolled_timed = timed(rolled_events)
    unrolled_timed = [t[:3] for t in timed(unrolled_events)]
    in_body = [t[3] > 1 for t in rolled_timed]
    if n < 2 or not any(in_body):
        assert [t[:3] for t in rolled_timed] == unrolled_timed
        return
    head = in_body.index(True)
    tail = len(in_body) - in_body[::-1].index(True)
    rolled_plain = [t[:3] for t in rolled_timed]
    assert rolled_plain[:head] == unrolled_timed[:head]
    assert rolled_plain[head:tail] == unrolled_timed[head:tail]
    post = len(rolled_plain) - tail
    assert rolled_plain[tail:] == unrolled_timed[len(unrolled_timed) - post :]


def test_loop_iterations_are_reported():
    events, _ = schedule_asap(_program("nested", 4, unroll=False))
    drives = [e for e in events if e.kind == OpKind.DRIVE]
    assert [e.iterations for e in drives] == [12]


def test_scheduling_time_does_not_grow_with_trip_count():
    small = _program("synced_lines", 8, unroll=False)
    large = _program("synced_lines", 100_000, unroll=False)
    _, m_small = schedule_alap(small)
    _, m_large = schedule_alap(large)
    assert m_large.compile_time_ms < max(5 * m_small.compile_time_ms, 5.0)


def test_symbolic_trip_count_is_rejected():
    program = _program("single_line", 4, unroll=False)
    loop = next(op for op in program.ops if op.kind == OpKind.FOR_LOOP)
    loop.attrs["ub"] = pulse.Parameter("n", 0, "i64")
    with pytest.raises(ScheduleError, match="non-constant trip count"):
        schedule_asap(program)
