# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A run-time wait holds back only the lines that depend on it: qubits that
wait on answers of their own keep their own timing, and lines wait for one
another only where a sync, a shared answer or a shared resource ties them."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Bits, Float

from tests._handles import Q
from tests.classical._schedule_matrix import line_timelines

POLICIES = ("asap", "alap")
TAKES = ("result", "timeout", "blocking")
LONG, RO, BOUNDS = 2000, 200, (16, 4000)


def _length(mode):
    return pulse.classical(
        lambda bits: 100.0 + 50.0 * bits[0],
        name=f"length_{mode}",
        args=(Bits(1),),
        results=(Float(),),
        max_latency_ns=50,
        latency_mode=mode,
    )


LENGTH = {"bounded": _length("bounded"), "blocking": _length("blocking")}


def _take(future, take):
    if take == "timeout":
        _, x = future.result(timeout=100 * pulse.ns)
        return x
    return future.result()


def _round(q, take):
    """Measure, ask for a wait's length from the bit, wait that long and
    drive."""
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, 0.1))
    m = readout(r, square(RO, 0.05))
    fn = LENGTH["blocking" if take == "blocking" else "bounded"]
    wait(d, _take(fn.submit([m]), take), bounds=BOUNDS)
    drive(d, square(40, 0.1))


def _busy(p, length=LONG):
    e = get_drive_line(p)
    drive(e, square(length, 0.01))
    return e, e.tone


def _own(low, k=0):
    """The pulses and waits on qudit *k*'s lines, with their timing."""
    qudit = low.qudits[k]
    return [
        (e.kind, e.start_cycles, e.duration_cycles)
        for e in low.schedule
        if e.link is None and e.qubit == qudit
    ]


def _timed(low):
    """When each line plays and waits at run time, with back-to-back
    pulses (which unrolling fuses) as one."""
    spans = sorted(
        (e.line, e.kind, e.start_cycles, e.start_cycles + e.duration_cycles)
        for e in low.schedule
        if e.kind in ("drive", "readout") or (e.kind == "wait" and e.duration_cycles == BOUNDS[0])
    )
    merged: list[list] = []
    for line, kind, start, end in spans:
        last = merged[-1] if merged else None
        if last and last[:2] == [line, kind] and kind != "wait" and last[3] == start:
            last[3] = end
        else:
            merged.append([line, kind, start, end])
    return merged


def _lower(kernel, *args, **options):
    return pulse.compile(kernel, **options).lower(*args)


@pytest.mark.parametrize("take", TAKES)
@pytest.mark.parametrize("policy", POLICIES)
def test_two_qubits_with_their_own_waits_do_not_stall_each_other(policy, take):
    def pair(length):
        @pulse.kernel
        def k(q, p):
            _round(q, take)
            _busy(p, length)
            _round(p, take)

        return _lower(k, Q[0], Q[1], schedule=policy)

    low = pair(LONG)
    timelines = line_timelines(low)
    assert len(set(timelines.values())) == 2
    # q runs the same however long p is busy before its own wait; they
    # share only the link.
    assert _own(low) == _own(pair(8))
    waits = [e for e in low.schedule if e.kind == "wait" and e.link is None]
    assert min(w.start_cycles for w in waits) < LONG


@pytest.mark.parametrize("take", TAKES)
@pytest.mark.parametrize("policy", POLICIES)
def test_a_sync_after_a_run_time_wait_still_aligns(policy, take):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        r = get_readout_line(q)
        e, u = _busy(p)
        m = readout(r, square(RO, 0.05))
        wait(d, _take(LENGTH["blocking" if take == "blocking" else "bounded"].submit([m]), take))
        sync(d, e)
        drive(d, square(40, 0.1))
        drive(e, square(40, 0.1), tone=u)

    low = _lower(k, Q[0], Q[1], schedule=policy)
    (variable,) = [e for e in low.schedule if e.kind == "wait" and e.link is None and e.line == 0]
    after = [e for e in low.schedule if e.kind == "drive" and e.duration_cycles == 40]
    assert len(after) == 2
    assert after[0].start_cycles == after[1].start_cycles >= variable.end
    assert after[0].start_cycles >= LONG


@pytest.mark.parametrize("take", TAKES)
@pytest.mark.parametrize("policy", POLICIES)
def test_an_answer_used_on_two_lines_holds_back_both(policy, take):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        r = get_readout_line(q)
        e, u = _busy(p)
        m = readout(r, square(RO, 0.05))
        x = _take(LENGTH["blocking" if take == "blocking" else "bounded"].submit([m]), take)
        wait(d, x, bounds=BOUNDS)
        wait(e, x, bounds=BOUNDS)
        drive(d, square(40, 0.1))

    low = _lower(k, Q[0], Q[1], schedule=policy)
    assert len(set(line_timelines(low).values())) == 1
    waits = [e for e in low.schedule if e.kind == "wait" and e.link is None]
    on_q, on_p = (next(w for w in waits if w.line == k) for k in (0, 2))
    # The wait on p waits for p's pulse, and q's for p's wait: one barrier
    # after the other.
    assert (
        on_p.start_cycles >= LONG
        and on_q.start_cycles >= on_p.end
        or (on_q.start_cycles >= LONG and on_p.start_cycles >= on_q.end)
    )


@pytest.mark.parametrize("take", ("result", "timeout"))
@pytest.mark.parametrize("inner", POLICIES)
@pytest.mark.parametrize("policy", POLICIES)
def test_a_called_kernel_with_its_own_schedule_feeds_only_its_lines(policy, inner, take):
    @pulse.kernel(schedule=inner)
    def measure(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        return LENGTH["bounded"].submit([readout(r, square(RO, 0.05))])

    def own(q):
        d = get_drive_line(q)
        wait(d, _take(measure(q), take), bounds=BOUNDS)
        drive(d, square(40, 0.1))

    def pair(length):
        @pulse.kernel
        def k(q, p):
            own(q)
            _busy(p, length)
            own(p)

        return _lower(k, Q[0], Q[1], schedule=policy)

    low = pair(LONG)
    assert len(set(line_timelines(low).values())) == 2
    assert _own(low) == _own(pair(8))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("unroll", [0, 10**6])
@pytest.mark.parametrize("take", TAKES)
@pytest.mark.parametrize("policy", POLICIES)
def test_loops_of_their_own_waits_do_not_stall_each_other(policy, take, unroll, shared):
    def pair(length, unroll):
        @pulse.kernel
        def k(q, p, n: int):
            _busy(p, length)
            if shared:
                for _ in range(n):
                    _round(q, take)
                    _round(p, take)
            else:
                for _ in range(n):
                    _round(q, take)
                for _ in range(n):
                    _round(p, take)

        return _lower(k, Q[0], Q[1], 3, schedule=policy, unroll_threshold=unroll)

    low = pair(LONG, unroll)
    assert len(set(line_timelines(low).values())) == 2
    assert _own(low) == _own(pair(8, unroll))
    # Rolled or unrolled, the loop runs the same.
    other = pair(LONG, 10**6 - unroll)
    assert _timed(low) == _timed(other)


@pytest.mark.parametrize("policy", POLICIES)
def test_one_line_keeps_its_program(policy):
    @pulse.kernel
    def k(q):
        _round(q, "result")

    low = _lower(k, Q[0], schedule=policy)
    assert "classical.timeline" not in low.mlir()
