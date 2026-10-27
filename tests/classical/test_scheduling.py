# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Scheduling classical calls under every policy, with property checks."""

from __future__ import annotations

import math
import time

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.compiler import MachineModel
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q
from tests.property_settings import examples

POLICIES = ("asap", "alap", "rcp", "alap_rcp")
CLOCK = 2.0  # the test device's clock, GHz
READOUT, ARM, LATENCY_NS = 300, 40, 1000
BUDGET = math.ceil(LATENCY_NS * CLOCK)


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=LATENCY_NS)
def decode(bits):
    return [bits[0]]


@pulse.classical(
    args=(Bits(1),),
    results=(Bits(1),),
    max_latency_ns=LATENCY_NS,
    submit="lazy",
    name="decode_lazy",
)
def decode_lazy(bits):
    return [bits[0]]


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=100, link=1)
def tune(x):
    return x


@pulse.classical(
    args=(Float(),), results=(Float(),), max_latency_ns=50, latency_mode="blocking", link=2
)
def calibrate(x):
    return x


@pulse.kernel
def feedback(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(READOUT, 0.05), rt)
    if decode.submit([m]).result()[0]:
        drive(d, square(ARM, 0.1), t)


def _events(low, kind):
    return [e for e in low.schedule if e.kind == kind]


def _makespan(low):
    return max(e.start_vtu + e.duration_vtu for e in low.schedule)


@pytest.mark.parametrize("policy", POLICIES)
def test_feedback_makespan_is_readout_submit_latency_and_arm(policy):
    low = pulse.compile(feedback, schedule=policy).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + ARM
    (call,) = low.classical_report()
    assert call.submit_vtu == READOUT and call.due_vtu == READOUT + 1 + BUDGET
    assert _events(low, "drive")[0].start_vtu == call.due_vtu


def test_readout_latency_delays_the_submission_under_rcp():
    machine = MachineModel(readout_latency_vtu=50)
    low = pulse.compile(feedback, schedule="rcp", machine=machine).lower(Q[0])
    (call,) = low.classical_report()
    assert call.submit_vtu == READOUT + 50


@pytest.mark.parametrize("feedback_ns", [0, 100, 250])
def test_feedback_latency_of_the_link_adds_to_the_budget(feedback_ns):
    machine = MachineModel(links={0: Link(feedback_ns=feedback_ns)})
    low = pulse.compile(feedback, machine=machine).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + math.ceil(feedback_ns * CLOCK) + ARM


def _late_consumer(fn):
    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        r, rt = get_readout_line(q)
        drive(e, square(8000, 0.01), u)
        m = readout(r, square(READOUT, 0.05), rt)
        if fn.submit([m]).result()[0]:
            drive(d, square(ARM, 0.1), t)

    return k


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
def test_eager_submission_keeps_its_asap_time_under_alap(policy):
    low = pulse.compile(_late_consumer(decode), schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.submit_vtu == READOUT or call.submit_vtu > READOUT
    readout_end = _events(low, "readout")[0].start_vtu + READOUT
    assert call.submit_vtu == readout_end
    # The conditional drive is pushed to the end, so the answer has slack.
    assert call.slack_vtu == 8000 - ARM - call.due_vtu


def test_lazy_submission_floats_to_just_in_time_under_alap():
    low = pulse.compile(_late_consumer(decode_lazy), schedule="alap").lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.slack_vtu == 0
    assert call.due_vtu == 8000 - ARM
    asap = pulse.compile(_late_consumer(decode_lazy), schedule="asap").lower(Q[0], Q[1])
    assert asap.classical_report()[0].submit_vtu == READOUT


def _rounds(n, fn=decode):
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        for _ in range(n):
            m = readout(r, square(READOUT, 0.05), rt)
            if fn.submit([m]).result()[0]:
                drive(d, square(ARM, 0.1), t)

    return k


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("rolled", [True, False])
def test_rounds_overlap_their_calls(policy, rolled):
    threshold = 0 if rolled else 10_000
    low = pulse.compile(_rounds(8), schedule=policy, unroll_threshold=threshold).lower(Q[0])
    calls = low.classical_report()
    assert len(calls) == 8
    assert [c.submit_vtu for c in calls] == [READOUT * (k + 1) for k in range(8)]
    # Eight requests are in flight at once on the default 64-slot link.
    in_flight = max(
        sum(c.submit_vtu <= t < c.due_vtu for c in calls) for t in (c.submit_vtu for c in calls)
    )
    assert in_flight == min(8, math.ceil((1 + BUDGET) / READOUT))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_link_slots_bound_requests_in_flight(policy, slots):
    machine = MachineModel(links={0: Link(max_in_flight=slots)})
    low = pulse.compile(_rounds(6), schedule=policy, machine=machine, unroll_threshold=0).lower(
        Q[0]
    )
    calls = low.classical_report()
    busy = [(c.submit_vtu, c.due_vtu) for c in calls]
    for start, _ in busy:
        assert sum(s <= start < e for s, e in busy) <= slots
    # Request k waits for the answer of request k - slots.
    for k in range(slots, len(calls)):
        assert calls[k].submit_vtu >= calls[k - slots].due_vtu


def test_submissions_on_one_link_go_out_in_order():
    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05), rt)
        futures = []
        for _ in range(5):
            futures.append(decode.submit([m]))
        for f in futures:
            f.result()

    machine = MachineModel(links={0: Link(fixed_ns=10)})
    calls = pulse.compile(k, machine=machine).lower(Q[0]).classical_report()
    starts = [c.submit_vtu for c in calls]
    assert starts == sorted(starts)
    for a, b in zip(calls, calls[1:]):
        assert b.submit_vtu >= a.submit_vtu + a.submit_duration_vtu


def test_links_are_independent():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        a = tune.submit(0.1)
        b = calibrate.submit(0.2)
        drive(d, square(ARM, a.result() + b.result()), t)

    low = pulse.compile(k, schedule="asap").lower(Q[0])
    calls = low.classical_report()
    assert [c.submit_vtu for c in calls] == [0, 0]
    assert {c.link for c in calls} == {1, 2}


@pytest.mark.parametrize("policy", POLICIES)
def test_nested_branches_on_answers(policy):
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05), rt)
        if decode.submit([m]).result()[0]:
            m2 = readout(r, square(READOUT, 0.05), rt)
            if decode.submit([m2]).result()[0]:
                drive(d, square(ARM, 0.1), t)

    low = pulse.compile(k, schedule=policy).lower(Q[0])
    first, second = low.classical_report()
    assert second.submit_vtu >= first.due_vtu + READOUT
    assert _makespan(low) == 2 * (READOUT + 1 + BUDGET) + ARM


@pytest.mark.parametrize("policy", ["asap", "alap"])
@pytest.mark.parametrize("flatten", [False, True])
def test_a_block_returns_its_future(policy, flatten):
    @pulse.kernel(schedule="alap")
    def measure_and_submit(r, rt):
        m = readout(r, square(READOUT, 0.05), rt)
        return decode.submit([m])

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        answer = measure_and_submit(r, rt)
        drive(d, square(ARM, 0.2), t)
        if answer.result()[0]:
            drive(d, square(ARM, 0.1), t)

    low = pulse.compile(k, schedule=policy, flatten_schedules=flatten).lower(Q[0])
    (call,) = low.classical_report()
    assert call.due_vtu == READOUT + 1 + BUDGET
    assert _makespan(low) == call.due_vtu + ARM


def test_timeout_budgets_the_timeout():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05), rt)
        ok, fix = decode.submit([m]).result(timeout_ns=400)
        if ok:
            drive(d, square(ARM, 0.1), t)

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.due_vtu == READOUT + 1 + math.ceil(400 * CLOCK)


def test_blocking_waits_start_epochs():
    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(ARM, 0.1), u)
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x), t)
        y = calibrate.submit(x).result()
        drive(d, square(ARM, y), t)

    for policy in ("asap", "alap"):
        low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
        calls = low.classical_report()
        assert [c.epoch for c in calls] == [0, 1]
        # Nothing before the first answer ends after it.
        first_answer = calls[0].taken_vtu
        assert _events(low, "drive")[0].start_vtu + ARM <= first_answer


def test_a_missed_deadline_is_rejected():
    tight = pulse.classical(
        lambda b: b,
        name="tight",
        args=(Bits(1),),
        results=(Bits(1),),
        max_latency_ns=1000,
        deadline_ns=800,
    )

    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        tight.submit([readout(r, square(READOUT, 0.05), rt)]).result()

    with pytest.raises(RuntimeError, match="cannot meet its deadline.*slack -401"):
        pulse.compile(k).lower(Q[0])


def test_deadline_slack_is_reported():
    loose = pulse.classical(
        lambda b: b,
        name="loose",
        args=(Bits(1),),
        results=(Bits(1),),
        max_latency_ns=1000,
        deadline_ns=1500,
    )

    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        loose.submit([readout(r, square(READOUT, 0.05), rt)]).result()

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.deadline_slack_vtu == math.ceil(1500 * CLOCK) - (1 + BUDGET)


def test_a_request_too_big_for_its_slot_is_rejected():
    machine = MachineModel(links={0: Link(slot_bytes=24 + 0 + 1 - 1 + 1)})
    big = pulse.classical(lambda b: b, name="big", args=(Bits(16),), results=(), max_latency_ns=1)

    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05), rt)
        big.submit([m] * 16)

    with pytest.raises(RuntimeError, match="a request of 'big' takes 26 bytes"):
        pulse.compile(k, machine=machine).lower(Q[0])


def _many(n, links):
    functions = [
        pulse.classical(
            lambda x: x,
            name=f"f{k}",
            args=(Float(),),
            results=(Float(),),
            max_latency_ns=20,
            link=k,
        )
        for k in range(links)
    ]

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        for i in range(n):
            a = functions[i % links].submit(0.1).result()
            drive(d, square(8, a), t)

    return k


@pytest.mark.parametrize("links", [1, 4])
def test_very_many_calls_schedule_in_linear_time(links):
    timings = {}
    for n in (500, 2000):
        kernel = _many(n, links)
        t0 = time.perf_counter()
        low = pulse.compile(kernel, schedule="asap").lower(Q[0])
        timings[n] = time.perf_counter() - t0
        calls = low.classical_report()
        assert len(calls) == n
        # Each call waits for the previous drive; answers are due 41 later.
        assert calls[-1].due_vtu == calls[-1].submit_vtu + 1 + 40
    # 4x the calls in well under 16x the time (quadratic would be 16x).
    assert timings[2000] < 8 * timings[500]


# ── Properties ───────────────────────────────────────────────────────────────

_STEP = st.tuples(
    st.sampled_from(("readout", "submit", "wait", "drive", "drive_from_answer")),
    st.integers(0, 2),
    st.integers(0, 1),
)


def _program(steps):
    @pulse.kernel
    def k(q0, q1, q2):
        qs = (q0, q1, q2)
        ms = []
        futures = []
        answers = []
        for kind, q, link in steps:
            d, t = get_drive_line(qs[q])
            r, rt = get_readout_line(qs[q])
            if kind == "readout":
                ms.append(readout(r, square(60, 0.05), rt))
            elif kind == "submit" and ms:
                fn = decode if link == 0 else decode_lazy
                futures.append(fn.submit([ms[-1]]))
            elif kind == "wait" and futures:
                answers.append(futures.pop(0).result())
            elif kind == "drive":
                drive(d, square(20, 0.1), t)
            elif kind == "drive_from_answer" and answers:
                if answers[-1][0]:
                    drive(d, square(20, 0.1), t)
        for f in futures:
            f.result()

    return k


@settings(max_examples=examples(40), deadline=None)
@given(steps=st.lists(_STEP, min_size=1, max_size=25), policy=st.sampled_from(POLICIES))
def test_scheduling_invariants(steps, policy):
    low = pulse.compile(_program(steps), schedule=policy).lower(Q[0], Q[1], Q[2])
    calls = low.classical_report()
    for call in calls:
        # S7: every answer is taken no earlier than it is due.
        assert call.taken_vtu is None or call.taken_vtu >= call.due_vtu
        assert call.first_use_vtu is None or call.first_use_vtu >= call.due_vtu
        assert call.due_vtu == call.submit_vtu + call.submit_duration_vtu + BUDGET
    # S9: one link sends one request at a time, in program order.
    starts = [c.submit_vtu for c in calls]
    assert starts == sorted(starts)
    # No two pulses overlap on a line, and the makespan covers everything.
    by_line: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for e in low.schedule:
        assert e.start_vtu >= 0
        if e.link is None:
            by_line.setdefault(("line", e.line), []).append((e.start_vtu, e.duration_vtu))
    for spans in by_line.values():
        spans.sort()
        for (s0, d0), (s1, _) in zip(spans, spans[1:]):
            assert s0 + d0 <= s1
    if calls:
        assert _makespan(low) >= max(c.taken_vtu or 0 for c in calls)


# ── Regressions ──────────────────────────────────────────────────────────────


def test_a_helper_passed_a_future_takes_its_answer():
    def answer(future):
        return future.result()[0]

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        if answer(decode.submit([readout(r, square(READOUT, 0.05), rt)])):
            drive(d, square(ARM, 0.1), t)

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.first_use_vtu == call.due_vtu


@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_blocking_wait_waits_for_every_line(policy):
    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(8000, 0.01), u)
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x), t)

    low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.taken_vtu == 8000
    assert _events(low, "drive")[1].start_vtu == 8000


@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_block_waits_for_a_slot_the_caller_holds(callee, policy):
    @pulse.kernel(schedule=callee)
    def twice(r, rt):
        m = readout(r, square(10, 0.05), rt)
        a = decode.submit([m])
        b = decode.submit([m])
        a.result()
        b.result()

    @pulse.kernel
    def k(q):
        r, rt = get_readout_line(q)
        f = decode.submit([readout(r, square(10, 0.05), rt)])
        twice(r, rt)
        f.result()

    machine = MachineModel(links={0: Link(max_in_flight=2)})
    calls = pulse.compile(k, schedule=policy, machine=machine).lower(Q[0]).classical_report()
    first, *block = calls
    # The block's second request needs the slot the first call frees.
    assert block[1].submit_vtu == first.due_vtu
    assert block[0].submit_vtu == block[1].submit_vtu - 1


@pytest.mark.parametrize("passes", [None, ()])
def test_a_blocking_wait_in_a_block_is_rejected_without_the_verifier(passes):
    @pulse.kernel(schedule="asap")
    def inner(d, t):
        drive(d, square(ARM, calibrate.submit(0.5).result()), t)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        inner(d, t)

    with pytest.raises(RuntimeError, match="return the future from the kernel and take the answer"):
        pulse.compile(k, passes=passes).lower(Q[0])
