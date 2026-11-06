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
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q
from tests._work import python_work
from tests.property_settings import examples

POLICIES = ("asap", "alap", "rcp", "alap_rcp")
CLOCK = 2.0  # the test device's clock, GHz
READOUT, ARM, LATENCY_NS = 300, 40, 1000
BUDGET = math.ceil(LATENCY_NS * CLOCK)


@pulse.classical(max_latency_ns=LATENCY_NS)
def decode(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=LATENCY_NS, submit="lazy", name="decode_lazy")
def decode_lazy(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=100, link=1)
def tune(x: float) -> float:
    return x


@pulse.classical(max_latency_ns=50, latency_mode="blocking", link=2)
def calibrate(x: float) -> float:
    return x


@pulse.kernel
def feedback(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(READOUT, 0.05))
    if decode.submit([m]).result()[0]:
        drive(d, square(ARM, 0.1))


def _events(low, kind):
    return [e for e in low.schedule if e.kind == kind]


def _makespan(low):
    return max(e.start_cycles + e.duration_cycles for e in low.schedule)


@pytest.mark.parametrize("policy", POLICIES)
def test_feedback_makespan_is_readout_submit_latency_and_arm(policy):
    low = pulse.compile(feedback, schedule=policy).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + ARM
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT and call.due_cycles == READOUT + 1 + BUDGET
    assert _events(low, "drive")[0].start_cycles == call.due_cycles


def test_readout_latency_delays_the_submission_under_rcp():
    machine = ControlSystem(controllers=[pulse.Controller(feedback_latency_cycles=50)])
    low = pulse.compile(feedback, schedule="rcp", control_system=machine).lower(Q[0])
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT + 50


@pytest.mark.parametrize("feedback_ns", [0, 100, 250])
def test_feedback_latency_of_the_link_adds_to_the_budget(feedback_ns):
    machine = ControlSystem(links={"host": Link(feedback_ns=feedback_ns)})
    low = pulse.compile(feedback, control_system=machine).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + math.ceil(feedback_ns * CLOCK) + ARM


@pulse.kernel
def answered_before_a_loop(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    n = decode.submit([readout(r, square(100, 0.05))]).result()[0]
    for _ in range(4):
        if readout(r, square(100, 0.05)):
            drive(d, square(40, 0.1))
        if n:
            drive(d, square(30, 0.1))
        wait(d, 200)


@pytest.mark.parametrize("schedule", POLICIES)
def test_an_answer_taken_before_a_loop_is_taken_before_its_first_iteration_tests_it(schedule):
    lowered = pulse.compile(answered_before_a_loop, schedule=schedule).lower(Q[0])
    events = lowered.export_controller()["events"]
    # Every iteration branches on the answer, the first at its due time.
    [answer] = [e for e in events if e["kind"] == "answer"]
    tested = [e for e in events if any(c["answers"] for c in e["conditions"])]
    assert len(tested) == 4
    assert min(e["start_cycles"] for e in tested) >= answer["end_cycles"]


def _late_consumer(fn):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        r = get_readout_line(q)
        drive(e, square(8000, 0.01))
        m = readout(r, square(READOUT, 0.05))
        if fn.submit([m]).result()[0]:
            drive(d, square(ARM, 0.1))

    return k


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
def test_eager_submission_keeps_its_asap_time_under_alap(policy):
    low = pulse.compile(_late_consumer(decode), schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT or call.submit_cycles > READOUT
    readout_end = _events(low, "readout")[0].start_cycles + READOUT
    assert call.submit_cycles == readout_end
    # The conditional drive is pushed to the end, so the answer has slack.
    assert call.slack_cycles == 8000 - ARM - call.due_cycles


def test_lazy_submission_floats_to_just_in_time_under_alap():
    low = pulse.compile(_late_consumer(decode_lazy), schedule="alap").lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.slack_cycles == 0
    assert call.due_cycles == 8000 - ARM
    asap = pulse.compile(_late_consumer(decode_lazy), schedule="asap").lower(Q[0], Q[1])
    assert asap.classical_report()[0].submit_cycles == READOUT


def _rounds(n, fn=decode):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(n):
            m = readout(r, square(READOUT, 0.05))
            if fn.submit([m]).result()[0]:
                drive(d, square(ARM, 0.1))

    return k


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("rolled", [True, False])
def test_rounds_overlap_their_calls(policy, rolled):
    threshold = 0 if rolled else 10_000
    low = pulse.compile(_rounds(8), schedule=policy, unroll_threshold=threshold).lower(Q[0])
    calls = low.classical_report()
    assert len(calls) == 8
    assert [c.submit_cycles for c in calls] == [READOUT * (k + 1) for k in range(8)]
    # Eight requests are in flight at once on the default 64-slot link.
    in_flight = max(
        sum(c.submit_cycles <= t < c.due_cycles for c in calls)
        for t in (c.submit_cycles for c in calls)
    )
    assert in_flight == min(8, math.ceil((1 + BUDGET) / READOUT))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_link_slots_bound_requests_in_flight(policy, slots):
    machine = ControlSystem(links={"host": Link(max_in_flight=slots)})
    low = pulse.compile(
        _rounds(6), schedule=policy, control_system=machine, unroll_threshold=0
    ).lower(Q[0])
    calls = low.classical_report()
    busy = [(c.submit_cycles, c.due_cycles) for c in calls]
    for start, _ in busy:
        assert sum(s <= start < e for s, e in busy) <= slots
    # Request k waits for the answer of request k - slots.
    for k in range(slots, len(calls)):
        assert calls[k].submit_cycles >= calls[k - slots].due_cycles


def test_submissions_on_one_link_go_out_in_order():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        futures = []
        for _ in range(5):
            futures.append(decode.submit([m]))
        for f in futures:
            f.result()

    machine = ControlSystem(links={"host": Link(fixed_ns=10)})
    calls = pulse.compile(k, control_system=machine).lower(Q[0]).classical_report()
    starts = [c.submit_cycles for c in calls]
    assert starts == sorted(starts)
    for a, b in zip(calls, calls[1:], strict=False):
        assert b.submit_cycles >= a.submit_cycles + a.submit_duration_cycles


def test_links_are_independent():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        a = tune.submit(0.1)
        b = calibrate.submit(0.2)
        drive(d, square(ARM, a.result() + b.result()))

    low = pulse.compile(k, schedule="asap").lower(Q[0])
    calls = low.classical_report()
    assert [c.submit_cycles for c in calls] == [0, 0]
    assert {c.link for c in calls} == {1, 2}


@pytest.mark.parametrize("policy", POLICIES)
def test_nested_branches_on_answers(policy):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        if decode.submit([m]).result()[0]:
            m2 = readout(r, square(READOUT, 0.05))
            if decode.submit([m2]).result()[0]:
                drive(d, square(ARM, 0.1))

    low = pulse.compile(k, schedule=policy).lower(Q[0])
    first, second = low.classical_report()
    assert second.submit_cycles >= first.due_cycles + READOUT
    assert _makespan(low) == 2 * (READOUT + 1 + BUDGET) + ARM


@pytest.mark.parametrize("policy", ["asap", "alap"])
@pytest.mark.parametrize("flatten", [False, True])
def test_a_block_returns_its_future(policy, flatten):
    @pulse.kernel(schedule="alap")
    def measure_and_submit(r):
        m = readout(r, square(READOUT, 0.05))
        return decode.submit([m])

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        answer = measure_and_submit(r)
        drive(d, square(ARM, 0.2))
        if answer.result()[0]:
            drive(d, square(ARM, 0.1))

    low = pulse.compile(k, schedule=policy, flatten_schedules=flatten).lower(Q[0])
    (call,) = low.classical_report()
    assert call.due_cycles == READOUT + 1 + BUDGET
    assert _makespan(low) == call.due_cycles + ARM


def test_timeout_budgets_the_timeout():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        ok, fix = decode.submit([m]).result(timeout=400 * pulse.ns)
        if ok:
            drive(d, square(ARM, 0.1))

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.due_cycles == READOUT + 1 + math.ceil(400 * CLOCK)


def test_blocking_waits_start_epochs():
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(ARM, 0.1))
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x))
        y = calibrate.submit(x).result()
        drive(d, square(ARM, y))

    for policy in ("asap", "alap"):
        low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
        calls = low.classical_report()
        assert [c.epoch for c in calls] == [0, 1]
        # Nothing before the first answer ends after it.
        first_answer = calls[0].taken_cycles
        assert _events(low, "drive")[0].start_cycles + ARM <= first_answer


def test_a_missed_deadline_is_rejected():
    @pulse.classical(max_latency_ns=1000, deadline_ns=800)
    def tight(b: pulse.Bits) -> pulse.Bits:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        tight.submit([readout(r, square(READOUT, 0.05))]).result()

    with pytest.raises(
        RuntimeError, match=r"cannot meet its deadline.*slack -401\); raise deadline_ns"
    ):
        pulse.compile(k).lower(Q[0])


def test_deadline_slack_is_reported():
    @pulse.classical(max_latency_ns=1000, deadline_ns=1500)
    def loose(b: pulse.Bits) -> pulse.Bits:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        loose.submit([readout(r, square(READOUT, 0.05))]).result()

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.deadline_slack_cycles == math.ceil(1500 * CLOCK) - (1 + BUDGET)


def test_a_request_too_big_for_its_slot_is_rejected():
    machine = ControlSystem(links={"host": Link(slot_bytes=24 + 0 + 1 - 1 + 1)})

    @pulse.classical(max_latency_ns=1)
    def big(b: pulse.Bits) -> None:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        big.submit([m] * 16)

    with pytest.raises(RuntimeError, match="a request of 'big__16' takes 26 bytes"):
        pulse.compile(k, control_system=machine).lower(Q[0])


def _many(n, links):
    def same(x: float) -> float:
        return x

    functions = [
        pulse.classical(same, name=f"f{k}", max_latency_ns=20, link=k) for k in range(links)
    ]

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        for i in range(n):
            a = functions[i % links].submit(0.1).result()
            drive(d, square(8, a))

    return k


@pytest.mark.slow
@pytest.mark.parametrize("links", [1, 4])
def test_very_many_calls_schedule_in_linear_time(links):
    # The Python work is counted, the same at any load. The time, which also
    # takes in native code, is the best of five in CPU time: one wall-clock
    # sample per size failed on a loaded machine.
    timings, work = {}, {}
    for n in (500, 2000):
        kernel = _many(n, links)

        def run(kernel=kernel):
            return pulse.compile(kernel, schedule="asap").lower(Q[0])

        work[n] = python_work(run)
        timings[n] = float("inf")
        for _ in range(5):
            t0 = time.process_time()
            low = run()
            timings[n] = min(timings[n], time.process_time() - t0)
        calls = low.classical_report()
        assert len(calls) == n
        # Each call waits for the previous drive; answers are due 41 later.
        assert calls[-1].due_cycles == calls[-1].submit_cycles + 1 + 40
    # 4x the calls in well under 16x the time (quadratic would be 16x).
    assert all(work[2000][count] <= 4 * work[500][count] for count in work[500]), work
    assert timings[2000] < 8 * timings[500], timings


@pulse.classical(max_latency_ns=200)
def first_guess() -> tuple[float, float]:
    return 0.0, 0.0


@pulse.classical(max_latency_ns=200)
def next_guess(bits: pulse.Bits) -> tuple[float, float]:
    return 0.0, 0.0


def _shots(n):
    @pulse.kernel
    def search(q):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        x, done = first_guess.submit().result()
        for _ in range(2):
            bits = []
            for _ in range(n):
                if done < 0.5:
                    shift_frequency(d.tone, x)
                    drive(d, square(64, 0.01))
                    shift_frequency(d.tone, -x)
                sync(d, ro)
                bits.append(readout(ro, square(40, 0.05)))
                sync(d, ro)
            x, done = next_guess.submit(bits).result()

    return search


@pytest.mark.slow
def test_shots_that_branch_on_one_answer_schedule_in_linear_time():
    # Every shot of a round branches on the answer the round before sent its
    # measurements to; the time, native, is the best of five.
    timings = {}
    for n in (100, 1600):
        program = pulse.compile(_shots(n))
        timings[n] = min(program.lower(Q[0]).metrics.schedule_ms for _ in range(5))
    # 16x the shots in well under 256x the time (quadratic would be 256x).
    assert timings[1600] < 48 * timings[100], timings


@pulse.classical(max_latency_ns=60, link="decoder", submit="lazy")
def record_syndrome(bits: pulse.Bits) -> pulse.Bits:
    return bits


@pulse.classical(max_latency_ns=120, link="decoder")
def decode_syndromes() -> pulse.Bits:
    return [0]


@pulse.kernel
def decoded(q, r):
    # The decoder keeps every round's syndrome: decode_syndromes() takes
    # none, and its answer depends on those of the calls before it on the
    # link alone.
    d = get_drive_line(q)
    ro = get_readout_line(r)
    for k in range(6):
        record_syndrome.submit([readout(ro, square(80, 0.05))])
        if k % 3 == 2 and decode_syndromes.submit().result()[0]:
            drive(d, square(16, 0.3))


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_correction_from_decoder_state_waits_for_every_syndrome_it_holds(schedule):
    machine = ControlSystem(links={"decoder": Link(fixed_ns=10)})
    lowered = pulse.compile(decoded, schedule=schedule, control_system=machine).lower(Q[0], Q[1])
    events = lowered.export_controller()["events"]
    corrections = [e for e in events if e["kind"] == "drive" and e["conditions"]]
    assert len(corrections) == 2
    for drive in corrections:
        [condition] = drive["conditions"]
        [answer] = condition["answers"]
        submit = max(e["index"] for e in events if e["kind"] == "submit" and e["index"] < answer)
        syndromes = [e for e in events if e["kind"] == "readout" and e["index"] < submit]
        assert syndromes
        assert drive["start_cycles"] >= max(e["end_cycles"] for e in syndromes)


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
            d = get_drive_line(qs[q])
            r = get_readout_line(qs[q])
            if kind == "readout":
                ms.append(readout(r, square(60, 0.05)))
            elif kind == "submit" and ms:
                fn = decode if link == 0 else decode_lazy
                futures.append(fn.submit([ms[-1]]))
            elif kind == "wait" and futures:
                answers.append(futures.pop(0).result())
            elif kind == "drive":
                drive(d, square(20, 0.1))
            elif kind == "drive_from_answer" and answers:
                if answers[-1][0]:
                    drive(d, square(20, 0.1))
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
        assert call.taken_cycles is None or call.taken_cycles >= call.due_cycles
        assert call.first_use_cycles is None or call.first_use_cycles >= call.due_cycles
        assert call.due_cycles == call.submit_cycles + call.submit_duration_cycles + BUDGET
    # S9: one link sends one request at a time, in program order.
    starts = [c.submit_cycles for c in calls]
    assert starts == sorted(starts)
    # No two pulses overlap on a line, and the makespan covers everything.
    by_line: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for e in low.schedule:
        assert e.start_cycles >= 0
        if e.link is None:
            by_line.setdefault(("line", e.line), []).append((e.start_cycles, e.duration_cycles))
    for spans in by_line.values():
        spans.sort()
        for (s0, d0), (s1, _) in zip(spans, spans[1:], strict=False):
            assert s0 + d0 <= s1
    if calls:
        assert _makespan(low) >= max(c.taken_cycles or 0 for c in calls)


# ── Regressions ──────────────────────────────────────────────────────────────


def test_a_helper_passed_a_future_takes_its_answer():
    def answer(future):
        return future.result()[0]

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        if answer(decode.submit([readout(r, square(READOUT, 0.05))])):
            drive(d, square(ARM, 0.1))

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.first_use_cycles == call.due_cycles


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_blocking_wait_waits_for_the_lines_it_feeds(policy, tied):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(8000, 0.01))
        if tied:
            sync(d, e)
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x))

    low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    fed = [e for e in _events(low, "drive") if e.line == 0]
    if tied:
        assert call.taken_cycles == 8000
        assert fed[0].start_cycles == 8000
    else:
        # Only the line that uses the answer waits for it.
        assert call.taken_cycles == call.due_cycles < 8000
        assert fed[0].start_cycles == call.due_cycles


@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_block_waits_for_a_slot_the_caller_holds(callee, policy):
    @pulse.kernel(schedule=callee)
    def twice(r):
        m = readout(r, square(10, 0.05))
        a = decode.submit([m])
        b = decode.submit([m])
        a.result()
        b.result()

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        f = decode.submit([readout(r, square(10, 0.05))])
        twice(r)
        f.result()

    machine = ControlSystem(links={"host": Link(max_in_flight=2)})
    calls = pulse.compile(k, schedule=policy, control_system=machine).lower(Q[0]).classical_report()
    first, *block = calls
    # The block's second request needs the slot the first call frees.
    assert block[1].submit_cycles == first.due_cycles
    assert block[0].submit_cycles == block[1].submit_cycles - 1


@pytest.mark.parametrize("passes", [None, ()])
def test_a_blocking_wait_in_a_block_is_rejected_without_the_verifier(passes):
    @pulse.kernel(schedule="asap")
    def inner(d):
        drive(d, square(ARM, calibrate.submit(0.5).result()))

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        inner(d)

    with pytest.raises(RuntimeError, match="return the future from the kernel and take the answer"):
        pulse.compile(k, passes=passes).lower(Q[0])


@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_future_taken_with_a_timeout_in_a_block_budgets_the_timeout(callee, policy):
    @pulse.kernel(schedule=callee)
    def take(future, d, t):
        ok, fix = future.result(timeout=400 * pulse.ns)
        if fix[0]:
            drive(d, square(ARM, 0.1), tone=t)

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        take(decode.submit([readout(r, square(READOUT, 0.05))]), d, d.tone)

    (call,) = pulse.compile(k, schedule=policy).lower(Q[0]).classical_report()
    assert call.due_cycles == READOUT + 1 + math.ceil(400 * CLOCK)
    assert call.taken_cycles >= call.due_cycles and call.first_use_cycles >= call.due_cycles
