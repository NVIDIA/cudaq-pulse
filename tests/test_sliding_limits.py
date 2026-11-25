# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What follows a wait whose length is computed at run time starts later in
a run than the schedule places it, and lines the wait does not hold back
keep their times. The limits on what plays at once (``max_concurrent_drives``,
``max_concurrent_readouts``, a shared line's ``max_concurrent_tones``) count
pulses as they can slide, so two the schedule places apart are refused when
some run plays them together."""

from __future__ import annotations

import itertools
import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, _slides
from cudaq_pulse._errors import UnsupportedError
from cudaq_pulse.controller import ControllerLimitError, ReadoutLine
from cudaq_pulse.quantum_systems import PhysicalQudit

_SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")


@pytest.fixture(autouse=True)
def _the_check_alone(monkeypatch):
    """These tests are of the check: asap and alap keep the schedule they
    place, which compile would otherwise place again as rcp and alap_rcp do
    past a declared limit (test_slides_placed_as_rcp_does below)."""
    from cudaq_pulse import compiler

    monkeypatch.setattr(compiler, "_PLACED_BY_LIMITS", {})


@pulse.classical(max_latency_ns=50)
def next_tau(x: float) -> float:
    return 2.0 * x


@pulse.classical(max_latency_ns=50, latency_mode="blocking")
def slow(x: float) -> float:
    return 0.1


def _system(limit=None, shared=False):
    qudits = {i: PhysicalQudit(i, 5.0e9 + 1.0e7 * i, 50.0, 30.0) for i in range(2)}
    readouts = [ReadoutLine([0, 1], max_concurrent_tones=1)] if shared else [0, 1]
    control = ControlSystem(
        controllers=[Controller(drive_lines=[0, 1], readout_lines=readouts)],
        **({"max_concurrent_drives": limit} if limit else {}),
    )
    return pulse.quantum_system(name="sliding", qudits=qudits, control_system=control)


def _kernel(body):
    source = (
        "def k(a, b, tau0: float):\n"
        "    da = get_drive_line(a)\n"
        "    db = get_drive_line(b)\n"
        "    ra = get_readout_line(a)\n"
        "    rb = get_readout_line(b)\n" + "".join(f"    {line}\n" for line in body)
    )
    scope = {"next_tau": next_tau, "slow": slow}
    exec(source, scope)  # noqa: S102
    return pulse.kernel(scope["k"])


def _compiled(body, schedule="asap", **system):
    qs = _system(**system)
    lowered = pulse.compile(_kernel(body), quantum_system=qs, schedule=schedule)
    return lowered.lower(qs.qudits[0], qs.qudits[1], 10.0)


_WAIT = "wait(da, next_tau.submit(tau0).result(), bounds=(16, 4000))"
# A drive after a wait of 16 to 4,000 cycles on one line, and one at cycle
# 600 on a line the wait does not hold back.
_TWO_DRIVES = [_WAIT, "drive(da, square(40, 0.1))", "wait(db, 600)", "drive(db, square(40, 0.1))"]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_drive_that_can_slide_onto_another_is_refused(schedule):
    with pytest.raises(ControllerLimitError) as refused:
        _compiled(_TWO_DRIVES, schedule, limit=1)
    message = str(refused.value)
    assert "2 drives can play at once from cycle 600" in message
    assert "the wait on d0 that ends at cycle 117 (up to 3984 cycles later)" in message
    assert "counted as independent" in message
    assert "at most 1 (max_concurrent_drives)" in message
    assert "sync the lines after the wait" in message


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_the_schedules_that_keep_to_the_limit_hold_both_lines_at_the_wait(schedule):
    lowered = _compiled(_TWO_DRIVES, schedule, limit=1)
    assert list(pulse.controller.check(lowered)) == []


@pytest.mark.parametrize("schedule", _SCHEDULES)
def test_two_drives_that_can_meet_fit_a_limit_of_two(schedule):
    assert list(pulse.controller.check(_compiled(_TWO_DRIVES, schedule, limit=2))) == []


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize(
    "body",
    [
        # The lines wait together, and play one after the other.
        [_WAIT, "sync(da, db)", "drive(da, square(40, 0.1))", "sync(da, db)", _TWO_DRIVES[3]],
        # The other line's drive is over before the wait can end.
        ["drive(db, square(40, 0.1))", _WAIT, "drive(da, square(40, 0.1))"],
        # One line.
        [_WAIT, "drive(da, square(40, 0.1))", _WAIT, "drive(da, square(40, 0.1))"],
        # A wait whose bounds keep the drives apart.
        [
            "wait(da, next_tau.submit(tau0).result(), bounds=(16, 300))",
            "drive(da, square(40, 0.1))",
            "wait(db, 600)",
            "drive(db, square(40, 0.1))",
        ],
    ],
    ids=["synced", "before the wait", "one line", "bounds apart"],
)
def test_pulses_that_cannot_meet_are_not_refused(body, schedule):
    assert list(pulse.controller.check(_compiled(body, schedule, limit=1))) == []


def test_a_wait_without_an_upper_bound_lets_what_follows_meet_anything_later():
    body = ["wait(da, next_tau.submit(tau0).result())", *_TWO_DRIVES[1:]]
    with pytest.raises(ControllerLimitError, match="any number of cycles later"):
        _compiled(body, limit=1)
    assert list(pulse.controller.check(_compiled(body, "rcp", limit=1))) == []


def test_waits_on_two_lines_slide_each_on_its_own():
    body = [
        _WAIT,
        "drive(da, square(40, 0.1))",
        "wait(da, next_tau.submit(tau0).result(), bounds=(16, 300))",
        "drive(da, square(40, 0.1))",
        "wait(db, next_tau.submit(tau0).result(), bounds=(16, 500))",
        "drive(db, square(40, 0.1))",
    ]
    with pytest.raises(ControllerLimitError) as refused:
        _compiled(body, limit=1)
    message = str(refused.value)
    assert "2 drives can play at once" in message
    assert message.count("the wait on d") == 3
    assert list(pulse.controller.check(_compiled(body, limit=2))) == []


@pytest.mark.parametrize("schedule", _SCHEDULES)
def test_tones_of_a_shared_line_that_can_meet_are_refused(schedule):
    body = [
        "wait(ra, next_tau.submit(tau0).result(), bounds=(16, 4000))",
        "readout(ra, square(200, 0.05))",
        "wait(rb, 600)",
        "readout(rb, square(200, 0.05))",
    ]
    with pytest.raises(ControllerLimitError) as refused:
        _compiled(body, schedule, shared=True)
    message = str(refused.value)
    assert "can play 2 tones at once from cycle 600" in message
    assert "the wait on r0 that ends at cycle 117 (up to 3984 cycles later)" in message
    assert "at most 1 (max_concurrent_tones)" in message
    synced = [body[0], "sync(ra, rb)", body[1], "sync(ra, rb)", body[3]]
    assert list(pulse.controller.check(_compiled(synced, schedule, shared=True))) == []


def test_a_blocking_wait_lets_what_follows_play_at_any_later_time():
    body = [
        "x = slow.submit(tau0).result()",
        "drive(da, square(40, x))",
        "wait(db, 600)",
        "drive(db, square(40, 0.1))",
    ]
    with pytest.raises(ControllerLimitError) as refused:
        _compiled(body, limit=1)
    message = str(refused.value)
    assert "2 drives can play at once from cycle 600" in message
    assert "the blocking wait on link" in message
    assert "any number of cycles later" in message
    assert list(pulse.controller.check(_compiled(body, limit=2))) == []


def test_the_arms_of_a_branch_after_a_wait_count_as_one():
    body = [
        "m = readout(ra, square(100, 0.05))",
        _WAIT,
        "if m:",
        "    drive(da, square(40, 0.1))",
        "else:",
        "    drive(da, square(60, 0.1))",
        "wait(db, 600)",
        "drive(db, square(40, 0.1))",
    ]
    with pytest.raises(ControllerLimitError, match="2 drives can play at once"):
        _compiled(body, limit=1)
    # Either arm and the other line's drive: two, not three.
    assert list(pulse.controller.check(_compiled(body, limit=2))) == []


def _rounds(trips, tail):
    return [
        f"for _ in range({trips}):",
        "    drive(da, square(40, 0.1))",
        f"    {_WAIT}",
        *tail,
    ]


@pytest.mark.parametrize("trips", [50, 2000])
def test_a_loop_with_a_wait_each_trip_slides_as_far_as_its_trips(trips):
    # Walked trip by trip (50), and kept rolled by the check's walk (2,000).
    other = ["wait(db, 9000)", "drive(db, square(40, 0.1))"]
    with pytest.raises(ControllerLimitError) as refused:
        _compiled(_rounds(trips, other), limit=1)
    message = str(refused.value)
    assert "2 drives can play at once from cycle 9000" in message
    assert f"{trips - 4} more such waits" in message
    assert list(pulse.controller.check(_compiled(_rounds(trips, other), limit=2))) == []
    # Alone on its line, or with the other line held at every wait.
    assert list(pulse.controller.check(_compiled(_rounds(trips, []), limit=1))) == []
    tied = [
        f"for _ in range({trips}):",
        "    drive(da, square(40, 0.1))",
        f"    {_WAIT}",
        "    sync(da, db)",
        "    drive(db, square(40, 0.1))",
        "    sync(da, db)",
    ]
    assert list(pulse.controller.check(_compiled(tied, limit=1))) == []


def test_a_loop_past_the_copies_a_check_counts_is_not_passed():
    other = ["wait(db, 9000)", "drive(db, square(40, 0.1))"]
    with pytest.raises(UnsupportedError, match="would count 2999998 copies of rolled loops"):
        _compiled(_rounds(3_000_000, other), limit=1)
    # Without a limit nothing is counted, and a loop alone on its line, or
    # one whose lines wait together, is counted from its summary as before.
    _compiled(_rounds(3_000_000, other))
    assert list(pulse.controller.check(_compiled(_rounds(3_000_000, []), limit=1))) == []
    tied = [
        "for _ in range(3000000):",
        "    drive(da, square(40, 0.1))",
        f"    {_WAIT}",
        "    sync(da, db)",
        "    drive(db, square(40, 0.1))",
        "    sync(da, db)",
    ]
    assert list(pulse.controller.check(_compiled(tied, limit=1))) == []


# ── The count ────────────────────────────────────────────────────────────────


def _value(steps, at):
    count = 0
    for cycle, value in steps:
        if cycle <= at:
            count = value
    return count


def _every_slide(pulses, after, ranges, span):
    """The most that play at once at each cycle over every slide."""
    best = [0] * span
    for slide in itertools.product(*(range(most + 1) for most in ranges)):
        shift = [0] * len(after)
        for k in range(1, len(after)):
            shift[k] = shift[after[k]] + slide[k]
        count = [0] * span
        for segment, played in pulses.items():
            for start, end, _ in played:
                for at in range(start + shift[segment], min(span, end + shift[segment])):
                    count[at] += 1
        best = [max(a, b) for a, b in zip(best, count, strict=True)]
    return best


@pytest.mark.parametrize("chains", [False, True])
def test_the_count_is_the_most_over_every_slide(chains):
    rng = random.Random(5)
    exact = 0
    for _ in range(400):
        n = rng.randint(2, 6)
        after = [None] + [
            k - 1 if chains and rng.random() < 0.8 else rng.randrange(k) for k in range(1, n)
        ]
        ranges = [0] + [rng.choice([0, 0, 1, 2, 3]) for _ in range(1, n)]
        pulses = {}
        for _ in range(rng.randint(1, 7)):
            start = rng.randint(0, 12)
            pulses.setdefault(rng.randrange(n), []).append((start, start + rng.randint(1, 4), {}))
        for played in pulses.values():
            played.sort(key=lambda pulse: pulse[:2])
        best = _every_slide(pulses, after, ranges, 50)
        tree = _slides.most_at_once(pulses, after, ranges, blocks=False)
        assert [_value(tree, at) for at in range(50)] == best
        # A run of segments counted as one block never counts fewer.
        blocks = [_value(_slides.most_at_once(pulses, after, ranges), at) for at in range(50)]
        assert all(a >= b for a, b in zip(blocks, best, strict=True))
        exact += blocks == best
    assert exact > 200


def test_a_slide_by_any_number_holds_the_most_reached_from_then_on():
    assert _slides.slid([(3, 1), (5, 2), (6, 0)], None) == [(3, 1), (5, 2)]
    assert _slides.slid([(3, 1), (5, 0)], 2) == [(3, 1), (7, 0)]
    assert _slides.added([[(0, 1), (3, 0)], [(1, 2), (2, 0)]]) == [(0, 1), (1, 3), (2, 1), (3, 0)]
    assert _slides.first_over([(0, 1), (5, 2), (9, 3), (10, 0)], 1) == (5, 2, 3)
    assert _slides.first_over([(0, 1), (5, 0)], 1) is None


def test_a_slide_by_any_number_never_counts_fewer_than_a_long_one():
    rng = random.Random(9)
    for _ in range(300):
        n = rng.randint(2, 5)
        after = [None] + [k - 1 if rng.random() < 0.6 else rng.randrange(k) for k in range(1, n)]
        ranges = [0] + [rng.choice([0, 2, None]) for _ in range(1, n)]
        pulses = {}
        for _ in range(rng.randint(1, 6)):
            start = rng.randint(0, 10)
            pulses.setdefault(rng.randrange(n), []).append((start, start + rng.randint(1, 4), {}))
        for played in pulses.values():
            played.sort(key=lambda pulse: pulse[:2])
        # Slides of up to 12 cycles reach every cycle before 12 that any can.
        capped = [12 if most is None else most for most in ranges]
        best = _every_slide(pulses, after, capped, 12)
        for blocks in (False, True):
            steps = _slides.most_at_once(pulses, after, ranges, blocks=blocks)
            assert all(_value(steps, at) >= best[at] for at in range(12))
        exact = _slides.most_at_once(pulses, after, ranges, blocks=False)
        assert [_value(exact, at) for at in range(12)] == best


def test_the_segments_of_a_long_loop_are_counted_as_one_block():
    trips = 200_000
    after = [None, *range(trips)]
    ranges = [0] + [3984] * trips
    pulses = {k: [(117 * k + 61, 117 * k + 101, {})] for k in range(trips + 1)}
    steps = _slides.most_at_once(pulses, after, ranges)
    assert max(count for _, count in steps) == 1
    pulses[0] = [*pulses[0], (9000, 9040, {})]
    assert _slides.first_over(_slides.most_at_once(pulses, after, ranges), 1) == (9000, 2, 2)


# ── What compile gives for the programs the check refuses ────────────────────

_PLACED = {"asap": "rcp", "alap": "alap_rcp"}
_REFUSED = {
    "a drive that can slide onto another": _TWO_DRIVES,
    "a wait without an upper bound": ["wait(da, next_tau.submit(tau0).result())", *_TWO_DRIVES[1:]],
    "waits on two lines": [
        _WAIT,
        "drive(da, square(40, 0.1))",
        "wait(da, next_tau.submit(tau0).result(), bounds=(16, 300))",
        "drive(da, square(40, 0.1))",
        "wait(db, next_tau.submit(tau0).result(), bounds=(16, 500))",
        "drive(db, square(40, 0.1))",
    ],
    "a blocking wait": [
        "x = slow.submit(tau0).result()",
        "drive(da, square(40, x))",
        "wait(db, 600)",
        "drive(db, square(40, 0.1))",
    ],
    "the arms of a branch after a wait": [
        "m = readout(ra, square(100, 0.05))",
        _WAIT,
        "if m:",
        "    drive(da, square(40, 0.1))",
        "else:",
        "    drive(da, square(60, 0.1))",
        "wait(db, 600)",
        "drive(db, square(40, 0.1))",
    ],
    "a loop of 50 with a wait each trip": _rounds(
        50, ["wait(db, 9000)", "drive(db, square(40, 0.1))"]
    ),
    "a loop of 2000 with a wait each trip": _rounds(
        2000, ["wait(db, 9000)", "drive(db, square(40, 0.1))"]
    ),
}


def _events(lowered):
    return [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("name", list(_REFUSED))
def test_slides_placed_as_rcp_does(name, schedule, monkeypatch):
    # Past max_concurrent_drives for some run, asap and alap place the
    # program as rcp and alap_rcp place it, which the check passes.
    from cudaq_pulse import compiler

    monkeypatch.setattr(compiler, "_PLACED_BY_LIMITS", _PLACED)
    body = _REFUSED[name]
    lowered = _compiled(body, schedule, limit=1)
    assert lowered.options.schedule == _PLACED[schedule]
    assert _events(lowered) == _events(_compiled(body, _PLACED[schedule], limit=1))
    assert list(pulse.controller.check(lowered)) == []
