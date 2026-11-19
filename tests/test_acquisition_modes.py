# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readouts that acquire a level (the default) or the integrated IQ point."""

from __future__ import annotations

import json
import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import CompilationError, InvalidValueError
from cudaq_pulse.controller import ControllerLimitError, ReadoutLine, unroll
from cudaq_pulse.quantum_systems import DispersiveReadout, PhysicalQudit, ReadoutPulse
from cudaq_pulse.runtime import _calls

PI_AMP = math.pi / 20
PLUS = [2**-0.5, 2**-0.5]


def _device(error_0to1=0.0, controllers=None):
    qudit = PhysicalQudit(
        index=0,
        frequency_hz=5e9,
        anharmonicity_hz=-2e8,
        t1_us=0.0,
        t2_star_us=0.0,
        readout=DispersiveReadout(
            dispersive_shift_hz=1e6, linewidth_hz=2e6, efficiency=0.5, error_0to1=error_0to1
        ),
        calibrations={pulse.gates.READOUT: ReadoutPulse()},
    )
    system = None if controllers is None else pulse.ControlSystem(controllers=controllers)
    return pulse.quantum_system(name="sim", qudits={0: qudit}, control_system=system)


DEV = _device()
Q = DEV.qudits[0]


@pulse.kernel
def plain(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, PI_AMP / 2))
    sync(d, r)
    return readout(r, square(400, 0.05))


@pulse.kernel
def level(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, PI_AMP / 2))
    sync(d, r)
    return readout(r, square(400, 0.05), acquire="level")


@pulse.kernel
def point(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, PI_AMP / 2))
    sync(d, r)
    return readout(r, square(400, 0.05), acquire="integrated")


@pulse.kernel
def both(q):
    r = get_readout_line(q)
    readout(r, square(400, 0.05))
    readout(r, square(400, 0.05), acquire="integrated")


@pulse.kernel(unroll_threshold=0)
def points_in_a_loop(q):
    r = get_readout_line(q)
    for _ in range(5):
        readout(r, square(400, 0.05), acquire="integrated")
        wait(r, 100)


def _lower(kernel, dev=DEV):
    return pulse.compile(kernel, quantum_system=dev).lower(dev.qudits[0])


def _readouts(program):
    return [e for e in program["events"] if e["kind"] == "readout"]


# ── IR and exports ───────────────────────────────────────────────────────────


def test_acquire_level_is_the_readout_it_always_was():
    default, given = _lower(plain), _lower(level)
    assert default.mlir() == given.mlir().replace("@level", "@plain")
    for version in (1, 2):
        assert json.dumps(default.export_to_controller(version=version)) == json.dumps(
            given.export_to_controller(version=version)
        ).replace('"level"', '"plain"')
    assert all("acquire" not in e for e in _readouts(default.export_to_controller()))


def test_an_integrated_readout_is_marked_in_the_ir_and_both_exports():
    lowered = _lower(both)
    text = lowered.mlir()
    assert ", level <{" in text and ", integrated <{" in text and ", iq <{" not in text
    level_line, point_line = [line for line in text.splitlines() if " pulse.readout " in line]
    assert level_line.endswith("!pulse.measurement") and point_line.endswith("!pulse.iq_data")
    for version in (1, 2):
        first, second = _readouts(lowered.export_to_controller(version=version))
        assert "acquire" not in first
        assert second["acquire"] == "integrated"
    flat = lowered.export_to_controller()
    assert list(_readouts(flat)[1])[-1] == "acquire"


def test_unrolling_a_rolled_loop_of_points_keeps_their_mode():
    lowered = _lower(points_in_a_loop)
    structured = lowered.export_to_controller(version=2)
    (loop,) = [e for e in structured["events"] if e["kind"] == "loop"]
    assert [e.get("acquire") for e in loop["body"] if e["kind"] == "readout"] == ["integrated"]
    flat = lowered.export_to_controller()
    assert [e["acquire"] for e in _readouts(flat)] == ["integrated"] * 5
    assert unroll(structured)["events"] == flat["events"]


# ── Capability ───────────────────────────────────────────────────────────────


def test_a_controller_without_acquire_integrated_refuses_points_only():
    others = [c for c in pulse.Capability if c is not pulse.Capability.ACQUIRE_INTEGRATED]
    box = pulse.Controller(drive_lines=[0], readout_lines=[0], supports=others)
    dev = _device(controllers=[box])
    _lower(plain, dev)
    with pytest.raises(ControllerLimitError, match="needs 'acquire_integrated'"):
        _lower(point, dev)
    full = pulse.Controller(drive_lines=[0], readout_lines=[0], supports=list(pulse.Capability))
    assert _readouts(_lower(point, _device(controllers=[full])).export_to_controller())


# ── What a point may be used for ─────────────────────────────────────────────


@pulse.classical(max_latency_ns=0)
def magnitude(p: complex) -> float:
    return abs(p)


@pulse.classical(max_latency_ns=0)
def parity(bits: pulse.Bits) -> pulse.Bits:
    return [sum(bits) % 2]


@pulse.classical(max_latency_ns=0)
def total(levels: list[int]) -> int:
    return sum(levels)


@pulse.kernel
def branches_on_a_point(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    p = readout(r, square(400, 0.05), acquire="integrated")
    sync(d, r)
    if p:
        drive(d, square(40, PI_AMP))


@pulse.kernel
def compares_a_point(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    p = readout(r, square(400, 0.05), acquire="integrated")
    sync(d, r)
    if p == 1:
        drive(d, square(40, PI_AMP))


@pulse.kernel
def sends_a_point_as_bits(q):
    r = get_readout_line(q)
    p = readout(r, square(400, 0.05), acquire="integrated")
    parity.submit([p]).result()


@pulse.kernel
def sends_points_as_levels(q):
    r = get_readout_line(q)
    p = readout(r, square(400, 0.05), acquire="integrated")
    total.submit([p, p]).result()


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (branches_on_a_point, "gives an IQ point, which cannot decide a branch"),
        (compares_a_point, "gives an IQ point, which cannot be compared"),
        (sends_a_point_as_bits, "the level of each readout, but an integrated readout"),
        (sends_points_as_levels, "the level of each readout, but an integrated readout"),
    ],
)
def test_a_point_has_no_level(kernel, message):
    with pytest.raises(CompilationError, match=message) as raised:
        _lower(kernel)
    assert 'acquire="level"' in (raised.value.fix or "")


@pytest.mark.parametrize(
    ("acquire", "message"),
    [
        ("trace", 'acquire is "level" or "integrated".*trace and counts are not supported yet'),
        ("raw", 'acquire is "level" or "integrated".*got \'raw\''),
        ("iq", 'acquire is "level" or "integrated".*got \'iq\''),
        (1, "got 1"),
    ],
)
def test_acquire_takes_level_or_integrated(acquire, message):
    @pulse.kernel
    def bad(q):
        r = get_readout_line(q)
        readout(r, square(400, 0.05), acquire=acquire)

    with pytest.raises(CompilationError, match=message):
        _lower(bad)


@pulse.kernel
def fits_a_point(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    p = readout(r, square(400, 0.05), acquire="integrated")
    sync(d, r)
    drive(d, square(40, 0.001 * magnitude.submit(p).result()))


def test_a_call_takes_a_point_as_complex():
    lowered = _lower(fits_a_point)
    (submit,) = [e for e in lowered.export_to_controller()["events"] if e["kind"] == "submit"]
    readout_index = _readouts(lowered.export_to_controller())[0]["index"]
    assert submit["measurements"] == [readout_index]
    result = pulse.simulate(lowered, shots_count=20, seed=3, levels=2)
    answers = [a for b in result.branches for a in b.answers("magnitude")]
    assert answers and all(a > 0 for a in answers)


# ── Scheduling ───────────────────────────────────────────────────────────────


@pulse.classical(max_latency_ns=200, reaction_deadline_ns=2000)
def react(p: complex) -> float:
    return abs(p)


def _scheduled_kernels(acquire):
    """Kernels that read out with *acquire* and send what they read to a
    call, in a rolled loop too or while a drive plays, carry it, pass it
    into or out of a kernel with its own schedule, branch on a call's answer
    to it, or leave it unread."""

    @pulse.kernel
    def sends(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        p = readout(r, square(400, 0.05), acquire=acquire)
        sync(d, r)
        drive(d, square(40, 0.001 * magnitude.submit(p).result()))

    @pulse.kernel
    def sends_while_driving(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        p = readout(r, square(400, 0.05), acquire=acquire)
        sync(d, r)
        react.submit(p).result()
        drive(d, square(40, PI_AMP))

    @pulse.kernel(unroll_threshold=0)
    def sends_in_a_loop(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(5):
            p = readout(r, square(400, 0.05), acquire=acquire)
            drive(d, square(40, 0.001 * magnitude.submit(p).result()))

    @pulse.kernel(schedule="alap")
    def reads(q):
        r = get_readout_line(q)
        return readout(r, square(400, 0.05), acquire=acquire)

    @pulse.kernel
    def sends_what_a_kernel_read(q):
        d = get_drive_line(q)
        p = reads(q)
        drive(d, square(40, 0.001 * magnitude.submit(p).result()))

    @pulse.kernel(schedule="alap")
    def sent_by_a_kernel(q, p):
        d = get_drive_line(q)
        drive(d, square(40, 0.001 * magnitude.submit(p).result()))

    @pulse.kernel
    def passes_to_a_kernel(q):
        r = get_readout_line(q)
        sent_by_a_kernel(q, readout(r, square(400, 0.05), acquire=acquire))

    @pulse.kernel
    def branches_on_the_answer(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        p = readout(r, square(400, 0.05), acquire=acquire)
        if magnitude.submit(p).result() > 0.5:
            drive(d, square(40, PI_AMP))

    @pulse.kernel(unroll_threshold=0)
    def reacts_in_a_loop(q):
        r = get_readout_line(q)
        for _ in range(5):
            react.submit(readout(r, square(100, 0.05), acquire=acquire)).result()

    @pulse.kernel(unroll_threshold=0)
    def reacts_to_a_carried_one(q):
        r = get_readout_line(q)
        p = readout(r, square(100, 0.05), acquire=acquire)
        for _ in range(5):
            react.submit(p).result()
            p = readout(r, square(100, 0.05), acquire=acquire)

    @pulse.kernel
    def leaves_it_unread(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        readout(r, square(400, 0.05), acquire=acquire)
        drive(d, square(400, PI_AMP))
        drive(d, square(400, PI_AMP))

    return {
        "sends": sends,
        "sends_while_driving": sends_while_driving,
        "sends_in_a_loop": sends_in_a_loop,
        "sends_what_a_kernel_read": sends_what_a_kernel_read,
        "passes_to_a_kernel": passes_to_a_kernel,
        "branches_on_the_answer": branches_on_the_answer,
        "reacts_in_a_loop": reacts_in_a_loop,
        "reacts_to_a_carried_one": reacts_to_a_carried_one,
        "leaves_it_unread": leaves_it_unread,
    }


_SYSTEMS = {
    "plain": None,
    # Outcomes usable 20 cycles after a readout ends.
    "latency": [pulse.Controller(drive_lines=[0], readout_lines=[0], feedback_latency_cycles=20)],
    # The readout on controllers[0], the drive on controllers[1].
    "two": [
        pulse.Controller(readout_lines=[0], feedback_latency_cycles=12, peer_latency_cycles=16),
        pulse.Controller(drive_lines=[0], feedback_latency_cycles=4),
    ],
    # Both block on calls: a call holds the lines of the controller that
    # reads out what it sends, and of those that use its answer.
    "blocking": [
        pulse.Controller(readout_lines=[0], blocks_on_calls=True),
        pulse.Controller(drive_lines=[0], blocks_on_calls=True),
    ],
    # The latency on a readout line delayed 6 ns (12 cycles), which an
    # outcome waits for under rcp.
    "delayed": [
        pulse.Controller(
            drive_lines=[0],
            readout_lines=[ReadoutLine(0, delay_ns=6.0)],
            feedback_latency_cycles=20,
        )
    ],
}
_SCHEDULED = [
    (system, schedule)
    for system in ("plain", "latency", "two", "blocking")
    for schedule in ("alap", "asap", "alap_rcp")
] + [("delayed", "alap_rcp")]


def _refusal(name, system, schedule):
    """What compiling kernel *name* is refused for, a level and a point
    alike: a kernel with its own schedule= under rcp, or one that makes a
    call on controllers that block on calls; a reaction deadline on a
    readout a loop carries."""
    if name == "reacts_to_a_carried_one":
        return "a measurement it carries comes out of a branch or a loop"
    if name in ("sends_what_a_kernel_read", "passes_to_a_kernel") and schedule == "alap_rcp":
        return "does not support kernels called with their own schedule= yet"
    if name == "passes_to_a_kernel" and system == "blocking":
        return "the controller blocks on calls"
    return None


def _as_level(text):
    """*text* as the program that reads out levels instead gives it."""
    text = re.sub(r", integrated( <| :)", r", level\1", text)
    return text.replace("!pulse.iq_data", "!pulse.measurement")


def _without_acquire(value):
    if isinstance(value, dict):
        return {k: _without_acquire(v) for k, v in value.items() if k != "acquire"}
    if isinstance(value, list):
        return [_without_acquire(v) for v in value]
    return value


def _built(kernel, system, schedule):
    """The IR and the exports (each the format allows: /1 and /2 of one
    controller, /1 alone with a delayed line, /1 of each of several) of
    *kernel* on *system*, or the error compiling it raises."""
    controllers = _SYSTEMS[system]
    dev = _device(controllers=controllers)
    try:
        compiled = pulse.compile(kernel, quantum_system=dev, schedule=schedule)
        lowered = compiled.lower(dev.qudits[0])
    except (CompilationError, RuntimeError) as e:
        return str(e)
    if controllers is not None and len(controllers) > 1:
        programs = [lowered.export_to_controller(controller=k) for k in range(len(controllers))]
    else:
        programs = [lowered.export_to_controller()]
        if system != "delayed":
            programs.append(lowered.export_to_controller(version=2))
    return lowered.mlir(), json.dumps(programs)


@pytest.mark.parametrize(("system", "schedule"), _SCHEDULED)
@pytest.mark.parametrize("name", list(_scheduled_kernels("level")))
def test_a_point_schedules_as_a_level_does(name, system, schedule):
    level = _built(_scheduled_kernels("level")[name], system, schedule)
    point = _built(_scheduled_kernels("integrated")[name], system, schedule)
    refused = _refusal(name, system, schedule)
    if refused:
        assert isinstance(level, str) and refused in level, level
        assert _as_level(point) == level
        return
    assert not isinstance(level, str), level
    assert not isinstance(point, str), point
    assert "!pulse.iq_data" in point[0] and "!pulse.iq_data" not in level[0]
    assert _as_level(point[0]) == level[0]
    assert json.dumps(_without_acquire(json.loads(point[1]))) == level[1]


# ── Simulation ───────────────────────────────────────────────────────────────


def test_points_are_drawn_per_shot_at_the_observed_level():
    shots = 4000
    result = pulse.simulate(_lower(point), shots_count=shots, seed=5, levels=2)
    points = result.acquisitions(0)
    assert points.shape == (shots, 1) and points.dtype == np.complex128
    by_level = [points[result.records[:, 0] == k, 0] for k in (0, 1)]
    # The tone sits midway between the pulled resonator frequencies, so the
    # two levels' points are mirror images, with vacuum noise of variance
    # 1/2 per quadrature around each.
    means = [p.mean() for p in by_level]
    error = 5 * math.sqrt(1.0 / min(len(p) for p in by_level))
    assert abs(means[0] - means[1].conjugate()) < error
    assert abs(means[0] - means[1]) > 1.0
    for p in by_level:
        assert np.var(p.real) == pytest.approx(0.5, rel=0.15)
        assert np.var(p.imag) == pytest.approx(0.5, rel=0.15)


def test_a_point_projects_as_a_level_readout_does():
    for options in ({}, {"shots_count": 300, "seed": 2}):
        by_level = pulse.simulate(_lower(level), levels=2, **options)
        by_point = pulse.simulate(_lower(point), levels=2, **options)
        np.testing.assert_allclose(by_point.density_matrix, by_level.density_matrix, atol=1e-12)
        if options:
            np.testing.assert_array_equal(by_point.records, by_level.records)


def test_assignment_errors_do_not_apply_to_points():
    dev = _device(error_0to1=0.4)
    result = pulse.simulate(_lower(both, dev), shots_count=400, seed=1, levels=2, readout="model")
    reported_level, point_level = result.records[:, 0], result.records[:, 1]
    assert (point_level == 0).all()
    assert 0.3 < reported_level.mean() < 0.5
    assert result.acquisitions(1).shape == (400, 1)


def test_acquisitions_follow_the_shots_postselect_keeps():
    result = pulse.simulate(_lower(both), shots_count=200, seed=4, levels=2, initial_state=PLUS)
    kept = result.postselect(lambda b: b.levels[0] == 0)
    assert kept.acquisitions(1).shape == (len(kept.records), 1)
    rows = result.records[:, 0] == 0
    np.testing.assert_array_equal(kept.acquisitions(1), result.acquisitions(1)[rows])


def test_acquisitions_name_the_integrated_readouts():
    result = pulse.simulate(_lower(both), shots_count=10, seed=0, levels=2)
    with pytest.raises(InvalidValueError, match="readout 0 acquires no IQ points; the integrated"):
        result.acquisitions(0)
    plain_result = pulse.simulate(_lower(plain), shots_count=10, seed=0, levels=2)
    with pytest.raises(InvalidValueError, match="the program has no integrated readout"):
        plain_result.acquisitions(0)
    with pytest.raises(InvalidValueError, match="simulate with shots_count"):
        pulse.simulate(_lower(both), levels=2).acquisitions(1)


def test_a_loop_of_points_has_one_column_per_execution():
    result = pulse.simulate(_lower(points_in_a_loop), shots_count=30, seed=0, levels=2)
    points = result.acquisitions(0)
    assert points.shape == (30, 5)
    assert not np.isnan(points).any()


def test_calling_the_kernel_returns_the_point():
    found = point(Q)
    assert isinstance(found, complex)
    assert isinstance(level(Q), int)


def test_drawing_points_costs_the_same_at_any_shot_count(monkeypatch):
    draws = []
    draw = _calls.draw_points

    def counted(*args):
        draws.append(args[-1])
        return draw(*args)

    monkeypatch.setattr(_calls, "draw_points", counted)
    lowered = _lower(points_in_a_loop)
    counts = []
    for shots in (10, 10_000):
        draws.clear()
        pulse.simulate(lowered, shots_count=shots, seed=0, levels=2, initial_state=PLUS)
        counts.append(len(draws))
        assert sum(draws) == 5 * shots
    # One draw per execution and outcome row (both levels here), not per shot.
    assert counts[0] == counts[1] == 5 * 2
