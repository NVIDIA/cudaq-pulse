# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Classical calls in every scheduling mode, from every place in a kernel,
and kernels calling kernels with every pair of schedules.

Every case runs by default. The cases and the properties are in
``_schedule_matrix``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel

from tests._handles import Q
from tests.classical import _schedule_matrix as matrix


def _params(cases):
    return [pytest.param(c, id=c.id) for c in cases]


CASES = matrix.all_cases()
VALID = [c for c in CASES if matrix.expected_error(c) is None]


# ── Every case: compile, verify, schedule, export ────────────────────────────


@pytest.mark.parametrize("case", _params(CASES))
def test_case(case):
    expected = matrix.expected_error(case)
    if expected is not None:
        error, message = expected
        with pytest.raises(error, match=message):
            matrix.lower_case(case, Q)
        return
    low = matrix.lower_case(case, Q)
    matrix.check_calls(case, low)
    matrix.check_lines(low)
    matrix.check_barriers(case, low)
    matrix.check_scope(case, low)
    matrix.check_reverify(low)
    matrix.check_export(case, low, matrix.lower_case(case, Q))
    if case.policy in ("asap", "alap") and not case.blocks:
        matrix.check_tight(case, low)


# The order of ASAP and ALAP is a property of the program, not of its root's
# schedule, which this test sets.
_ORDER = [c for c in VALID if c.root == "default"]


@pytest.mark.parametrize(
    "case",
    _params(_ORDER),
)
def test_asap_is_early_and_alap_late(case):
    matrix.check_policy_order(case, Q)


# ── Kernels calling kernels ──────────────────────────────────────────────────

_SHAPED = [c for c in VALID if c.blocks and c.placement in matrix.SHAPED]


@pytest.mark.parametrize("case", _params(_SHAPED))
def test_a_called_kernel_with_its_own_schedule_keeps_its_shape(case):
    low = matrix.lower_case(case, Q)
    own = matrix.compile_case(case, kernel=matrix.alone(case)).lower(
        Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
    )
    matrix.check_callee_shape(low, own)


_INLINED = [c for c in VALID if c.placement in matrix.CALLEE_PLACEMENTS and c.callee is None]


@pytest.mark.parametrize("case", _params(_INLINED))
def test_a_called_kernel_without_a_schedule_is_scheduled_as_inlined(case):
    as_kernel = matrix.lower_case(case, Q)
    helper = matrix.compile_case(case, kernel=matrix.program(case, callee_as_helper=True))
    assert matrix.timeline(
        helper.lower(Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU)
    ) == matrix.timeline(as_kernel)


_FLAT = [c for c in VALID if c.flatten]


@pytest.mark.parametrize("case", _params(_FLAT))
def test_flattening_ignores_every_called_kernels_schedule(case):
    plain = dataclasses.replace(case, callee=None, flatten=False)
    assert matrix.timeline(matrix.lower_case(case, Q)) == matrix.timeline(
        matrix.lower_case(plain, Q)
    )


_NESTED = list(
    itertools.product(matrix.CALLEES, matrix.CALLEES, matrix.CALLEES, matrix.LINKS, matrix.TAKES)
)


def _nested(outer, middle, inner, link, take):
    """Three kernels deep, each with its own schedule: the innermost
    measures and submits, the middle one plays beside it and passes the
    future up, the outer one waits beside them; the program takes it."""
    decode, _ = matrix.FUNCTIONS[(link, take)]
    answer, _, _ = matrix._calls(matrix.Case(link=link, take=take))

    @pulse.kernel(schedule=inner)
    def measure(q):
        r = get_readout_line(q)
        return decode.submit([readout(r, square(matrix.RO, 0.05))])

    @pulse.kernel(schedule=middle)
    def beside(q, p):
        e = get_drive_line(p)
        future = measure(q)
        drive(e, square(400, 0.05))
        return future

    @pulse.kernel(schedule=outer)
    def round_(q, p):
        d = get_drive_line(q)
        drive(d, square(matrix.X90, 0.1))
        return beside(q, p)

    @pulse.kernel
    def program(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(matrix.SPECTATOR, 0.05))
        if answer(round_(q, p)):
            drive(d, square(matrix.ARM, 0.3))

    return program, measure


@pytest.mark.parametrize(
    "nested",
    [pytest.param(n, id="-".join(str(x) for x in n)) for n in _NESTED],
)
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_three_kernels_deep_with_mixed_schedules(nested, policy):
    outer, middle, inner, link, take = nested
    program, measure = _nested(*nested)
    case = matrix.Case(link=link, take=take)
    low = pulse.compile(program, schedule=policy, machine=matrix.MACHINE).lower(Q[0], Q[1])
    matrix.check_calls(case, low)
    matrix.check_lines(low)
    matrix.check_reverify(low)
    (call,) = matrix.calls(low)
    events = low.schedule
    readout_event = next(e for e in events if e.kind == "readout")
    # The request goes out as soon as its readout ends: every schedule keeps
    # an eager submission at its earliest time within its block.
    assert call.submit == readout_event.end
    # The conditional pulse waits for the answer, and under ASAP no longer:
    # it is taken when due, or for a blocking call once every line is done.
    feedback = events[-1]
    assert feedback.kind == "drive" and feedback.start >= call.taken >= call.due
    if policy == "asap":
        assert feedback.start == max(call.taken, events[1].end)


# ── Simulation against an explicitly scheduled reference ─────────────────────

_SIMULATED = [c for c in VALID if c.root in ("default", "asap", "alap") and c.link == "controller"]


@pytest.mark.parametrize("case", _params(_SIMULATED))
def test_simulation_matches_an_explicit_schedule(case):
    low = matrix.lower_case(case, Q)
    shifts = matrix.epoch_shifts(case, low)
    reference = matrix.explicit_reference(low, shifts).lower(Q[0], Q[1])
    assert matrix.pulses(reference) == matrix.pulses(low, shifts)
    simulated, expected = pulse.simulate(low, levels=2), pulse.simulate(reference, levels=2)
    np.testing.assert_allclose(simulated.density_matrix, expected.density_matrix, atol=1e-9)
    by_levels = lambda branch: branch.levels  # noqa: E731
    paths = sorted(simulated.branches, key=by_levels)
    reference_paths = sorted(expected.branches, key=by_levels)
    assert [b.levels for b in paths] == [b.levels for b in reference_paths]
    np.testing.assert_allclose(
        [b.probability for b in paths], [b.probability for b in reference_paths], atol=1e-9
    )


# ── Kinds of call ────────────────────────────────────────────────────────────


def _in_flight(n, slots, schedule, callee):
    machine = MachineModel(links={"controller": Link(max_in_flight=slots)})
    decode, _ = matrix.FUNCTIONS[("controller", "result")]

    def burst(r, rt):
        m = readout(r, square(10, 0.05), tone=rt)
        futures = [decode.submit([m]) for _ in range(n)]
        for future in futures:
            future.result()

    inner = pulse.kernel(schedule=callee)(burst)

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        inner(r, r.tone)

    return pulse.compile(k, schedule=schedule, machine=machine).lower(Q[0])


@pytest.mark.parametrize("callee", matrix.CALLEES)
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
@pytest.mark.parametrize("over", [0, 1, 3])
def test_requests_beyond_max_in_flight_wait_for_a_slot(callee, schedule, over):
    slots = 3
    if schedule == "rcp" and callee is not None:
        with pytest.raises(pulse.CompilationError, match="flatten_schedules=True"):
            _in_flight(slots + over, slots, schedule, callee)
        return
    made = matrix.calls(_in_flight(slots + over, slots, schedule, callee))
    first = made[0].submit
    # Up to max_in_flight go out back to back; each one more waits for the
    # slot of the one max_in_flight before it.
    assert [c.submit - first for c in made[:slots]] == list(range(slots))
    for k in range(slots, len(made)):
        assert made[k].submit == made[k - slots].due
    spans = [(c.submit, c.due) for c in made]
    assert max(sum(s <= t < e for s, e in spans) for t, _ in spans) == slots


@pytest.mark.parametrize("link", matrix.LINKS)
@pytest.mark.parametrize(("deadline", "meets"), [(150.0 + 50.0 + 20.0, True), (150.0, False)])
def test_a_latency_meets_its_deadline_or_names_the_fix(link, deadline, meets):
    @pulse.classical(
        name=f"deadline_{link}", max_latency_ns=matrix.LATENCY_NS, deadline_ns=deadline, link=link
    )
    def tight(bits: pulse.Bits) -> pulse.Bits:
        return bits

    @pulse.kernel(schedule="asap")
    def inner(r, rt):
        return tight.submit([readout(r, square(matrix.RO, 0.05), tone=rt)])

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        inner(r, r.tone).result()

    compiled = pulse.compile(k, machine=matrix.MACHINE)
    budget = math.ceil(matrix.LATENCY_NS * matrix.CLOCK) + math.ceil(
        matrix.LINKS[link].feedback_ns * matrix.CLOCK
    )
    satisfiable = matrix.LINKS[link].fixed_ns * matrix.CLOCK + 1 + budget <= deadline * matrix.CLOCK
    if meets and satisfiable:
        (call,) = matrix.calls(compiled.lower(Q[0]))
        assert call.deadline_slack >= 0
    else:
        with pytest.raises(RuntimeError, match="cannot meet its deadline.*raise deadline_ns"):
            compiled.lower(Q[0])


def test_limits_without_timeouts_reject_a_timeout_and_name_the_fix():
    from cudaq_pulse.controller import ControllerLimitError, ControllerLimits

    case = matrix.Case("asap", "alap", "future_in", take="timeout")
    machine = MachineModel(
        links=dict(matrix.LINKS), controller=ControllerLimits(supports_timeouts=False)
    )
    with pytest.raises(ControllerLimitError, match=re.escape("take it with result() (no timeout)")):
        matrix.lower_case(case, Q, machine=machine)


def test_a_future_is_taken_in_the_loop_iteration_that_submitted_it():
    decode, _ = matrix.FUNCTIONS[("controller", "result")]

    @pulse.kernel(schedule="asap")
    def take(future):
        future.result()

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        future = decode.submit([readout(r, square(matrix.RO, 0.05))])
        for _ in range(2):
            take(future)

    with pytest.raises(pulse.CompilationError, match="take it in the same loop body"):
        pulse.compile(k, machine=matrix.MACHINE, unroll_threshold=0).lower(Q[0])
