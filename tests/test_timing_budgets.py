# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Timing budgets: ``@kernel(max_duration_cycles=...)`` bounds every
execution of a kernel, and ``@classical(reaction_deadline_ns=...)`` bounds how long
after its readouts a call's answer is due."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest

from tests._handles import Q


def cycle_of(budget: int | None, **options):
    @pulse.kernel(schedule="asap", max_duration_cycles=budget, **options)
    def cycle(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.1, 10.0))
        drive(d, gaussian(40, 0.1, 10.0))

    return cycle


def repeated(cycle, rounds: int = 1000):
    @pulse.kernel
    def program(q):
        for _ in range(rounds):
            cycle(q)

    return program


# ── max_duration_cycles ─────────────────────────────────────────────────


def test_a_called_kernel_within_its_budget_compiles_as_without_one():
    with_budget = repeated(cycle_of(80)).lower(Q[0])
    without = repeated(cycle_of(None)).lower(Q[0])
    assert with_budget.schedule == without.schedule


def test_a_called_kernel_over_its_budget_names_the_kernel_and_the_fix():
    with pytest.raises(
        RuntimeError,
        match=r"each call of kernel 'cycle' takes 80 cycles, more than "
        r"max_duration_cycles=79; let its operations overlap, shorten them or raise",
    ):
        repeated(cycle_of(79)).lower(Q[0])


def test_a_budget_needs_a_kernel_with_its_own_schedule():
    @pulse.kernel(max_duration_cycles=100)
    def inlined(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.1, 10.0))

    with pytest.raises(pulse.CompilationError, match=r'give it schedule="asap" or schedule="alap"'):
        repeated(inlined).lower(Q[0])


def test_flattening_a_budgeted_kernel_is_an_error():
    program = repeated(cycle_of(80))
    with pytest.raises(pulse.CompilationError, match="compile without flatten_schedules"):
        pulse.compile(program, flatten_schedules=True).lower(Q[0])


def test_the_compiled_kernel_bounds_the_whole_program():
    def program(budget: int):
        @pulse.kernel(max_duration_cycles=budget)
        def run(q):
            d = get_drive_line(q)
            for _ in range(100):
                drive(d, gaussian(40, 0.1, 10.0))

        return run

    program(4000).lower(Q[0])
    pulse.compile(program(4000)).lower(Q[0])
    with pytest.raises(RuntimeError, match=r"kernel 'run' takes 4000 cycles, more than"):
        program(3999).lower(Q[0])


def test_a_budget_bounds_a_kernel_of_runtime_qudits():
    def program(budget: int):
        @pulse.kernel(max_duration_cycles=budget)
        def run(q: pulse.Qudit, amps: list[float]):
            d = get_drive_line(q)
            for a in amps:
                drive(d, gaussian(40, a, 10.0))

        return run

    program(80).lower(Q[0], [0.1, 0.2])
    with pytest.raises(RuntimeError, match=r"kernel 'run' takes 80 cycles, more than"):
        program(79).lower(Q[0], [0.1, 0.2])


def test_a_variable_wait_counts_at_its_highest_length():
    def program(budget: int):
        @pulse.kernel(max_duration_cycles=budget)
        def run(q, tau: float):
            d = get_drive_line(q)
            wait(d, tau, bounds=(0, 400))
            drive(d, gaussian(40, 0.1, 10.0))

        return run

    # The program compiles once for every length, so the budget holds for the
    # highest: 400 + 40.
    program(440).lower(Q[0], 10.0)
    with pytest.raises(RuntimeError, match=r"takes up to 440 cycles \(40 as scheduled"):
        program(439).lower(Q[0], 10.0)


def test_a_wait_without_a_highest_length_has_no_bound():
    @pulse.kernel(max_duration_cycles=1000)
    def run(q, tau: float):
        d = get_drive_line(q)
        wait(d, tau)
        drive(d, gaussian(40, 0.1, 10.0))

    with pytest.raises(RuntimeError, match=r"no highest length; bound it with wait\(\.\.\."):
        run.lower(Q[0], 10.0)


def test_a_blocking_answer_has_no_bound():
    @pulse.classical(max_latency_ns=100, latency_mode="blocking", name="block_bit")
    def bit() -> pulse.Bits:
        return [0]

    @pulse.kernel(max_duration_cycles=10_000)
    def run(q):
        d = get_drive_line(q)
        if bit.submit().result()[0]:
            drive(d, gaussian(40, 0.1, 10.0))

    with pytest.raises(RuntimeError, match=r'blocking classical function.*latency_mode="bounded"'):
        run.lower(Q[0])


@pytest.mark.parametrize("value", [0, -3, 1.5, True, "80"])
def test_max_duration_cycles_is_a_positive_whole_number(value):
    with pytest.raises(ValueError, match="max_duration_cycles must be a whole number of cycles"):
        pulse.kernel(max_duration_cycles=value)


def test_a_program_without_budgets_keeps_its_module():
    text = repeated(cycle_of(None)).lower(Q[0]).mlir()
    assert "max_duration_cycles" not in text


# ── reaction_deadline_ns ─────────────────────────────────────────────────────────


def responder(reaction_deadline_ns: float, max_latency_ns: float = 400.0, name: str = "react"):
    @pulse.classical(
        max_latency_ns=max_latency_ns, reaction_deadline_ns=reaction_deadline_ns, name=name
    )
    def respond(m: pulse.Bits) -> pulse.Bits:
        return [m[0]]

    return respond


def reacting(respond, rounds: int = 1):
    @pulse.kernel(unroll_threshold=0)
    def run(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(rounds):
            m = readout(r, square(100, 0.05))
            if respond.submit([m]).result()[0]:
                drive(d, gaussian(40, 0.1, 10.0))

    return run


def test_a_call_within_its_reaction_time_compiles_as_without_one():
    lowered = reacting(responder(2000.0)).lower(Q[0])
    (call,) = lowered.classical_report()
    # Submitted as the readout ends, due 1 + 800 cycles later at 2 GHz.
    assert call.due_cycles - call.submit_cycles == 801


def test_a_call_over_its_reaction_time_names_the_slack_and_the_fix():
    with pytest.raises(
        RuntimeError,
        match=r"'react__1' cannot react in time: in iteration \(0\) its answer is due 801 "
        r"cycles after "
        r"the readouts it carries end, but reaction_deadline_ns allows 600 \(slack -201\); "
        r"submit it sooner, lower max_latency_ns",
    ):
        reacting(responder(300.0)).lower(Q[0])


def test_a_reaction_time_holds_in_every_iteration_of_a_loop():
    reacting(responder(2000.0), rounds=1000).lower(Q[0])
    with pytest.raises(RuntimeError, match="cannot react in time"):
        reacting(responder(300.0), rounds=1000).lower(Q[0])


def test_a_call_without_readouts_reacts_from_its_submission():
    @pulse.classical(max_latency_ns=400, reaction_deadline_ns=300, name="react_no_readout")
    def respond() -> pulse.Bits:
        return [0]

    @pulse.kernel
    def run(q):
        r = get_readout_line(q)
        readout(r, square(100, 0.05))
        respond.submit().result()

    with pytest.raises(RuntimeError, match="801 cycles after the submission starts"):
        run.lower(Q[0])


def test_a_bounded_wait_before_the_submission_counts_at_its_highest_length():
    respond = responder(600.0, max_latency_ns=200.0, name="react_after_wait")

    def program(highest: int):
        @pulse.kernel
        def run(q, tau: float):
            r = get_readout_line(q)
            m = readout(r, square(100, 0.05))
            wait(r, tau, bounds=(0, highest))
            respond.submit([m]).result()

        return run

    # 1 + 400 cycles to answer, plus the wait.
    program(799).lower(Q[0], 0.0)
    with pytest.raises(RuntimeError, match=r"due 1201 cycles after the readouts .* allows 1200"):
        program(800).lower(Q[0], 0.0)


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan")])
def test_reaction_deadline_ns_is_a_positive_number(value):
    with pytest.raises(ValueError, match="reaction_deadline_ns must be a positive number"):
        responder(value)


def test_a_reaction_time_is_kept_on_the_function():
    assert responder(250).reaction_deadline_ns == 250.0
    assert responder(250, name="other").deadline_ns is None
    assert responder(0.25 * pulse.us, name="timed").reaction_deadline_ns == 250.0
