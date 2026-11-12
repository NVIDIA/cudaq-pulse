# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a compiled module says of the systems it plays on and of its own
schedule: ``pulse.control_system``, ``pulse.quantum_system`` and each
function's ``pulse.schedule_state``."""

from __future__ import annotations

import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._clocks import clock_period_fs
from cudaq_pulse.controller import DriveLine, ReadoutLine
from cudaq_pulse.quantum_systems import PhysicalQudit

from tests._handles import Q, ladder_target


@pulse.kernel
def _one_pulse(q):
    drive(get_drive_line(q), square(40, 0.1))


@pulse.kernel
def _two_lines(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(8, 0.1))
    drive(e, square(8, 0.1))
    sync(d, e)
    drive(d, square(8, 0.1))


def _attribute(text: str, name: str) -> str:
    """The value of module attribute ``pulse.<name>`` in *text*."""
    start = text.index(f"pulse.{name} = ") + len(f"pulse.{name} = ")
    depth = 0
    for end, char in enumerate(text[start:], start):
        depth += {"<": 1, ">": -1}.get(char, 0)
        if char == ">" and depth == 0:
            return text[start : end + 1]
    raise AssertionError(text)


@pytest.mark.parametrize(
    ("clock_ghz", "period"),
    [
        (2.0, "500000"),
        (1.0, "1000000"),
        (2.4, "1250000/3"),
        (1.8, "5000000/9"),
        (0.3, "10000000/3"),
        (12.0, "250000/3"),
        (1 / 3, "3000000"),
        (4.5, "2000000/9"),
        (0.25, "4000000"),
    ],
)
def test_the_clock_period_is_an_exact_fraction_of_femtoseconds(clock_ghz, period):
    qs = ladder_target(2, clock_ghz=clock_ghz)
    text = pulse.compile(_one_pulse, quantum_system=qs).lower(qs.qudits[0]).mlir()
    assert f"clock_period_fs = {period}>" in text
    # The packed builder and the writer of a whole control system agree.
    assert clock_period_fs(clock_ghz) == period
    numerator, _, denominator = period.partition("/")
    assert float(1_000_000 * int(denominator or 1)) / int(numerator) == clock_ghz


def test_a_program_without_a_controller_gives_only_its_clock():
    text = pulse.compile(_one_pulse).lower(Q[0]).mlir()
    assert _attribute(text, "control_system") == ("#pulse.control_system<clock_period_fs = 500000>")


@pulse.kernel(max_duration_cycles=100)
def _budgeted(q):
    drive(get_drive_line(q), square(40, 0.1))


def test_a_budget_is_the_controllers():
    text = pulse.compile(_budgeted).lower(Q[0]).mlir()
    assert "controllers = [#pulse.controller<max_duration_cycles = 100>]" in text


def _delayed(delays):
    box = pulse.Controller(
        drive_lines=[DriveLine(q, delay_ns=delays.get(("drive", q), 0.0)) for q in range(2)],
        readout_lines=[ReadoutLine(q) for q in range(2)],
    )
    return pulse.ControlSystem(controllers=[box])


def test_scheduling_marks_the_function_timed_or_placed():
    plain = pulse.compile(_two_lines, schedule="asap").lower(Q[0], Q[1]).mlir()
    assert "attributes {pulse.schedule_state = #pulse.schedule_state<timed>}" in plain
    machine = _delayed({("drive", 0): 1.5})
    delayed = pulse.compile(_two_lines, schedule="asap", control_system=machine)
    text = delayed.lower(Q[0], Q[1]).mlir()
    assert "attributes {pulse.schedule_state = #pulse.schedule_state<placed>}" in text
    assert "lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>]" in text


def test_a_change_of_delays_alone_gives_the_new_lines():
    first = pulse.compile(_two_lines, schedule="asap", control_system=_delayed({("drive", 0): 1.5}))
    first.lower(Q[0], Q[1])
    later = pulse.compile(_two_lines, schedule="asap", control_system=_delayed({("drive", 1): 2.0}))
    text = later.lower(Q[0], Q[1]).mlir()
    assert "lines = [#pulse.line<drive, qudits = [1], delay_cycles = 4>]" in text
    assert "#pulse.schedule_state<placed>" in text


def test_a_system_of_several_controllers_lists_each_line_and_controller():
    near = pulse.Controller(drive_lines=[0], readout_lines=[0])
    far = pulse.Controller(drive_lines=[1], readout_lines=[1], blocks_on_calls=True)
    machine = pulse.ControlSystem(controllers=[near, far])
    text = pulse.compile(_two_lines, control_system=machine).lower(Q[0], Q[1]).mlir()
    system = _attribute(text, "control_system")
    assert system.startswith(
        "#pulse.control_system<clock_period_fs = 500000, lines = ["
        "#pulse.line<drive, qudits = [0]>, #pulse.line<drive, qudits = [1], controller = 1>, "
        "#pulse.line<readout, qudits = [0]>, #pulse.line<readout, qudits = [1], controller = 1>], "
        "controllers = [#pulse.controller<outcome_latency_cycles = "
    )
    controllers = re.findall(r"#pulse\.controller<[^>]*>", system)
    assert len(controllers) == 2
    assert "blocks_on_calls" not in controllers[0] and "blocks_on_calls" in controllers[1]


def test_a_compiled_module_gives_the_frequencies_it_is_compiled_for():
    qs = ladder_target(2)
    text = pulse.compile(_one_pulse, quantum_system=qs).lower(qs.qudits[1]).mlir()
    assert _attribute(text, "quantum_system") == (
        "#pulse.quantum_system<qudit_frequency_hz = [5.000000e+09, 5.100000e+09]>"
    )


def test_simulate_writes_the_simulated_systems_own_frequencies():
    compiled_for = ladder_target(1)
    detuned = pulse.quantum_system(
        name="detuned", qudits={0: PhysicalQudit(0, 5.001e9, -300e6, 50.0, 30.0)}
    )
    lowered = pulse.compile(_one_pulse, quantum_system=compiled_for).lower(compiled_for.qudits[0])
    pulse.simulate(lowered, dt_ns=0.125, quantum_system=detuned)
    system = _attribute(lowered.mlir(), "quantum_system")
    assert system.startswith("#pulse.quantum_system<qudit_frequency_hz = [5.001000e+09], ")
    assert "level_counts = [3]" in system and "t1_ns = [5.000000e+04]" in system
    # The program keeps the tone it was compiled with.
    assert "frequency_hz = 5.000000e+09 : f64" in lowered.mlir()
