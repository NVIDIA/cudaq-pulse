# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""controller.condition_table: whether an exported event plays, for every
outcome of the readouts its branch conditions read."""

from __future__ import annotations

import itertools
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._errors import InvalidValueError
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.controller import condition_table

from tests._handles import Q


@pulse.kernel
def nested(q, r):
    d = get_drive_line(q)  # noqa: F821
    e = get_drive_line(r)  # noqa: F821
    a = get_readout_line(q)  # noqa: F821
    b = get_readout_line(r)  # noqa: F821
    m0 = readout(a, square(80, 0.05))  # noqa: F821
    m1 = readout(b, square(80, 0.05))  # noqa: F821
    sync(d, e, a, b)  # noqa: F821
    if m0:
        if not m1:
            drive(d, square(16, 0.1))  # noqa: F821
        else:
            drive(e, square(24, 0.1))  # noqa: F821
    else:
        drive(d, square(32, 0.1))  # noqa: F821


def _drives(program):
    return [e for e in program["events"] if e["kind"] == "drive"]


def test_each_event_plays_where_every_condition_is_its_arm():
    lowered = pulse.compile(nested).lower(Q[0], Q[1])
    both, set_clear, clear = _drives(lowered.export_to_controller())
    assert condition_table(both["conditions"]) == {
        (0, 0): False,
        (0, 1): False,
        (1, 0): False,
        (1, 1): True,
    }
    assert condition_table(set_clear["conditions"]) == {
        (0, 0): False,
        (0, 1): False,
        (1, 0): True,
        (1, 1): False,
    }
    # One condition is the path of that condition alone.
    [otherwise] = clear["conditions"]
    assert condition_table(otherwise) == {(0,): True, (1,): False}
    assert condition_table(otherwise | {"arm": True}) == {(0,): False, (1,): True}


def test_the_table_plays_what_the_simulator_plays_for_each_outcome():
    lowered = pulse.compile(nested).lower(Q[0], Q[1])
    program = lowered.export_to_controller()
    readouts = [e["index"] for e in program["events"] if e["kind"] == "readout"]
    module = lowered._concrete().simulation_module(*lowered.options.schedule_args())
    for levels in itertools.product((0, 1), repeat=2):
        path = []
        for level in levels:
            readout, iteration = module.next_measurement(path)[:2]
            path.append([readout, *iteration, level])
        played = {start for _, _, start, _, _ in module.drive_frames(path)[0]}
        observed = dict(zip(readouts, levels, strict=True))
        expected = set()
        for event in _drives(program):
            read = sorted({r for c in event["conditions"] for r in c["readouts"]})
            if condition_table(event["conditions"])[tuple(observed[r] for r in read)]:
                expected.add(event["start_cycles"])
        assert played == expected, levels


def test_the_keys_follow_the_readouts_event_indices_whatever_the_order_of_conditions():
    lowered = pulse.compile(nested).lower(Q[0], Q[1])
    set_clear = _drives(lowered.export_to_controller())[1]
    conditions = set_clear["conditions"]
    assert condition_table(conditions[::-1]) == condition_table(conditions)
    assert condition_table([]) == {(): True}


@pulse.kernel
def on_level(q):
    d = get_drive_line(q)  # noqa: F821
    ro = get_readout_line(q)  # noqa: F821
    m = readout(ro, square(80, 0.05))  # noqa: F821
    sync(d, ro)  # noqa: F821
    if m:
        drive(d, square(16, 0.1))  # noqa: F821


def test_a_condition_on_a_readout_level_tabulates_each_level():
    lowered = pulse.compile(on_level).lower(Q[0])
    (event,) = _drives(lowered.export_to_controller())
    # A set readout is any level but 0.
    assert condition_table(event["conditions"], levels=3) == {(0,): False, (1,): True, (2,): True}
    # The same branch on level 1 exactly, as an expression of the readout.
    text = lowered._concrete().print()
    line = next(line for line in text.splitlines() if "pulse.is_set" in line)
    condition, measurement = re.match(r"\s*(%\S+) = pulse.is_set (%\S+) :", line).groups()
    level = (
        f"%k = pulse.outcome {measurement} : !pulse.measurement -> i64\n"
        f"%one = arith.constant 1 : i64\n{condition} = arith.cmpi eq, %k, %one : i64"
    )
    module = MLIRPipeline().parse_to_module(text.replace(line, level))
    object.__setattr__(lowered, "_module", module)
    (event,) = _drives(lowered.export_to_controller())
    [found] = event["conditions"]
    assert found["test"] == "expression"
    assert condition_table(found, levels=3) == {(0,): False, (1,): True, (2,): False}


@pulse.classical(max_latency_ns=200, name="echo")
def echo(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def on_answer(q):
    d = get_drive_line(q)  # noqa: F821
    ro = get_readout_line(q)  # noqa: F821
    m = readout(ro, square(80, 0.05))  # noqa: F821
    if echo.submit([m]).result()[0]:
        drive(d, square(16, 0.1))  # noqa: F821


def test_a_condition_on_an_answer_has_no_table():
    (event,) = _drives(pulse.compile(on_answer).lower(Q[0]).export_to_controller())
    answer = event["conditions"][0]["answers"][0]
    with pytest.raises(InvalidValueError, match=rf"answer events \[{answer}\]"):
        condition_table(event["conditions"])


@pytest.mark.parametrize("levels", [1, 0, True, 2.0, "2"])
def test_levels_is_an_int_of_at_least_2(levels):
    with pytest.raises(InvalidValueError, match="levels must be an int of at least 2"):
        condition_table([], levels=levels)
