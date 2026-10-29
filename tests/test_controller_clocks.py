# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Clocks of a control system: the program clock its controllers give a
device, and the reference clock they lock to."""

from __future__ import annotations

import json

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import DeviceDefinitionError

from tests._handles import ladder_target


def _system(*clocks, **kwargs):
    """A control system of one controller per clock, each playing the drive
    and readout lines of two qudits of four."""
    return pulse.ControlSystem(
        controllers=[
            pulse.Controller(
                drive_lines=[2 * k, 2 * k + 1], readout_lines=[2 * k, 2 * k + 1], clock_ghz=c
            )
            for k, c in enumerate(clocks)
        ],
        **kwargs,
    )


def _device(*clocks, system=None, **config):
    control = system or _system(*clocks)
    return pulse.device({"target": ladder_target(4), "control_system": control, **config})


# ── The program clock ────────────────────────────────────────────────────────


def test_the_program_clock_is_the_fastest_controller_clock_the_others_divide():
    assert _device(1.0, 0.5).clock_ghz == 1.0
    assert _device(0.25, 1.0).clock_ghz == 1.0
    # A controller without a clock runs on the program clock.
    assert _device(None, 0.5).clock_ghz == 0.5
    # Without any controller clock, the quantum system's.
    assert _device(None, None).clock_ghz == ladder_target(4).clock_ghz


def test_uneven_controller_clocks_need_a_declared_program_clock():
    with pytest.raises(DeviceDefinitionError) as caught:
        _device(2.4, 1.0)
    [problem] = caught.value.problems
    assert problem.path == "clock_ghz"
    # Every clock that works, never one picked for the user.
    assert "set 'clock_ghz' in its config to one of 12 GHz" in problem.message
    assert "2.4 GHz (controllers[0]'s; then controllers[1] every 12 program cycles)" in (
        problem.message
    )
    assert _device(2.4, 1.0, clock_ghz=12.0).clock_ghz == 12.0
    assert _device(2.4, 1.0, clock_ghz=1.0).clock_ghz == 1.0


def test_a_declared_program_clock_has_a_whole_cycle_ratio_to_every_controller():
    with pytest.raises(DeviceDefinitionError, match="no whole-cycle ratio") as caught:
        _device(1.0, 0.5, clock_ghz=3.1415926)
    assert [p.path for p in caught.value.problems] == ["clock_ghz"]
    assert _device(1.0, 0.5, clock_ghz=2.0).clock_ghz == 2.0


def test_a_device_built_directly_derives_its_program_clock_too():
    target = ladder_target(4)
    assert pulse.Device(target, name="t", control_system=_system(1.0, 0.5)).clock_ghz == 1.0
    with pytest.raises(DeviceDefinitionError, match="do not divide evenly"):
        pulse.Device(target, name="t", control_system=_system(2.4, 1.0))


def test_a_derived_program_clock_round_trips_through_the_system_file():
    device = _device(1.0, 0.5)
    again = pulse.device(json.loads(json.dumps(device.as_dict())))
    assert again.clock_ghz == 1.0 and again == device


# ── The reference clock ──────────────────────────────────────────────────────


def test_every_controller_clock_is_a_whole_multiple_of_the_reference_clock():
    assert _device(1.0, 0.5, system=_system(1.0, 0.5, reference_clock_ghz=0.25)).clock_ghz == 1.0
    with pytest.raises(DeviceDefinitionError) as caught:
        _device(system=_system(1.0, None, reference_clock_ghz=0.3))
    paths = [p.path for p in caught.value.problems]
    # The second runs on the program clock, 1 GHz: neither is a multiple.
    assert paths == [
        "control_system.controllers[0].clock_ghz",
        "control_system.controllers[1].clock_ghz",
    ]
    assert "its 1 GHz clock (the program clock) is not a whole multiple of the 0.3 GHz" in str(
        caught.value
    )
