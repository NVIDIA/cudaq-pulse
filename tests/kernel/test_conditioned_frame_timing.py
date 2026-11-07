# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Frame ops in the arms of a measurement-conditioned if.

An arm that only shifts or sets a tone's phase or frequency changes a value
the measurement chooses, so a drive on that tone after the if cannot start
before the measurement is in, even with no sync between the readout and
drive lines. Each kernel here must schedule and simulate exactly like its
control, which syncs the lines after the readout.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests._handles import Q
from tests.runtime._lowering import EVOLVE_PASSES, _handles
from tests.runtime.test_simulate import _HALF_PI, _target


@pulse.kernel
def shifted_phase(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    if m:
        shift_phase(d.tone, math.pi)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def shifted_phase_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        shift_phase(d.tone, math.pi)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def set_phases(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    if m:
        set_phase(d.tone, math.pi)
    else:
        set_phase(d.tone, 0.0)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def set_phases_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        set_phase(d.tone, math.pi)
    else:
        set_phase(d.tone, 0.0)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def shifted_frequency(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    if m:
        shift_frequency(d.tone, 2.0e6)
    drive(d, square(400, _HALF_PI))


@pulse.kernel
def shifted_frequency_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        shift_frequency(d.tone, 2.0e6)
    drive(d, square(400, _HALF_PI))


@pulse.kernel
def set_frequencies(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    if m:
        set_frequency(d.tone, 5.002e9)
    else:
        set_frequency(d.tone, 4.999e9)
    drive(d, square(400, _HALF_PI))


@pulse.kernel
def set_frequencies_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        set_frequency(d.tone, 5.002e9)
    else:
        set_frequency(d.tone, 4.999e9)
    drive(d, square(400, _HALF_PI))


@pulse.kernel
def other_qudits_tone(q0, q1):
    # The drive in the arm waits for the measurement, but only on its own
    # tone: the other qudit's tone is ready when the measurement is.
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d1, square(40, _HALF_PI))
    m = readout(r0, square(200, 0.05))
    if m:
        shift_phase(d1.tone, math.pi)
        drive(d0, square(40, _HALF_PI))
    drive(d1, square(40, _HALF_PI))


@pulse.kernel
def other_qudits_tone_synced(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d1, square(40, _HALF_PI))
    m = readout(r0, square(200, 0.05))
    sync(d0, d1, r0)
    if m:
        shift_phase(d1.tone, math.pi)
        drive(d0, square(40, _HALF_PI))
    drive(d1, square(40, _HALF_PI))


@pulse.kernel
def elif_phases(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    first = readout(r, square(200, 0.05))
    second = readout(r, square(200, 0.05))
    if first:
        shift_phase(d.tone, math.pi)
    elif second:
        shift_phase(d.tone, _HALF_PI)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def elif_phases_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    first = readout(r, square(200, 0.05))
    second = readout(r, square(200, 0.05))
    sync(d, r)
    if first:
        shift_phase(d.tone, math.pi)
    elif second:
        shift_phase(d.tone, _HALF_PI)
    drive(d, square(40, _HALF_PI))


@pulse.kernel
def nested_frequencies(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    first = readout(r, square(200, 0.05))
    second = readout(r, square(200, 0.05))
    if first:
        if second:
            shift_frequency(d.tone, 2.0e6)
        else:
            set_phase(d.tone, math.pi)
    drive(d, square(400, _HALF_PI))


@pulse.kernel
def nested_frequencies_synced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, _HALF_PI))
    sync(d, r)
    first = readout(r, square(200, 0.05))
    second = readout(r, square(200, 0.05))
    sync(d, r)
    if first:
        if second:
            shift_frequency(d.tone, 2.0e6)
        else:
            set_phase(d.tone, math.pi)
    drive(d, square(400, _HALF_PI))


_PAIRS = [
    (shifted_phase, shifted_phase_synced, 1),
    (set_phases, set_phases_synced, 1),
    (shifted_frequency, shifted_frequency_synced, 1),
    (set_frequencies, set_frequencies_synced, 1),
    (other_qudits_tone, other_qudits_tone_synced, 2),
    (elif_phases, elif_phases_synced, 1),
    (nested_frequencies, nested_frequencies_synced, 1),
]


def _lower(kernel, n, schedule):
    device = pulse.device(_target(n))
    compiled = pulse.compile(kernel, passes=EVOLVE_PASSES, schedule=schedule)
    return compiled.lower(*_handles(device, [Q[i] for i in range(n)]))


def _timing(lowered):
    return [
        (e.kind, e.line, e.start_cycles, e.duration_cycles)
        for e in lowered.schedule
        if e.kind != "sync"
    ]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize(
    ("kernel", "control", "n"), _PAIRS, ids=[kernel.__name__ for kernel, _, _ in _PAIRS]
)
def test_a_frame_op_in_an_arm_waits_for_the_measurement(kernel, control, n, schedule):
    lowered = _lower(kernel, n, schedule)
    expected = _lower(control, n, schedule)
    assert _timing(lowered) == _timing(expected)
    readouts = [e for e in lowered.schedule if e.kind == "readout"]
    last = lowered.schedule[-1]
    assert last.kind == "drive"
    assert last.start_cycles >= max(e.start_cycles + e.duration_cycles for e in readouts)
    actual = pulse.simulate(lowered, shots=400, seed=5).density_matrix
    reference = pulse.simulate(expected, shots=400, seed=5).density_matrix
    np.testing.assert_allclose(actual, reference, atol=1e-12)
