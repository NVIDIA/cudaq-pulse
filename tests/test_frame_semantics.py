# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How frames behave, from a kernel (the rules of the Frames section of the
dialect docs; test/Conversion/pulse-to-qop-frames.mlir has the IR half):
each read back from the controller export's frequency and phase per event."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest


@pytest.fixture(scope="module")
def qs():
    return pulse.quantum_system("transmon_surface17", layout=[0, 1])


def _played(kernel, qs):
    events = (
        pulse.compile(kernel, quantum_system=qs)
        .lower(qs.qudits[0])
        .export_to_controller()["events"]
    )
    return [e for e in events if e["kind"] in ("drive", "readout", "frame_update")]


def test_a_frame_change_takes_effect_at_the_next_play(qs):
    @pulse.kernel
    def shifted(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.5, 10.0))
        shift_phase(d.tone, 0.25)
        drive(d, gaussian(40, 0.5, 10.0))

    first, second = _played(shifted, qs)
    assert first["phase_rad"] == 0.0
    assert second["phase_rad"] == pytest.approx(0.25)


def test_a_frame_change_after_the_last_play_is_a_frame_update(qs):
    @pulse.kernel
    def trailing(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.5, 10.0))
        shift_phase(d.tone, 0.25)

    drive_event, update = _played(trailing, qs)
    assert drive_event["phase_rad"] == 0.0
    assert (update["kind"], update["op"], update["value"]) == ("frame_update", "shift_phase", 0.25)


def test_a_wait_does_not_advance_a_tone_s_phase(qs):
    # Unlike OpenPulse, where a delay advances the frame: on the qudit's own
    # frequency the drive after a wait has the phase it has without it.
    @pulse.kernel
    def waited(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.5, 10.0))
        wait(d, 400)
        drive(d, gaussian(40, 0.5, 10.0))

    @pulse.kernel
    def unwaited(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.5, 10.0))
        drive(d, gaussian(40, 0.5, 10.0))

    with_wait, without = _played(waited, qs), _played(unwaited, qs)
    assert with_wait[1]["phase_rad"] == without[1]["phase_rad"] == 0.0
    assert with_wait[1]["start_cycles"] > without[1]["start_cycles"]


def test_a_frame_rotates_through_a_sync(qs):
    @pulse.kernel
    def synced(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        shift_phase(d.tone, 0.5)
        readout(r, square(200, 0.05))
        sync(d, r)
        drive(d, gaussian(40, 0.5, 10.0))

    readout_event, drive_event = _played(synced, qs)
    assert drive_event["start_cycles"] >= readout_event["start_cycles"] + 200
    assert drive_event["phase_rad"] == pytest.approx(0.5)


def test_a_readout_takes_its_tone_s_frame_and_gives_a_measurement(qs):
    @pulse.kernel
    def measured(q):
        r = get_readout_line(q)
        shift_phase(r.tone, 0.75)
        m = readout(r, square(200, 0.05))
        return m

    (readout_event,) = _played(measured, qs)
    assert readout_event["phase_rad"] == pytest.approx(0.75)
