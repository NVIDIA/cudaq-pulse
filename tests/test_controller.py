# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exporting a compiled kernel to a controller, and checking its limits."""

from __future__ import annotations

import json
import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Bits, qec
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import (
    FORMAT,
    ControllerLimitError,
    ControllerLimits,
    check,
    sample_waveform,
    shape_key,
)

from tests._handles import Q


@pulse.kernel
def drive_and_branch(q, r):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    d1, t1 = get_drive_line(r)
    drive(d, gaussian(16, 0.5, 4.0), t)
    shift_phase(t, 0.5)
    drive(d, drag(16, 0.25, 4.0, 0.3), t)
    sync(d, ro)
    m = readout(ro, square(100, 0.05), rt)
    sync(ro, d1)
    if m:
        drive(d1, wf_scale(square(20, 0.3), -0.5), t1)
    else:
        drive(d1, gaussian(20, 0.1, 5.0), t1)


def _program(kernel=drive_and_branch, schedule="asap", machine=None, **kwargs):
    compiled = pulse.compile(kernel, schedule=schedule, machine=machine)
    return compiled.lower(Q[0], Q[1]).export_controller(**kwargs)


def test_export_lists_every_timed_operation_with_its_waveform_and_tone():
    program = _program()
    assert program["format"] == FORMAT
    json.dumps(program)
    assert program["lines"] == [
        {"line": 0, "role": "drive", "qudit": 0},
        {"line": 1, "role": "readout", "qudit": 0},
        {"line": 2, "role": "drive", "qudit": 1},
    ]
    kinds = [(e["kind"], e["line"], e["start_vtu"], e["duration_vtu"]) for e in program["events"]]
    assert kinds == [
        ("drive", 0, 0, 16),
        ("drive", 0, 16, 16),
        ("readout", 1, 32, 100),
        ("drive", 2, 132, 20),
        ("drive", 2, 132, 20),
    ]
    first, second, readout, then, otherwise = program["events"]
    assert first["waveform"] == {
        "shape": "gaussian",
        "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0},
    }
    assert second["waveform"]["shape"] == "drag"
    assert second["waveform"]["params"]["beta"] == pytest.approx(0.3)
    # The frame change before a drive is the frame it plays in.
    assert first["phase_rad"] == 0.0
    assert second["phase_rad"] == pytest.approx(0.5)
    assert first["frequency_hz"] == pytest.approx(5.0e9)
    assert then["frequency_hz"] == pytest.approx(5.1e9)
    assert readout["amplitude"] == {"re": 0.05, "im": 0.0}
    assert then["waveform"]["shape"] == "scale"
    assert then["amplitude"]["re"] == pytest.approx(-0.15)


def test_export_marks_the_branch_and_the_measurement_it_depends_on():
    events = _program()["events"]
    readout, then, otherwise = events[2], events[3], events[4]
    assert readout["conditions"] == []
    assert then["conditions"] == [
        {"branch": 0, "arm": True, "readouts": [2], "answers": [], "test": "is_set"}
    ]
    assert otherwise["conditions"][0]["arm"] is False
    assert otherwise["conditions"][0]["readouts"] == [2]


def test_export_samples_envelopes_at_a_multiple_of_the_clock():
    program = _program(sample_rate_ghz=4 * pulse.get_device().clock_ghz)
    first = program["events"][0]
    samples = np.asarray(first["samples"]["re"]) + 1j * np.asarray(first["samples"]["im"])
    assert len(samples) == 16 * 4
    expected = sample_waveform(first["waveform"], 16, 4)
    np.testing.assert_allclose(samples, expected)
    t = np.arange(64) / 4 - 8
    np.testing.assert_allclose(samples.real, 0.5 * np.exp(-0.5 * t**2 / 16.0))
    with pytest.raises(ValueError, match="whole multiple"):
        _program(sample_rate_ghz=1.5 * pulse.get_device().clock_ghz)


def test_drag_samples_carry_the_derivative_in_quadrature():
    node = {
        "shape": "drag",
        "params": {"duration": 10, "amplitude": 1.0, "sigma": 2.0, "beta": 0.5},
    }
    samples = sample_waveform(node, 10)
    t = np.arange(10) - 5.0
    g = np.exp(-0.5 * t**2 / 4.0)
    np.testing.assert_allclose(samples, g - 1j * 0.5 * t * g / 4.0)


def test_shape_key_ignores_overall_amplitude_only():
    a = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0}}
    b = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.1, "sigma": 4.0}}
    c = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 3.0}}
    scaled = {"shape": "scale", "params": {"factor": -2.0}, "of": [b]}
    assert shape_key(a) == shape_key(b) == shape_key(scaled)
    assert shape_key(a) != shape_key(c)


@pulse.kernel
def tone_in_one_arm(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    sync(d, r)
    if m:
        drive(d, square(20, 0.3), t)
    drive(d, gaussian(20, 0.2, 5.0), t)


@pulse.kernel
def phase_in_one_arm(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(20, 0.2, 5.0), t)
    m = readout(r, square(100, 0.05), rt)
    sync(d, r)
    if m:
        shift_phase(t, 0.5)
        drive(d, square(20, 0.3), t)
    drive(d, gaussian(20, 0.2, 5.0), t)


def test_a_frame_only_one_arm_touches_is_unchanged_in_the_other():
    last = pulse.compile(tone_in_one_arm).lower(Q[0]).export_controller()["events"][-1]
    assert last["frequency_hz"] == pytest.approx(5.0e9)
    assert last["phase_rad"] == 0.0
    last = pulse.compile(phase_in_one_arm).lower(Q[0]).export_controller()["events"][-1]
    assert last["frequency_hz"] is None and last["phase_rad"] is None


@pulse.kernel
def two_readouts(q, r):
    ra, ta = get_readout_line(q)
    rb, tb = get_readout_line(r)
    readout(ra, square(40, 0.05), ta)
    readout(rb, square(40, 0.05), tb)


@pulse.kernel
def close_drives(q, r):
    d, t = get_drive_line(q)
    drive(d, square(5, 0.1), t)
    drive(d, gaussian(8, 0.1, 2.0), t)


def test_limits_report_issue_spacing_acquirers_and_feedback():
    limits = ControllerLimits(issue_vtu=4, min_wait_vtu=4)
    found = check(pulse.compile(close_drives).lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "starts 5 cycles after" in found[0]
    limits = ControllerLimits(acquirers={0: 0, 1: 0}, acquirer_hold_vtu=30)
    found = check(pulse.compile(two_readouts, schedule="asap").lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "acquirer 0" in found[0]
    # Serialized by the scheduler, the readouts fit.
    machine = MachineModel(max_concurrent_readouts=1, line_switch_penalty_vtu=30)
    assert (
        check(
            pulse.compile(two_readouts, schedule="rcp", machine=machine).lower(Q[0], Q[1]), limits
        )
        == []
    )
    limits = ControllerLimits(feedback_latency_vtu=50)
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    assert len(check(lowered, limits)) == 2
    machine = MachineModel(readout_latency_vtu=50)
    lowered = pulse.compile(drive_and_branch, schedule="rcp", machine=machine).lower(Q[0], Q[1])
    assert check(lowered, limits) == []


def test_limits_report_clock_tones_memory_and_size():
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    clock = lowered.target.clock_ghz
    found = check(lowered, ControllerLimits(clock_ghz=clock / 2))
    assert len(found) == 1 and "clock" in found[0]
    assert check(lowered, ControllerLimits(tones_per_line=1)) == []
    # Line 0 plays two shapes of 16 cycles, the readout line one of 100,
    # line 2 two of 20.
    found = check(lowered, ControllerLimits(samples_per_vtu=2, waveform_samples=64))
    assert [f.split(":")[0] for f in found] == ["line 1", "line 2"]
    found = check(lowered, ControllerLimits(max_events=4))
    assert "at most 4" in found[0]


def test_limits_on_the_machine_model_fail_the_compilation():
    machine = MachineModel(controller=ControllerLimits(issue_vtu=4, min_wait_vtu=4))
    with pytest.raises(ControllerLimitError, match="starts 5 cycles after") as caught:
        pulse.compile(close_drives, machine=machine).lower(Q[0], Q[1])
    assert len(caught.value.violations) == 1
    assert isinstance(caught.value, pulse.CompilationError)
    with pytest.raises(TypeError, match="ControllerLimits"):
        MachineModel(controller={"issue_vtu": 4})  # type: ignore[arg-type]


def test_limits_are_part_of_the_machine_model_identity():
    a = MachineModel(controller=ControllerLimits(issue_vtu=4, acquirers={0: 0}))
    b = MachineModel(controller=ControllerLimits(issue_vtu=4, acquirers={0: 0}))
    c = MachineModel(controller=ControllerLimits(issue_vtu=2))
    assert a == b and hash(a) == hash(b)
    assert a != c and a != MachineModel()
    with pytest.raises(ValueError, match="issue_vtu"):
        ControllerLimits(issue_vtu=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        ControllerLimits(acquirer_hold_vtu=-1)


decoder = qec.Decoder(0, syndromes=1, corrections=1, max_latency_ns=400, decode=lambda b: [b[0]])


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=400)
def check_bit(bits):
    return [bits[0]]


@pulse.kernel
def decoded(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    qec.enqueue_syndromes(decoder, [m], 0)
    fix = qec.get_corrections(decoder).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3), t)


@pulse.kernel
def timed_out(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    ok, fix = check_bit.submit([m]).result(timeout_ns=300)
    sync(d, ro)
    if ok:
        drive(d, square(20, 0.3), t)


def test_export_carries_calls_with_their_measurements_and_waits():
    program = pulse.compile(decoded, schedule="asap").lower(Q[0]).export_controller()
    submits = [e for e in program["events"] if e["kind"] == "submit"]
    answer = next(e for e in program["events"] if e["kind"] == "answer")
    readout = next(e for e in program["events"] if e["kind"] == "readout")
    enqueue, get = submits
    assert enqueue["function"].startswith("enqueue_syndromes")
    assert enqueue["measurements"] == [readout["index"]]
    assert get["measurements"] == []
    assert answer["wait"] == "bounded" and answer["timeout_ns"] is None
    fix = next(e for e in program["events"] if e["conditions"])
    assert fix["conditions"][0]["answers"] == [answer["index"]]
    assert fix["conditions"][0]["test"] == "answer"
    assert {f["name"] for f in program["manifest"]["functions"]} == {
        enqueue["function"],
        get["function"],
    }


def test_a_controller_without_answer_timeouts_rejects_a_timed_wait():
    limits = ControllerLimits(answer_timeouts=False)
    assert check(pulse.compile(decoded).lower(Q[0]), limits) == []
    lowered = pulse.compile(timed_out).lower(Q[0])
    answer = next(e for e in lowered.export_controller()["events"] if e["kind"] == "answer")
    assert answer["wait"] == "timeout" and answer["timeout_ns"] == pytest.approx(300)
    with pytest.raises(ControllerLimitError, match="cannot stop waiting"):
        pulse.compile(timed_out, machine=MachineModel(controller=limits)).lower(Q[0])


@pulse.kernel
def ramsey(q, phase: float):
    d, t = get_drive_line(q)
    drive(d, gaussian(16, 0.25, 4.0), t)
    shift_phase(t, phase)
    drive(d, gaussian(16, 0.25, 4.0), t)


def test_float_arguments_are_exported_at_their_values():
    lowered = pulse.compile(ramsey).lower(Q[0], 0.75)
    assert lowered.generic is not None
    with pytest.raises(ValueError, match="float parameters"):
        lowered.generic.export_controller()
    program = lowered.export_controller()
    assert program["events"][1]["phase_rad"] == pytest.approx(0.75)
    assert not math.isnan(program["events"][0]["phase_rad"])
