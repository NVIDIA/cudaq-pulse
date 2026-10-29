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
from cudaq_pulse.classical import Bits, Float, qec
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
    assert program["format"] == FORMAT == "cudaq-pulse.controller/1"
    assert json.loads(json.dumps(program)) == program
    assert program["lines"] == [
        {"line": 0, "role": "drive", "qudit": 0},
        {"line": 1, "role": "readout", "qudit": 0},
        {"line": 2, "role": "drive", "qudit": 1},
    ]
    kinds = [(e["kind"], e["line"], e["start_cycles"], e["end_cycles"]) for e in program["events"]]
    assert kinds == [
        ("drive", 0, 0, 16),
        ("drive", 0, 16, 32),
        ("sync", None, 32, 32),
        ("readout", 1, 32, 132),
        ("sync", None, 132, 132),
        ("drive", 2, 132, 152),
        ("drive", 2, 132, 152),
    ]
    assert [e["index"] for e in program["events"]] == list(range(7))
    first, second, _, readout, _, then, _ = program["events"]
    assert first["waveform"] == {
        "shape": "gaussian",
        "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0},
    }
    assert first["duration_cycles"] == 16 and first["iteration"] == []
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
    readout, then, otherwise = events[3], events[5], events[6]
    assert readout["conditions"] == []
    assert then["conditions"] == [
        {"branch": 0, "arm": True, "readouts": [3], "answers": [], "test": "is_set"}
    ]
    assert otherwise["conditions"] == [
        {"branch": 0, "arm": False, "readouts": [3], "answers": [], "test": "is_set"}
    ]


def test_export_names_the_lines_a_sync_joins():
    events = _program()["events"]
    assert events[2]["lines"] == [0, 1] and events[4]["lines"] == [1, 2]
    assert events[2]["qudit"] is None and events[2]["duration_cycles"] == 0


@pulse.kernel
def staggered(q, r):
    d, t = get_drive_line(q)
    d1, t1 = get_drive_line(r)
    ro, rt = get_readout_line(q)
    for _ in range(3):
        drive(d, gaussian(24, 0.5, 6.0), t)
        drive(d1, square(8, 0.1), t1)
        sync(d, d1, ro)
        readout(ro, square(40, 0.05), rt)
        drive(d1, square(12, 0.1), t1)
        sync(d, d1, ro)


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_a_sync_is_a_barrier_between_the_events_of_its_lines(schedule):
    events = _program(staggered, schedule=schedule)["events"]
    syncs = [e for e in events if e["kind"] == "sync"]
    assert len(syncs) == 6
    assert [s["iteration"] for s in syncs] == [[0], [0], [1], [1], [2], [2]]
    for barrier in syncs:
        joined = set(barrier["lines"])
        assert joined == {0, 1, 2}
        for event in events:
            if event["kind"] == "sync" or event["line"] not in joined:
                continue
            if event["index"] < barrier["index"]:
                assert event["end_cycles"] <= barrier["start_cycles"], (barrier, event)
            else:
                assert event["start_cycles"] >= barrier["start_cycles"], (barrier, event)


@pulse.kernel
def detuned(q, r):
    d, t = get_drive_line(q)
    t = shift_frequency(t, 2.0e6)
    drive(d, square(16, 0.1), t)
    wait(d, 100)
    drive(d, square(16, 0.1), t)
    t = shift_phase(t, 0.25)
    drive(d, square(16, 0.1), t)


def test_a_detuned_tone_gains_phase_between_its_drives():
    # fold_phases folds the phase shift into the last drive.
    lowered = pulse.compile(detuned, schedule="asap").lower(Q[0], Q[1])
    assert "frame_phase_offset" in lowered.mlir()
    program = lowered.export_controller()
    clock = program["clock_ghz"]
    drives = [e for e in program["events"] if e["kind"] == "drive"]
    assert [e["start_cycles"] for e in drives] == [0, 116, 132]
    for event, extra in zip(drives, (0.0, 0.0, 0.25), strict=True):
        assert event["frequency_hz"] == pytest.approx(5.0e9 + 2.0e6)
        expected = 2 * math.pi * 2.0e6 * event["start_cycles"] / clock * 1e-9 + extra
        assert event["phase_rad"] == pytest.approx(expected)


@pulse.kernel
def tone_in_one_arm(q, r):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    sync(d, ro)
    if m:
        drive(d, square(20, 0.3), t)
    drive(d, gaussian(20, 0.2, 5.0), t)


@pulse.kernel
def phase_in_one_arm(q, r):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    drive(d, gaussian(20, 0.2, 5.0), t)
    m = readout(ro, square(100, 0.05), rt)
    sync(d, ro)
    if m:
        shift_phase(t, 0.5)
        drive(d, square(20, 0.3), t)
    drive(d, gaussian(20, 0.2, 5.0), t)


def test_a_frame_only_one_arm_touches_is_unchanged_in_the_other():
    last = _program(tone_in_one_arm)["events"][-1]
    assert last["frequency_hz"] == pytest.approx(5.0e9)
    assert last["phase_rad"] == 0.0
    last = _program(phase_in_one_arm)["events"][-1]
    assert last["frequency_hz"] is None and last["phase_rad"] is None


@pulse.kernel
def idle_readout_line(q, r):
    ro, rt = get_readout_line(q)
    wait(ro, 10)


def test_a_line_is_listed_with_its_role_even_when_it_only_waits():
    program = _program(idle_readout_line)
    assert program["lines"] == [{"line": 0, "role": "readout", "qudit": 0}]
    (event,) = program["events"]
    assert event["kind"] == "wait" and event["end_cycles"] == 10
    assert "waveform" not in event


def test_export_samples_envelopes_at_a_multiple_of_the_clock():
    program = _program(sample_rate_ghz=4 * pulse.device().clock_ghz)
    first = program["events"][0]
    samples = np.asarray(first["samples"]["re"]) + 1j * np.asarray(first["samples"]["im"])
    assert len(samples) == 16 * 4
    expected = sample_waveform(first["waveform"], 16, 4)
    np.testing.assert_allclose(samples, expected)
    t = np.arange(64) / 4 - 8
    np.testing.assert_allclose(samples.real, 0.5 * np.exp(-0.5 * t**2 / 16.0))
    assert "samples" not in program["events"][2]
    with pytest.raises(ValueError, match="whole multiple"):
        _program(sample_rate_ghz=1.5 * pulse.device().clock_ghz)


def test_drag_samples_carry_the_derivative_in_quadrature():
    node = {
        "shape": "drag",
        "params": {"duration": 10, "amplitude": 1.0, "sigma": 2.0, "beta": 0.5},
    }
    samples = sample_waveform(node, 10)
    t = np.arange(10) - 5.0
    g = np.exp(-0.5 * t**2 / 4.0)
    np.testing.assert_allclose(samples, g - 1j * 0.5 * t * g / 4.0)


def test_gaussian_square_samples_are_flat_between_their_edges():
    node = {
        "shape": "gaussian_square",
        "params": {"duration": 40, "amplitude": 0.5, "sigma": 3.0, "risefall": 10},
    }
    samples = sample_waveform(node, 40).real
    np.testing.assert_allclose(samples[10:30], 0.5)
    np.testing.assert_allclose(samples[:10], 0.5 * np.exp(-0.5 * (np.arange(10) - 10) ** 2 / 9))
    np.testing.assert_allclose(samples[30:], 0.5 * np.exp(-0.5 * np.arange(10) ** 2 / 9))


def ramp(t):
    return t / 8


@pulse.kernel
def sampled_envelopes(q, r):
    d, t = get_drive_line(q)
    drive(d, custom(8, ramp), t)
    drive(d, custom_samples([0.0, 0.5, 1.0, 0.5]), t)


def test_custom_envelopes_are_exported_as_their_samples():
    program = _program(sampled_envelopes, sample_rate_ghz=pulse.device().clock_ghz)
    traced, given = program["events"]
    expected = [(k + 0.5) / 8 for k in range(8)]
    assert traced["waveform"]["shape"] == "custom_samples"
    assert traced["waveform"]["samples"] == pytest.approx(expected)
    assert traced["samples"]["re"] == pytest.approx(expected)
    assert given["waveform"] == {"shape": "custom_samples", "samples": [0.0, 0.5, 1.0, 0.5]}
    assert traced["amplitude"]["re"] == pytest.approx(expected[-1])


def test_a_waveform_a_python_function_plays_cannot_be_sampled():
    node = {"shape": "custom", "name": "ramp_envelope", "params": {"duration": 32.0}}
    with pytest.raises(ValueError, match="cannot sample a 'custom' waveform"):
        sample_waveform(node, 32)


def test_shape_key_ignores_overall_amplitude_only():
    a = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0}}
    b = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.1, "sigma": 4.0}}
    c = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 3.0}}
    scaled = {"shape": "scale", "params": {"factor": -2.0}, "of": [b]}
    assert shape_key(a) == shape_key(b) == shape_key(scaled)
    assert shape_key(a) != shape_key(c)
    one = {"shape": "custom", "name": "one", "params": {"duration": 8.0}}
    two = {"shape": "custom", "name": "two", "params": {"duration": 8.0}}
    assert shape_key(one) != shape_key(two)


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
    limits = ControllerLimits(issue_cycles=4, min_wait_cycles=4)
    found = check(pulse.compile(close_drives).lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "starts 5 cycles after" in found[0]
    limits = ControllerLimits(acquirers={0: 0, 1: 0}, acquirer_hold_cycles=30)
    found = check(pulse.compile(two_readouts, schedule="asap").lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "acquirer 0" in found[0]
    # Serialized by the scheduler, the readouts fit.
    machine = MachineModel(max_concurrent_readouts=1, line_switch_penalty_cycles=30)
    lowered = pulse.compile(two_readouts, schedule="rcp", machine=machine).lower(Q[0], Q[1])
    assert check(lowered, limits) == []
    limits = ControllerLimits(feedback_latency_cycles=50)
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    assert len(check(lowered, limits)) == 2
    machine = MachineModel(readout_latency_cycles=50)
    lowered = pulse.compile(drive_and_branch, schedule="rcp", machine=machine).lower(Q[0], Q[1])
    assert check(lowered, limits) == []


def test_limits_report_clock_tones_memory_and_size():
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    clock = lowered.target.clock_ghz
    # Every operation starts and lasts whole cycles of a controller at half
    # the clock; at an eighth, the readout and the two drives after it do not.
    assert check(lowered, ControllerLimits(clock_ghz=clock / 2)) == []
    found = check(lowered, ControllerLimits(clock_ghz=clock / 8))
    assert len(found) == 3 and all("moves in timesteps of 8 cycles" in f for f in found)
    assert check(lowered, ControllerLimits(tones_per_line=1)) == []
    # Line 0 plays two shapes of 16 cycles, the readout line one of 100,
    # line 2 two of 20.
    found = check(lowered, ControllerLimits(sample_rate_ghz=4.0, waveform_samples=64))
    assert [f.split(":")[0] for f in found] == ["line 1", "line 2"]
    found = check(lowered, ControllerLimits(max_events=4))
    assert "5 drives and readouts" in found[0] and "at most 4" in found[0]
    # Both arms of a branch play on one line at one time without conflict.
    assert check(lowered, ControllerLimits(issue_cycles=4)) == []


def test_limits_on_the_machine_model_fail_the_compilation():
    machine = MachineModel(controller=ControllerLimits(issue_cycles=4, min_wait_cycles=4))
    with pytest.raises(ControllerLimitError, match="starts 5 cycles after") as caught:
        pulse.compile(close_drives, machine=machine).lower(Q[0], Q[1])
    assert len(caught.value.violations) == 1
    assert isinstance(caught.value, pulse.CompilationError)
    with pytest.raises(TypeError, match="ControllerLimits"):
        MachineModel(controller={"issue_cycles": 4})  # type: ignore[arg-type]


@pulse.kernel
def strong_drives(q, amplitude: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, square(16, 1.5), t)
    drive(d, gaussian(16, amplitude, 4.0), t)
    drive(d, wf_scale(drag(16, 0.5, 4.0, 16.0), 1.5), t)
    readout(r, square(32, 0.1), rt)


def test_only_the_controller_limits_the_amplitude():
    lowered = pulse.compile(strong_drives).lower(Q[0], 3.0)
    assert lowered.export_controller()["events"][0]["amplitude"]["re"] == 1.5
    found = check(lowered, ControllerLimits(max_amplitude=1.0))
    assert found[0] == (
        "line 0 (drive of qudit 0): the drive at 0 peaks at amplitude 1.5; the controller "
        "plays at most 1 (ControllerLimits.max_amplitude)"
    )
    # The quadrature of the DRAG pulse takes its peak above 0.75.
    assert [f.split(": the ")[1].split(" peaks")[0] for f in found] == [
        "drive at 0",
        "drive at 16",
        "drive at 32",
    ]
    assert check(lowered, ControllerLimits(max_amplitude=3.0)) == []
    assert "max_amplitude" not in ControllerLimits(clock_ghz=1.0).as_dict()
    machine = MachineModel(controller=ControllerLimits(max_amplitude=2.0))
    compiled = pulse.compile(strong_drives, machine=machine)
    assert compiled.lower(Q[0], 0.5).export_controller()["limits"]["max_amplitude"] == 2.0
    with pytest.raises(ControllerLimitError, match=r"peaks at amplitude 3; .* at most 2 \("):
        compiled.lower(Q[0], 3.0).export_controller()
    for wrong in (0, -1.0, math.inf, math.nan, True, "1"):
        with pytest.raises(pulse.InvalidValueError, match="max_amplitude must be a positive"):
            ControllerLimits(max_amplitude=wrong)  # type: ignore[arg-type]


def test_a_limit_error_lists_ten_violations_and_counts_the_rest():
    error = ControllerLimitError([f"violation {k}" for k in range(13)])
    assert "violation 9" in str(error) and "violation 10" not in str(error)
    assert "and 3 more" in str(error) and len(error.violations) == 13


def test_limits_are_part_of_the_machine_model_identity():
    a = MachineModel(controller=ControllerLimits(issue_cycles=4, acquirers={0: 0}))
    b = MachineModel(controller=ControllerLimits(issue_cycles=4, acquirers={0: 0}))
    c = MachineModel(controller=ControllerLimits(issue_cycles=2))
    assert a == b and hash(a) == hash(b)
    assert a != c and a != MachineModel()
    assert ControllerLimits(acquirers={0: 0, 1: 0}).as_dict()["acquirers"] == {"0": 0, "1": 0}
    with pytest.raises(ValueError, match="issue_cycles"):
        ControllerLimits(issue_cycles=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        ControllerLimits(acquirer_hold_cycles=-1)
    with pytest.raises(ValueError, match="positive or None"):
        ControllerLimits(waveform_samples=0)


def test_the_limits_travel_with_the_program():
    limits = ControllerLimits(issue_cycles=1, tones_per_line=2)
    program = _program(machine=MachineModel(controller=limits))
    assert program["limits"] == limits.as_dict()
    assert _program()["limits"] is None


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


def test_a_kernel_with_float_parameters_is_checked_when_compiled():
    machine = MachineModel(controller=ControllerLimits(issue_cycles=4, min_wait_cycles=40))
    with pytest.raises(ControllerLimitError, match="starts 16 cycles after"):
        pulse.compile(ramsey, machine=machine).lower(Q[0], 0.75)


# ── Classical calls ──────────────────────────────────────────────────────────

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
    ok, fix = check_bit.submit([m]).result(timeout=300 * pulse.ns)
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
    assert all(e["end_cycles"] == e["start_cycles"] + e["duration_cycles"] for e in submits)
    fix = next(e for e in program["events"] if e["conditions"])
    assert fix["conditions"][0]["answers"] == [answer["index"]]
    assert fix["conditions"][0]["test"] == "answer"
    assert {f["name"] for f in program["manifest"]["functions"]} == {
        enqueue["function"],
        get["function"],
    }
    # Links are not lines.
    assert all(line["role"] in ("drive", "readout") for line in program["lines"])


def test_events_after_a_variable_wait_are_in_its_epoch():
    @pulse.classical(results=(Float(),), max_latency_ns=100)
    def tau():
        return 8.0

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(20, 0.1), t)
        wait(d, tau.submit().result(), bounds=(4, 40))
        drive(d, square(20, 0.1), t)

    program = pulse.compile(k).lower(Q[0]).export_controller()
    drives = [e for e in program["events"] if e["kind"] == "drive"]
    assert [e["epoch"] for e in drives] == [0, 1]
    assert len(program["epochs"]) == 2


def test_a_controller_without_answer_timeouts_rejects_a_timed_wait():
    limits = ControllerLimits(answer_timeouts=False)
    assert check(pulse.compile(decoded).lower(Q[0]), limits) == []
    lowered = pulse.compile(timed_out).lower(Q[0])
    answer = next(e for e in lowered.export_controller()["events"] if e["kind"] == "answer")
    assert answer["wait"] == "timeout" and answer["timeout_ns"] == pytest.approx(300)
    with pytest.raises(ControllerLimitError, match="cannot stop waiting"):
        pulse.compile(timed_out, machine=MachineModel(controller=limits)).lower(Q[0])


# ── Values known at run time ────────────────────────────────────────────────


@pulse.classical(args=(Bits(1),), results=(Float(), Float(), Float()), max_latency_ns=200)
def settings(bits):
    return 0.1 * bits[0], 1.0e5 * bits[0], 40.0 + bits[0]


@pulse.kernel
def answered(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    amp, detuning, delay = settings.submit([m]).result()
    shift_frequency(t, detuning)
    drive(d, square(20, amp * 0.5), t)
    wait(d, delay, bounds=(8, 400))
    shift_phase(t, amp)
    drive(d, square(20, 0.2), t)


def test_values_from_answers_are_exported_as_expressions():
    program = pulse.compile(answered).lower(Q[0]).export_controller()
    events = program["events"]
    answer = next(e for e in events if e["kind"] == "answer")["index"]
    first, second = (e for e in events if e["kind"] == "drive")
    assert first["amplitude"] is None and first["frequency_hz"] is None
    assert first["waveform"]["runtime"] == {
        "amplitude": {
            "op": "arith.mulf",
            "of": [{"answer": answer, "result": 0}, {"const": 0.5}],
        }
    }
    [shift] = first["frame_updates"]
    [phase] = second["frame_updates"]
    assert shift == {
        "id": shift["id"],
        "op": "shift_frequency",
        "value": {"answer": answer, "result": 1},
        "conditions": [],
    }
    assert phase == {
        "id": phase["id"],
        "op": "shift_phase",
        "value": {"answer": answer, "result": 0},
        "conditions": [],
    }
    assert phase["id"] != shift["id"]
    assert "runtime" not in second["waveform"]
    readout = next(e for e in events if e["kind"] == "readout")
    # The readout tone's frame does not depend on run time.
    assert first["tone"] == second["tone"] and "tone" not in readout
    wait = next(e for e in events if e["kind"] == "wait")
    assert (wait["min_cycles"], wait["max_cycles"]) == (8, 400)

    def answers(expr):
        if "answer" in expr:
            return [(expr["answer"], expr["result"])]
        return [a for child in expr.get("of", []) for a in answers(child)]

    assert answers(wait["length"]) == [(answer, 2)]
    submit = next(e for e in events if e["kind"] == "submit")
    assert submit["operands"] == [{"readout": readout["index"]}]
    assert json.loads(json.dumps(events)) == events


@pulse.classical(args=(Bits(2), Float()), results=(Float(),), max_latency_ns=200)
def scaled(bits, value):
    return value * (bits[0] + bits[1])


@pulse.kernel
def chained(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    a = readout(ro, square(100, 0.05), rt)
    b = readout(ro, square(100, 0.05), rt)
    first = scaled.submit([a, b], 0.5).result()
    second = scaled.submit([b, a], first * 2.0).result()
    drive(d, square(20, second), t)


def test_a_submission_carries_its_arguments_in_order():
    events = pulse.compile(chained).lower(Q[0]).export_controller()["events"]
    a, b = (e["index"] for e in events if e["kind"] == "readout")
    first, second = (e for e in events if e["kind"] == "submit")
    answer = next(e for e in events if e["kind"] == "answer")["index"]
    assert first["operands"] == [{"readout": a}, {"readout": b}, {"const": 0.5}]
    assert second["operands"][:2] == [{"readout": b}, {"readout": a}]
    assert second["operands"][2] == {
        "op": "arith.mulf",
        "of": [{"answer": answer, "result": 0}, {"const": 2.0}],
    }


@pulse.kernel
def chosen_by_readout(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    if m:
        value = 0.3
    else:
        value = 0.1
    shift_phase(t, value)
    drive(d, square(20, value), t)


def test_a_value_a_readout_selects_at_run_time_is_exported_as_its_select(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
    events = pulse.compile(chosen_by_readout).lower(Q[0]).export_controller()["events"]
    readout = next(e for e in events if e["kind"] == "readout")["index"]
    [drive] = (e for e in events if e["kind"] == "drive")
    select = {
        "op": "arith.select",
        "of": [
            {"op": "pulse.is_set", "of": [{"readout": readout}]},
            {"const": 0.3},
            {"const": 0.1},
        ],
    }
    assert drive["conditions"] == []
    assert drive["waveform"]["params"]["amplitude"] is None
    assert drive["waveform"]["runtime"] == {"amplitude": select}
    [shift] = drive["frame_updates"]
    assert (shift["op"], shift["value"]) == ("shift_phase", select)


@pulse.kernel
def frame_in_arms(q, r):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    shift_phase(t, 0.25)
    m = readout(ro, square(100, 0.05), rt)
    sync(d, ro)
    if m:
        drive(d, square(20, 0.1), t)
        shift_phase(t, 0.5)
    else:
        shift_frequency(t, 1.0e6)
    drive(d, square(20, 0.1), t)


def test_frame_updates_carry_the_arm_they_happen_in():
    program = _program(frame_in_arms)
    readout = next(e for e in program["events"] if e["kind"] == "readout")["index"]
    in_arm, after = (e for e in program["events"] if e["kind"] == "drive")
    assert [(u["op"], u["value"]) for u in in_arm["frame_updates"]] == [("shift_phase", 0.25)]
    # The drive in the if arm took the shift from before the branch, so
    # after the branch it is pending only on the else arm's path.
    ops = [
        (u["op"], u["value"], [(c["readouts"], c["arm"]) for c in u["conditions"]])
        for u in after["frame_updates"]
    ]
    assert ops == [
        ("shift_phase", 0.25, [([readout], False)]),
        ("shift_phase", 0.5, [([readout], True)]),
        ("shift_frequency", 1.0e6, [([readout], False)]),
    ]
    # The shift from before the branch is one change, listed on both paths.
    assert after["frame_updates"][0]["id"] == in_arm["frame_updates"][0]["id"]
    assert len({u["id"] for u in after["frame_updates"]}) == 3
    assert all("length" not in e for e in program["events"] if e["kind"] == "wait")


@pulse.kernel
def decided_at_compile_time(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    for k in range(3):
        shift_phase(t, 0.1 * k)
        drive(d, square(20, 0.1 * (k + 1)), t)
    m = readout(ro, square(100, 0.05), rt)
    sync(d, ro)
    if m:
        drive(d, square(20, 0.1), t)
    wait(d, 16)
    drive(d, square(20, 0.1), t)


def test_a_program_computing_nothing_at_run_time_exports_no_run_time_keys():
    events = pulse.compile(decided_at_compile_time).lower(Q[0]).export_controller()["events"]
    keys = {"tone", "frame_updates", "phase_offset_rad", "length", "operands"}
    assert all(not keys & e.keys() for e in events)
    assert all("runtime" not in (e.get("waveform") or {}) for e in events)
