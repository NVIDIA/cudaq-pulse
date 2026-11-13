# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waveform ids and parameter roles in format /2, and the waveform memory
model of ``Controller``."""

from __future__ import annotations

import json
import tomllib

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, Processor
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import ControllerLimitError, WaveformMemoryScope, check, unroll

from tests._handles import Q


@pulse.kernel
def tuned(q, amp: float, phi: float, df: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, amp, 10.0))
    shift_phase(d.tone, phi)
    drive(d, gaussian(40, 0.5, 10.0))
    shift_frequency(d.tone, df)
    drive(d, wf_scale(drag(40, 0.3, 10.0, 0.2), amp))
    drive(d, square(16, 0.1))


@pulse.kernel
def widths(q, sigma: float, wait_cycles: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, 0.5, sigma))
    wait(d, wait_cycles, bounds=(0, 100))
    drive(d, gaussian(40, 0.5, 10.0))


@pulse.kernel
def two_lines(a, b):
    d = get_drive_line(a)
    e = get_drive_line(b)
    drive(d, gaussian(40, 0.5, 10.0))
    drive(e, gaussian(40, 0.25, 10.0))
    drive(e, square(16, 0.1))


def _ids(program):
    return [(w["id"], w.get("shape_id")) for w in program["waveforms"]]


# ── Waveform ids ────────────────────────────────────────────────────────


def test_waveform_ids_depend_on_content_alone():
    compiled = pulse.compile(tuned)
    first = compiled.lower(Q[0], 0.25, 0.1, 0.0).export_controller(version=2)
    again = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0).export_controller(version=2)
    other = compiled.lower(Q[1], 0.25, 0.1, 0.0).export_controller(version=2)
    assert _ids(first) == _ids(again) == _ids(other)
    ids = [w["id"] for w in first["waveforms"]]
    assert len(set(ids)) == len(ids) == 4
    assert all(len(i) == 16 and int(i, 16) >= 0 for i in ids)


def test_a_new_amplitude_keeps_the_shape_id_and_changes_the_id():
    compiled = pulse.compile(tuned)
    a = compiled.lower(Q[0], 0.25, 0.1, 0.0).export_controller(version=2)["waveforms"]
    b = compiled.lower(Q[0], 0.75, 0.3, 0.0).export_controller(version=2)["waveforms"]
    assert [w["shape_id"] for w in a] == [w["shape_id"] for w in b]
    # The waveforms amp scales change; the others do not.
    assert [x["id"] == y["id"] for x, y in zip(a, b, strict=True)] == [False, True, False, True]
    # Two Gaussians of one width at different amplitudes are one shape.
    assert a[0]["shape_id"] == a[1]["shape_id"] != a[2]["shape_id"]


def test_unrolling_drops_the_ids():
    program = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0).export_controller(version=2)
    flat = unroll(program)
    assert "parameters" not in flat
    assert all(
        "id" not in e["waveform"] and "shape_id" not in e["waveform"]
        for e in flat["events"]
        if e["kind"] == "drive"
    )


# ── Parameter roles ─────────────────────────────────────────────────────


def test_the_export_says_what_each_float_parameter_reaches():
    program = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0).export_controller(version=2)
    assert program["parameters"] == {
        "amp": {"value": 0.25, "reaches": ["amplitude"]},
        "phi": {"value": 0.1, "reaches": ["phase"]},
        "df": {"value": 0.0, "reaches": ["frequency"]},
    }
    program = pulse.compile(widths).lower(Q[0], 8.0, 12.0).export_controller(version=2)
    assert {k: v["reaches"] for k, v in program["parameters"].items()} == {
        "sigma": ["shape"],
        "wait_cycles": ["timing"],
    }


def test_a_program_without_float_parameters_lists_none():
    program = pulse.compile(two_lines).lower(Q[0], Q[1]).export_controller(version=2)
    assert "parameters" not in program


# ── The memory model ────────────────────────────────────────────────────


def test_waveform_samples_is_the_memory_of_each_line():
    lowered = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0)
    # gaussian, drag and square: 40 + 40 + 16 samples.
    assert check(lowered, Controller(waveform_memory_samples=96)) == []
    assert check(lowered, Controller(waveform_memory_samples=95)) == [
        "line 0: its waveforms take 96 samples; the controller's waveform memory holds 95"
    ]
    per_line = Controller(waveform_memory_samples=95, waveform_memory="per_line")
    assert per_line == Controller(waveform_memory_samples=95)
    assert check(lowered, per_line) == check(lowered, Controller(waveform_memory_samples=95))


def test_a_shared_memory_holds_the_waveforms_of_every_line_once():
    lowered = pulse.compile(two_lines).lower(Q[0], Q[1])
    # A gaussian at two amplitudes is one shape: 40 + 16 samples over both lines.
    assert check(lowered, Controller(waveform_memory_samples=56, waveform_memory="shared")) == []
    assert check(lowered, Controller(waveform_memory_samples=55, waveform_memory="shared")) == [
        "its waveforms take 56 samples; the controller's shared waveform memory holds 55"
    ]
    # Per line, the second line alone takes 56.
    assert check(lowered, Controller(waveform_memory_samples=55)) == [
        "line 1: its waveforms take 56 samples; the controller's waveform memory holds 55"
    ]


def test_the_memory_reaches_the_export_only_when_shared():
    shared = Controller(waveform_memory_samples=4096, waveform_memory="shared")
    lowered = pulse.compile(two_lines, control_system=ControlSystem(controllers=[shared])).lower(
        Q[0], Q[1]
    )
    program = lowered.export_controller(version=2)
    assert program["controller"]["waveform_memory_samples"] == 4096
    assert program["controller"]["waveform_memory"] == "shared"
    assert lowered.export_controller()["limits"]["waveform_memory"] == "shared"
    assert unroll(program)["limits"] == shared.as_dict()
    json.dumps(program)
    per_line = Controller(waveform_memory_samples=4096)
    assert "waveform_memory" not in per_line.as_dict()
    plain = pulse.compile(two_lines, control_system=ControlSystem(controllers=[per_line])).lower(
        Q[0], Q[1]
    )
    assert "waveform_memory" not in plain.export_controller(version=2)["controller"]
    assert unroll(plain.export_controller(version=2))["limits"] == per_line.as_dict()


def test_the_memory_takes_its_strings_or_its_members():
    by_string = Controller(waveform_memory_samples=8, waveform_memory="shared")
    by_member = Controller(waveform_memory_samples=8, waveform_memory=WaveformMemoryScope.SHARED)
    assert by_string == by_member and hash(by_string) == hash(by_member)
    assert by_string.waveform_memory is WaveformMemoryScope.SHARED
    assert Controller().waveform_memory == "per_line"
    with pytest.raises(pulse.InvalidValueError, match="'per_line' or 'shared', got 'global'"):
        Controller(waveform_memory="global")


def test_a_controller_description_in_toml_gives_the_same_limits():
    description = tomllib.loads('waveform_memory_samples = 65536\nwaveform_memory = "shared"\n')
    assert Controller(**description) == Controller(
        waveform_memory_samples=65536, waveform_memory="shared"
    )


def test_the_structured_format_writes_the_controller_and_names_its_links_anew():
    @pulse.classical(max_latency_ns=200, link="ctl")
    def flip(bits: pulse.Bits) -> pulse.Bits:
        return [1 - bits[0]]

    @pulse.kernel
    def feedback(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(40, 0.05))
        bit = flip.submit([m]).result()
        sync(d, r)
        if bit:
            drive(d, square(16, 0.1))

    limits = Controller(
        instruction_cycles=1,
        waveform_memory_samples=4096,
        processors=[Processor(links=("ctl",), max_ops_per_function=64)],
    )
    links = {"ctl": Link(fixed_ns=4, bytes_per_ns=2.0, feedback_ns=8, slots=4)}
    machine = ControlSystem(controllers=[limits], links=links)
    lowered = pulse.compile(feedback, control_system=machine).lower(Q[0])
    program = lowered.export_controller(version=2)
    assert "limits" not in program
    controller = program["controller"]
    assert controller["instruction_cycles"] == 1
    assert controller["waveform_memory_samples"] == 4096
    assert {"min_idle_cycles", "detectors", "detector_recovery_cycles", "sample_rate_ghz"} < set(
        controller
    )
    assert {"max_tones_per_line", "max_pulses", "supports_timeouts"} < set(controller)
    (processor,) = controller["processors"]
    assert processor == limits.processors[0].as_dict()
    assert processor["links"] == ["ctl"] and processor["max_ops_per_function"] == 64
    assert processor["fixed_cost_only"] is False
    assert processor["max_concurrent_calls"] == 1
    assert processor["unsupported"] == []
    assert None not in processor["cycles_per_op"].values()
    assert program["quantum_system"] == lowered.target.name and "target" not in program
    (link,) = program["links"].values()
    assert link["send_ns"] == 4 and link["bandwidth_bytes_per_ns"] == 2.0
    assert link["answer_delay_ns"] == 8 and link["slots"] == 4 and link["slot_bytes"] == 1024
    flat = lowered.export_controller()
    back = unroll(program)
    assert back["limits"] == flat["limits"] and list(back["limits"]) == list(flat["limits"])
    assert back["links"] == flat["links"]
    assert back["target"] == flat["target"] and list(back) == list(flat)
    # A controller without processors lists none.
    plain = ControlSystem(controllers=[Controller()], links=links)
    program = pulse.compile(feedback, control_system=plain).lower(Q[0]).export_controller(version=2)
    assert program["controller"]["processors"] == []
    assert unroll(program)["limits"] == Controller().as_dict()


def test_values_no_limit_depends_on_are_bound_without_a_check(monkeypatch):
    from cudaq_pulse import controller

    checked = []
    original = controller._call_violations

    def counting(view):
        checked.append(view.arguments)
        return original(view)

    monkeypatch.setattr(controller, "_call_violations", counting)
    machine = ControlSystem(
        controllers=[Controller(waveform_memory_samples=16, max_tones_per_line=1)]
    )

    @pulse.kernel
    def sweep(q, amp: float):
        line = get_drive_line(q)
        for _ in range(1000):
            drive(line, gaussian(16, amp, 4.0))
            drive(line, gaussian(16, 0.5, 4.0))

    # One shape at any amplitude fits whatever the value: no check per call.
    compiled = pulse.compile(sweep, control_system=machine)
    for value in (0.1, 0.2, 0.1):
        compiled.lower(Q[0], value)
    assert checked == []

    @pulse.kernel
    def widths(q, sigma: float):
        line = get_drive_line(q)
        drive(line, gaussian(16, 0.5, sigma))
        drive(line, gaussian(16, 0.5, 4.0))

    # A width sets whether the two share memory: each distinct value is checked once.
    compiled = pulse.compile(widths, control_system=machine)
    for value in (4.0, 4.0):
        compiled.lower(Q[0], value)
    with pytest.raises(ControllerLimitError, match="waveform memory"):
        compiled.lower(Q[0], 5.0)
    assert [tuple(arguments) for arguments in checked] == [(4.0,), (5.0,)]
