# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Waveform ids and parameter roles in format /2, and the waveform memory
model of ``ControllerLimits``."""

from __future__ import annotations

import json
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import (
    ControllerLimitError,
    ControllerLimits,
    WaveformMemory,
    check,
    unroll,
)

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


def test_waveform_samples_is_a_memory_of_that_many_samples():
    lowered = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0)
    for samples in (95, 96):
        short = ControllerLimits(waveform_samples=samples)
        full = ControllerLimits(waveform_memory=WaveformMemory(samples=samples))
        assert check(lowered, short) == check(lowered, full)
    # gaussian, drag and square: 40 + 40 + 16 samples.
    assert check(lowered, ControllerLimits(waveform_samples=96)) == []
    assert check(lowered, ControllerLimits(waveform_samples=95)) == [
        "line 0: its waveforms take 96 samples; the controller's waveform memory holds 95"
    ]


@pytest.mark.parametrize(
    ("memory", "used"),
    [
        (WaveformMemory(samples=1, granule=64), 192),
        (WaveformMemory(samples=1, scaling=False), 136),
        (WaveformMemory(samples=1, hold=True), 80),
        (WaveformMemory(samples=1, synthesized=frozenset({"gaussian", "drag"})), 16),
    ],
    ids=["granule", "no scaling", "hold", "synthesized"],
)
def test_the_memory_model_decides_what_a_program_takes(memory, used):
    lowered = pulse.compile(tuned).lower(Q[0], 0.25, 0.1, 0.0)
    (message,) = check(lowered, ControllerLimits(waveform_memory=memory))
    assert f"take {used} samples" in message


def test_a_shared_memory_holds_the_waveforms_of_every_line_once():
    lowered = pulse.compile(two_lines).lower(Q[0], Q[1])
    per_line = ControllerLimits(waveform_memory=WaveformMemory(samples=56))
    shared = ControllerLimits(waveform_memory=WaveformMemory(samples=56, shared=True))
    assert check(lowered, per_line) == []
    assert check(lowered, shared) == []
    assert check(
        lowered, ControllerLimits(waveform_memory=WaveformMemory(samples=55, shared=True))
    ) == ["its waveforms take 56 samples; the controller's shared waveform memory holds 55"]


def test_without_scaling_a_float_amplitude_is_checked_at_the_call():
    @pulse.kernel
    def two_amplitudes(q, a: float, b: float):
        d, t = get_drive_line(q)
        drive(d, gaussian(40, a, 10.0), t)
        drive(d, gaussian(40, b, 10.0), t)

    memory = WaveformMemory(samples=60, scaling=False)
    compiled = pulse.compile(
        two_amplitudes, machine=MachineModel(controller=ControllerLimits(waveform_memory=memory))
    )
    assert len(compiled.lower(Q[0], 0.5, 0.5).export_controller()["events"]) == 2
    with pytest.raises(ControllerLimitError, match="80 samples"):
        compiled.lower(Q[0], 0.5, 0.25)
    # With scaling, any two amplitudes share one waveform: compiling decides.
    scaled = MachineModel(controller=ControllerLimits(waveform_memory=WaveformMemory(samples=60)))
    lowered = pulse.compile(two_amplitudes, machine=scaled).lower(Q[0], 0.5, 0.25)
    assert type(lowered.generic) is pulse.Lowered


def test_the_memory_reaches_the_export():
    memory = WaveformMemory(
        samples=4096,
        granule=16,
        synthesized=frozenset({"square"}),
        upload_samples_per_cycle=0.5,
        upload_while_playing=True,
        persistent=True,
    )
    machine = MachineModel(controller=ControllerLimits(waveform_memory=memory))
    program = (
        pulse.compile(two_lines, machine=machine).lower(Q[0], Q[1]).export_controller(version=2)
    )
    assert program["controller"]["waveform_memory"] == {
        "samples": 4096,
        "granule": 16,
        "shared": False,
        "scaling": True,
        "synthesized": ["square"],
        "hold": False,
        "upload_samples_per_cycle": 0.5,
        "upload_while_playing": True,
        "persistent": True,
    }
    json.dumps(program)


def test_a_memory_is_validated():
    with pytest.raises(ValueError, match="not both"):
        ControllerLimits(waveform_samples=10, waveform_memory=WaveformMemory(samples=10))
    with pytest.raises(ValueError, match="must be positive"):
        WaveformMemory(samples=0)
    with pytest.raises(ValueError, match="must be positive"):
        WaveformMemory(samples=8, granule=0)
    with pytest.raises(ValueError, match="unknown shapes"):
        WaveformMemory(samples=8, synthesized=frozenset({"chirp"}))
    with pytest.raises(ValueError, match="upload_samples_per_cycle"):
        WaveformMemory(samples=8, upload_samples_per_cycle=0.0)
    shapes: Any = {"drag"}
    limits = ControllerLimits(waveform_memory=WaveformMemory(samples=8, synthesized=shapes))
    assert limits.waveform_memory is not None and limits.waveform_memory.synthesized == frozenset(
        {"drag"}
    )
    same = ControllerLimits(
        waveform_memory=WaveformMemory(samples=8, synthesized=frozenset({"drag"}))
    )
    assert limits == same and hash(limits) == hash(same)


def test_the_structured_format_names_the_controller_and_its_links_anew():
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

    limits = ControllerLimits(
        issue_cycles=1, waveform_samples=4096, max_ops=64, compute_links=("ctl",)
    )
    links = {"ctl": Link(fixed_ns=4, bytes_per_ns=2.0, feedback_ns=8, max_in_flight=4)}
    machine = MachineModel(controller=limits, links=links)
    lowered = pulse.compile(feedback, machine=machine).lower(Q[0])
    program = lowered.export_controller(version=2)
    assert "limits" not in program
    controller = program["controller"]
    assert controller["instruction_cycles"] == 1
    assert controller["waveform_memory_samples"] == 4096
    assert {"min_idle_cycles", "detectors", "detector_recovery_cycles", "samples_per_cycle"} < set(
        controller
    )
    assert {"max_tones_per_line", "max_pulses", "supports_timeouts"} < set(controller)
    assert not {"issue_cycles", "acquirers", "max_ops", "op_cycles"} & set(controller)
    (processor,) = controller["processors"]
    assert list(processor) == [
        "links",
        "clock_ghz",
        "cycles_per_op",
        "unsupported",
        "max_concurrent_calls",
        "fixed_cost_only",
        "max_ops",
        "max_state_bytes",
        "max_calls_per_shot",
        "max_bytes_per_shot",
    ]
    assert processor["links"] == ["ctl"] and processor["max_ops"] == 64
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
    # Without compute limits, the controller has no processor.
    plain = MachineModel(controller=ControllerLimits(), links=links)
    program = pulse.compile(feedback, machine=plain).lower(Q[0]).export_controller(version=2)
    assert program["controller"]["processors"] == []
    assert unroll(program)["limits"] == ControllerLimits().as_dict()
