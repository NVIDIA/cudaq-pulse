# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exporting for, and running in, a virtual-time controller emulation."""

from __future__ import annotations

import dataclasses
import json
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Bits, Float, Link, emulation, wire
from cudaq_pulse.compiler import MachineModel

from tests._handles import Q


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=1000)
def decode(bits):
    return [bits[0]]


@pulse.classical(args=(Bits(1),), results=(Float(),), max_latency_ns=10)
def next_wait(bits):
    return 400.0 * bits[0]


@pulse.kernel
def feedback(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(300, 0.05))
    ok, fix = decode.submit([m]).result(timeout=500 * pulse.ns)
    if ok:
        if fix[0]:
            drive(d, square(40, 0.1))


@pulse.kernel
def adaptive(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(300, 0.05))
    w = next_wait.submit([m]).result()
    drive(d, square(40, 0.1))
    wait(d, w, bounds=(0, 1000))
    sync(d, r)
    readout(r, square(300, 0.05))


def test_export_and_manifest_are_json():
    machine = MachineModel(links={0: Link(slot_bytes=256, max_in_flight=8)})
    low = pulse.compile(adaptive, machine=machine).lower(Q[0])
    artifact = json.loads(json.dumps(low.export_emulation()))
    assert artifact["format"] == emulation.FORMAT
    assert [e["opened_by"] for e in artifact["epochs"]] == ["start", "wait"]
    after = [e for e in artifact["events"] if e["epoch"] == 1]
    assert after and all(e["start_cycles"] >= e["epoch_start_cycles"] for e in after)
    (entry,) = artifact["manifest"]["functions"]
    assert entry["function_id"] == next_wait.id and entry["slot_bytes"] == 256
    assert entry["schema"] == wire.schema(next_wait)
    assert entry["request_bytes"] == 25 and entry["answer_bytes"] == 32
    assert artifact["links"]["0"]["max_in_flight"] == 8


def test_a_call_exports_its_fields_in_order():
    (call,) = pulse.compile(adaptive).lower(Q[0]).classical_report()
    fields = {**dataclasses.asdict(call), "iteration": list(call.iteration)}
    del fields["controller"]
    assert list(emulation._call_data(call).items()) == list(fields.items())
    placed = dataclasses.replace(call, controller=1)
    assert list(emulation._call_data(placed).items()) == list({**fields, "controller": 1}.items())


def test_variable_waits_take_their_emulated_lengths():
    low = pulse.compile(adaptive).lower(Q[0])
    result = emulation.emulate(
        low, shots=20, seed=1, outcomes=lambda q, _, rng: int(rng.integers(0, 2))
    )
    clock = low.target.clock_ghz
    for outcomes, end in zip(result.outcomes, result.ends_ns):
        # A 1 waits 400 cycles longer before the second readout.
        assert outcomes[0] in (0, 1)
    ends = {o[0]: e for o, e in zip(result.outcomes, result.ends_ns)}
    assert ends[1] - ends[0] == pytest.approx(400 / clock)


def test_late_answers_take_the_timeout_arm_at_the_modelled_rate():
    low = pulse.compile(feedback).lower(Q[0])
    # Each direction: 200 ns fixed plus exponential jitter of mean 100 ns.
    # Late when the sum of two jitters exceeds 100 ns: P = (1 + 1) e^-1.
    transport = emulation.Transport(fixed_ns=200, jitter_ns=100)
    result = emulation.emulate(low, shots=4000, transport=transport, service_time="none", seed=5)
    expected = 2 * math.exp(-1)
    assert result.late_fraction == pytest.approx(
        expected, abs=4 * math.sqrt(expected * (1 - expected) / 4000)
    )
    assert all(len(shot) == 1 for shot in result.calls)
    assert all(c.budget_ns == 500 for shot in result.calls for c in shot)


def test_udp_gives_the_same_answers_as_in_process():
    low = pulse.compile(adaptive).lower(Q[0])
    functions = low.classical
    handlers = {fn.name: fn.implementation for fn in functions}
    _, address, stop = emulation.serve_udp(functions, handlers)
    try:
        udp = emulation.emulate(low, shots=10, seed=3, channel=emulation.UdpChannel(address))
    finally:
        stop()
    local = emulation.emulate(low, shots=10, seed=3)
    assert udp.outcomes == local.outcomes
    assert [[c.answer for c in s] for s in udp.calls] == [
        [c.answer for c in s] for s in local.calls
    ]
    assert udp.ends_ns == local.ends_ns


def test_throughput_is_reported():
    low = pulse.compile(feedback).lower(Q[0])
    result = emulation.emulate(low, shots=50, seed=1)
    assert result.calls_per_second > 0


def test_the_realtime_channel_and_device_call_export_are_seams():
    with pytest.raises(NotImplementedError, match="InProcess"):
        emulation.RealtimeChannel("libcudaq-realtime.so")
    low = pulse.compile(feedback).lower(Q[0])
    with pytest.raises(NotImplementedError, match="export_emulation"):
        emulation.device_call_text(low)


def test_exports_with_calls_in_loops_are_json_data():
    import json

    @pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=100, name="json_decode")
    def json_decode(bits):
        return [bits[0]]

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(2):
            if json_decode.submit([readout(r, square(100, 0.05))]).result()[0]:
                drive(d, square(40, 0.1))

    low = pulse.compile(k, unroll_threshold=0).lower(Q[0])
    for program in (low.export_emulation(), low.export_controller()):
        assert json.loads(json.dumps(program)) == program
        assert [c["iteration"] for c in program["calls"]] == [[0], [1]]
