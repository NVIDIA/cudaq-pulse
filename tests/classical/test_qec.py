# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time decoding with CUDA-QX decoder messages."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import emulation, qec, wire

from tests.classical.test_simulation import PI_AMP, _device

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "realtime_wire.json").read_text())
CUDAQX = FIXTURE["cudaqx_decoder"]


def _decoder(decode=None):
    return qec.Decoder(3, syndrome_count=12, correction_count=2, max_latency_ns=500, decode=decode)


def _key(request):
    return struct.unpack_from("<I", request, 4)[0]


def test_calls_carry_the_cudaqx_ids_and_payloads():
    decoder = _decoder()
    bits = [1, 0] * 6
    enqueue = wire.encode_request(decoder.enqueue_function, (3, 5, 0, 12, bits))
    assert _key(enqueue) == CUDAQX["function_ids"]["enqueue_syndromes"]
    header = CUDAQX["payload_bytes"]["enqueue_syndromes_header"]
    assert len(enqueue) - 24 == header + 2
    assert struct.unpack_from("<qqqq", enqueue, 24) == (3, 5, 0, 12)
    get = wire.encode_request(decoder.corrections_function, (3, 2, 1))
    assert _key(get) == CUDAQX["function_ids"]["get_corrections"]
    assert len(get) - 24 == CUDAQX["payload_bytes"]["get_corrections"]
    reset = wire.encode_request(decoder.reset_function, (3,))
    assert _key(reset) == CUDAQX["function_ids"]["reset_decoder"]
    assert len(reset) - 24 == CUDAQX["payload_bytes"]["reset_decoder"]


def _two_decoders():
    small = qec.Decoder(
        0, syndrome_count=1, correction_count=1, max_latency_ns=100, decode=lambda b: [b[0]]
    )
    same = qec.Decoder(
        1, syndrome_count=1, correction_count=1, max_latency_ns=100, decode=lambda b: [0]
    )
    large = qec.Decoder(
        2, syndrome_count=2, correction_count=1, max_latency_ns=100, decode=lambda b: [b[0] ^ b[1]]
    )
    return small, same, large


def test_decoders_share_the_cudaqx_keys_in_one_program():
    decoders = _two_decoders()

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        for decoder in decoders:
            qec.enqueue_syndromes(decoder, [m] * decoder.syndrome_count)
            qec.get_corrections(decoder).result()

    dev = _device()
    low = pulse.compile(k, device=dev).lower(dev.qudit_handles[0])
    keys = {c.dispatch_key for c in low.classical_report()}
    assert keys == {CUDAQX["function_ids"][n] for n in ("enqueue_syndromes", "get_corrections")}


def test_a_udp_server_tells_decoders_that_share_a_key_apart():
    decoders = _two_decoders()
    functions = [f for d in decoders for f in (d.enqueue_function, d.corrections_function)]
    handlers = {n: h for d in decoders for n, h in d.handlers().items()}
    served = {}
    for d in decoders:
        get = d.corrections_function.name
        served[get] = lambda *args, decoder=d: [decoder.decoder_id % 2]
    _, address, stop = emulation.serve_udp(functions, {**handlers, **served})
    channel = emulation.UdpChannel(address)
    try:
        for d in decoders:
            fn = d.enqueue_function
            request = wire.encode_request(
                fn, (d.decoder_id, 0, 0, d.syndrome_count, [1] * d.syndrome_count)
            )
            answer, _ = channel.call(fn, request)
            assert wire.decode_answer(fn, answer)[0] == 0
            fn = d.corrections_function
            request = wire.encode_request(fn, (d.decoder_id, 1, 0))
            answer, _ = channel.call(fn, request)
            assert wire.decode_answer(fn, answer)[2] == [d.decoder_id % 2]
    finally:
        channel.close()
        stop()


def test_each_path_keeps_its_own_decoder_state():
    decoder = qec.Decoder(
        0,
        syndrome_count=1,
        correction_count=1,
        max_latency_ns=100,
        decode=lambda bits: [sum(bits) % 2],
    )

    @pulse.kernel
    def k(a, b, flag):
        qec.reset_decoder(decoder)
        for q in (a, b):
            r = get_readout_line(q)
            qec.enqueue_syndromes(decoder, [readout(r, square(100, 0.05))])
        f = get_drive_line(flag)
        if qec.get_corrections(decoder).result()[0]:
            drive(f, square(40, PI_AMP))

    dev = _device(n=3)
    low = pulse.compile(k, device=dev).lower(*dev.qudit_handles[:3])
    plus = np.array([1, 1]) / np.sqrt(2)
    state = np.kron(np.array([1, 0]), np.kron(plus, plus))  # a, b in |+>, flag in |0>
    result = pulse.simulate(low, initial_state=state, levels=2)
    assert len(result.branches) == 4
    for branch in result.branches:
        assert branch.probability == pytest.approx(0.25)
    # The flag is flipped exactly on the odd-parity paths: P(flag) = 1/2.
    flag = sum(result.density_matrix[i, i].real for i in range(8) if i & 4)
    assert flag == pytest.approx(0.5, abs=1e-6)
    for branch in result.branches:
        assert [c.function.split("_")[0] for c in branch.calls] == [
            "reset",
            "enqueue",
            "enqueue",
            "get",
        ]
        assert branch.calls[-1].answer == [(branch.levels[0] + branch.levels[1]) % 2]


def _parity_decoder():
    return qec.Decoder(
        0,
        syndrome_count=2,
        correction_count=1,
        max_latency_ns=100,
        decode=lambda bits: [sum(bits) % 2],
    )


def test_emulation_decodes_what_each_shot_enqueued():
    decoder = _parity_decoder()

    @pulse.kernel
    def k(a, b):
        ra, rb = get_readout_line(a), get_readout_line(b)
        qec.reset_decoder(decoder)
        for r in range(2):
            qec.enqueue_syndromes(
                decoder, [readout(ra, square(100, 0.05)), readout(rb, square(100, 0.05))], r
            )
        qec.get_corrections(decoder).result()

    dev = _device(n=2)
    low = pulse.compile(k, device=dev).lower(*dev.qudit_handles[:2])
    run = emulation.emulate(low, shots=40, seed=3, service_time="none")
    parities = set()
    for outcomes, calls in zip(run.outcomes, run.calls, strict=True):
        assert calls[-1].answer == [sum(outcomes) % 2]
        parities.add(sum(outcomes) % 2)
    assert parities == {0, 1}


def test_serve_udp_keeps_a_decoders_state_as_a_server_does():
    decoder = _parity_decoder()
    enqueue, get, reset = (
        decoder.enqueue_function,
        decoder.corrections_function,
        decoder.reset_function,
    )
    _, address, stop = emulation.serve_udp([enqueue, get, reset], decoder.handlers())
    channel = emulation.UdpChannel(address)
    request_id = iter(range(100))

    def call(fn, *args):
        answer, _ = channel.call(fn, wire.encode_request(fn, args, next(request_id)))
        status, _, value = wire.decode_answer(fn, answer)
        return status, value

    not_ready = CUDAQX["status"]["NOT_READY"]
    try:
        assert call(reset, 0) == (0, None)
        assert call(get, 0, 1, 0) == (not_ready, None)
        assert call(enqueue, 0, 0, 0, 2, [1, 0]) == (0, None)
        assert call(enqueue, 0, 1, 0, 2, [1, 1]) == (0, None)
        assert call(get, 0, 1, 0) == (0, [1])
        assert call(get, 0, 1, 1) == (0, [1])  # again, then dropped
        assert call(get, 0, 1, 0) == (not_ready, None)
        # After a decode, an enqueue starts a new volume.
        call(enqueue, 0, 0, 0, 2, [1, 1])
        assert call(get, 0, 1, 0) == (0, [0])
        call(enqueue, 0, 0, 0, 2, [0, 1])
        assert call(get, 0, 1, 0) == (0, [1])
        call(reset, 0)
        assert call(get, 0, 1, 0) == (not_ready, None)
    finally:
        channel.close()
        stop()


def _nothing_enqueued(timeout):
    decoder = _parity_decoder()

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        qec.reset_decoder(decoder)
        if timeout:
            ok, fix = qec.get_corrections(decoder).result(timeout=200 * pulse.ns)
            if ok:
                drive(d, square(40, PI_AMP))
        else:
            if qec.get_corrections(decoder).result()[0]:
                drive(d, square(40, PI_AMP))

    dev = _device()
    return pulse.compile(k, device=dev).lower(dev.qudit_handles[0])


def test_a_get_with_nothing_enqueued_is_late():
    low = _nothing_enqueued(timeout=True)
    (branch,) = pulse.simulate(low, levels=2).branches
    assert not branch.calls[-1].in_time and branch.calls[-1].answer is None
    run = emulation.emulate(low, shots=2, service_time="none")
    assert [shot[-1].in_time for shot in run.calls] == [False, False]
    low = _nothing_enqueued(timeout=False)
    with pytest.raises(pulse.ExecutionError, match=r"nothing to decode \(NOT_READY"):
        pulse.simulate(low, levels=2)
    with pytest.raises(pulse.ExecutionError, match=r"NOT_READY \(status 4\)"):
        emulation.emulate(low, service_time="none")


def test_a_get_decodes_what_was_enqueued_since_the_last_decode():
    decoder = qec.Decoder(
        0,
        syndrome_count=1,
        correction_count=1,
        max_latency_ns=100,
        decode=lambda bits: [len(bits) % 2],
    )

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        qec.reset_decoder(decoder)
        qec.enqueue_syndromes(decoder, [readout(r, square(100, 0.05))])
        qec.get_corrections(decoder).result()
        qec.enqueue_syndromes(decoder, [readout(r, square(100, 0.05))])
        qec.get_corrections(decoder).result()

    dev = _device()
    low = pulse.compile(k, device=dev).lower(dev.qudit_handles[0])
    for branch in pulse.simulate(low, levels=2).branches:
        assert [c.answer for c in branch.calls if c.function.startswith("get")] == [[1], [1]]
    for shot in emulation.emulate(low, shots=4, seed=1, service_time="none").calls:
        assert [c.answer for c in shot if c.function.startswith("get")] == [[1], [1]]


def test_late_fraction_counts_the_answers_the_program_waits_on():
    decoder = qec.Decoder(
        0, syndrome_count=1, correction_count=1, max_latency_ns=2000, decode=lambda b: b
    )

    @pulse.kernel
    def k(q):
        d, r = get_drive_line(q), get_readout_line(q)
        qec.reset_decoder(decoder)
        qec.enqueue_syndromes(decoder, [readout(r, square(100, 0.05))])
        ok, fix = qec.get_corrections(decoder).result(timeout=2000 * pulse.ns)
        if ok:
            drive(d, square(40, PI_AMP))

    dev = _device()
    low = pulse.compile(k, device=dev).lower(dev.qudit_handles[0])
    kw = {"shots": 5, "seed": 1, "service_time": "none"}
    run = emulation.emulate(low, transport=emulation.Transport(fixed_ns=500), **kw)
    for reset, enqueue, get in run.calls:
        # Sent without waiting: back after their budget, which counts for nothing.
        assert not reset.in_time and not enqueue.in_time and get.in_time
    assert run.late_fraction == 0.0
    run = emulation.emulate(low, transport=emulation.Transport(fixed_ns=1500), **kw)
    assert run.late_fraction == 1.0


LATENCIES = {"min_latency_ns": 1000, "nominal_latency_ns": 2000, "max_latency_ns": 5000}
latency_decoder = qec.Decoder(
    0, syndrome_count=1, correction_count=1, timeout_ns=3000, decode=lambda b: b, **LATENCIES
)


@pulse.classical.external(latency_mode="timeout", timeout_ns=3000, **LATENCIES)
def plain_decode(syndrome: pulse.Length[pulse.Bits, 1]) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits()


@pulse.kernel
def with_decoder(q):
    d, r = get_drive_line(q), get_readout_line(q)
    m = readout(r, square(100, 0.05))
    qec.enqueue_syndromes(latency_decoder, [m])
    ok, fix = qec.get_corrections(latency_decoder).result()
    if ok:
        if fix[0]:
            drive(d, square(40, PI_AMP))


@pulse.kernel
def with_function(q):
    d, r = get_drive_line(q), get_readout_line(q)
    m = readout(r, square(100, 0.05))
    ok, fix = plain_decode.submit([m]).result()
    if ok:
        if fix[0]:
            drive(d, square(40, PI_AMP))


def test_a_decoder_s_latencies_work_as_a_classical_function_s():
    dev = _device()
    lows = [
        pulse.compile(k, device=dev).lower(dev.qudit_handles[0])
        for k in (with_decoder, with_function)
    ]
    handlers = {"plain_decode": lambda s: s}
    (*_, by_decoder), (by_function,) = (low.classical_report() for low in lows)
    assert (
        by_decoder.due_cycles - by_decoder.submit_cycles
        == by_function.due_cycles - by_function.submit_cycles
    )
    for branching in (False, True):
        in_time = [
            sum(
                b.probability
                for b in pulse.simulate(
                    low, levels=2, handlers=handlers, latency_branching=branching
                ).branches
                if b.calls[-1].in_time
            )
            for low in lows
        ]
        # P(latency <= 3000) over a triangle on [1000, 5000] with its mode at 2000.
        expected = 1 - 2000**2 / (4000 * 3000) if branching else 1.0
        assert in_time == pytest.approx([expected, expected])
    kw = {"shots": 2, "seed": 1, "service_time": "none", "handlers": handlers}
    budgets = [emulation.emulate(low, **kw).calls[0][-1].budget_ns for low in lows]
    assert budgets[0] == budgets[1] > 3000


def test_a_decoder_without_decode_needs_handlers():
    decoder = _decoder()
    with pytest.raises(TypeError, match="has no body to run here. Give it a handler"):
        decoder.corrections_function(3, 2, 0)
    handlers = qec.Decoder(3, syndrome_count=12, correction_count=2, max_latency_ns=500).handlers()
    assert set(handlers) == {
        decoder.enqueue_function.name,
        decoder.corrections_function.name,
        decoder.reset_function.name,
    }


def test_a_mapped_enqueue_sends_its_mapping_row():
    decoder = _decoder()
    seen = []

    @pulse.kernel
    def rounds(q):
        r = get_readout_line(q)
        ms = []
        for _ in range(12):
            ms.append(readout(r, square(100, 0.05)))
        qec.enqueue_syndromes_mapped(decoder, ms, 4, 2)

    def enqueue(decoder_id, round, mapping, count, bits):
        seen.append((decoder_id, round, mapping, count))

    dev = _device()
    lowered = pulse.compile(rounds, device=dev).lower(dev.qudit_handles[0])
    pulse.simulate(
        lowered, handlers={decoder.enqueue_function.name: enqueue}, shots=1, seed=0, levels=2
    )
    assert set(seen) == {(3, 4, 2, 12)}
