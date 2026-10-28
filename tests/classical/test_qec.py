# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Real-time decoding with CUDA-QX decoder messages."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import qec, wire

from tests.classical.test_simulation import PI_AMP, _device

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "realtime_wire.json").read_text())
CUDAQX = FIXTURE["cudaqx_decoder"]


def _decoder(decode=None):
    return qec.Decoder(3, syndromes=12, corrections=2, max_latency_ns=500, decode=decode)


def test_calls_carry_the_cudaqx_ids_and_payloads():
    decoder = _decoder()
    assert decoder.enqueue_function.id == CUDAQX["function_ids"]["enqueue_syndromes"]
    assert decoder.corrections_function.id == CUDAQX["function_ids"]["get_corrections"]
    assert decoder.reset_function.id == CUDAQX["function_ids"]["reset_decoder"]
    bits = [1, 0] * 6
    enqueue = wire.encode_request(decoder.enqueue_function, (3, 5, 0, 12, bits))
    header = CUDAQX["payload_bytes"]["enqueue_syndromes_header"]
    assert len(enqueue) - 24 == header + 2
    assert struct.unpack_from("<qqqq", enqueue, 24) == (3, 5, 0, 12)
    get = wire.encode_request(decoder.corrections_function, (3, 2, 1))
    assert len(get) - 24 == CUDAQX["payload_bytes"]["get_corrections"]
    reset = wire.encode_request(decoder.reset_function, (3,))
    assert len(reset) - 24 == CUDAQX["payload_bytes"]["reset_decoder"]


def test_each_path_keeps_its_own_decoder_state():
    decoder = qec.Decoder(
        0, syndromes=1, corrections=1, max_latency_ns=100, decode=lambda bits: [sum(bits) % 2]
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


def test_a_decoder_without_decode_needs_handlers():
    decoder = _decoder()
    with pytest.raises(TypeError, match="has no body to run here. Give it a handler"):
        decoder.corrections_function(3, 2, 0)
    handlers = qec.Decoder(3, syndromes=12, corrections=2, max_latency_ns=500).handlers()
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
