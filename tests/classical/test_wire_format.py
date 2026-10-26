# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The wire format against the CUDA-Q Realtime protocol and CUDA-QX decoders.

The facts come from ``fixtures/realtime_wire.json``, copied from the public
sources it names. With ``CUDAQ_SOURCE_DIR`` (and ``CUDAQX_SOURCE_DIR``) set
to checkouts of those projects, the fixture is checked against them too.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import classical
from cudaq_pulse.classical import IQ, Bits, Byte, Float, Floats, Integer, Integers, Levels, wire

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "realtime_wire.json").read_text())


def test_header_layout_and_magics():
    assert wire.HEADER_BYTES == FIXTURE["header_bytes"]
    assert wire.REQUEST_MAGIC == FIXTURE["request_magic"]
    assert wire.RESPONSE_MAGIC == FIXTURE["response_magic"]
    fn = pulse.classical(lambda: None, name="probe", max_latency_ns=0)
    request = wire.encode_request(fn, (), request_id=7, timestamp=2**40)
    magic, fid, length, rid, ts = struct.unpack("<IIIIQ", request)
    assert (magic, fid, length, rid, ts) == (wire.REQUEST_MAGIC, fn.id, 0, 7, 2**40)


def test_payload_type_ids():
    ids = FIXTURE["payload_types"]
    fn = pulse.classical(
        lambda *a: None,
        name="all",
        args=(Byte, Integer(32), Integer(), Float(32), Float(), Floats(2, 32), Floats(2), Bits(3)),
        results=(two_bytes := classical.Bytes(2), Integers(2), Levels(2), IQ(1)),
        max_latency_ns=0,
    )
    s = wire.schema(fn)
    assert [a["type_id"] for a in s["args"]] == [
        ids["UINT8"],
        ids["INT32"],
        ids["INT64"],
        ids["FLOAT32"],
        ids["FLOAT64"],
        ids["ARRAY_FLOAT32"],
        ids["ARRAY_FLOAT64"],
        ids["BIT_PACKED"],
    ]
    assert [r["type_id"] for r in s["results"]] == [
        ids["ARRAY_UINT8"],
        ids["ARRAY_INT32"],
        ids["ARRAY_UINT8"],
        ids["ARRAY_FLOAT32"],
    ]
    assert s["args"][-1] == {"type_id": ids["BIT_PACKED"], "size_bytes": 1, "num_elements": 3}
    assert s["results"][-1] == {"type_id": ids["ARRAY_FLOAT32"], "size_bytes": 8, "num_elements": 2}
    assert two_bytes.descriptor == "u8[2]"
    assert len(fn.args) <= FIXTURE["schema_limits"]["args"]


def test_protocol_examples():
    example1, example2, example3, _ = FIXTURE["protocol_examples"]
    process = pulse.classical(
        lambda c, t: None, name="process", args=(Integer(32), Float(32)), max_latency_ns=0
    )
    request = wire.encode_request(process, (5, 0.25))
    assert len(request) - 24 == example1["arg_len"]
    assert struct.unpack_from("<i", request, example1["offsets"][0])[0] == 5
    assert struct.unpack_from("<f", request, example1["offsets"][1])[0] == 0.25
    decode = pulse.classical(
        lambda b, n: None, name="decode", args=(Bits(128), Integer(32)), max_latency_ns=0
    )
    request = wire.encode_request(decode, ([1] * 128, 128))
    assert len(request) - 24 == example2["arg_len"]
    assert struct.unpack_from("<I", request, example2["offsets"][1])[0] == 128
    answer_fn = pulse.classical(lambda: 3, name="answer", results=(Byte,), max_latency_ns=0)
    answer = wire.encode_answer(answer_fn, 3)
    assert answer[24] == example3["result_byte_24"]
    assert struct.unpack_from("<I", answer, 8)[0] == example3["result_len"]


def test_bits_pack_lsb_first():
    case = FIXTURE["bit_packing"]
    fn = pulse.classical(
        lambda b: None, name="bits", args=(Bits(len(case["bits"])),), max_latency_ns=0
    )
    request = wire.encode_request(fn, (case["bits"],))
    assert list(request[24:]) == case["bytes"]
    assert wire.decode_request(fn, request)[2] == [case["bits"]]


def test_round_trips():
    fn = pulse.classical(
        lambda *a: None,
        name="mixed",
        args=(Bits(10), Levels(3), IQ(2), Float(), Integer(32), Floats(2, 32)),
        results=(Float(), Bits(3)),
        max_latency_ns=0,
    )
    args = ([1, 0] * 5, [0, 2, 1], [[0.5, -0.5], [1.0, 2.0]], 3.25, -7, [1.5, 2.5])
    request = wire.encode_request(fn, args, request_id=9)
    rid, _, decoded = wire.decode_request(fn, request)
    assert rid == 9
    assert decoded[:2] == [args[0], args[1]] and decoded[2] == [0.5, -0.5, 1.0, 2.0]
    assert decoded[3:] == [3.25, -7, [1.5, 2.5]]
    answer = wire.encode_answer(fn, (0.125, [1, 1, 0]), request_id=9)
    assert wire.decode_answer(fn, answer) == (0, 9, (0.125, [1, 1, 0]))
    failed = wire.encode_answer(fn, (0.0, [0, 0, 0]), status=3)
    assert wire.decode_answer(fn, failed)[0] == 3
    with pytest.raises(ValueError, match="not a request"):
        wire.decode_request(fn, answer)


def test_cudaqx_decoder_ids():
    for name, fid in FIXTURE["cudaqx_decoder"]["function_ids"].items():
        assert classical.function_id(name) == fid


_C_HANDLER = r"""
#include <stdint.h>
#include <string.h>
#include <stddef.h>
struct header { uint32_t magic, function_id, length, request_id; uint64_t timestamp; };
struct answer { uint32_t magic; int32_t status; uint32_t length, request_id; uint64_t timestamp; };
/* The parity of 4 bit-packed syndrome bits, as one bit. */
void parity(const void *request, void *out, size_t slot) {
  const struct header *h = request;
  const uint8_t *bits = (const uint8_t *)request + 24;
  uint8_t p = (bits[0] ^ (bits[0] >> 1) ^ (bits[0] >> 2) ^ (bits[0] >> 3)) & 1;
  struct answer a = {0x43555153u, 0, 1, h->request_id, h->timestamp};
  memcpy(out, &a, sizeof a);
  ((uint8_t *)out)[24] = p;
}
"""


@pytest.fixture(scope="module")
def handler_library(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler")
    directory = tmp_path_factory.mktemp("handler")
    source = directory / "parity.c"
    source.write_text(_C_HANDLER)
    library = directory / "libparity.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    return str(library)


parity = pulse.classical(
    lambda bits: [sum(bits) % 2],
    name="parity",
    args=(Bits(4),),
    results=(Bits(1),),
    max_latency_ns=0,
)


def test_a_native_handler_answers_like_the_python_one(handler_library):
    native = wire.native(handler_library, "parity", parity)
    for bits in ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]):
        assert native(bits) == parity(bits)


def test_a_native_handler_drives_a_simulation(handler_library):
    from tests.classical.test_simulation import PI_AMP, _device

    @pulse.kernel
    def k(a, b, c, d, flag):
        ms = []
        for q in (a, b, c, d):
            r, rt = get_readout_line(q)
            ms.append(readout(r, square(100, 0.05), rt))
        f, ft = get_drive_line(flag)
        if parity.submit(ms).result()[0]:
            drive(f, square(40, PI_AMP), ft)

    dev = _device(n=5)
    low = pulse.compile(k, device=dev).lower(*dev.qudit_handles[:5])
    import numpy as np

    rho0 = np.zeros(32)
    rho0[0b00111] = 1.0  # three data qubits in |1>: odd parity
    python = pulse.simulate(low, initial_state=rho0)
    native = pulse.simulate(
        low, initial_state=rho0, handlers={"parity": wire.native(handler_library, "parity", parity)}
    )
    np.testing.assert_allclose(native.density_matrix, python.density_matrix, atol=1e-12)
    assert native.density_matrix[0b10111, 0b10111].real == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(
    not os.environ.get("CUDAQ_SOURCE_DIR"), reason="set CUDAQ_SOURCE_DIR to check the fixture"
)
def test_fixture_matches_cuda_q_sources():
    root = Path(os.environ["CUDAQ_SOURCE_DIR"])
    header = (root / FIXTURE["sources"]["cuda-quantum"]["files"][0]).read_text()
    assert (
        int(re.search(r"CUDAQ_RPC_MAGIC_REQUEST (0x[0-9A-Fa-f]+)u", header).group(1), 16)
        == FIXTURE["request_magic"]
    )
    assert (
        int(re.search(r"CUDAQ_RPC_MAGIC_RESPONSE (0x[0-9A-Fa-f]+)u", header).group(1), 16)
        == FIXTURE["response_magic"]
    )
    assert (
        int(re.search(r"CUDAQ_RPC_HEADER_SIZE (\d+)u", header).group(1)) == FIXTURE["header_bytes"]
    )
    api = (root / FIXTURE["sources"]["cuda-quantum"]["files"][1]).read_text()
    for name, value in FIXTURE["payload_types"].items():
        assert int(re.search(rf"CUDAQ_TYPE_{name} = (0x[0-9A-Fa-f]+)", api).group(1), 16) == value
    assert re.search(r"cudaq_type_desc_t args\[(\d+)\]", api).group(1) == str(
        FIXTURE["schema_limits"]["args"]
    )
    assert re.search(r"cudaq_type_desc_t results\[(\d+)\]", api).group(1) == str(
        FIXTURE["schema_limits"]["results"]
    )


@pytest.mark.skipif(
    not os.environ.get("CUDAQX_SOURCE_DIR"), reason="set CUDAQX_SOURCE_DIR to check the fixture"
)
def test_fixture_matches_cudaqx_sources():
    root = Path(os.environ["CUDAQX_SOURCE_DIR"])
    text = (root / FIXTURE["sources"]["cudaqx"]["files"][0]).read_text()
    expected = FIXTURE["cudaqx_decoder"]
    for name, fid in expected["function_ids"].items():
        assert re.search(rf'fnv1a_hash\("{name}"\)', text)
        assert (
            f"{fid:#X}".replace("0X", "0x") in text.replace("u,", ",").replace("u\n", "\n")
            or f"0x{fid:08X}" in text
        )
    for name, value in expected["status"].items():
        assert re.search(rf"\b{name} = {value}\b", text)
    assert "sizeof(EnqueueRequestPayload) == 32" in text
    assert "sizeof(GetCorrectionsRequestPayload) == 17" in text
    assert "sizeof(ResetRequestPayload) == 8" in text
