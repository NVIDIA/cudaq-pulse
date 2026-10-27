# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The wire format of a call: what a request and an answer look like.

A request is a 24-byte header (magic ``'CUQR'``, function id, argument
length, request id, timestamp) followed by the arguments, tightly packed and
little-endian in the order of the function's argument types; an answer is a
24-byte header (magic ``'CUQS'``, status, result length, request id,
timestamp) followed by the results. This is the layout of the CUDA-Q Realtime
messaging protocol, declared here so cudaq-pulse needs nothing from CUDA-Q;
``core/mlir/include/cudaq-pulse/Runtime/WireFormat.h`` declares the same for
C++.

``native(library, symbol, function)`` wraps a host handler written in C with
the signature ``void handler(const void *request, void *answer, size_t
slot_bytes)``, the shape of a CUDA-Q Realtime host-call handler, as a Python
handler for simulation.
"""

from __future__ import annotations

import ctypes
import struct
from collections.abc import Sequence
from typing import Any

from . import ClassicalFunction, ValueType

__all__ = [
    "HEADER_BYTES",
    "REQUEST_MAGIC",
    "RESPONSE_MAGIC",
    "decode_answer",
    "decode_request",
    "encode_answer",
    "encode_request",
    "native",
    "schema",
]

HEADER_BYTES = 24
REQUEST_MAGIC = 0x43555152  # 'CUQR'
RESPONSE_MAGIC = 0x43555153  # 'CUQS'
_REQUEST = struct.Struct("<IIIIQ")
_RESPONSE = struct.Struct("<IiIIQ")

# Payload type ids, and the element format of each argument descriptor.
_TYPES = {
    "u8": (0x10, "B"),
    "i32": (0x11, "i"),
    "i64": (0x12, "q"),
    "f32": (0x13, "f"),
    "f64": (0x14, "d"),
}
_ARRAYS = {"u8": (0x20, "B"), "i32": (0x21, "i"), "f32": (0x22, "f"), "f64": (0x23, "d")}
_BIT_PACKED = 0x30


def _layout(spec: ValueType) -> tuple[int, int, int, str]:
    """(type id, bytes, wire elements, element format) of one type."""
    d = spec.descriptor
    if d.startswith("bits:"):
        return _BIT_PACKED, spec.wire_bytes, spec.count, ""
    if d.startswith("levels:"):
        return _ARRAYS["u8"][0], spec.count, spec.count, "B"
    if d.startswith("iq:"):
        return _ARRAYS["f32"][0], 8 * spec.count, 2 * spec.count, "f"
    if "[" in d:
        element = d[: d.index("[")]
        type_id, fmt = _ARRAYS[element]
        return type_id, spec.wire_bytes, spec.count, fmt
    type_id, fmt = _TYPES[d]
    return type_id, spec.wire_bytes, 1, fmt


def schema(function: ClassicalFunction) -> dict[str, Any]:
    """The function's handler schema: per argument and result its payload
    type id, size in bytes and element count, as a dispatcher function table
    entry declares them."""

    def describe(specs: Sequence[ValueType]) -> list[dict[str, int]]:
        out = []
        for spec in specs:
            type_id, size, elements, _ = _layout(spec)
            out.append({"type_id": type_id, "size_bytes": size, "num_elements": elements})
        return out

    return {
        "function_id": function.id,
        "args": describe(function.args),
        "results": describe(function.results),
    }


def _pack(specs: Sequence[ValueType], values: Sequence[Any], what: str) -> bytes:
    if len(values) != len(specs):
        raise ValueError(f"{len(specs)} {what} expected, got {len(values)}")
    out = bytearray()
    for spec, value in zip(specs, values):
        _, size, _, fmt = _layout(spec)
        items = [value] if spec.is_scalar else list(value)
        if spec.descriptor.startswith("bits:"):
            packed = bytearray(size)
            for i, bit in enumerate(items):
                if bit:
                    packed[i // 8] |= 1 << (i % 8)
            out += packed
        else:
            if spec.descriptor.startswith("iq:") and len(items) == spec.count:
                items = [c for point in items for c in point]
            out += struct.pack("<" + fmt * len(items), *items)
    return bytes(out)


def _unpack(specs: Sequence[ValueType], data: bytes) -> list[Any]:
    values: list[Any] = []
    cursor = 0
    for spec in specs:
        _, size, elements, fmt = _layout(spec)
        chunk = data[cursor : cursor + size]
        cursor += size
        if len(chunk) != size:
            raise ValueError("payload shorter than its schema")
        if spec.descriptor.startswith("bits:"):
            items: list[Any] = [(chunk[i // 8] >> (i % 8)) & 1 for i in range(spec.count)]
        else:
            items = list(struct.unpack("<" + fmt * elements, chunk))
            if spec.descriptor.startswith("iq:"):
                items = list(zip(items[0::2], items[1::2]))
        values.append(items[0] if spec.is_scalar else items)
    return values


def encode_request(
    function: ClassicalFunction, args: Sequence[Any], request_id: int = 0, timestamp: int = 0
) -> bytes:
    """A request to *function* with *args* (as a handler receives them)."""
    payload = _pack(function.args, args, "arguments")
    header = _REQUEST.pack(REQUEST_MAGIC, function.id, len(payload), request_id, timestamp)
    return header + payload


def decode_request(function: ClassicalFunction, data: bytes) -> tuple[int, int, list[Any]]:
    """``(request id, timestamp, arguments)`` of a request to *function*."""
    magic, fid, length, request_id, timestamp = _REQUEST.unpack_from(data)
    if magic != REQUEST_MAGIC:
        raise ValueError(f"not a request: magic {magic:#x}")
    if fid != function.id:
        raise ValueError(f"request for function {fid:#x}, not {function.name} ({function.id:#x})")
    return request_id, timestamp, _unpack(function.args, data[HEADER_BYTES : HEADER_BYTES + length])


def encode_answer(
    function: ClassicalFunction,
    answer: Any,
    request_id: int = 0,
    timestamp: int = 0,
    status: int = 0,
) -> bytes:
    """The answer of *function* (as a handler returns it) to one request."""
    results = function.results
    values = list(answer) if len(results) > 1 else ([answer] if results else [])
    payload = _pack(results, values, "results")
    return _RESPONSE.pack(RESPONSE_MAGIC, status, len(payload), request_id, timestamp) + payload


def decode_answer(function: ClassicalFunction, data: bytes) -> tuple[int, int, Any]:
    """``(status, request id, answer)`` of an answer from *function*."""
    magic, status, length, request_id, _ = _RESPONSE.unpack_from(data)
    if magic != RESPONSE_MAGIC:
        raise ValueError(f"not an answer: magic {magic:#x}")
    values = (
        _unpack(function.results, data[HEADER_BYTES : HEADER_BYTES + length]) if status == 0 else []
    )
    if not function.results or status != 0:
        return status, request_id, None
    return status, request_id, values[0] if len(values) == 1 else tuple(values)


class _Native:
    """A C host handler, called through its request and answer slots."""

    def __init__(self, handler: Any, function: ClassicalFunction, slot_bytes: int) -> None:
        self._handler = handler
        self._function = function
        self._slot_bytes = slot_bytes
        self._request = ctypes.create_string_buffer(slot_bytes)
        self._answer = ctypes.create_string_buffer(slot_bytes)
        self._next_id = 0

    def __call__(self, *args: Any) -> Any:
        request = encode_request(self._function, args, self._next_id)
        if len(request) > self._slot_bytes:
            raise ValueError(
                f"{self._function.name}: request of {len(request)} bytes exceeds the slot"
            )
        ctypes.memmove(self._request, request, len(request))
        self._handler(self._request, self._answer, self._slot_bytes)
        status, request_id, answer = decode_answer(self._function, self._answer.raw)
        if request_id != self._next_id:
            raise RuntimeError(
                f"{self._function.name}: answer to request {request_id}, expected {self._next_id}"
            )
        self._next_id = (self._next_id + 1) & 0xFFFFFFFF
        if status != 0:
            raise RuntimeError(f"{self._function.name} answered with status {status}")
        return answer


def native(library: str, symbol: str, function: ClassicalFunction, slot_bytes: int = 1024) -> Any:
    """A handler for *function* that calls the C function *symbol* in the
    shared library *library* with a request and an answer slot."""
    handler = getattr(ctypes.CDLL(library), symbol)
    handler.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    handler.restype = None
    return _Native(handler, function, slot_bytes)
