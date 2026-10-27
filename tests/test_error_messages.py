# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Error messages say what is wrong, where, and what to write instead."""

from __future__ import annotations

import socket

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Bits, emulation, wire
from cudaq_pulse.kernel.ir_builder import Parameter, SymbolicValueError

from tests._handles import Q

NOT_CALLABLE = object()


def test_qudit_handles_in_a_list_say_how_to_pass_them():
    @pulse.kernel
    def k(qs):
        d, t = get_drive_line(qs[0])
        drive(d, square(8, 0.1), t)

    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(k).lower([Q[0], Q[1]])
    assert str(info.value) == (
        "get_drive_line(): argument 1 is a qudit handle that reached the kernel inside a "
        "list; pass qudits as separate arguments, a Register (dev.qudit_handles[0:4]) or a "
        "tuple, not a list"
    )


def test_a_name_used_before_it_is_assigned_is_a_plain_compilation_error():
    @pulse.kernel
    def k(q, flag: int):
        if flag:
            w = square(8, 0.1)
        d, t = get_drive_line(q)
        drive(d, w, t)

    pulse.compile(k).lower(Q[0], 1)
    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(k).lower(Q[0], 0)
    assert type(info.value) is pulse.CompilationError
    assert str(info.value) == (
        "'w' is used before it is assigned; assign it on every path that reaches this "
        "line (an if on a value known while tracing takes only one arm)"
    )
    assert "drive(d, w, t)" in "".join(info.value.__notes__)


def test_unpacking_what_an_op_returns_says_it_returns_nothing():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        _, _ = wait(d, 8)

    with pytest.raises(pulse.CompilationError, match=r"cannot unpack None into 2 names: a call"):
        pulse.compile(k).lower(Q[0])


def test_unpacking_too_few_values_counts_them():
    @pulse.kernel
    def k(q):
        _, _ = (1, 2, 3)

    with pytest.raises(pulse.CompilationError, match="cannot unpack 3 values into 2 names"):
        pulse.compile(k).lower(Q[0])


def test_calling_what_a_kernel_cannot_call_names_its_type():
    @pulse.kernel
    def k(q):
        NOT_CALLABLE()

    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(k).lower(Q[0])
    assert str(info.value).startswith("cannot call an object of type 'object' in a kernel; ")


def test_needing_a_float_arguments_value_says_how_to_keep_it_symbolic():
    with pytest.raises(SymbolicValueError) as info:
        int(Parameter("x", 0, "f64"))
    assert str(info.value) == (
        "int() needs the value of float argument 'x', which is only known when the kernel "
        "runs; use it only as a pulse op argument (amplitude, phase, frequency, wait "
        "length), or annotate it as int to compile one kernel per value"
    )
    assert info.value.names == ("x",)


def test_qudits_of_two_devices_of_one_target_name_their_layouts():
    a = pulse.device({"target": "transmon_generic:4"})
    b = pulse.device({"target": "transmon_generic:4", "layout": [2, 3]})

    @pulse.kernel
    def k(q0, q1):
        d, t = get_drive_line(q0)
        drive(d, square(8, 0.1), t)

    with pytest.raises(pulse.DeviceMismatchError) as info:
        k.lower(a.qudit_handles[0], b.qudit_handles[1])
    assert str(info.value) == (
        "k() got qudits of two different devices, both of target 'transmon_generic:4': one "
        "with layout 0..3, the other with layout 2..3; pass handles of one device"
    )


def test_a_failed_call_names_the_function_and_request():
    error = wire.call_failed("decode", 17, 3)
    assert isinstance(error, pulse.ExecutionError)
    assert str(error) == (
        "classical function 'decode' failed on its host: request 17 was answered with "
        "status 3 (0 means success); see the handler's log"
    )


def test_a_udp_call_without_an_answer_is_a_timeout_and_an_execution_error():
    silent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    silent.bind(("127.0.0.1", 0))
    host, port = silent.getsockname()

    f = pulse.classical.external("f", args=(Bits(1),), max_latency_ns=1)
    channel = emulation.UdpChannel((host, port), timeout_s=0.05)
    try:
        with pytest.raises(TimeoutError) as info:
            channel.call(f, wire.encode_request(f, [[1]], 0))
    finally:
        channel.close()
        silent.close()
    assert isinstance(info.value, pulse.ExecutionError)
    assert str(info.value) == (
        f"classical function 'f' got no answer from {host}:{port} within 0.05 s; check "
        "that serve_udp is running there, or raise UdpChannel(timeout_s=...)"
    )


def test_an_unsupported_cast_is_an_internal_error():
    with pytest.raises(pulse.InternalError, match="unsupported symbolic cast to i1; this is a bug"):
        Parameter("x", 0, "f64").cast("i1")
