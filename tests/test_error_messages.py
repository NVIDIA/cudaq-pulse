# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error messages say what is wrong, where, and what to write instead."""

from __future__ import annotations

import inspect
import socket

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import ops
from cudaq_pulse.classical import emulation, wire
from cudaq_pulse.kernel.ir_builder import Parameter, SymbolicValueError

from tests._handles import Q

NOT_CALLABLE = object()


@pytest.mark.parametrize("name", ops.__all__)
def test_an_op_called_outside_a_kernel_is_a_compilation_error_naming_it(name):
    op = getattr(ops, name)
    required = [
        p
        for p in inspect.signature(op).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    message = rf"^{name}\(\) must be called inside a @cudaq_pulse\.kernel function$"
    with pytest.raises(pulse.CompilationError, match=message):
        op(*[None] * len(required))


def test_qudit_handles_in_a_list_say_how_to_pass_them():
    @pulse.kernel
    def k(qs):
        d = get_drive_line(qs[0])
        drive(d, square(8, 0.1))

    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(k).lower([Q[0], Q[1]])
    assert str(info.value) == (
        "get_drive_line(): argument 1 is a qudit handle that reached the kernel inside a "
        "list; pass qudits as separate arguments, a Register (qs.qudits[0:4]) or a "
        "tuple, not a list"
    )


def test_a_name_used_before_it_is_assigned_is_a_plain_compilation_error():
    @pulse.kernel
    def k(q, flag: int):
        if flag:
            w = square(8, 0.1)
        d = get_drive_line(q)
        drive(d, w)

    pulse.compile(k).lower(Q[0], 1)
    with pytest.raises(pulse.CompilationError) as info:
        pulse.compile(k).lower(Q[0], 0)
    assert type(info.value) is pulse.CompilationError
    assert str(info.value) == (
        "'w' is used before it is assigned; assign it on every path that reaches this "
        "line (an if on a value known while tracing takes only one arm)"
    )
    assert "drive(d, w)" in "".join(info.value.__notes__)


def test_unpacking_what_an_op_returns_says_it_returns_nothing():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
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
    a = pulse.quantum_system("transmon_lattice:4")
    b = pulse.quantum_system("transmon_lattice:4", layout=[2, 3])

    @pulse.kernel
    def k(q0, q1):
        d = get_drive_line(q0)
        drive(d, square(8, 0.1))

    with pytest.raises(pulse.QuantumSystemMismatchError) as info:
        k.lower(a.qudits[0], b.qudits[1])
    assert str(info.value) == (
        "k() got qudits of two different quantum systems, both named 'transmon_lattice:4': one "
        "with layout 0..3, the other with layout 2..3; pass handles of one quantum system"
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

    @pulse.classical.external(max_latency_ns=1)
    def f(bits: pulse.Length[pulse.Bits, 1]) -> None:
        pass

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


@pytest.fixture(scope="module")
def idle():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wait(d, 8)

    return k.lower(Q[0])


@pytest.mark.parametrize(
    ("shots_count", "error"),
    [
        (0, pulse.InvalidValueError),
        (-1, pulse.InvalidValueError),
        (2.5, pulse.InvalidTypeError),
        ("ten", pulse.InvalidTypeError),
        (True, pulse.InvalidTypeError),
    ],
)
@pytest.mark.parametrize("run", ["simulate", "emulate"])
def test_shots_is_a_positive_int(idle, run, shots_count, error):
    call = pulse.simulate if run == "simulate" else emulation.emulate
    with pytest.raises(error) as info:
        call(idle, shots_count=shots_count)
    assert str(info.value) == f"shots_count must be a positive int, got {shots_count!r}"


def test_no_shots_is_the_documented_simulate_default(idle):
    assert pulse.simulate(idle, shots_count=None).records is None


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (999, "qudit index 999 is out of range: quantum system 'transmon_surface17' has 17 qudits"),
        (-18, "qudit index -18 is out of range: quantum system 'transmon_surface17' has 17 qudits"),
        ([0, 999], "qudit index 999 is out of range"),
    ],
)
def test_a_qudit_index_past_the_device_says_how_many_qudits_it_has(index, message):
    handles = pulse.quantum_system("transmon_surface17").qudits
    with pytest.raises(pulse.IndexOutOfRangeError, match=message) as info:
        handles[index]
    assert isinstance(info.value, IndexError)
    assert str(info.value).endswith("indexed 0 to 16 (or -17 to -1)")


def test_a_qudit_index_past_a_register_says_how_many_qudits_it_has():
    register = pulse.quantum_system("transmon_surface17").qudits[0:3]
    with pytest.raises(pulse.IndexOutOfRangeError) as info:
        register[5]
    assert str(info.value) == (
        "qudit index 5 is out of range: the register has 3 qudits, indexed 0 to 2 (or -3 to -1)"
    )
