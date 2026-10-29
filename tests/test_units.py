# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Times with units: ``20 * pulse.ns`` in every duration, converted to cycles
of the device's clock exactly when the kernel compiles."""

from __future__ import annotations

import pickle
import re
from fractions import Fraction

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimits

from tests._handles import Q

ns, us = pulse.ns, pulse.us


def _events(lowered):
    return [(e.kind, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


# ── The arithmetic ───────────────────────────────────────────────────────────


def test_a_time_holds_its_exact_value():
    assert (0.1 * us).exact_ns == 100
    assert (20 * ns).exact_ns == Fraction(20)
    assert (1 / 3 * ns).exact_ns == Fraction("0.3333333333333333")
    assert isinstance(20 * ns, pulse.Time) and isinstance(20 * ns, float)
    assert float(4 * us) == 4000.0
    assert [repr(t) for t in (20 * ns, 4 * us, 2.5 * pulse.ms, 3 * pulse.s)] == [
        "20 ns",
        "4 us",
        "2.5 ms",
        "3 s",
    ]
    assert pickle.loads(pickle.dumps(0.1 * us)).exact_ns == 100


def test_arithmetic_keeps_the_unit_where_it_means_a_time():
    t = 20 * ns
    for value, want in [
        (t * 2, 40),
        (2 * t, 40),
        (t + 4 * ns, 24),
        (t - 4 * ns, 16),
        (t / 4, 5),
        (-t, -20),
        (abs(-t), 20),
        (sum([t, t, t]), 60),
    ]:
        assert value.__class__ is pulse.Time and value.exact_ns == want
    assert t / (4 * ns) == 5.0 and (t / (4 * ns)).__class__ is float
    assert t // (3 * ns) == 6 and (t % (3 * ns)).exact_ns == 2
    assert 4 * ns < t <= 20 * ns and t > 0 * ns and t == 20 * ns and t != 4 * ns
    assert float(t) == 20.0 and float(2 * us) == 2000.0


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda t: t + 4, "mixes a time with 4 cycles"),
        (lambda t: 4 - t, "mixes a time with 4 cycles"),
        (lambda t: t * t, "a time times a time"),
        (lambda t: 3 / t, "a number over a time"),
        (lambda t: t < 5, "compares a time with a bare number"),
        (lambda t: t**2, "is not a time"),
    ],
)
def test_arithmetic_that_is_not_a_time_is_an_error(expression, message):
    with pytest.raises(TypeError, match=message):
        expression(20 * ns)


@pytest.mark.parametrize(
    "compare",
    [
        lambda t, n: t == n,
        lambda t, n: n == t,
        lambda t, n: t != n,
        lambda t, n: t < n,
        lambda t, n: n <= t,
        lambda t, n: t > n,
        lambda t, n: t >= n,
    ],
)
@pytest.mark.parametrize("number", [20, 20.0, 0])
def test_a_time_compared_with_a_bare_number_is_an_error(compare, number):
    message = r"compares a time with a bare number; compare it with a time \(.* \* pulse\.ns\)"
    with pytest.raises(TypeError, match=message + r", or compare float\(t\)"):
        compare(20 * ns, number)
    with pytest.raises(TypeError, match="compares a time with a bare number"):
        _ = 20 * ns == Fraction(20)


def test_a_time_never_hashes_like_a_number():
    times = [20 * ns, 0 * ns, -3 * us, 0.1 * us, 1 / 3 * ns]
    for t in times:
        assert hash(t) != hash(float(t)) and hash(t) != hash(t.exact_ns)
        assert hash(t) == hash(pulse.Time(t.exact_ns))
    keys = {20 * ns: "time", 20.0: "float"}
    assert len(keys) == 2 and keys[20 * ns] == "time" and keys[20] == "float"
    assert 20.0 not in {20 * ns} and 20 * ns in {20 * ns, 20.0}
    assert {20 * ns, 20 * ns, 0.02 * us} == {20 * ns}
    with pytest.raises(TypeError, match="compares a time with a bare number"):
        _ = 20.0 in [20 * ns]
    assert 20 * ns in [20 * ns, 4 * ns] and "a" != 20 * ns


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda: float("nan") * ns, "finite number of nanoseconds, not nan"),
        (lambda: float("inf") * ns, "finite number of nanoseconds, not inf"),
        (lambda: ns * float("-inf"), "finite number of nanoseconds, not -inf"),
        (lambda: ns / float("inf"), "finite number of nanoseconds, not inf"),
        (lambda: 10**400 * ns, r"at most 1\.7976931348623157e\+308 ns"),
        (lambda: 1e308 * pulse.us, r"at most 1\.7976931348623157e\+308 ns"),
    ],
)
def test_a_time_that_is_not_a_finite_float_is_an_error(expression, message):
    with pytest.raises(pulse.InvalidValueError, match=message):
        expression()


_LONGEST = "a duration is at most 2**53 cycles"
_SHORTEST = "a waveform lasts at least 1 cycle"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("wait(d, -5)", "wait(): duration is -5 cycles; a wait cannot be negative"),
        ("wait(d, -4 * pulse.ns)", "wait(): duration -4 ns is -8 cycles; a wait cannot be"),
        ("drive(d, square(-8, 0.1), t)", f"square(): duration is -8 cycles; {_SHORTEST}"),
        ("drive(d, square(0, 0.1), t)", f"square(): duration is 0 cycles; {_SHORTEST}"),
        (
            "drive(d, square(0 * pulse.ns, 0.1), t)",
            f"square(): duration 0 ns is 0 cycles; {_SHORTEST}",
        ),
        ("drive(d, gaussian(0, 0.1, 10.0), t)", f"gaussian(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, drag(-1, 0.1, 10.0, 0.5), t)", f"drag(): duration is -1 cycles; {_SHORTEST}"),
        ("drive(d, cosine(0, 0.1), t)", f"cosine(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, tanh_ramp(0, 0.1, 4.0), t)", f"tanh_ramp(): duration is 0 cycles; {_SHORTEST}"),
        (
            "drive(d, gaussian_square(64, 0.1, 4.0, -1), t)",
            "gaussian_square(): width is -1 cycles; a duration cannot be negative",
        ),
        ("wait(d, 2**53 + 1)", f"wait(): duration is {2**53 + 1} cycles; {_LONGEST}"),
        ("drive(d, square(2**62, 0.1), t)", f"square(): duration is {2**62} cycles; {_LONGEST}"),
        ("drive(d, square(2**63, 0.1), t)", f"square(): duration is {2**63} cycles; {_LONGEST}"),
        ("wait(d, 1e30 * pulse.ns)", f"wait(): duration 1e+21 s is {2 * 10**30} cycles"),
    ],
)
def test_a_duration_outside_its_cycles_is_an_error_naming_the_line(call, message):
    source = f"def k(q):\n    d, t = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.kernel(namespace["k"]).lower(Q[0])
    assert str(caught.value).startswith(message)
    assert 'File "<kernel>", line 3, in k' in "\n".join(caught.value.__notes__)


def test_a_negative_wait_parameter_is_an_error():
    @pulse.kernel
    def k(q, n: int):
        d, t = get_drive_line(q)
        wait(d, n)

    with pytest.raises(pulse.CompilationError, match="duration is -1 cycles; a wait cannot be"):
        k.lower(Q[0], -1)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("wait(d, np.int64(-5))", "wait(): duration is -5 cycles; a wait cannot be negative"),
        ("drive(d, square(np.int32(0), 0.1), t)", f"square(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, square(np.int64(2**60), 0.1), t)", f"is {2**60} cycles; {_LONGEST}"),
        ("drive(d, square(True, 0.1), t)", "square duration must be an integer, got bool"),
    ],
)
def test_an_integer_of_another_type_follows_the_same_rule(call, message):
    source = f"def k(q):\n    d, t = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse, "np": np}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError, match=re.escape(message)):
        pulse.kernel(namespace["k"]).lower(Q[0])


@pytest.mark.filterwarnings("ignore::cudaq_pulse.PulseWarning")
def test_a_float_argument_compiled_into_a_duration_follows_the_same_rule():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        drive(d, square(tau, 0.1), t)

    assert k.lower(Q[0], 8.0) is not None
    with pytest.raises(pulse.CompilationError, match="duration is 0 cycles; a waveform lasts"):
        k.lower(Q[0], 0.0)


def test_the_longest_duration_is_2_53_cycles():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        wait(d, 2**53)
        wait(d, 2**52 * ns)
        wait(d, 0)

    waits = [e.duration_cycles for e in k.lower(Q[0]).schedule if e.kind == "wait"]
    assert waits == [2**53, 2**53, 0]


# ── Durations ────────────────────────────────────────────────────────────────


def test_a_float_duration_is_an_error_that_shows_both_spellings():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, gaussian(20.0, 0.1, 4.0), t)

    with pytest.raises(pulse.CompilationError) as caught:
        k.lower(Q[0])
    assert str(caught.value).split("\nwhile tracing")[0] == (
        "gaussian(): duration is a float (20.0). Durations are an int number of cycles, "
        "or a time with units:\n"
        "    gaussian(20, ...)               # 20 cycles\n"
        "    gaussian(20 * pulse.ns, ...)    # 20 nanoseconds"
    )


@pytest.mark.parametrize(
    ("call", "what"),
    [
        ("drive(d, square(8.0, 0.1), t)", "square(): duration"),
        ("drive(d, gaussian_square(40, 0.1, 4.0, 20.0), t)", "gaussian_square(): width"),
        ("wait(d, 8.0)", "wait(): duration"),
        ("wait(d, 8, bounds=(0, 40.0))", "wait(): bounds"),
        ("wait(d, pulse.round_duration(d, 2.5))", "round_duration(): duration"),
    ],
)
def test_every_duration_rejects_a_float(call, what):
    source = f"def k(q):\n    d, t = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError, match="^" + re.escape(f"{what} is a float")):
        pulse.kernel(namespace["k"]).lower(Q[0])


def test_shape_times_take_fractional_cycles():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, drag(40, 0.1, 4.5, 0.5), t)

    assert k.lower(Q[0]).schedule


def _same_module(with_units, in_cycles, *args):
    def body(kernel):
        return kernel.lower(Q[0], *args).mlir().split("\n", 1)[1]

    assert body(with_units) == body(in_cycles)


@pulse.kernel
def waveforms_with_units(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(20 * ns, 0.1, 2.5 * ns), t)
    drive(d, square(4 * ns, 0.1), t)
    drive(d, drag(40 * ns, 0.1, 5 * ns, 0.25 * ns), t)
    drive(d, cosine(10 * ns, 0.1), t)
    drive(d, tanh_ramp(10 * ns, 0.1, 1 * ns), t)
    drive(d, gaussian_square(40 * ns, 0.1, 2 * ns, 20 * ns), t)
    wait(d, 0.1 * us)
    sync(d, r)
    readout(r, square(1 * us, 0.05), rt)


@pulse.kernel
def waveforms_in_cycles(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(40, 0.1, 5.0), t)
    drive(d, square(8, 0.1), t)
    drive(d, drag(80, 0.1, 10.0, 0.5), t)
    drive(d, cosine(20, 0.1), t)
    drive(d, tanh_ramp(20, 0.1, 2.0), t)
    drive(d, gaussian_square(80, 0.1, 4.0, 40), t)
    wait(d, 200)
    sync(d, r)
    readout(r, square(2000, 0.05), rt)


def test_times_compile_to_the_module_of_their_cycles():
    _same_module(waveforms_with_units, waveforms_in_cycles)


def _envelope(t):
    return 0.1


@pulse.kernel
def custom_with_units(q):
    d, t = get_drive_line(q)
    drive(d, custom(4 * ns, _envelope), t)


@pulse.kernel
def custom_in_cycles(q):
    d, t = get_drive_line(q)
    drive(d, custom(8, _envelope), t)


def test_a_custom_envelope_takes_a_time():
    _same_module(custom_with_units, custom_in_cycles)


def test_a_time_that_is_not_whole_cycles_is_an_error():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        wait(d, 0.3 * ns)

    with pytest.raises(pulse.CompilationError) as caught:
        k.lower(Q[0])
    assert str(caught.value).startswith(
        "wait(): duration 0.3 ns is 0.6 cycles of the 2 GHz program clock, and durations "
        "are whole cycles; round it with pulse.round_duration(line, 0.3 * pulse.ns)"
    )


def test_round_duration_rounds_a_time_onto_the_grid():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(pulse.round_duration(d, 10.3 * ns), 0.1), t)

    machine = MachineModel(controller=ControllerLimits(clock_ghz=0.5))
    assert [e[2] for e in _events(k.lower(Q[0]))] == [21]
    assert [e[2] for e in _events(pulse.compile(k, machine=machine).lower(Q[0]))] == [24]


def test_loops_and_parameters_take_times():
    @pulse.kernel
    def k(q, step, n: int):
        d, t = get_drive_line(q)
        for i in range(n):
            drive(d, square(8, 0.1), t)
            wait(d, step * (i + 1))

    events = _events(k.lower(Q[0], 10 * ns, 3))
    assert [e for e in events if e[0] == "wait"] == [
        ("wait", 8, 20),
        ("wait", 36, 40),
        ("wait", 84, 60),
    ]
    assert k.cache_info().compiled == 1
    k.lower(Q[0], 5 * ns, 3)
    assert k.cache_info().compiled == 2

    @pulse.kernel(unroll_threshold=0)
    def rolled(q, step):
        d, t = get_drive_line(q)
        for _ in range(40):
            drive(d, square(8, 0.1), t)
            wait(d, step)

    lowered = rolled.lower(Q[0], 10 * ns)
    assert "scf.for" in (lowered.generic or lowered).mlir()
    assert lowered.schedule[-1].start_cycles == 39 * 28 + 8


def test_a_float_parameter_in_nanoseconds_is_one_run_time_multiply():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        drive(d, square(8, 0.1), t)
        wait(d, tau * ns, bounds=(0, 1 * us))
        drive(d, square(8, 0.1), t)

    compiled = pulse.compile(k)
    assert [e[1] for e in _events(compiled.lower(Q[0], 30.0)) if e[0] == "drive"] == [0, 68]
    assert [e[1] for e in _events(compiled.lower(Q[0], 50.0)) if e[0] == "drive"] == [0, 108]
    assert compiled.cache_info().compiled == 1
    with pytest.raises(TypeError, match=r"argument 'tau' is a time \(30 ns\)"):
        compiled(Q[0], 30 * ns)


def test_mixing_units_in_a_run_time_length_is_an_error():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        wait(d, tau * ns + 4)

    with pytest.raises(pulse.CompilationError, match="mixes a time with a number of cycles"):
        k.lower(Q[0], 1.0)


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=60, link="controller")
def wait_length(x):
    return x


@pulse.kernel
def answer_in_nanoseconds(q):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    f = wait_length.submit(1.0)
    wait(d, f.result() * pulse.ns, bounds=(0, 4 * pulse.us))
    drive(d, square(8, 0.1), t)


def test_an_answer_in_nanoseconds_converts_with_one_multiply():
    machine = MachineModel(links={"controller": Link(fixed_ns=10)})
    text = pulse.compile(answer_in_nanoseconds, machine=machine).lower(Q[0]).mlir()
    taken = text.index("classical.wait")
    converted = text[taken:]
    assert converted.count("arith.mulf") == 1
    assert "arith.constant 2.000000e+00 : f64" in text
    assert "pulse.max_cycles = 8000" in converted


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=400)
def check_bit(bits):
    return [bits[0]]


def test_a_timeout_takes_a_time():
    @pulse.kernel
    def timed(q, timeout):
        d, t = get_drive_line(q)
        ro, rt = get_readout_line(q)
        m = readout(ro, square(100, 0.05), rt)
        ok, fix = check_bit.submit([m]).result(timeout=timeout)
        sync(d, ro)
        if ok:
            drive(d, square(20, 0.3), t)

    with_units = timed.lower(Q[0], 0.3 * us)
    in_cycles = timed.lower(Q[0], 600)
    assert with_units.mlir() == in_cycles.mlir()
    exports = (with_units.export_controller(), in_cycles.export_controller())
    answers = [next(e for e in p["events"] if e["kind"] == "answer") for p in exports]
    assert answers[0]["timeout_ns"] == answers[1]["timeout_ns"] == 300


def _timeout_error(timeout):
    source = (
        "def k(q):\n"
        "    ro, rt = get_readout_line(q)\n"
        "    m = readout(ro, square(100, 0.05), rt)\n"
        f"    check_bit.submit([m]).result({timeout})\n"
    )
    namespace = {"pulse": pulse, "check_bit": check_bit}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.kernel(namespace["k"]).lower(Q[0])
    return str(caught.value).split("\nwhile tracing")[0]


def test_a_float_timeout_is_the_float_duration_error():
    assert _timeout_error("timeout=350.0") == (
        "result(): timeout is a float (350.0). Durations are an int number of cycles, "
        "or a time with units:\n"
        "    f(...).result(timeout=350)               # 350 cycles\n"
        "    f(...).result(timeout=350 * pulse.ns)    # 350 nanoseconds"
    )


def test_a_timeout_off_the_cycles_is_an_error():
    assert _timeout_error("timeout=0.3 * pulse.ns").startswith(
        "result(): timeout 0.3 ns is 0.6 cycles of the 2 GHz program clock, and durations "
        "are whole cycles; round it with pulse.round_duration(line, 0.3 * pulse.ns)"
    )


@pytest.mark.parametrize(
    ("timeout", "message"),
    [
        ("timeout_ns=300", "result() got an unexpected keyword 'timeout_ns'"),
        ("timeout=0", "result(): timeout must be positive, got 0"),
        ("timeout=-2 * pulse.ns", "result(): timeout must be positive"),
    ],
)
def test_timeout_misuse(timeout, message):
    assert _timeout_error(timeout).startswith(message)


# ── One kernel, two clocks ───────────────────────────────────────────────────


cal = {"x_duration": 20 * ns, "x_sigma": 5 * ns, "x_amplitude": 0.2, "gap": 0.1 * us}


@pulse.kernel
def x_then_wait(q):
    d, t = get_drive_line(q)
    drive(d, gaussian(cal["x_duration"], cal["x_amplitude"], cal["x_sigma"]), t)
    wait(d, cal["gap"])
    drive(d, gaussian(cal["x_duration"], cal["x_amplitude"], cal["x_sigma"]), t)


def test_one_kernel_compiles_for_devices_with_different_clocks():
    fast = pulse.device({"target": "transmon_krinner_17q", "clock_ghz": 2.0})
    slow = pulse.device({"target": "transmon_krinner_17q", "clock_ghz": 1.0})
    on_fast = pulse.compile(x_then_wait, device=fast).lower(fast.qudit_handles[0])
    on_slow = pulse.compile(x_then_wait, device=slow).lower(slow.qudit_handles[0])
    assert _events(on_fast) == [("drive", 0, 40), ("wait", 40, 200), ("drive", 240, 40)]
    assert _events(on_slow) == [("drive", 0, 20), ("wait", 20, 100), ("drive", 120, 20)]
    # The same nanoseconds on both.
    for lowered, device in ((on_fast, fast), (on_slow, slow)):
        end = lowered.schedule[-1].start_cycles + lowered.schedule[-1].duration_cycles
        assert end / device.clock_ghz == 140.0
