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
    assert 4 * ns < t <= 20 * ns and t > 0 and t == 20 * ns and t != 20.0


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda t: t + 4, "mixes a time with 4 cycles"),
        (lambda t: 4 - t, "mixes a time with 4 cycles"),
        (lambda t: t * t, "a time times a time"),
        (lambda t: 3 / t, "a number over a time"),
        (lambda t: t < 5, "give the number with units"),
        (lambda t: t**2, "is not a time"),
    ],
)
def test_arithmetic_that_is_not_a_time_is_an_error(expression, message):
    with pytest.raises(TypeError, match=message):
        expression(20 * ns)


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
        ok, fix = check_bit.submit([m]).result(timeout_ns=timeout)
        sync(d, ro)
        if ok:
            drive(d, square(20, 0.3), t)

    with_units = timed.lower(Q[0], 0.3 * us).export_controller()
    in_ns = timed.lower(Q[0], 300).export_controller()
    answers = [next(e for e in p["events"] if e["kind"] == "answer") for p in (with_units, in_ns)]
    assert answers[0]["timeout_ns"] == answers[1]["timeout_ns"] == 300


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
