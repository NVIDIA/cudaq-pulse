# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Calling kernels: binding, specialization, caching, guards, devices, returns."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets import Qubit, Target
from cudaq_pulse.targets._registry import ENV_VAR

_PI_AMPLITUDE = np.pi / 20.0  # a square pulse of 40 cycles (20 ns) at this rate is a pi pulse


def _device(n=4, **kwargs):
    target = Target(
        name="call-test",
        qubits={
            i: Qubit(
                i,
                5.0e9 + 1e8 * i,
                -300e6,
                0.0,
                0.0,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
            )
            for i in range(n)
        },
    )
    return pulse.device({"target": target, **kwargs})


@pulse.kernel
def rabi(q, amp: float, *, duration: int = 40):
    d, t = get_drive_line(q)
    drive(d, square(duration, amp), t)


@pulse.kernel
def flip_and_measure(q):
    d, t = get_drive_line(q)
    drive(d, square(40, _PI_AMPLITUDE), t)
    r, rt = get_readout_line(q)
    sync(d, r)
    return readout(r, square(20, 0.05), rt)


# ── Binding ──────────────────────────────────────────────────────────────


def test_arguments_bind_like_a_python_call():
    q = _device().qudit_handles[0]
    assert rabi.lower(q, 0.1).schedule[0].duration_vtu == 40
    assert rabi.lower(q, amp=0.1, duration=8).schedule[0].duration_vtu == 8
    with pytest.raises(TypeError, match=r"rabi\(\) takes 2 positional arguments but 3"):
        rabi.lower(q, 0.1, 8)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'amp'"):
        rabi.lower(q)
    with pytest.raises(TypeError, match="unexpected keyword argument 'dur'"):
        rabi.lower(q, 0.1, dur=8)


def test_varargs_kernels_are_rejected():
    with pytest.raises(TypeError, match=r"cannot take \*args"):

        @pulse.kernel
        def k(*qs):
            pass


def test_annotations_split_static_and_runtime_parameters():
    assert rabi.runtime_parameters == ("amp",)
    assert rabi.static_parameters == ("q", "duration")


# ── Compile-only and caching ─────────────────────────────────────────────


def test_compile_takes_no_kernel_arguments_and_float_values_never_recompile():
    q = _device().qudit_handles[0]
    ck = pulse.compile(rabi)
    texts = {ck.lower(q, a).mlir() for a in (0.1, 0.2, 0.3)}
    assert len(texts) == 3
    assert ck.cache_info().compiled == 1
    ck.lower(q, 0.1, duration=8)
    assert ck.cache_info().compiled == 2


def test_a_pinned_static_free_kernel_is_built_by_compile():
    @pulse.kernel
    def drive_all(amp: float):
        q = pulse.qudit_ref()
        d, t = get_drive_line(q)
        drive(d, square(40, amp), t)

    dev = _device()
    ck = pulse.compile(drive_all, device=dev)
    assert ck.parameters == ("amp",)
    assert "%arg0: f64" in ck.mlir()
    assert ck.cache_info().compiled == 1


def test_compiled_kernel_without_a_specialization_explains_lower():
    with pytest.raises(TypeError, match=r"specialized on q, duration; inspect one .*\.lower"):
        pulse.compile(rabi).mlir()


def test_the_lru_holds_cache_size_specializations(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "2")

    @pulse.kernel
    def wait_for(q, n: int):
        d, t = get_drive_line(q)
        wait(d, n)

    q = _device().qudit_handles[0]
    for n in (4, 8, 12, 4):
        wait_for.lower(q, n)
    info = wait_for.cache_info()
    assert (info.size, info.compiled, info.evicted, info.capacity) == (2, 4, 2, 2)


def test_cache_size_is_validated(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "lots")
    with pytest.raises(ValueError, match="CUDAQ_PULSE_CACHE_SIZE must be a positive integer"):

        @pulse.kernel
        def k(q):
            pass

    with pytest.raises(ValueError, match="cache_size"):
        pulse.kernel(cache_size=0)(lambda q: None)
    monkeypatch.delenv("CUDAQ_PULSE_CACHE_SIZE")
    assert pulse.kernel(lambda q, amp=0.5: None).lower is not None


# ── Python semantics of globals and closures ─────────────────────────────

AMPLITUDE = 0.1
SETTINGS = {"duration": 40}


@pulse.kernel
def reads_globals(q):
    d, t = get_drive_line(q)
    drive(d, square(SETTINGS["duration"], AMPLITUDE), t)


def test_a_changed_global_is_seen_on_the_next_call():
    global AMPLITUDE
    q = _device().qudit_handles[0]
    first = reads_globals.lower(q)
    assert reads_globals.lower(q) is first
    AMPLITUDE = 0.2
    try:
        second = reads_globals.lower(q)
        assert second is not first and "2.000000e-01" in second.mlir()
        SETTINGS["duration"] = 8
        assert reads_globals.lower(q).schedule[0].duration_vtu == 8
    finally:
        AMPLITUDE = 0.1
        SETTINGS["duration"] = 40
    assert reads_globals.cache_info().compiled == 3


def test_a_changed_closure_cell_is_seen_on_the_next_call():
    amplitude = 0.1

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(40, amplitude), t)

    q = _device().qudit_handles[0]
    assert "1.000000e-01" in k.lower(q).mlir()
    amplitude = 0.3
    assert "3.000000e-01" in k.lower(q).mlir()


def test_a_trace_that_calls_an_impure_host_function_is_never_reused():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(40, random.random()), t)

    q = _device().qudit_handles[0]
    k.lower(q)
    k.lower(q)
    assert k.cache_info().compiled == 2


# ── Devices ──────────────────────────────────────────────────────────────


def test_the_device_comes_from_the_qudit_handles():
    a, b = _device(), _device(layout=[3, 2, 1, 0])
    assert "frequency_hz = 5.000000e+09" in rabi.lower(a.qudit_handles[0], 0.1).mlir()
    assert "frequency_hz = 5.300000e+09" in rabi.lower(b.qudit_handles[0], 0.1).mlir()
    assert rabi.lower(b.qudit_handles[0], 0.1).qudits == (3,)


def test_devices_from_the_same_config_are_equal_and_share_specializations():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        wait(d, 4)

    a, b = _device(), _device()
    assert a == b and a is not b
    assert k.lower(a.qudit_handles[1]) is k.lower(b.qudit_handles[1])


def test_mixing_devices_in_one_call_is_an_error():
    @pulse.kernel
    def pair(q0, q1):
        sync(get_drive_line(q0)[0], get_drive_line(q1)[0])

    a, other = _device(), _device(n=5)
    with pytest.raises(pulse.DeviceMismatchError, match="two devices"):
        pair.lower(a.qudit_handles[0], other.qudit_handles[1])
    pair.lower(a.qudit_handles[0], _device().qudit_handles[1])


def test_a_pinned_compile_rejects_another_device():
    ck = pulse.compile(rabi, device=_device())
    with pytest.raises(pulse.DeviceMismatchError, match="compiled for device"):
        ck.lower(_device(n=5).qudit_handles[0], 0.1)


def test_a_call_without_a_device_raises(monkeypatch):
    monkeypatch.delenv(ENV_VAR)
    with pytest.raises(pulse.NoDeviceError, match="needs a device"):
        rabi.lower(pulse.qudit_ref(), 0.1)


def test_registers_index_like_sequences():
    dev = _device(n=10)
    reg = dev.qudit_handles[0:5]
    assert [q.index for q in reg[1:3]] == [1, 2]
    assert [q.index for q in dev.qudit_handles[[5, 7]]] == [5, 7]
    assert reg[0] == dev.qudit_handles[0]

    @pulse.kernel
    def all_of(qs):
        for q in qs:
            d, t = get_drive_line(q)
            wait(d, 4)

    assert all_of.lower(dev.qudit_handles[[5, 7]]).qudits == (5, 7)


def test_device_values_are_read_only():
    dev = _device()
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        dev.qubit_freq[0] = 5.01e9
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        dev.transition_freq[0, 0, 1] = 5.01e9
    assert dev.transition_freq[1, 0, 1] == dev.qubit_freq[1] == 5.1e9
    assert dev.transition_freq[1, 1, 2] == pytest.approx(5.1e9 - 300e6)
    with pytest.raises(AttributeError):
        dev.target.name = "other"


def test_a_device_from_a_config_file(tmp_path: Path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps({"target": "transmon_krinner_17q", "layout": [4, 5]}))
    dev = pulse.device(path)
    assert [q.index for q in dev.qudit_handles] == [4, 5]
    assert dev == pulse.device({"target": "transmon_krinner_17q", "layout": [4, 5]})
    assert dev != pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    with pytest.raises(ValueError, match="unknown device config keys: lattice"):
        pulse.device({"target": "transmon_krinner_17q", "lattice": []})
    with pytest.raises(NotImplementedError, match="zones are reserved"):
        pulse.device({"target": "transmon_krinner_17q", "zones": []})


def test_a_waveform_as_a_frequency_is_reserved():
    @pulse.kernel
    def chirp(q):
        d, t = get_drive_line(q)
        set_frequency(t, square(8, 0.1))

    with pytest.raises(CompilationError, match="time-varying frequencies"):
        chirp.lower(_device().qudit_handles[0])


def test_a_qudit_the_target_lacks_is_an_error():
    dev = _device(n=2)
    with pytest.raises(CompilationError, match="uses qudit 5, which target 'call-test' does not"):
        pulse.compile(rabi, device=dev).lower(pulse.qudit_ref(5), 0.1)


# ── Frequencies as float arguments ───────────────────────────────────────


@pulse.kernel
def detuned(q, frequency: float):
    d, t = get_drive_line(q)
    t = set_frequency(t, frequency)
    drive(d, square(40, 0.02), t)


def test_a_float_argument_used_as_a_frequency_stays_symbolic():
    q = _device().qudit_handles[0]
    ck = pulse.compile(detuned)
    populations = []
    for offset in (0.0, 10e6, 40e6):
        lowered = ck.lower(q, 5.0e9 + offset)
        assert "%arg0: f64" in lowered.generic.mlir()
        populations.append(pulse.simulate(lowered, dt_ns=0.1).density_matrix[1, 1].real)
    assert ck.cache_info().compiled == 1
    assert populations[0] > populations[1] > populations[2]


# ── Execution and return values ──────────────────────────────────────────


def test_calling_a_kernel_runs_it_and_returns_none_without_a_return():
    assert rabi(_device().qudit_handles[0], 0.1) is None


def test_a_returned_measurement_is_its_outcome():
    q = _device().qudit_handles[0]
    assert flip_and_measure(q) == 1
    assert pulse.compile(flip_and_measure)(q) == 1


def test_returned_values_known_at_compile_time_and_tuples_of_measurements():
    @pulse.kernel
    def both(q0, q1):
        r0, t0 = get_readout_line(q0)
        r1, t1 = get_readout_line(q1)
        return readout(r0, square(20, 0.05), t0), readout(r1, square(20, 0.05), t1), 7

    dev = _device()
    assert both(dev.qudit_handles[0], dev.qudit_handles[1]) == (0, 0, 7)


def test_returning_a_value_of_a_float_parameter_is_an_error():
    @pulse.kernel
    def echo(q, amp: float):
        return amp * 2

    with pytest.raises(CompilationError, match="computed from a float parameter"):
        echo.lower(_device().qudit_handles[0], 0.1)


# ── Launcher ─────────────────────────────────────────────────────────────


def test_the_launcher_sets_the_process_target(tmp_path: Path):
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import cudaq_pulse as pulse
            print(pulse.get_device().name, sys.argv[1:])
            """
        )
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != ENV_VAR}
    run = subprocess.run(
        [sys.executable, "-m", "cudaq_pulse", "--target", "transmon_generic:3", str(script), "-x"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "transmon_generic:3 ['-x']"
    assert "target transmon_generic:3" in run.stderr
    bad = subprocess.run(
        [sys.executable, "-m", "cudaq_pulse", "--target", "nope", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert bad.returncode != 0 and "no registered target 'nope'" in bad.stderr


# ── Floats that shape the program are specialized on ─────────────────────


def test_a_float_that_decides_control_flow_or_a_duration_is_specialized_on():
    @pulse.kernel
    def shaped(q, amp: float, width: float):
        d, t = get_drive_line(q)
        if amp > 0.5:
            wait(d, 4)
        drive(d, square(int(width), amp), t)

    q = _device().qudit_handles[0]
    ck = pulse.compile(shaped)
    assert ck.lower(q, 0.7, 16.0).schedule[-1].duration_vtu == 16
    assert len(ck.lower(q, 0.2, 8.0).schedule) == 1
    assert shaped.static_parameters == ("q", "amp", "width")
    assert shaped.runtime_parameters == ()


def test_a_float_only_used_as_a_value_stays_symbolic():
    q = _device().qudit_handles[0]
    lowered = rabi.lower(q, 0.3)
    assert lowered.arguments == (0.3,)
    assert "%arg0: f64" in lowered.generic.mlir()
    assert "3.000000e-01" in lowered.mlir()


def test_simulating_new_float_values_shares_the_module():
    q = _device().qudit_handles[0]
    first = rabi.lower(q, 0.01)
    second = rabi.lower(q, 0.02)
    a = pulse.simulate(first, dt_ns=0.5).density_matrix[1, 1].real
    b = pulse.simulate(second, dt_ns=0.5).density_matrix[1, 1].real
    assert first.generic is second.generic
    # The CPU path evaluates the lowering at the values: no copy of the module.
    assert first._concrete_module is None and second._concrete_module is None
    assert b > a > 0


def test_concurrent_simulations_of_one_specialization_agree():
    from concurrent.futures import ThreadPoolExecutor

    q = _device().qudit_handles[0]
    calls = [rabi.lower(q, 0.01 * (i % 4 + 1)) for i in range(16)]
    expected = [pulse.simulate(c, dt_ns=0.5).density_matrix for c in calls]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda c: pulse.simulate(c, dt_ns=0.5).density_matrix, calls))
    for g, e in zip(got, expected):
        np.testing.assert_allclose(g, e, atol=1e-12)


def test_a_qudit_passed_twice_is_an_error():
    @pulse.kernel
    def pair(q0, q1):
        pass

    q = _device().qudit_handles[0]
    with pytest.raises(TypeError, match="got qudit 0 more than once"):
        pair.lower(q, q)


def test_each_call_returns_its_own_container():
    @pulse.kernel
    def listing(q):
        return [1, 2]

    q = _device().qudit_handles[0]
    first, second = listing(q), listing(q)
    assert first == second == [1, 2] and first is not second


def test_the_schedule_option_is_part_of_the_compiled_kernel():
    @pulse.kernel
    def two(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        drive(d0, square(40, 0.1), t0)
        drive(d0, square(40, 0.1), t0)
        drive(d1, square(20, 0.1), t1)
        sync(d0, d1)

    dev = _device()
    args = (dev.qudit_handles[0], dev.qudit_handles[1])
    starts = {
        policy: [e.start_vtu for e in pulse.compile(two, schedule=policy).lower(*args).schedule]
        for policy in ("asap", "alap")
    }
    assert starts["asap"] != starts["alap"]
    with pytest.raises(ValueError, match="Unknown schedule policy"):
        pulse.compile(two, schedule="soon")


# ── Kernels calling kernels ──────────────────────────────────────────────


@pulse.kernel
def pulse_once(q, amp: float):
    d, t = get_drive_line(q)
    drive(d, square(8, amp), t)


def test_a_kernel_called_in_a_range_loop_keeps_the_loop_rolled():
    @pulse.kernel
    def play(d, t, amp: float):
        drive(d, square(8, amp), t)

    @pulse.kernel
    def train(q, n: int):
        d, t = get_drive_line(q)
        for _ in range(n):
            play(d, t, 0.01)

    q = _device().qudit_handles[0]
    small, large = train.lower(q, 4), train.lower(q, 64)
    assert small.metrics.op_count == large.metrics.op_count
    assert large.mlir().count("scf.for") == 1
    assert sum(e.kind == "drive" for e in large.schedule) == 64


def test_a_kernel_called_with_the_loop_variable_sees_each_value():
    @pulse.kernel
    def idle(q, i):
        d, t = get_drive_line(q)
        wait(d, 4 * (i + 1))

    @pulse.kernel
    def ramp(q, n: int):
        for i in range(n):
            idle(q, i)

    q = _device().qudit_handles[0]
    waits = [e.duration_vtu for e in ramp.lower(q, 3).schedule]
    assert waits == [4, 8, 12]


def test_a_kernel_called_on_a_qudit_indexed_by_the_loop_variable():
    @pulse.kernel
    def each(qs, n: int):
        for i in range(n):
            pulse_once(qs[i], 0.02)

    dev = _device()
    lowered = each.lower(dev.qudit_handles[0:4], 4)
    assert sorted(e.qubit for e in lowered.schedule) == [0, 1, 2, 3]


def test_nested_kernel_calls_and_tone_state_carried_across_iterations():
    @pulse.kernel
    def advance(d, t):
        shift_phase(t, 0.25)
        drive(d, square(8, 0.02), t)

    @pulse.kernel
    def block(d, t):
        advance(d, t)
        advance(d, t)

    @pulse.kernel
    def carried(q, n: int):
        d, t = get_drive_line(q)
        for _ in range(n):
            block(d, t)

    q = _device().qudit_handles[0]
    lowered = pulse.compile(carried, passes=()).lower(q, 5)
    text = lowered.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.shift_phase") == 2
    # The tone, with its phase, is carried from one iteration to the next.
    assert "!pulse.tone" in next(line for line in text.splitlines() if "scf.yield" in line)
    assert sum(e.kind == "drive" for e in lowered.schedule) == 10


def test_a_helper_that_calls_a_kernel_is_inlined():
    def helper(q):
        pulse_once(q, 0.05)

    @pulse.kernel
    def via_helper(q):
        helper(q)

    assert (
        sum(e.kind == "drive" for e in via_helper.lower(_device().qudit_handles[0]).schedule) == 1
    )


# ── One drive channel per qudit ──────────────────────────────────────────


@pulse.kernel
def x90(q):
    d, t = get_drive_line(q)
    drive(d, square(16, 0.02), t)


def test_two_calls_that_acquire_the_same_drive_line_play_in_sequence():
    @pulse.kernel
    def twice(q):
        x90(q)
        x90(q)

    # Without fusion, which would merge the two pulses into one.
    lowered = pulse.compile(twice, passes=()).lower(_device().qudit_handles[0])
    assert [(e.start_vtu, e.duration_vtu) for e in lowered.schedule] == [(0, 16), (16, 16)]
    assert lowered.mlir().count("pulse.get_drive_line") == 1


def test_a_callee_shares_the_line_its_caller_holds():
    @pulse.kernel
    def outer(q):
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.5)
        drive(d, square(8, 0.02), t)
        x90(q)
        drive(d, square(8, 0.02), t)

    lowered = pulse.compile(outer, passes=()).lower(_device().qudit_handles[0])
    assert [e.start_vtu for e in lowered.schedule] == [0, 8, 24]
    # The callee's pulse carries the caller's phase.
    assert lowered.mlir().count("pulse.shift_phase") == 1


def test_two_names_for_one_line_stay_in_step():
    @pulse.kernel
    def aliased(q):
        d1, t1 = get_drive_line(q)
        d2, t2 = get_drive_line(q)
        drive(d1, square(8, 0.02), t1)
        drive(d2, square(8, 0.02), t2)
        drive(d1, square(8, 0.02), t1)

    lowered = pulse.compile(aliased, passes=()).lower(_device().qudit_handles[0])
    assert [e.start_vtu for e in lowered.schedule] == [0, 8, 16]


def test_a_callee_acquiring_the_line_in_a_rolled_loop_is_hoisted():
    @pulse.kernel
    def train(q, n: int):
        for _ in range(n):
            x90(q)

    q = _device().qudit_handles[0]
    small, large = train.lower(q, 4), train.lower(q, 64)
    assert small.metrics.op_count == large.metrics.op_count
    text = large.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.get_drive_line") == 1
    assert text.index("pulse.get_drive_line") < text.index("scf.for")
    assert [e.start_vtu for e in large.schedule] == [16 * i for i in range(64)]


def test_a_qudit_allocated_in_a_loop_body_still_unrolls():
    @pulse.kernel
    def fresh_each(n: int):
        for _ in range(n):
            x90(pulse.qudit_ref())

    lowered = pulse.compile(fresh_each, device=_device()).lower(3)
    assert "scf.for" not in lowered.mlir()
    assert len({e.line for e in lowered.schedule}) == 3
