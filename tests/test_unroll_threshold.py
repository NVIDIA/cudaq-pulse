# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small static loops are unrolled while tracing, larger ones stay rolled;
either way the program is the same."""

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from cudaq_pulse.kernel._unroll import DEFAULT_UNROLL_THRESHOLD
from cudaq_pulse.kernel.bytecode_bridge import compile_kernel_bytecode
from cudaq_pulse.targets import Coupling, MicrowaveDrive, Qubit, TargetDescription

TARGET = TargetDescription(
    name="unroll-pair",
    qubits={
        i: Qubit(
            i,
            5.0e9 + 0.2e9 * i,
            -300e6,
            0.0,
            0.0,
            drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
        )
        for i in range(2)
    },
    couplings=[Coupling((0, 1), 2e6)],
)
DEV = pulse.device(TARGET)
Q = DEV.qudit_handles


def _train(n):
    @pulse.kernel
    def train(q):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(4, 0.02))
            wait(d, 2)

    return train


def _rolled(lowered):
    return "scf.for" in lowered.mlir()


def test_the_default_unrolls_a_loop_up_to_the_threshold():
    assert DEFAULT_UNROLL_THRESHOLD == 8
    # Three calls per iteration (drive, square, wait): 2 x 3 <= 8 < 3 x 3.
    assert not _rolled(_train(2).lower(Q[0]))
    assert _rolled(_train(3).lower(Q[0]))


@pytest.mark.parametrize(("threshold", "rolled"), [(0, True), (11, True), (12, False)])
def test_the_estimate_is_the_trip_count_times_the_calls_in_the_body(threshold, rolled):
    lowered = pulse.compile(_train(4), unroll_threshold=threshold).lower(Q[0])
    assert _rolled(lowered) is rolled


def test_the_kernel_option_and_compile_override():
    @pulse.kernel(unroll_threshold=1000)
    def k(q):
        d = get_drive_line(q)
        for _ in range(10):
            drive(d, square(4, 0.02))

    assert not _rolled(k.lower(Q[0]))
    assert not _rolled(pulse.compile(k).lower(Q[0]))
    assert _rolled(pulse.compile(k, unroll_threshold=0).lower(Q[0]))
    message = r"must be a non-negative int \(0 means never unroll\), got "
    for bad in (-1, 1.5, "8", True, np.int64(8)):
        with pytest.raises(ValueError, match=message):
            pulse.compile(k, unroll_threshold=bad)
        with pytest.raises(ValueError, match=message):
            pulse.kernel(unroll_threshold=bad)
        with pytest.raises(ValueError, match=message):
            compile_kernel_bytecode(k.__wrapped__, unroll_threshold=bad)


def test_a_loop_whose_variable_is_read_is_unrolled_at_any_threshold():
    @pulse.kernel
    def ramp(q):
        d = get_drive_line(q)
        for i in range(20):
            drive(d, square(4, 0.001 * i))

    assert not _rolled(pulse.compile(ramp, unroll_threshold=0).lower(Q[0]))


def test_a_compiled_kernel_keeps_its_threshold():
    k = _train(2)
    rolled = pulse.compile(k, unroll_threshold=0)
    assert _rolled(rolled.lower(Q[0]))
    assert not _rolled(k.lower(Q[0]))
    k(Q[0])
    rolled(Q[0])
    assert _rolled(rolled.lower(Q[0]))
    assert k.cache_info().compiled == 1


@pulse.kernel
def pulse_pair(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    drive(d0, square(6, 0.03))
    sync(d0, d1)
    drive(d1, square(6, 0.03))


def _kernels():
    @pulse.kernel
    def train(q0, q1):
        d = get_drive_line(q0)
        for _ in range(6):
            drive(d, square(4, 0.02))
            shift_phase(d.tone, 0.3)
            wait(d, 2)

    @pulse.kernel
    def nested(q0, q1):
        d = get_drive_line(q0)
        for _ in range(3):
            for _ in range(2):
                drive(d, gaussian(8, 0.05, 2.0))
            wait(d, 3)

    @pulse.kernel
    def calls(q0, q1):
        for _ in range(4):
            pulse_pair(q0, q1)

    @pulse.kernel
    def detuned(q0, q1):
        d = get_drive_line(q0)
        set_frequency(d.tone, 5.01e9)
        for _ in range(5):
            drive(d, square(4, 0.02))
            wait(d, 5)

    return {"train": train, "nested": nested, "calls": calls, "detuned": detuned}


def _events(lowered):
    return sorted((e.line, e.kind, e.start_cycles, e.duration_cycles) for e in lowered.schedule)


@pytest.mark.parametrize("name", sorted(_kernels()))
def test_rolled_and_unrolled_programs_are_the_same(name):
    kernel = _kernels()[name]
    rolled = pulse.compile(kernel, unroll_threshold=0, passes=()).lower(Q[0], Q[1])
    unrolled = pulse.compile(kernel, unroll_threshold=10**6, passes=()).lower(Q[0], Q[1])
    assert _rolled(rolled) and not _rolled(unrolled)
    # Without passes every pulse is scheduled alike.
    assert _events(rolled) == _events(unrolled)
    # With the default passes, fuse_pulses may merge back-to-back pulses of the
    # unrolled program; the evolution is the same.
    a = pulse.compile(kernel, unroll_threshold=0).lower(Q[0], Q[1])
    b = pulse.compile(kernel, unroll_threshold=10**6).lower(Q[0], Q[1])
    end = max(e.start_cycles + e.duration_cycles for e in a.schedule)
    assert end == max(e.start_cycles + e.duration_cycles for e in b.schedule)
    rho_a = pulse.simulate(a, levels=3, dt_ns=0.05).density_matrix
    rho_b = pulse.simulate(b, levels=3, dt_ns=0.05).density_matrix
    assert np.allclose(rho_a, rho_b, atol=1e-10)
    assert abs(rho_a[0, 0] - 1.0) > 1e-3


POLICIES = ("asap", "alap", "rcp", "alap_rcp")
MACHINE = ControlSystem(links={"host": Link(fixed_ns=4, slots=4)})


def _first(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


FIRST = pulse.classical(_first, max_latency_ns=100, link=0)


def _loops(n1, n2, n3):
    """Kernels of several loops over one name, each rolled when n is large."""

    @pulse.kernel
    def sequential(q0, q1):
        d = get_drive_line(q0)
        for _ in range(n1):
            drive(d, square(4, 0.02))
            wait(d, 2)
        for _ in range(n2):
            drive(d, square(6, 0.02))
            wait(d, 3)
        for _ in range(n3):
            shift_phase(d.tone, 0.3)
            drive(d, square(8, 0.02))

    @pulse.kernel
    def calls(q0, q1):
        for _ in range(n1):
            pulse_pair(q0, q1)
        for _ in range(n2):
            pulse_pair(q1, q0)
        for _ in range(n3):
            pulse_pair(q0, q1)

    @pulse.kernel
    def readouts(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        for _ in range(n1):
            drive(d, square(4, 0.02))
            sync(d, r)
            readout(r, square(40, 0.05))
            sync(r, d)
        for _ in range(n2):
            drive(d, square(6, 0.02))
            sync(d, r)
            readout(r, square(40, 0.05))
            sync(r, d)
        for _ in range(n3):
            pulse_pair(q0, q1)
            sync(d, r)
            readout(r, square(40, 0.05))
            sync(r, d)

    @pulse.kernel
    def nested(q0, q1):
        d = get_drive_line(q0)
        for _ in range(n1):
            for _ in range(n2):
                drive(d, square(4, 0.02))
            for _ in range(n3):
                drive(d, square(6, 0.02))
            wait(d, 3)
        for _ in range(n2):
            for _ in range(n3):
                drive(d, gaussian(8, 0.05, 2.0))

    @pulse.kernel
    def answers(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        for _ in range(n1):
            drive(d, square(8, 0.1))
        for _ in range(n2):
            sync(d, r)
            m = readout(r, square(40, 0.05))
            if FIRST.submit([m]).result()[0]:
                drive(d, square(16, 0.1))
        for _ in range(n3):
            sync(d, r)
            m = readout(r, square(40, 0.05))
            if FIRST.submit([m]).result()[0]:
                drive(d, square(16, 0.1))
            else:
                wait(d, 8)

    return {
        "sequential": sequential,
        "calls": calls,
        "readouts": readouts,
        "nested": nested,
        "answers": answers,
    }


_FOR_LOOPS = {"sequential": 3, "calls": 3, "readouts": 3, "nested": 5, "answers": 3}


@pytest.mark.parametrize("name", sorted(_FOR_LOOPS))
def test_every_loop_over_one_name_stays_rolled(name):
    def lowered(n2, n3):
        compiled = pulse.compile(
            _loops(3, n2, n3)[name], unroll_threshold=0, control_system=MACHINE
        )
        return compiled.lower(Q[0], Q[1])

    small, large = lowered(4, 5), lowered(400, 5000)
    assert small.mlir().count("scf.for ") == _FOR_LOOPS[name]
    assert large.metrics.op_count == small.metrics.op_count


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", sorted(_FOR_LOOPS))
def test_several_rolled_loops_schedule_and_simulate_as_unrolled(name, policy):
    kernel = _loops(3, 4, 5)[name]

    def lowered(threshold, passes=None):
        options = {} if passes is None else {"passes": passes}
        compiled = pulse.compile(
            kernel, unroll_threshold=threshold, schedule=policy, control_system=MACHINE, **options
        )
        return compiled.lower(Q[0], Q[1])

    rolled, unrolled = lowered(0, ()), lowered(10**6, ())
    assert rolled.mlir().count("scf.for ") == _FOR_LOOPS[name] and not _rolled(unrolled)
    assert _events(rolled) == _events(unrolled)
    a, b = lowered(0), lowered(10**6)
    end = max(e.start_cycles + e.duration_cycles for e in a.schedule)
    assert end == max(e.start_cycles + e.duration_cycles for e in b.schedule)
    if name in ("readouts", "answers"):
        # Measurement makes the simulation slow; the schedules are compared.
        return
    rho_a = pulse.simulate(a, levels=3, dt_ns=0.05).density_matrix
    rho_b = pulse.simulate(b, levels=3, dt_ns=0.05).density_matrix
    assert np.allclose(rho_a, rho_b, atol=1e-10)
    assert abs(rho_a[0, 0] - 1.0) > 1e-3


def test_a_loop_variable_keeps_its_python_value_across_rolled_loops():
    def kernel(n):
        @pulse.kernel
        def k(q0, q1):
            d = get_drive_line(q0)
            for _ in range(n):
                drive(d, square(4, 0.02))
            for _ in range(0):
                drive(d, square(5, 0.02))
            drive(d, square(4 + _, 0.02))
            for _ in range(n):
                drive(d, square(6, 0.02))
            drive(d, square(4 + _, 0.02))
            _ = get_drive_line(q1)
            drive(_, square(4, 0.02))
            for _ in range(n):
                wait(d, 3)
            drive(d, square(5 + _, 0.02))

        return k

    rolled = pulse.compile(kernel(30), unroll_threshold=0, passes=()).lower(Q[0], Q[1])
    unrolled = pulse.compile(kernel(30), unroll_threshold=10**6, passes=()).lower(Q[0], Q[1])
    assert rolled.mlir().count("scf.for ") == 4
    assert _events(rolled) == _events(unrolled)
    after = [e.duration_cycles for e in unrolled.schedule if e.kind == "drive" and e.line == 0]
    assert after[30] == after[-2] == after[0] + 29
    assert after[-1] == after[0] + 30
