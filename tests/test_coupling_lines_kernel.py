# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupling lines in a kernel: ``get_coupling_line``, ``drive`` with no
tone, what they trace to, how they schedule, and what refuses them."""

from __future__ import annotations

import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import CouplingLine
from cudaq_pulse.ops import (
    drag,
    drive,
    gaussian,
    get_coupling_line,
    get_drive_line,
    get_readout_line,
    readout,
    shift_phase,
    square,
    sync,
    wait,
    wf_neg,
    wf_scale,
    wf_sub,
)
from cudaq_pulse.quantum_systems import (
    ChargeCoupling,
    Coupling,
    ExchangeControl,
    FluxTuning,
    PhysicalQudit,
)

_TUNING = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.0)
_EXCHANGE = ExchangeControl(
    unit="flux_quantum", amplitudes=(-0.4, 0.0, 0.4), exchange_hz=(9.0e6, 4.0e6, 9.0e6)
)


def _system(control_system=None, tuned=(0, 1, 2)):
    """Four qudits, *tuned* with a tuning: 0-1 has its own control, 1-2 an
    interaction made through its qudits, 2-3 is always on."""
    return pulse.quantum_system(
        name="tuned",
        qudits={
            i: PhysicalQudit(
                i,
                5.0e9 + 1.0e8 * i,
                50.0,
                30.0,
                anharmonicity_hz=-3.0e8,
                tuning=_TUNING if i in tuned else None,
            )
            for i in range(4)
        },
        couplings=[
            Coupling((0, 1), 4.0e6, control=_EXCHANGE),
            Coupling(
                (1, 2),
                4.0e6,
                interaction=ChargeCoupling(charge_coupling_hz=3e6, tunings=(_TUNING, _TUNING)),
            ),
            Coupling((2, 3), 1.0e6),
        ],
        control_system=control_system,
    )


QS = _system()
A, D, E = QS.qudits[0], QS.qudits[1], QS.qudits[2]
POLICIES = ("asap", "alap", "rcp", "alap_rcp")


def _events(lowered):
    return [(e.kind, e.line, e.qudit, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


@pulse.kernel
def detune(a, d, amplitude: float):
    ca = get_coupling_line(a)
    pair = get_coupling_line(d, a)
    da = get_drive_line(a)
    drive(ca, square(60 * pulse.ns, amplitude))
    drive(da, gaussian(40, 0.2, 10.0))
    wait(pair, 20 * pulse.ns)
    sync(ca, pair, da)
    drive(pair, square(40, -0.1))


def test_a_kernel_traces_to_the_coupling_line_ops():
    text = pulse.compile(detune, quantum_system=QS, passes=()).lower(A, D, 0.31).mlir()
    lines = [line.strip() for line in text.splitlines()]
    # One qudit's line, and the pair's, named by its qudits in the system's order.
    assert any(
        line.endswith("pulse.get_coupling_line %0 : (!pulse.qudit) -> !pulse.coupling_line")
        for line in lines
    ), text
    assert sum("pulse.get_coupling_line" in line for line in lines) == 2
    pair = next(line for line in lines if "(!pulse.qudit, !pulse.qudit)" in line)
    assert pair.endswith(": (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line")
    # A drive on it is its own op, with no tone, and a drive on a drive
    # line prints as ever.
    baseband = [line for line in lines if "pulse.drive_baseband " in line]
    assert len(baseband) == 2 and all(
        line.endswith(": !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line")
        for line in baseband
    )
    toned = ": !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone"
    assert sum(line.endswith(toned) for line in lines) == 1
    assert "(!pulse.coupling_line, !pulse.duration) -> !pulse.coupling_line" in text
    assert (
        "!pulse.coupling_line, !pulse.coupling_line, !pulse.drive_line -> "
        "!pulse.coupling_line, !pulse.coupling_line, !pulse.drive_line"
    ) in text
    # The module says that it holds a coupling line.
    assert "pulse.coupling_lines" in text


@pytest.mark.parametrize("policy", POLICIES)
def test_a_coupling_line_schedules_as_a_line(policy):
    lowered = pulse.compile(detune, quantum_system=QS, schedule=policy).lower(A, D, 0.31)
    events = _events(lowered)
    # Lines in allocation order: 0 the line of a, 1 the pair's, 2 a's drive
    # line. An event on a coupling line names the lowest of its qudits.
    late = policy.startswith("alap")
    assert events == [
        ("drive", 0, 0, 0, 120),
        ("drive", 2, 0, 80 if late else 0, 40),
        ("wait", 1, 0, 80 if late else 0, 40),
        ("drive", 1, 0, 120, 40),
    ]
    assert lowered.duration == 80 * pulse.ns
    assert lowered.qudits == (0, 1)


def test_the_line_is_fetched_once_whatever_the_order_of_its_qudits():
    @pulse.kernel
    def helper(a, d):
        drive(get_coupling_line(d, a), square(20, 0.1))

    @pulse.kernel
    def twice(a, d):
        pair = get_coupling_line(a, d)
        drive(pair, square(20, 0.1))
        helper(a, d)
        again = get_coupling_line(d, a)
        drive(again, square(20, 0.1))
        drive(pair, square(20, 0.1))

    lowered = twice.lower(A, D)
    assert lowered.mlir().count("pulse.get_coupling_line") == 1
    assert _events(lowered) == [("drive", 0, 0, 20 * k, 20) for k in range(4)]


def test_a_coupling_line_plays_beside_its_qudits_other_lines():
    @pulse.kernel
    def beside(a):
        c, d, r = get_coupling_line(a), get_drive_line(a), get_readout_line(a)
        drive(c, square(100, 0.2))
        drive(d, gaussian(40, 0.2, 10.0))
        readout(r, square(60, 0.05))

    assert _events(pulse.compile(beside, quantum_system=QS, schedule="asap").lower(A)) == [
        ("drive", 0, 0, 0, 100),
        ("drive", 1, 0, 0, 40),
        ("readout", 2, 0, 0, 60),
    ]


def test_a_measurement_conditioned_pulse_on_a_coupling_line():
    @pulse.kernel
    def conditioned(a):
        c, r = get_coupling_line(a), get_readout_line(a)
        if readout(r, square(40, 0.05)):
            drive(c, square(20, 0.2))
        drive(c, square(20, -0.2))

    lowered = conditioned.lower(A)
    text = lowered.mlir()
    assert "-> (!pulse.coupling_line)" in text and "scf.if" in text
    starts = [e.start_cycles for e in lowered.schedule if e.line == 0]
    # The arm's pulse after the measurement, the next one after the arm.
    assert starts[0] >= 40 and starts[1] == starts[0] + 20


@pytest.mark.parametrize("policy", POLICIES)
def test_a_rolled_loop_carries_a_coupling_line(policy):
    @pulse.kernel
    def train(a, d):
        ca, pair, da = get_coupling_line(a), get_coupling_line(a, d), get_drive_line(d)
        for _ in range(12):
            drive(ca, square(8, 0.05))
            wait(ca, 4)
            drive(da, gaussian(16, 0.1, 4.0))
            sync(ca, pair, da)
            drive(pair, square(6, -0.1))

    rolled = pulse.compile(train, quantum_system=QS, schedule=policy, unroll_threshold=0)
    unrolled = pulse.compile(train, quantum_system=QS, schedule=policy, unroll_threshold=10**6)
    text = rolled.lower(A, D).mlir()
    assert "scf.for" in text and "iter_args" in text and "!pulse.coupling_line" in text
    assert "scf.for" not in unrolled.lower(A, D).mlir()
    assert _events(rolled.lower(A, D)) == _events(unrolled.lower(A, D))
    assert len(_events(rolled.lower(A, D))) == 12 * 4


def test_a_line_first_fetched_in_a_loop_body_unrolls_the_loop():
    @pulse.kernel
    def lazy(a):
        for _ in range(4):
            drive(get_coupling_line(a), square(8, 0.05))

    lowered = pulse.compile(lazy, quantum_system=QS, unroll_threshold=0).lower(A)
    assert "scf.for" not in lowered.mlir()
    assert _events(lowered) == [("drive", 0, 0, 8 * k, 8) for k in range(4)]


def _raises(kernel, *args, error=pulse.CompilationError):
    with pytest.raises(error) as raised:
        kernel.lower(*args)
    return str(raised.value)


def test_get_coupling_line_names_what_is_wrong():
    @pulse.kernel
    def none(a):
        get_coupling_line()

    @pulse.kernel
    def twice(a):
        get_coupling_line(a, a)

    @pulse.kernel
    def number(a):
        get_coupling_line(a, 3)

    @pulse.kernel
    def a_line(a):
        get_coupling_line(get_drive_line(a))

    assert "takes the qudits the line reaches" in _raises(none, A) and "got none" in _raises(
        none, A
    )
    assert "got the same qudit twice" in _raises(twice, A)
    assert "argument 2 must be a qudit, got int 3" in _raises(number, A)
    assert "argument 1 must be a qudit, got a drive line" in _raises(a_line, A)


def test_a_coupling_line_has_no_tone_and_no_frame():
    @pulse.kernel
    def toned(a):
        drive(get_coupling_line(a), square(20, 0.1), tone=get_drive_line(a).tone)

    @pulse.kernel
    def tone_read(a):
        shift_phase(get_coupling_line(a).tone, 0.1)

    @pulse.kernel
    def tones_read(a):
        return get_coupling_line(a).tones[(1, 2)]

    @pulse.kernel
    def framed(a):
        shift_phase(get_coupling_line(a), 0.1)

    @pulse.kernel
    def read(a):
        readout(get_coupling_line(a), square(20, 0.1))

    @pulse.kernel
    def calibrated(a):
        c = get_coupling_line(a)
        drive(c, c.get_calibration(pulse.gates.CZ))

    @pulse.kernel
    def not_a_waveform(a):
        drive(get_coupling_line(a), 0.3)

    assert "drive() on a coupling line takes no tone" in _raises(toned, A)
    assert "a coupling line has no attribute 'tone'" in _raises(tone_read, A)
    assert "it has no tone and no frame" in _raises(tones_read, A)
    assert "shift_phase(): argument 1 must be a tone, got a coupling line" in _raises(framed, A)
    assert "readout() takes a readout line, got a coupling line" in _raises(read, A)
    assert "the calibrated gates of a coupling line are not available yet" in _raises(calibrated, A)
    assert "argument 2 must be a waveform, got float 0.3" in _raises(not_a_waveform, A)


def test_a_coupling_line_plays_a_real_waveform():
    @pulse.kernel
    def complex_square(a):
        drive(get_coupling_line(a), square(20, 0.1 + 0.2j))

    @pulse.kernel
    def dragged(a):
        drive(get_coupling_line(a), drag(20, 0.1, 5.0, 0.5))

    @pulse.kernel
    def real(a):
        c = get_coupling_line(a)
        drive(c, square(20, -0.1))
        drive(c, gaussian(20, 0.1, 5.0))
        drive(c, drag(20, 0.1, 5.0, 0.0))

    @pulse.kernel
    def combined(a):
        mixed = wf_sub(gaussian(20, 0.1, 5.0), drag(20, 0.1, 5.0, 0.5))
        drive(get_coupling_line(a), wf_scale(mixed, 0.5))

    @pulse.kernel
    def in_a_loop(a):
        c = get_coupling_line(a)
        for _ in range(3):
            drive(c, square(20, 0.1))
            drive(c, wf_neg(square(20, 0.2j)))

    # Known while the kernel is traced: refused there, at the kernel's line.
    for kernel, made_from in (
        (complex_square, "square() with the amplitude's imaginary part 0.2"),
        (dragged, "drag() with beta 0.5"),
        (combined, "drag() with beta 0.5"),
        (in_a_loop, "square() with the amplitude's imaginary part 0.2"),
    ):
        message = _raises(kernel, A)
        assert "drive() on a coupling line plays a real waveform" in message
        assert f"this one is made from {made_from}" in message
    assert len(real.lower(A).schedule) == 3

    @pulse.kernel
    def retraced(a, count: int):
        c = get_coupling_line(a)
        for i in range(count):
            # A body that differs by iteration is undone and traced again.
            drive(c, square(20 + 4 * i, 0.1))
        drive(c, square(20, 0.1))
        drive(c, wf_neg(square(20, 0.2j)))

    assert "square() with the amplitude's imaginary part 0.2" in _raises(retraced, A, 3)

    @pulse.kernel
    def beta_given(a, beta: float):
        drive(get_coupling_line(a), drag(20, 0.1, 5.0, beta))

    # Not known until the program is verified: the same kind of error.
    message = _raises(beta_given, A, 0.0)
    assert "plays a real waveform" in message
    assert "this pulse.drag may have an imaginary part" in message


def test_a_calibrated_pulse_does_not_play_on_a_coupling_line_yet():
    shipped = pulse.quantum_system("transmon_surface17")
    d4, x2 = shipped.qudits[3], shipped.qudits[10]
    cz = shipped.coupling(x2, d4).get_calibration(pulse.gates.CZ)

    @pulse.kernel
    def gate(a):
        drive(get_coupling_line(a), get_drive_line(a).get_calibration(pulse.gates.X))

    @pulse.kernel
    def measured(a):
        drive(get_coupling_line(a), get_readout_line(a).get_calibration(pulse.gates.READOUT))

    @pulse.kernel
    def flux_cz(a):
        drive(get_coupling_line(a), cz)

    for kernel, args, named in (
        (gate, (d4,), "DragPulse("),
        (measured, (d4,), "ReadoutPulse("),
        (flux_cz, (d4,), "FluxCZ("),
    ):
        message = _raises(kernel, *args)
        assert f"drive(): {named}" in message
        assert "the calibrated gates of a coupling line are not available yet" in message
        # The same under a tracer that checks units.
        with pytest.raises(pulse.CompilationError, match="calibrated gates of a coupling line"):
            pulse.compile(kernel, quantum_system=shipped, unit_warnings=True).lower(*args)

    @pulse.kernel
    def on_its_own_line(a):
        d = get_drive_line(a)
        drive(get_coupling_line(a), square(20, 0.1))
        drive(d, d.get_calibration(pulse.gates.X))

    # The pulse plays where it is calibrated, beside the coupling line.
    assert len(on_its_own_line.lower(d4).schedule) == 2


def test_exports_refuse_a_program_that_holds_a_coupling_line():
    lowered = detune.lower(A, D, 0.31)
    for export in (
        lowered.export_to_controller,
        lambda: lowered.export_to_controller(version=2),
        lowered.export_to_emulator,
    ):
        with pytest.raises(pulse.UnsupportedError) as raised:
            export()
        message = str(raised.value)
        assert "coupling lines are not exported in formats /1 and /2" in message
        assert "cudaq-pulse.controller/1" in message and "cudaq-pulse.emulation/1" in message

    @pulse.kernel
    def plain(a):
        drive(get_drive_line(a), gaussian(40, 0.2, 10.0))

    # A program without one exports as ever.
    assert plain.lower(A).export_to_controller()["format"] == "cudaq-pulse.controller/1"
    assert plain.lower(A).export_to_emulator()["format"] == "cudaq-pulse.emulation/1"


def test_a_check_of_limits_refuses_it_and_the_program_pickles():
    lowered = detune.lower(A, D, 0.31)
    with pytest.raises(pulse.UnsupportedError, match="limits are not checked on them yet"):
        pulse.controller.check(lowered, pulse.Controller(min_pulse_cycles=4))
    back = pickle.loads(pickle.dumps(lowered))
    assert _events(back) == _events(lowered) and "pulse.coupling_lines" in back.mlir()
    with pytest.raises(pulse.UnsupportedError, match="formats /1 and /2"):
        back.export_to_controller()


def test_simulating_a_pulse_on_a_coupling_line_is_refused_for_now():
    from cudaq_pulse.classical.emulation import emulate

    lowered = detune.lower(A, D, 0.31)
    for run in (lambda: pulse.simulate(lowered, levels=2), lambda: emulate(lowered, seed=1)):
        with pytest.raises(pulse.UnsupportedError) as raised:
            run()
        assert "a program with a coupling line is not simulated or emulated yet" in str(
            raised.value
        )


@pulse.classical(max_latency_ns=200)
def coupling_test_decide(bits: pulse.Length[pulse.Bits, 1]) -> float:
    return 0.1 + 0.1 * bits[0]


def test_what_the_passes_do_not_do_beside_a_coupling_line_is_refused():
    @pulse.kernel
    def answered(a):
        c = get_coupling_line(a)
        m = readout(get_readout_line(a), square(100, 0.2))
        amplitude = coupling_test_decide.submit([m]).result()
        drive(get_drive_line(a), gaussian(40, amplitude, 10.0))
        drive(c, square(40, 0.1))

    @pulse.kernel
    def fetched_after(a):
        m = readout(get_readout_line(a), square(100, 0.2))
        amplitude = coupling_test_decide.submit([m]).result()
        drive(get_drive_line(a), gaussian(40, amplitude, 10.0))
        drive(get_coupling_line(a), square(40, 0.1))

    for kernel in (answered, fetched_after):
        with pytest.raises(pulse.UnsupportedError) as raised:
            kernel.lower(A)
        assert "a program that holds a coupling line cannot hold a classical call yet" in str(
            raised.value
        )

    @pulse.kernel(schedule="alap")
    def late(a):
        drive(get_coupling_line(a), square(40, 0.1))

    @pulse.kernel
    def calls_scheduled(a):
        drive(get_drive_line(a), gaussian(40, 0.2, 10.0))
        late(a)

    with pytest.raises(pulse.UnsupportedError) as raised:
        calls_scheduled.lower(A)
    assert "cannot hold a kernel with its own schedule (schedule=)" in str(raised.value)
    # Flattened, the called kernel's pulses are scheduled with the caller's.
    flat = pulse.compile(calls_scheduled, quantum_system=QS, flatten_schedules=True).lower(A)
    assert len(flat.schedule) == 2


def test_a_coupling_line_that_is_fetched_and_never_used_changes_nothing():
    @pulse.kernel
    def fetched(a):
        get_coupling_line(a)
        drive(get_drive_line(a), gaussian(40, 0.2, 10.0))

    @pulse.kernel
    def plain(a):
        drive(get_drive_line(a), gaussian(40, 0.2, 10.0))

    # It is dropped as any line that is never used: the program exports and
    # simulates as the one without it.
    lowered, same = fetched.lower(A), plain.lower(A)
    assert "get_coupling_line" not in lowered.mlir()
    assert lowered.export_to_controller() == same.export_to_controller()
    assert lowered.export_to_emulator() == same.export_to_emulator()
    assert (
        pulse.simulate(lowered, levels=2).density_matrix
        == pulse.simulate(same, levels=2).density_matrix
    ).all()


@pulse.kernel
def tuned(a):
    drive(get_coupling_line(a), square(20, 0.1))


@pulse.kernel
def paired(a, d):
    drive(get_coupling_line(a, d), square(20, 0.1))


def test_a_line_the_quantum_system_declares_no_control_for_is_refused():
    q = QS.qudits
    # With no control system, the line exists when the quantum system
    # declares the component for exactly those qudits.
    assert len(tuned.lower(q[2]).schedule) == 1 and len(paired.lower(q[1], q[0]).schedule) == 1
    assert "qudit 3 has no tuning, so no coupling line reaches it" in _raises(tuned, q[3])
    message = _raises(paired, q[1], q[2])
    assert "the coupling of qudits 1 and 2 has no control of its own" in message
    assert "reach its qudits' tunings: CouplingLine(1) or CouplingLine(2)" in message
    assert "the coupling of qudits 2 and 3 has no control: it is always on" in _raises(
        paired, q[3], q[2]
    )
    assert "qudits 0 and 2 are not coupled" in _raises(paired, q[0], q[2])

    @pulse.kernel
    def three(a, b, c):
        get_coupling_line(a, b, c)

    assert "nothing says what a line of three or more does" in _raises(three, q[0], q[1], q[2])
    # The shipped system has the tunings its calibration gives it.
    shipped = pulse.quantum_system("transmon_surface17")
    assert len(tuned.lower(shipped.qudits[3]).schedule) == 1
    assert "has no control of its own" in _raises(paired, shipped.qudits[3], shipped.qudits[10])
    untuned = pulse.quantum_system("transmon_lattice:2")
    assert "qudit 0 has no tuning" in _raises(tuned, untuned.qudits[0])


def test_a_run_time_qudits_line_needs_a_control_system():
    @pulse.kernel
    def bound(a: pulse.Qudit):
        drive(get_coupling_line(a), square(20, 0.1))

    # The line of a qudit a call binds is checked when it is bound, by the
    # control system's check of the call: with none, it is refused.
    message = _raises(bound, A)
    assert "the line of a qudit that a call binds" in message
    assert "a kernel with no control system does not do" in message
    # Passed as a static argument, the line is checked when the kernel is traced.
    assert len(tuned.lower(A).schedule) == 1


def test_a_helper_kernel_plays_on_the_line_it_is_given():
    @pulse.kernel
    def helper(line, amplitude):
        drive(line, square(20, amplitude))
        wait(line, 8)

    @pulse.kernel
    def caller(a):
        c = get_coupling_line(a)
        helper(c, 0.1)
        helper(c, -0.1)

    @pulse.kernel
    def gives_a_gate(a):
        c = get_coupling_line(a)
        helper_gate(c, get_drive_line(a).get_calibration(pulse.gates.X))

    @pulse.kernel
    def helper_gate(line, gate):
        drive(line, gate)

    assert [e[0] for e in _events(caller.lower(A))] == ["drive", "wait", "drive", "wait"]
    shipped = pulse.quantum_system("transmon_surface17")
    assert "the calibrated gates of a coupling line are not available yet" in _raises(
        gives_a_gate, shipped.qudits[3]
    )


def test_systems_that_differ_in_tunings_share_no_specialization():
    def system(tuning):
        return pulse.quantum_system(
            name="one",
            qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-3.0e8, tuning=tuning)},
        )

    with_tuning, without = system(_TUNING), system(None)
    assert with_tuning.fingerprint != without.fingerprint
    assert len(tuned.lower(with_tuning.qudits[0]).schedule) == 1
    # The kernel compiled for the tuned system is not reused for the other.
    assert "qudit 0 has no tuning" in _raises(tuned, without.qudits[0])
    assert len(tuned.lower(with_tuning.qudits[0]).schedule) == 1


def test_a_line_the_control_system_does_not_have_is_refused():
    box = pulse.Controller(coupling_lines=[CouplingLine(0), CouplingLine((0, 1)), CouplingLine(2)])
    qs = _system(pulse.ControlSystem(controllers=[box]))
    message = _raises(tuned, qs.qudits[1])
    assert "no coupling line of the control system acts on exactly qudit 1" in message
    assert "the coupling lines that reach it are CouplingLine((0, 1))" in message
    # A line the controller has: what a controller's limits say of it is
    # not checked yet, so the program is refused for now, and says so.
    with pytest.raises(pulse.UnsupportedError, match="limits are not checked on them yet"):
        tuned.lower(qs.qudits[0])


def test_what_scheduling_does_not_do_yet_is_refused():
    several = pulse.ControlSystem(
        controllers=[
            pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1]),
            pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3]),
        ]
    )
    with pytest.raises(pulse.UnsupportedError, match="several controllers"):
        pulse.compile(tuned, quantum_system=QS, control_system=several).lower(A)
    # A control system with no coupling line has none that acts on the qudit.
    plain = pulse.ControlSystem(controllers=[pulse.Controller()])
    message = _raises(pulse.compile(tuned, quantum_system=QS, control_system=plain), A)
    assert "no coupling line of the control system acts on exactly qudit 0" in message
    assert "coupling_lines=[CouplingLine(0)]" in message
    delayed = pulse.Controller(coupling_lines=[CouplingLine(0, delay_ns=2.0)])
    with pytest.raises(pulse.UnsupportedError, match=r"the delay of CouplingLine\(0, delay_ns"):
        pulse.compile(tuned, quantum_system=QS, control_system=delayed).lower(A)


def test_the_tracing_module_loads_only_for_a_kernel_with_a_coupling_line():
    code = (
        "import sys, cudaq_pulse as pulse\n"
        "qs = pulse.quantum_system('transmon_surface17')\n"
        "@pulse.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(20, 0.1))\n"
        "    wait(d, 4)\n"
        "low = k.lower(qs.qudits[0])\n"
        "low.export_to_controller(); low.export_to_emulator()\n"
        "print('cudaq_pulse.kernel._coupling' in sys.modules)\n"
        "print('pulse.coupling_lines' in low.mlir())\n"
        "@pulse.kernel\n"
        "def c(q):\n"
        "    drive(get_coupling_line(q), square(20, 0.1))\n"
        "c.lower(qs.qudits[0])\n"
        "print('cudaq_pulse.kernel._coupling' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False", "True"]
