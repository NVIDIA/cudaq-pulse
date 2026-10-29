# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A millisecond calibration chain that runs on the controller.

After

    M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta, D. Kjær,
    F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen, T. Madsen,
    Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
    "Millisecond-Scale Calibration and Benchmarking of Superconducting Qubits",
    arXiv:2602.11912 (2026).

and, for adaptive T1 estimation,

    F. Berritta, J. Benestad, J. A. Krzywda, O. Krause, M. A. Marciniak,
    S. Krøjer, C. W. Warren, E. Hogedal, A. Nylander, I. Ahmad, A. Osman,
    J. Biznárová, M. Rommel, A. F. Roudsari, J. Bylander, G. Tancredi,
    J. Danon, J. Hastrup, F. Kuemmeth and M. Kjaergaard, "Real-time adaptive
    tracking of fluctuating relaxation rates in superconducting qubits",
    arXiv:2506.09576 (2025).

Builds on ``19_classical_calls.py`` and ``20_adaptive_calibration.py``. The
paper's calibration chain runs as one pulse program whose estimators are
classical functions on the controller's own link: nothing goes to a host or
a GPU between steps. Each step's answer sets the next step's pulses:

1. Readout assignment errors, from shots prepared in |0> and |1>.
2. The drive frequency, by sparse phase estimation on Ramsey shots at three
   phases; the drive tone is shifted onto the qubit.
3. The pi-pulse amplitude, by sparse phase estimation on a train of 21 pi
   pulses at three amplitude scales.
4. The pi/2-pulse amplitude, the same way on a train of 42 pi/2 pulses.
5. T1, by the analytical decay estimator at three delays spaced by the last
   estimate.
6. T1 again, by the adaptive Bayesian tracker of Berritta et al.: a gamma
   posterior updated after every shot, which picks the next wait, with the
   prior centred on step 5's estimate and the likelihood using step 1's
   assignment errors.

Then the frequency binary search of

    F. Berritta, J. Benestad, L. Pahl, M. Mathews, J. A. Krzywda, R. Assouly,
    Y. Sung, D. K. Kim, B. M. Niedzielski, K. Serniak, M. E. Schwartz,
    J. L. Yoder, A. Chatterjee, J. A. Grover, J. Danon, W. D. Oliver and
    F. Kuemmeth, "Efficient Qubit Calibration by Binary-Search Hamiltonian
    Tracking", arXiv:2501.05386 (2025),

which narrows a detuning shot by shot without resetting the qubit, and the
resource report: what each estimator costs the controller, checked against
a declared ``ControllerLimits``.

Compile with ALAP scheduling (the default): the pulse before each variable
wait then ends where the wait starts, so the qubit decays for exactly the
wait the estimator chose, not also through the call's latency.

The device is qubit 0 of the Krinner 17-qubit transmon twin with its T1 set
to 18.3 us (the analytical decay estimate reported by Marciniak et al.),
detuned by 60 kHz from the drive and with a 1.5% drive-amplitude error.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import dataclasses
import math
import time

import cudaq_pulse as pulse
from cudaq_pulse import controller
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.targets import transmon_krinner_17q

CLOCK_GHZ = 2.0  # the program clock: the estimators count waits and T1 in its cycles
PULSE = 20 * pulse.ns  # square pulses
PI_AMP = math.pi / 20  # square(PULSE, PI_AMP) is a pi pulse on the twin
READOUT = 300 * pulse.ns
READOUT_AMP = 0.05
SHOTS = 50  # per setting
TRAIN = 21  # pi pulses per amplitude train
RAMSEY = 2 * pulse.us  # between the centres of the Ramsey pi/2 pulses
RAMSEY_NS = RAMSEY / pulse.ns
T0 = 16 * pulse.ns  # the decay estimator's first delay
ATT_SHOTS = 50
ATT_SCALE = 0.51
SETTLE = 100 * pulse.us  # a passive reset, more than 5 T1


def in_cycles(time):
    """*time* in whole cycles of the program clock, the estimators' unit."""
    return round(time / pulse.ns * CLOCK_GHZ)


PULSE_CYCLES = in_cycles(PULSE)
T0_CYCLES = in_cycles(T0)
WAITS = (T0_CYCLES, in_cycles(1 * pulse.ms))  # every variable wait, in cycles: 16 ns to 1 ms

# A generic controller: 4 ns per cycle, with these cycles per operation.
LIMITS = controller.ControllerLimits(
    op_cycles={"div": 8, "sqrt": 8, "exp": 12, "log": 12, "trig": 12, "popcount": 1},
    compute_clock_ghz=0.25,
    controller_form=True,
    max_ops=256,
    state_bytes=512,
)
MACHINE = MachineModel(
    links={"controller": Link(fixed_ns=4, feedback_ns=8, max_in_flight=4)},
    controller=LIMITS,
)


def _fraction(bits):
    return cal.count_ones(bits) / SHOTS


# ── Estimators on the controller ────────────────────────────────────────────


@pulse.classical(
    args=(Bits(SHOTS), Bits(SHOTS)),
    results=(Float(), Float()),
    max_latency_ns=500,
    link="controller",
)
def assignment_errors(ground, excited):
    """|1> read as 0, and |0> read as 1."""
    return 1.0 - _fraction(excited), _fraction(ground)


@pulse.classical(
    args=(Bits(SHOTS), Bits(SHOTS), Bits(SHOTS), Float()),
    results=(Float(), Float()),
    max_latency_ns=1000,
    link="controller",
)
def ramsey_estimator(minus, zero, plus, mean_hz):
    """The detuning from Ramsey shots at phases -pi/2, 0 and +pi/2 about the
    estimate *mean_hz*, and its standard deviation."""
    theta, sigma = cal.sparse_phase(_fraction(minus), _fraction(zero), _fraction(plus), SHOTS)
    detuning = cal.ramsey_detuning(mean_hz, theta, RAMSEY_NS)
    return detuning, sigma / (2.0 * math.pi * RAMSEY_NS * 1e-9)


@pulse.classical(
    args=(Bits(SHOTS), Bits(SHOTS), Bits(SHOTS), Float()),
    results=(Float(), Float()),
    max_latency_ns=1000,
    link="controller",
)
def amplitude_estimator(minus, zero, plus, amplitude):
    """The corrected amplitude from a train that turns TRAIN times pi, at
    amplitude scales 1 - 1/(2 TRAIN), 1 and 1 + 1/(2 TRAIN), and the
    standard deviation of its relative correction."""
    theta, sigma = cal.sparse_phase(_fraction(minus), _fraction(zero), _fraction(plus), SHOTS)
    return cal.pi_train_amplitude(amplitude, theta, TRAIN), sigma / (TRAIN * math.pi)


@pulse.classical(args=(Float(),), results=(Float(), Float()), max_latency_ns=200, link="controller")
def decay_delays(t1_cycles):
    """The decay estimator's second and third delays, spaced by the last
    T1 estimate."""
    spacing = min(max(t1_cycles, 64.0), (WAITS[1] - T0_CYCLES) / 3.0)
    return T0_CYCLES + spacing, T0_CYCLES + 3.0 * spacing


@pulse.classical(
    args=(Bits(SHOTS), Bits(SHOTS), Bits(SHOTS), Float(), Float()),
    results=(Float(), Float()),
    max_latency_ns=1000,
    link="controller",
)
def decay_estimator(first, second, third, spacing_cycles, t1_cycles):
    """T1 and its standard deviation (in clock cycles) from the three
    delays; the last estimate *t1_cycles* when none can be formed."""
    t1, sigma, ok = cal.decay_time(
        _fraction(first), _fraction(second), _fraction(third), spacing_cycles, SHOTS
    )
    if ok:
        return t1, sigma
    return t1_cycles, t1_cycles


@pulse.classical(args=(Float(), Float()), results=(Float(),), max_latency_ns=200, link="controller")
def first_wait(shape, rate):
    return min(max(cal.adaptive_t1_wait(shape, rate, ATT_SCALE), WAITS[0]), WAITS[1])


@pulse.classical(
    args=(Bits(1), Float(), Float(), Float(), Float(), Float()),
    results=(Float(), Float(), Float()),
    max_latency_ns=2000,
    link="controller",
)
def adaptive_t1(bit, shape, rate, wait_cycles, error_1to0, error_0to1):
    """Update the gamma posterior over 1/T1 (rate in clock cycles) from one
    shot; pick the next wait."""
    shape, rate = cal.adaptive_t1_update(shape, rate, bit[0], wait_cycles, error_1to0, error_0to1)
    wait_cycles = min(max(cal.adaptive_t1_wait(shape, rate, ATT_SCALE), WAITS[0]), WAITS[1])
    return shape, rate, wait_cycles


# ── Shots and steps ─────────────────────────────────────────────────────────


@pulse.kernel
def shot_end(q, pi_amp: float):
    """Read out, then reset from the result (active reset)."""
    d = get_drive_line(q)
    r = get_readout_line(q)
    sync(d, r)
    m = readout(r, square(READOUT, READOUT_AMP))
    sync(d, r)
    if m:
        drive(d, square(PULSE, pi_amp))
    return m


@pulse.kernel
def settle_and_read(q, settle):
    """Read out, then wait *settle* for the qubit to relax (passive reset)."""
    d = get_drive_line(q)
    r = get_readout_line(q)
    sync(d, r)
    m = readout(r, square(READOUT, READOUT_AMP))
    sync(d, r)
    wait(d, settle)
    return m


@pulse.kernel
def assignment(q, pi_amp: float, settle=0 * pulse.ns):
    """Assignment errors; with a *settle* time, a passive reset of that length
    after each readout instead of the active one."""
    d = get_drive_line(q)
    ground = []
    excited = []
    if settle:
        wait(d, settle)
    else:
        shot_end(q, pi_amp)
    for _ in range(SHOTS):
        if settle:
            ground.append(settle_and_read(q, settle))
            drive(d, square(PULSE, pi_amp))
            excited.append(settle_and_read(q, settle))
        else:
            ground.append(shot_end(q, pi_amp))
            drive(d, square(PULSE, pi_amp))
            excited.append(shot_end(q, pi_amp))
    return assignment_errors.submit(ground, excited).result()


@pulse.kernel
def ramsey(q, pi_amp: float, mean_hz: float):
    d = get_drive_line(q)
    minus = []
    zero = []
    plus = []
    phase0 = -2.0 * math.pi * mean_hz * RAMSEY_NS * 1e-9
    for _ in range(SHOTS):
        for k in range(3):
            # Second pi/2 with its tone shifted by pi - phase: P(1) is
            # (1 - cos(2 pi D t + phase)) / 2.
            shift = math.pi - (phase0 + (k - 1) * math.pi / 2)
            drive(d, square(PULSE, pi_amp * 0.5))
            wait(d, RAMSEY - PULSE)
            shift_phase(d.tone, shift)
            drive(d, square(PULSE, pi_amp * 0.5))
            shift_phase(d.tone, -shift)
            m = shot_end(q, pi_amp)
            if k == 0:
                minus.append(m)
            elif k == 1:
                zero.append(m)
            else:
                plus.append(m)
    return ramsey_estimator.submit(minus, zero, plus, mean_hz).result()


@pulse.kernel
def train_shot(q, amplitude: float, pulses: int, pi_amp: float):
    d = get_drive_line(q)
    for _ in range(pulses):
        drive(d, square(PULSE, amplitude))
    return shot_end(q, pi_amp)


@pulse.kernel
def amplitude(q, amplitude: float, pulses: int, pi_amp: float):
    """Correct the amplitude of a pulse; *pulses* of them turn TRAIN pi."""
    minus = []
    zero = []
    plus = []
    for _ in range(SHOTS):
        minus.append(train_shot(q, amplitude * (1.0 - 0.5 / TRAIN), pulses, pi_amp))
        zero.append(train_shot(q, amplitude, pulses, pi_amp))
        plus.append(train_shot(q, amplitude * (1.0 + 0.5 / TRAIN), pulses, pi_amp))
    return amplitude_estimator.submit(minus, zero, plus, amplitude).result()


@pulse.kernel
def decay(q, pi_amp: float, t1_cycles: float):
    d = get_drive_line(q)
    second, third = decay_delays.submit(t1_cycles).result()
    first_bits = []
    second_bits = []
    third_bits = []
    for _ in range(SHOTS):
        drive(d, square(PULSE, pi_amp))
        wait(d, T0)
        first_bits.append(shot_end(q, pi_amp))
        drive(d, square(PULSE, pi_amp))
        wait(d, second, bounds=WAITS)
        second_bits.append(shot_end(q, pi_amp))
        drive(d, square(PULSE, pi_amp))
        wait(d, third, bounds=WAITS)
        third_bits.append(shot_end(q, pi_amp))
    return decay_estimator.submit(
        first_bits, second_bits, third_bits, second - T0_CYCLES, t1_cycles
    ).result()


@pulse.kernel
def tracked_t1(
    q,
    pi_amp: float,
    shape: float,
    rate: float,
    error_1to0: float,
    error_0to1: float,
    settle=0 * pulse.ns,
):
    """Adaptive T1; with a *settle* time, a passive reset of that length after
    each readout instead of the active one."""
    d = get_drive_line(q)
    w = first_wait.submit(shape, rate).result()
    for _ in range(ATT_SHOTS):
        drive(d, square(PULSE, pi_amp))
        wait(d, w, bounds=WAITS)
        if settle:
            m = settle_and_read(q, settle)
        else:
            m = shot_end(q, pi_amp)
        shape, rate, w = adaptive_t1.submit([m], shape, rate, w, error_1to0, error_0to1).result()
    return shape, rate


@pulse.kernel
def calibration_chain(q, t1_guess_cycles: float):
    d = get_drive_line(q)
    # The adaptive tracker's likelihood holds only assignment errors: an
    # active reset would add a preparation error that depends on the last
    # outcome, so its shots and the errors it uses reset passively.
    error_1to0, error_0to1 = assignment(q, PI_AMP, SETTLE)
    detuning, _ = ramsey(q, PI_AMP, 0.0)
    shift_frequency(d.tone, detuning)
    pi_amp, _ = amplitude(q, PI_AMP, TRAIN, PI_AMP)
    half_amp, _ = amplitude(q, pi_amp * 0.5, 2 * TRAIN, pi_amp)
    t1, _ = decay(q, pi_amp, t1_guess_cycles)
    tracked_t1(q, pi_amp, 3.0, 3.0 * t1, error_1to0, error_0to1, SETTLE)


# ── Frequency binary search ─────────────────────────────────────────────────

SIGMA0_HZ = 1e6
COHERENCE_NS = 10_000.0
SEARCH_WAITS = (PULSE_CYCLES, in_cycles(20 * pulse.us))


def _search_setting(mean_hz, sigma_hz):
    """The next binary-search wait (between the pi/2 pulses, in clock
    cycles) and phase."""
    wait_ns = cal.binary_search_wait(sigma_hz, COHERENCE_NS)
    wait_cycles = min(max(wait_ns * CLOCK_GHZ - PULSE_CYCLES, SEARCH_WAITS[0]), SEARCH_WAITS[1])
    return wait_cycles, cal.binary_search_phase(mean_hz, (wait_cycles + PULSE_CYCLES) / CLOCK_GHZ)


WAIT0, PHASE0 = _search_setting(0.0, SIGMA0_HZ)


@pulse.classical(
    args=(Bits(2), Float(), Float(), Float()),
    results=(Float(), Float(), Float(), Float()),
    max_latency_ns=2000,
    link="controller",
)
def search_step(bits, mean_hz, sigma_hz, wait_cycles):
    """Without a reset, a shot's outcome is whether the readout changed."""
    outcome = 2 * abs(bits[0] - bits[1]) - 1
    mean_hz, sigma_hz = cal.binary_search_update(
        mean_hz,
        sigma_hz,
        outcome,
        (wait_cycles + PULSE_CYCLES) / CLOCK_GHZ,
        -0.02,
        0.6,
        COHERENCE_NS,
    )
    wait_cycles, phase = _search_setting(mean_hz, sigma_hz)
    return mean_hz, sigma_hz, wait_cycles, phase


@pulse.kernel
def binary_search(q, rounds: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    previous = readout(r, square(READOUT, READOUT_AMP))
    sync(d, r)
    mean, sigma, w, phase = 0.0, SIGMA0_HZ, WAIT0, PHASE0
    drive(d, square(PULSE, PI_AMP * 0.5))
    wait(d, int(WAIT0))
    shift_phase(d.tone, math.pi - PHASE0)
    drive(d, square(PULSE, PI_AMP * 0.5))
    for i in range(rounds):
        sync(d, r)
        m = readout(r, square(READOUT, READOUT_AMP))
        mean, sigma, w, phase = search_step.submit([m, previous], mean, sigma, w).result()
        previous = m
        if i < rounds - 1:
            sync(d, r)
            drive(d, square(PULSE, PI_AMP * 0.5))
            wait(d, w, bounds=SEARCH_WAITS)
            shift_phase(d.tone, math.pi - phase)
            drive(d, square(PULSE, PI_AMP * 0.5))


# ── Devices ─────────────────────────────────────────────────────────────────


def chain_plant(target, t1_us=18.3, offset_hz=60e3, amplitude_scale=1.015):
    """Qubit 0 of *target* as the chain's device: its T1, a detuning from
    the drive, a drive-amplitude error and small readout errors."""
    qubit = target.qubits[0]
    drive = dict(qubit.drive_params)
    drive["amplitude_scale_rad_per_ns"] = target.drive_amplitude_scale(0) * amplitude_scale
    readout_params = {**qubit.readout_params, "error_1to0": 0.03, "error_0to1": 0.02}
    plant = dataclasses.replace(
        qubit,
        t1_us=t1_us,
        frequency_hz=qubit.frequency_hz + offset_hz,
        drive_params=drive,
        readout_params=readout_params,
    )
    return dataclasses.replace(target, qubits={**target.qubits, 0: plant})


def search_plant(target, offset_hz):
    """A device like the binary-search paper's: T2* 12 us, T1 80 us, and
    readout errors that give its fringe contrast of 0.6 between shots."""
    qubit = target.qubits[0]
    readout_params = {**qubit.readout_params, "error_1to0": 0.115, "error_0to1": 0.105}
    plant = dataclasses.replace(
        qubit,
        t1_us=80.0,
        t2_star_us=12.0,
        thermal_population=0.0,
        frequency_hz=qubit.frequency_hz + offset_hz,
        readout_params=readout_params,
    )
    return dataclasses.replace(target, qubits={**target.qubits, 0: plant})


def chain_results(calls):
    """Each step's answer, from one shot's calls."""
    answers = {}
    for call in calls:
        answers[call.function] = call.answer
    shape, rate, _ = answers["adaptive_t1"]
    return {
        "readout errors": answers["assignment_errors"],
        "detuning (kHz)": answers["ramsey_estimator"][0] / 1e3,
        "pi amplitude": [c.answer[0] for c in calls if c.function == "amplitude_estimator"][0],
        "pi/2 amplitude": [c.answer[0] for c in calls if c.function == "amplitude_estimator"][1],
        "T1, decay estimator (us)": answers["decay_estimator"][0] / CLOCK_GHZ / 1e3,
        "T1, adaptive (us)": rate / shape / CLOCK_GHZ / 1e3,
    }


if __name__ == "__main__":
    print(
        "The millisecond calibration chain of Marciniak et al., arXiv:2602.11912 (2026),\n"
        "with the adaptive T1 tracker of Berritta et al., arXiv:2506.09576 (2025).\n"
    )
    target = transmon_krinner_17q()
    dev = pulse.device({"target": target})
    q = dev.qudit_handles[0]
    plant = chain_plant(target)

    print("=== 1. The calibration chain, on the controller ===")
    start = time.perf_counter()
    compiled = pulse.compile(calibration_chain, device=dev, machine=MACHINE)
    lowered = compiled.lower(q, 20_000.0 * CLOCK_GHZ)
    compile_s = time.perf_counter() - start
    end_cycles = max(
        e.start_cycles + e.duration_cycles
        for e in lowered.schedule
        if e.start_cycles is not None and e.duration_cycles is not None
    )
    print(
        f"  {sum(1 for e in lowered.schedule if e.kind == 'readout')} readouts, "
        f"{len(lowered.classical_report())} calls, {len(lowered.epochs())} epochs; "
        f"compiled in {compile_s:.2f} s"
    )
    print(f"  the static schedule (every variable wait at its least) is {end_cycles / 2e3:.0f} us")
    start = time.perf_counter()
    result = pulse.simulate(lowered, plant=plant, shots=1, seed=7, paths="sampled", readout="model")
    print(f"  one shot of the chain simulated in {time.perf_counter() - start:.1f} s")
    for name, value in chain_results(result.calls[0]).items():
        shown = ", ".join(f"{v:.3f}" for v in value) if isinstance(value, tuple) else f"{value:.4f}"
        print(f"  {name:<26} {shown}")
    print(
        f"  truth: detuning 60 kHz, pi amplitude {PI_AMP / 1.015:.4f}, "
        f"pi/2 {PI_AMP / 2 / 1.015:.4f}, T1 18.3 us"
    )

    print("\n=== 2. Frequency binary search, 15 shots, no reset ===")
    rounds = 15
    lowered = pulse.compile(binary_search, device=dev, machine=MACHINE).lower(q, rounds)
    errors, sigmas = [], []
    for seed in range(4):
        offset = (seed - 1.5) * 300e3
        found = pulse.simulate(
            lowered,
            plant=search_plant(target, offset),
            shots=1,
            seed=seed,
            paths="sampled",
            readout="model",
        )
        mean, sigma, _, _ = found.calls[0][-1].answer
        errors.append(mean - offset)
        sigmas.append(sigma)
        print(
            f"  detuning {offset / 1e3:+6.0f} kHz: estimate {mean / 1e3:+6.0f} +- {sigma / 1e3:.0f} kHz"
        )
    print(
        f"  posterior width from {SIGMA0_HZ / 1e3:.0f} kHz to {sum(sigmas) / len(sigmas) / 1e3:.0f} kHz"
    )

    print("\n=== 3. What the estimators cost the controller ===")
    lowered = compiled.lower(q, 20_000.0 * CLOCK_GHZ)
    report = controller.resources(lowered)
    print(report)
    over = dataclasses.replace(LIMITS, op_cycles={**LIMITS.op_cycles, "exp": 400})
    print("  with exp at 400 cycles:")
    for violation in controller.resources(lowered, over).violations:
        print(f"    {violation}")
