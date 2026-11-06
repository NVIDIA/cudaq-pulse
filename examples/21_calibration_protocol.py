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
a declared ``Controller``.

Compile with ALAP scheduling (the default): the pulse before each variable
wait then ends where the wait starts, so the qubit decays for exactly the
wait the estimator chose, not also through the call's latency.

The device is qubit 0 of the 17-qubit transmon twin with its T1 set
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
from cudaq_pulse.classical import Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.targets import transmon_17q

clock_ghz = 2.0  # the program clock: the estimators count waits and T1 in its cycles
pulse_time = 20 * pulse.ns  # square pulses
# square(pulse_time, nominal_pi_amplitude) is a pi pulse on the twin.
nominal_pi_amplitude = math.pi / 20
readout_time = 300 * pulse.ns
readout_amplitude = 0.05
shots_per_setting = 50
pulses_per_train = 21  # pi pulses per amplitude train
ramsey_delay = 2 * pulse.us  # between the centres of the Ramsey pi/2 pulses
ramsey_delay_ns = ramsey_delay / pulse.ns
first_delay = 16 * pulse.ns  # the decay estimator's first delay
tracker_shots = 50
tracker_wait_scale = 0.51
settle_time = 100 * pulse.us  # a passive reset, more than 5 T1


def in_cycles(time):
    """*time* in whole cycles of the program clock, the estimators' unit."""
    return round(time / pulse.ns * clock_ghz)


pulse_cycles = in_cycles(pulse_time)
first_delay_cycles = in_cycles(first_delay)
wait_range_cycles = (
    first_delay_cycles,
    in_cycles(1 * pulse.ms),
)  # every variable wait, in cycles: 16 ns to 1 ms

# A generic controller whose processor runs the estimators: 4 ns per cycle,
# with these cycles per operation, up to one call per slot of its link at once.
processor = pulse.Processor(
    links=["controller"],
    cycles_per_op={"div": 8, "sqrt": 8, "exp": 12, "log": 12, "trig": 12, "popcount": 1},
    clock_ghz=0.25,
    max_concurrent_calls=4,
    fixed_cost_only=True,
    max_ops_per_function=256,
    max_state_bytes_per_function=512,
)
links = {"controller": Link(fixed_ns=4, feedback_ns=8, max_in_flight=4)}
control_system = pulse.ControlSystem(
    controllers=[pulse.Controller(processors=[processor])], links=links
)


def _fraction(bits):
    return cal.count_ones(bits) / len(bits)


# ── Estimators on the controller ────────────────────────────────────────────


@pulse.classical(max_latency_ns=500, link="controller")
def assignment_errors(ground: pulse.Bits, excited: pulse.Bits) -> tuple[float, float]:
    """|1> read as 0, and |0> read as 1."""
    return 1.0 - _fraction(excited), _fraction(ground)


@pulse.classical(max_latency_ns=1000, link="controller")
def ramsey_estimator(
    minus: pulse.Bits, zero: pulse.Bits, plus: pulse.Bits, mean_hz: float
) -> tuple[float, float]:
    """The detuning from Ramsey shots at phases -pi/2, 0 and +pi/2 about the
    estimate *mean_hz*, and its standard deviation."""
    theta, sigma = cal.sparse_phase(
        _fraction(minus), _fraction(zero), _fraction(plus), shots_per_setting
    )
    detuning = cal.ramsey_detuning(mean_hz, theta, ramsey_delay_ns)
    return detuning, sigma / (2.0 * math.pi * ramsey_delay_ns * 1e-9)


@pulse.classical(max_latency_ns=1000, link="controller")
def amplitude_estimator(
    minus: pulse.Bits,
    zero: pulse.Bits,
    plus: pulse.Bits,
    amplitude: float,
) -> tuple[float, float]:
    """The corrected amplitude from a train of n = pulses_per_train pi
    pulses, at amplitude scales 1 - 1/(2 n), 1 and 1 + 1/(2 n), and the
    standard deviation of its relative correction."""
    theta, sigma = cal.sparse_phase(
        _fraction(minus), _fraction(zero), _fraction(plus), shots_per_setting
    )
    return cal.pi_train_amplitude(amplitude, theta, pulses_per_train), sigma / (
        pulses_per_train * math.pi
    )


@pulse.classical(max_latency_ns=200, link="controller")
def decay_delays(t1_cycles: float) -> tuple[float, float]:
    """The decay estimator's second and third delays, spaced by the last
    T1 estimate."""
    spacing = min(max(t1_cycles, 64.0), (wait_range_cycles[1] - first_delay_cycles) / 3.0)
    return first_delay_cycles + spacing, first_delay_cycles + 3.0 * spacing


@pulse.classical(max_latency_ns=1000, link="controller")
def decay_estimator(
    first: pulse.Bits,
    second: pulse.Bits,
    third: pulse.Bits,
    spacing_cycles: float,
    t1_cycles: float,
) -> tuple[float, float]:
    """T1 and its standard deviation (in clock cycles) from the three
    delays; the last estimate *t1_cycles* when none can be formed."""
    t1, sigma, ok = cal.decay_time(
        _fraction(first), _fraction(second), _fraction(third), spacing_cycles, shots_per_setting
    )
    if ok:
        return t1, sigma
    return t1_cycles, t1_cycles


@pulse.classical(max_latency_ns=200, link="controller")
def first_wait(shape: float, rate: float) -> float:
    return min(
        max(cal.adaptive_t1_wait(shape, rate, tracker_wait_scale), wait_range_cycles[0]),
        wait_range_cycles[1],
    )


@pulse.classical(max_latency_ns=2000, link="controller")
def adaptive_t1(
    bit: pulse.Bits,
    shape: float,
    rate: float,
    wait_cycles: float,
    error_1to0: float,
    error_0to1: float,
) -> tuple[float, float, float]:
    """Update the gamma posterior over 1/T1 (rate in clock cycles) from one
    shot; pick the next wait."""
    shape, rate = cal.adaptive_t1_update(shape, rate, bit[0], wait_cycles, error_1to0, error_0to1)
    wait_cycles = min(
        max(cal.adaptive_t1_wait(shape, rate, tracker_wait_scale), wait_range_cycles[0]),
        wait_range_cycles[1],
    )
    return shape, rate, wait_cycles


# ── Shots and steps ─────────────────────────────────────────────────────────


@pulse.kernel
def shot_end(q, pi_amp: float):
    """Read out, then reset from the result (active reset)."""
    d = get_drive_line(q)
    r = get_readout_line(q)
    sync(d, r)
    m = readout(r, square(readout_time, readout_amplitude))
    sync(d, r)
    if m:
        drive(d, square(pulse_time, pi_amp))
    return m


@pulse.kernel
def settle_and_read(q, settle):
    """Read out, then wait *settle* for the qubit to relax (passive reset)."""
    d = get_drive_line(q)
    r = get_readout_line(q)
    sync(d, r)
    m = readout(r, square(readout_time, readout_amplitude))
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
    for _ in range(shots_per_setting):
        if settle:
            ground.append(settle_and_read(q, settle))
            drive(d, square(pulse_time, pi_amp))
            excited.append(settle_and_read(q, settle))
        else:
            ground.append(shot_end(q, pi_amp))
            drive(d, square(pulse_time, pi_amp))
            excited.append(shot_end(q, pi_amp))
    return assignment_errors.submit(ground, excited).result()


@pulse.kernel
def ramsey(q, pi_amp: float, mean_hz: float):
    d = get_drive_line(q)
    minus = []
    zero = []
    plus = []
    phase0 = -2.0 * math.pi * mean_hz * ramsey_delay_ns * 1e-9
    for _ in range(shots_per_setting):
        for k in range(3):
            # Second pi/2 with its tone shifted by pi - phase: P(1) is
            # (1 - cos(2 pi D t + phase)) / 2.
            shift = math.pi - (phase0 + (k - 1) * math.pi / 2)
            drive(d, square(pulse_time, pi_amp * 0.5))
            wait(d, ramsey_delay - pulse_time)
            shift_phase(d.tone, shift)
            drive(d, square(pulse_time, pi_amp * 0.5))
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
        drive(d, square(pulse_time, amplitude))
    return shot_end(q, pi_amp)


@pulse.kernel
def amplitude(q, amplitude: float, pulses: int, pi_amp: float):
    """Correct the amplitude of a pulse from trains of *pulses* of them."""
    minus = []
    zero = []
    plus = []
    for _ in range(shots_per_setting):
        minus.append(train_shot(q, amplitude * (1.0 - 0.5 / pulses_per_train), pulses, pi_amp))
        zero.append(train_shot(q, amplitude, pulses, pi_amp))
        plus.append(train_shot(q, amplitude * (1.0 + 0.5 / pulses_per_train), pulses, pi_amp))
    return amplitude_estimator.submit(minus, zero, plus, amplitude).result()


@pulse.kernel
def decay(q, pi_amp: float, t1_cycles: float):
    d = get_drive_line(q)
    second, third = decay_delays.submit(t1_cycles).result()
    first_bits = []
    second_bits = []
    third_bits = []
    for _ in range(shots_per_setting):
        drive(d, square(pulse_time, pi_amp))
        wait(d, first_delay)
        first_bits.append(shot_end(q, pi_amp))
        drive(d, square(pulse_time, pi_amp))
        wait(d, second, bounds=wait_range_cycles)
        second_bits.append(shot_end(q, pi_amp))
        drive(d, square(pulse_time, pi_amp))
        wait(d, third, bounds=wait_range_cycles)
        third_bits.append(shot_end(q, pi_amp))
    return decay_estimator.submit(
        first_bits, second_bits, third_bits, second - first_delay_cycles, t1_cycles
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
    for _ in range(tracker_shots):
        drive(d, square(pulse_time, pi_amp))
        wait(d, w, bounds=wait_range_cycles)
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
    error_1to0, error_0to1 = assignment(q, nominal_pi_amplitude, settle_time)
    detuning, _ = ramsey(q, nominal_pi_amplitude, 0.0)
    shift_frequency(d.tone, detuning)
    pi_amp, _ = amplitude(q, nominal_pi_amplitude, pulses_per_train, nominal_pi_amplitude)
    half_amp, _ = amplitude(q, pi_amp * 0.5, 2 * pulses_per_train, pi_amp)
    t1, _ = decay(q, pi_amp, t1_guess_cycles)
    tracked_t1(q, pi_amp, 3.0, 3.0 * t1, error_1to0, error_0to1, settle_time)


# ── Frequency binary search ─────────────────────────────────────────────────

initial_sigma_hz = 1e6
coherence_ns = 10_000.0
search_wait_range_cycles = (pulse_cycles, in_cycles(20 * pulse.us))


def _search_setting(mean_hz, sigma_hz):
    """The next binary-search wait (between the pi/2 pulses, in clock
    cycles) and phase."""
    wait_ns = cal.binary_search_wait(sigma_hz, coherence_ns)
    wait_cycles = min(
        max(wait_ns * clock_ghz - pulse_cycles, search_wait_range_cycles[0]),
        search_wait_range_cycles[1],
    )
    return wait_cycles, cal.binary_search_phase(mean_hz, (wait_cycles + pulse_cycles) / clock_ghz)


initial_wait_cycles, initial_phase = _search_setting(0.0, initial_sigma_hz)


@pulse.classical(max_latency_ns=2000, link="controller")
def search_step(
    bits: pulse.Bits,
    mean_hz: float,
    sigma_hz: float,
    wait_cycles: float,
) -> tuple[float, float, float, float]:
    """Without a reset, a shot's outcome is whether the readout changed."""
    outcome = 2 * abs(bits[0] - bits[1]) - 1
    mean_hz, sigma_hz = cal.binary_search_update(
        mean_hz,
        sigma_hz,
        outcome,
        (wait_cycles + pulse_cycles) / clock_ghz,
        -0.02,
        0.6,
        coherence_ns,
    )
    wait_cycles, phase = _search_setting(mean_hz, sigma_hz)
    return mean_hz, sigma_hz, wait_cycles, phase


@pulse.kernel
def binary_search(q, rounds: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    previous = readout(r, square(readout_time, readout_amplitude))
    sync(d, r)
    mean, sigma, w, phase = 0.0, initial_sigma_hz, initial_wait_cycles, initial_phase
    drive(d, square(pulse_time, nominal_pi_amplitude * 0.5))
    wait(d, int(initial_wait_cycles))
    shift_phase(d.tone, math.pi - initial_phase)
    drive(d, square(pulse_time, nominal_pi_amplitude * 0.5))
    for i in range(rounds):
        sync(d, r)
        m = readout(r, square(readout_time, readout_amplitude))
        mean, sigma, w, phase = search_step.submit([m, previous], mean, sigma, w).result()
        previous = m
        if i < rounds - 1:
            sync(d, r)
            drive(d, square(pulse_time, nominal_pi_amplitude * 0.5))
            wait(d, w, bounds=search_wait_range_cycles)
            shift_phase(d.tone, math.pi - phase)
            drive(d, square(pulse_time, nominal_pi_amplitude * 0.5))


# ── Devices ─────────────────────────────────────────────────────────────────


def chain_plant(target, t1_us=18.3, offset_hz=60e3, amplitude_scale=1.015):
    """Qubit 0 of *target* as the chain's device: its T1, a detuning from
    the drive, a drive-amplitude error and small readout errors."""
    qubit = target.qubits[0]
    scale = target.drive_amplitude_scale(0) * amplitude_scale
    plant = dataclasses.replace(
        qubit,
        t1_us=t1_us,
        frequency_hz=qubit.frequency_hz + offset_hz,
        drive=qubit.drive.replace(amplitude_scale_rad_per_ns=scale),
        readout=qubit.readout.replace(error_1to0=0.03, error_0to1=0.02),
    )
    return dataclasses.replace(target, qubits={**target.qubits, 0: plant})


def search_plant(target, offset_hz):
    """A device like the binary-search paper's: T2* 12 us, T1 80 us, and
    readout errors that give its fringe contrast of 0.6 between shots."""
    qubit = target.qubits[0]
    plant = dataclasses.replace(
        qubit,
        t1_us=80.0,
        t2_star_us=12.0,
        thermal_population=0.0,
        frequency_hz=qubit.frequency_hz + offset_hz,
        readout=qubit.readout.replace(error_1to0=0.115, error_0to1=0.105),
    )
    return dataclasses.replace(target, qubits={**target.qubits, 0: plant})


def chain_results(branch):
    """Each step's answer, from the branch one shot took."""
    shape, rate, _ = branch.answers("adaptive_t1")[-1]
    pi_amplitude, half_pi_amplitude = branch.answers("amplitude_estimator")[:2]
    return {
        "readout errors": branch.answers("assignment_errors")[-1],
        "detuning (kHz)": branch.answers("ramsey_estimator")[-1][0] / 1e3,
        "pi amplitude": pi_amplitude[0],
        "pi/2 amplitude": half_pi_amplitude[0],
        "T1, decay estimator (us)": branch.answers("decay_estimator")[-1][0] / clock_ghz / 1e3,
        "T1, adaptive (us)": rate / shape / clock_ghz / 1e3,
    }


if __name__ == "__main__":
    print(
        "The millisecond calibration chain of Marciniak et al., arXiv:2602.11912 (2026),\n"
        "with the adaptive T1 tracker of Berritta et al., arXiv:2506.09576 (2025).\n"
    )
    target = transmon_17q()
    dev = pulse.device({"target": target})
    q = dev.qudit_handles[0]
    plant = chain_plant(target)

    print("=== 1. The calibration chain, on the controller ===")
    start = time.perf_counter()
    compiled = pulse.compile(calibration_chain, device=dev, control_system=control_system)
    lowered = compiled.lower(q, 20_000.0 * clock_ghz)
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
    result = pulse.simulate(
        lowered, plant=plant, shots=1, seed=7, branches="sampled", readout="model"
    )
    print(f"  one shot of the chain simulated in {time.perf_counter() - start:.1f} s")
    for name, value in chain_results(result.branches[0]).items():
        shown = ", ".join(f"{v:.3f}" for v in value) if isinstance(value, tuple) else f"{value:.4f}"
        print(f"  {name:<26} {shown}")
    print(
        f"  truth: detuning 60 kHz, pi amplitude {nominal_pi_amplitude / 1.015:.4f}, "
        f"pi/2 {nominal_pi_amplitude / 2 / 1.015:.4f}, T1 18.3 us"
    )

    print("\n=== 2. Frequency binary search, 15 shots, no reset ===")
    rounds = 15
    lowered = pulse.compile(binary_search, device=dev, control_system=control_system).lower(
        q, rounds
    )
    errors, sigmas = [], []
    for seed in range(4):
        offset = (seed - 1.5) * 300e3
        found = pulse.simulate(
            lowered,
            plant=search_plant(target, offset),
            shots=1,
            seed=seed,
            branches="sampled",
            readout="model",
        )
        mean, sigma, _, _ = found.branches[0].answers("search_step")[-1]
        errors.append(mean - offset)
        sigmas.append(sigma)
        print(
            f"  detuning {offset / 1e3:+6.0f} kHz: estimate {mean / 1e3:+6.0f} +- {sigma / 1e3:.0f} kHz"
        )
    print(
        f"  posterior width from {initial_sigma_hz / 1e3:.0f} kHz to {sum(sigmas) / len(sigmas) / 1e3:.0f} kHz"
    )

    print("\n=== 3. What the estimators cost the controller ===")
    lowered = compiled.lower(q, 20_000.0 * clock_ghz)
    report = controller.resources(lowered)
    print(report)
    slow = dataclasses.replace(processor, cycles_per_op={**processor.cycles_per_op, "exp": 400})
    slow_system = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[slow])], links=links
    )
    print("  with exp at 400 cycles:")
    try:
        pulse.compile(calibration_chain, device=dev, control_system=slow_system).lower(
            q, 20_000.0 * clock_ghz
        )
    except controller.ControllerLimitError as error:
        for violation in error.violations:
            print(f"    {violation}")
