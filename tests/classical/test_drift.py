# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parameter drift: the processes, the drift clock, a drifting system's
equality with its non-drifting twin (compiling, caching, pickling,
exporting), custom models, and the outcome probabilities against full
simulation."""

from __future__ import annotations

import dataclasses
import math
import pickle

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.quantum_systems import PhysicalQudit
from cudaq_pulse.quantum_systems.drift import (
    OrnsteinUhlenbeck,
    ParameterDrift,
    ParameterDriftModel,
    TelegraphT1,
)

PI_AMP = math.pi / 20


def _target(t1_us=20.0, qudit_count=1):
    qudits = {
        i: PhysicalQudit(
            index=i, frequency_hz=5e9 + 1e8 * i, anharmonicity_hz=-2e8, t1_us=t1_us, t2_star_us=0.0
        )
        for i in range(qudit_count)
    }
    return pulse.quantum_system(name="sim", qudits=qudits)


def _frequency_only(sigma_hz, correlation_time):
    return ParameterDrift(
        frequency=OrnsteinUhlenbeck(sigma_hz, correlation_time), t1=None, amplitude=None
    )


@dataclasses.dataclass(frozen=True)
class LinearFrequencyDrift:
    """A user model: the frequency walks at a fixed rate."""

    hz_per_s: float
    correlation_time: pulse.Time | None = None

    def start(self, calibrated, rng):
        return calibrated

    def advance(self, true, calibrated, seconds, rng):
        return true.replace(frequency_hz=true.frequency_hz + self.hz_per_s * seconds)


@pulse.kernel
def t1_shot(q, wait_ns: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, PI_AMP))
    wait(d, wait_ns)
    sync(d, r)
    readout(r, square(40, 0.05))


@pulse.kernel
def rabi(q, scale: float):
    d = get_drive_line(q)
    drive(d, square(40, PI_AMP * scale))


@pulse.kernel
def ramsey(q, phase: float):
    d = get_drive_line(q)
    drive(d, square(40, PI_AMP / 2))
    wait(d, 400)
    shift_phase(d.tone, phase)
    drive(d, square(40, PI_AMP / 2))


# ── The processes ────────────────────────────────────────────────────────────


def test_telegraph_t1_switches_at_its_rate_and_spends_half_its_time_low():
    t1 = TelegraphT1(low_fraction=1 / 6, dwell_time=100 * pulse.ms)
    qs = _target(t1_us=30.0).replace(
        parameter_drift=ParameterDrift(frequency=None, t1=t1, amplitude=None),
        parameter_drift_seed=1,
    )
    # It starts at the calibrated value.
    assert qs.true_qudits[0].t1_us == 30.0
    values = []
    for _ in range(20000):
        qs.advance(1 * pulse.ms)
        values.append(qs.true_qudits[0].t1_us)
    t1_us = np.array(values)
    assert set(np.unique(t1_us)) == {5.0, 30.0}
    assert np.mean(t1_us == 30.0) == pytest.approx(0.5, abs=0.05)
    switches = np.count_nonzero(np.diff(t1_us))
    # 10 Hz for 20 s, less the pairs that cancel within one step.
    assert switches == pytest.approx(200 * math.exp(-2 * 10 * 1e-3), rel=0.2)


def test_ornstein_uhlenbeck_drift_has_its_variance_and_correlation():
    sigma, tau, dt = 50e3, 0.5, 0.05
    qs = _target().replace(
        parameter_drift=_frequency_only(sigma, tau * pulse.s), parameter_drift_seed=2
    )
    calibrated_hz = qs.physical_qudits[0].frequency_hz
    # The offset starts at 0; ten correlation times forget that.
    qs.advance(10 * tau * pulse.s)
    offsets = []
    for _ in range(40000):
        qs.advance(dt * pulse.s)
        offsets.append(qs.true_qudits[0].frequency_hz - calibrated_hz)
    x = np.array(offsets)
    assert np.std(x) == pytest.approx(sigma, rel=0.1)
    lag = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag == pytest.approx(math.exp(-dt / tau), abs=0.02)


def test_amplitude_drift_scales_the_drive_around_its_calibrated_scale():
    drift = ParameterDrift(frequency=None, t1=None, amplitude=OrnsteinUhlenbeck(0.05, 1 * pulse.s))
    qs = _target().replace(parameter_drift=drift, parameter_drift_seed=3)
    nominal = qs.drive_amplitude_scale(0)
    scales = []
    for _ in range(2000):
        qs.advance(1 * pulse.s)
        scales.append(qs.true_qudits[0].drive.amplitude_scale_rad_per_ns / nominal)
    assert np.mean(scales) == pytest.approx(1.0, abs=0.01)
    assert np.std(scales) == pytest.approx(0.05, rel=0.15)
    assert qs.physical_qudits[0].drive is None


def test_the_default_model_is_slow_and_documented():
    model = ParameterDrift()
    assert model.frequency == OrnsteinUhlenbeck(20e3, 3600 * pulse.s)
    assert model.t1 == TelegraphT1(0.5, 600 * pulse.s)
    assert model.amplitude == OrnsteinUhlenbeck(0.002, 3600 * pulse.s)
    assert model.correlation_time == 600 * pulse.s
    qs = pulse.quantum_system("transmon_surface17", parameter_drift=True, parameter_drift_seed=0)
    assert set(qs.parameter_drift) == set(qs.physical_qudits)
    qs.advance(1 * pulse.s)
    moved = [
        abs(qs.true_qudits[q].frequency_hz - qs.physical_qudits[q].frequency_hz)
        for q in qs.physical_qudits
    ]
    # A second of drift on an hour's correlation: hundreds of Hz.
    assert 0 < max(moved) < 5e3


# ── Seeds and qudits ─────────────────────────────────────────────────────────


def _trace(qs, steps=50):
    out = []
    for _ in range(steps):
        qs.advance(100 * pulse.ms)
        out.append({q: (t.frequency_hz, t.t1_us) for q, t in qs.true_qudits.items()})
    return out


def test_a_seed_repeats_the_drift_and_each_qudit_drifts_on_its_own_stream():
    drift = ParameterDrift(
        frequency=OrnsteinUhlenbeck(1e4, 1 * pulse.s),
        t1=TelegraphT1(0.5, 1 * pulse.s),
        amplitude=None,
    )
    one = _target(qudit_count=2)
    a = _trace(one.replace(parameter_drift={0: drift}, parameter_drift_seed=5))
    assert a == _trace(one.replace(parameter_drift={0: drift}, parameter_drift_seed=5))
    assert a != _trace(one.replace(parameter_drift={0: drift}, parameter_drift_seed=6))
    # Adding a drifting qudit leaves qudit 0's trajectory as it was.
    both = _trace(one.replace(parameter_drift=drift, parameter_drift_seed=5))
    assert [step[0] for step in both] == [step[0] for step in a]
    assert both[-1][1] != a[-1][1]
    # Without a seed each system drifts its own way.
    assert _trace(one.replace(parameter_drift=drift)) != _trace(one.replace(parameter_drift=drift))


# ── Building ─────────────────────────────────────────────────────────────────


def test_parameter_drift_takes_true_a_model_or_a_dict_by_qudit():
    target = _target(qudit_count=3)
    assert set(target.replace(parameter_drift=True).parameter_drift) == {0, 1, 2}
    model = _frequency_only(1e3, 1 * pulse.s)
    every = target.replace(parameter_drift=model)
    assert dict(every.parameter_drift) == {0: model, 1: model, 2: model}
    some = target.replace(parameter_drift={2: model})
    assert dict(some.parameter_drift) == {2: model}
    some.advance(1 * pulse.s)
    assert some.true_qudits[0] is target.physical_qudits[0]
    assert some.true_qudits[2] != target.physical_qudits[2]
    assert type(some.true_qudits[2]) is type(some.physical_qudits[2])
    with pytest.raises(ValueError, match="names qudit 5, which quantum system 'sim' does not"):
        target.replace(parameter_drift={5: model})
    with pytest.raises(TypeError, match="parameter_drift is True, a drift model"):
        target.replace(parameter_drift="fast")
    with pytest.raises(TypeError, match="parameter_drift_seed is a non-negative int"):
        target.replace(parameter_drift=True, parameter_drift_seed=1.5)
    with pytest.raises(ValueError, match="parameter_drift_seed seeds parameter_drift"):
        target.replace(parameter_drift_seed=1)
    with pytest.raises(ValueError, match="drifts nothing"):
        ParameterDrift(frequency=None, t1=None, amplitude=None)
    with pytest.raises(TypeError, match="correlation_time is a time"):
        OrnsteinUhlenbeck(1e3, 10.0)
    with pytest.raises(ValueError, match="low_fraction is in"):
        TelegraphT1(0.0, 1 * pulse.s)


def test_pulse_quantum_system_takes_the_drift_keywords():
    qs = pulse.quantum_system(
        "transmon_surface17", parameter_drift={3: ParameterDrift()}, parameter_drift_seed=1
    )
    twin = pulse.quantum_system("transmon_surface17")
    assert qs == twin and hash(qs) == hash(twin) and qs.fingerprint == twin.fingerprint
    assert qs.calibration == twin.calibration
    assert qs.as_dict() == twin.as_dict()
    assert repr(qs) == "<QuantumSystem transmon_surface17 (17 qudits, parameter drift on 1)>"
    built = pulse.quantum_system(name="sim", qudits=_target().physical_qudits, parameter_drift=True)
    assert built == _target() and built.parameter_drift_time == 0 * pulse.ns
    with pytest.raises(NotImplementedError, match="has no simulation model"):
        pulse.quantum_system(
            name="sim", qudits=_target().physical_qudits, model=None, parameter_drift=True
        )


def test_replace_without_drift_makes_a_non_drifting_variant():
    qs = _target().replace(parameter_drift=True, parameter_drift_seed=0)
    calm = qs.replace(parameter_drift=None)
    assert calm.parameter_drift is None and calm == qs
    moved = qs.replace(layout=[0])
    assert moved.parameter_drift is None


def test_a_system_without_drift_has_its_calibrated_values_as_true_ones():
    qs = _target()
    assert qs.true_qudits is qs.physical_qudits
    assert qs.parameter_drift is None and qs.parameter_drift_time is None
    with pytest.raises(NotImplementedError, match="built without parameter drift"):
        qs.advance(1 * pulse.s)


def test_advance_takes_a_time():
    qs = _target().replace(parameter_drift=True)
    with pytest.raises(TypeError, match=r"advance\(\) takes a time, as 20 \* pulse.ms"):
        qs.advance(0.02)
    with pytest.raises(TypeError, match="takes a time"):
        qs.advance(1)
    with pytest.raises(ValueError, match="only moves forward"):
        qs.advance(-1 * pulse.ms)
    qs.advance(0 * pulse.ms)
    qs.advance(20 * pulse.ms)
    assert qs.parameter_drift_time == 20 * pulse.ms


# ── Custom models ────────────────────────────────────────────────────────────


def test_a_user_model_plugs_in():
    model = LinearFrequencyDrift(1e3)
    assert isinstance(model, ParameterDriftModel)
    qs = _target().replace(parameter_drift={0: model})
    qs.advance(2 * pulse.s)
    assert qs.true_qudits[0].frequency_hz == pytest.approx(5e9 + 2e3)

    @dataclasses.dataclass(frozen=True)
    class Broken:
        def start(self, calibrated, rng):
            return calibrated

        def advance(self, true, calibrated, seconds, rng):
            return true.frequency_hz

    broken = _target().replace(parameter_drift=Broken())
    with pytest.raises(TypeError, match=r"advance\(\) returns the qudit's PhysicalQudit"):
        broken.advance(1 * pulse.s)


def test_a_model_may_start_away_from_the_calibration():
    @dataclasses.dataclass(frozen=True)
    class Miscalibrated:
        def start(self, calibrated, rng):
            return calibrated.replace(frequency_hz=calibrated.frequency_hz + 5e5)

        def advance(self, true, calibrated, seconds, rng):
            return true

    qs = _target().replace(parameter_drift=Miscalibrated())
    assert qs.true_qudits[0].frequency_hz == 5e9 + 5e5
    assert qs.physical_qudits[0].frequency_hz == 5e9


# ── The drift clock ──────────────────────────────────────────────────────────


def test_simulate_advances_the_clock_by_its_shots():
    qs = _target(t1_us=20.0).replace(parameter_drift={0: LinearFrequencyDrift(0.0)})
    program = rabi.lower(qs.qudits[0], 1.0)
    duration = program.duration
    pulse.simulate(program, levels=2)
    # One shot without shots_count; the reset is 5 T1 by default.
    assert qs.parameter_drift_time == duration + 100 * pulse.us
    pulse.simulate(program, levels=2, shots_count=10, reset_time=200 * pulse.us)
    assert qs.parameter_drift_time == duration + 100 * pulse.us + 10 * (duration + 200 * pulse.us)
    # A kernel call runs a shot on the system too.
    before = qs.parameter_drift_time
    rabi(qs.qudits[0], 1.0)
    assert qs.parameter_drift_time > before


def test_reset_time_is_a_time_and_changes_nothing_without_drift():
    qs = _target()
    program = rabi.lower(qs.qudits[0], 0.5)
    plain = pulse.simulate(program, levels=2, shots_count=20, seed=1)
    reset = pulse.simulate(program, levels=2, shots_count=20, seed=1, reset_time=1 * pulse.ms)
    np.testing.assert_array_equal(plain.density_matrix, reset.density_matrix)
    np.testing.assert_array_equal(plain.records, reset.records)
    for system in (qs, qs.replace(parameter_drift=True)):
        with pytest.raises(TypeError, match="reset_time is a time, as 100 \\* pulse.us"):
            pulse.simulate(program, levels=2, quantum_system=system, reset_time=1e5)
    with pytest.raises(ValueError, match="must not be negative"):
        pulse.simulate(program, levels=2, reset_time=-1 * pulse.us)


def test_simulate_plays_the_true_values_and_holds_them_for_the_call():
    qs = _target(t1_us=0.0).replace(parameter_drift={0: LinearFrequencyDrift(1e6)})
    program = ramsey.lower(qs.qudits[0], 0.0)
    calm = pulse.simulate(program, levels=2)
    qs.advance(1 * pulse.s)  # 1 MHz off
    twin = pulse.quantum_system(name="sim", qudits=_target(t1_us=0.0).physical_qudits)
    offset = qs.replace(qudits=dict(qs.true_qudits))
    detuned = pulse.simulate(program, levels=2, quantum_system=offset)
    assert offset.parameter_drift is None and twin == qs
    played = pulse.simulate(program, levels=2)
    np.testing.assert_allclose(played.density_matrix, detuned.density_matrix, atol=1e-12)
    assert abs(played.density_matrix[1, 1] - calm.density_matrix[1, 1]) > 0.1


def test_a_long_call_is_split_into_chunks_of_its_correlation_time():
    # 1 ms per shot (with the reset), a 2.5 ms correlation time: 20 shots
    # are 8 chunks of 2 or 3 shots, each at the frequency at its start.
    model = LinearFrequencyDrift(2e6, correlation_time=2.5 * pulse.ms)
    qs = _target(t1_us=0.0).replace(parameter_drift={0: model})
    program = t1_shot.lower(qs.qudits[0], 0.0)
    reset = 1 * pulse.ms - program.duration
    result = pulse.simulate(program, levels=2, shots_count=20, seed=4, reset_time=reset)
    assert qs.parameter_drift_time == 20 * pulse.ms
    assert result.records.shape == (20, 1)
    assert sum(b.probability for b in result.branches) == pytest.approx(1.0)
    assert np.trace(result.density_matrix).real == pytest.approx(1.0)
    # Seeded, it repeats.
    again = _target(t1_us=0.0).replace(parameter_drift={0: model})
    repeat = pulse.simulate(
        program, levels=2, shots_count=20, seed=4, reset_time=reset, quantum_system=again
    )
    np.testing.assert_array_equal(result.records, repeat.records)
    # One advance per chunk: 8 here; one for a model without a correlation
    # time, or a call shorter than it.
    for correlation_time, chunks in ((2.5 * pulse.ms, 8), (None, 1), (1 * pulse.s, 1)):
        counted = Counted(correlation_time)
        system = _target(t1_us=0.0).replace(parameter_drift={0: counted})
        pulse.simulate(program, levels=2, shots_count=20, reset_time=reset, quantum_system=system)
        assert counted.seconds == pytest.approx([chunk / 1e3 for chunk in _sizes(20, chunks)])


def _sizes(shots, chunks):
    base, extra = divmod(shots, chunks)
    return [base + (i < extra) for i in range(chunks)]


class Counted:
    """A model that records how far each advance moved."""

    def __init__(self, correlation_time):
        self.correlation_time = correlation_time
        self.seconds = []

    def start(self, calibrated, rng):
        return calibrated

    def advance(self, true, calibrated, seconds, rng):
        self.seconds.append(seconds)
        return true


def test_chunks_play_the_drift_within_the_call():
    # The frequency runs 25 kHz per ms: 40 shots of 1 ms span a Ramsey
    # fringe's worth of detuning, so chunked, the call's state is the average
    # over the drift; unchunked it is the start's.
    fast = LinearFrequencyDrift(25e6, correlation_time=1 * pulse.ms)
    slow = LinearFrequencyDrift(25e6)
    results = []
    for model in (fast, slow):
        qs = _target(t1_us=0.0).replace(parameter_drift={0: model})
        program = ramsey.lower(qs.qudits[0], 0.0)
        reset = 1 * pulse.ms - program.duration
        results.append(pulse.simulate(program, levels=2, shots_count=40, reset_time=reset))
    chunked, whole = results
    assert abs(chunked.density_matrix[1, 1] - whole.density_matrix[1, 1]) > 0.05


# ── A drifting system compiles as its twin ───────────────────────────────────


def test_programs_are_shared_with_the_twin_and_bound_to_the_system_lowered_for():
    @pulse.kernel
    def pi(q):
        d = get_drive_line(q)
        drive(d, square(40, PI_AMP))

    twin = _target()
    qs = twin.replace(parameter_drift=True, parameter_drift_seed=0)
    other = twin.replace(parameter_drift=True, parameter_drift_seed=1)
    first = pi.lower(twin.qudits[0])
    drifting = pi.lower(qs.qudits[0])
    assert pi.cache_info().compiled == 1
    assert first.quantum_system is twin and drifting.quantum_system is qs
    assert pi.lower(other.qudits[0]).quantum_system is other
    assert pi.lower(twin.qudits[0]).quantum_system is twin
    assert pi.cache_info().compiled == 1
    assert drifting.mlir() == first.mlir()
    assert drifting.export_to_controller() == first.export_to_controller()
    assert drifting.export_to_emulator() == first.export_to_emulator()

    # Compiled the other way round: a drifting system's first, then its twin.
    @pulse.kernel
    def half(q):
        d = get_drive_line(q)
        drive(d, square(40, PI_AMP / 2))

    assert half.lower(qs.qudits[0]).quantum_system is qs
    assert half.lower(twin.qudits[0]).quantum_system is twin
    compiled = pulse.compile(half, quantum_system=qs)
    assert compiled.lower(qs.qudits[0]).quantum_system is qs
    before = twin.parameter_drift_time
    pulse.simulate(half.lower(twin.qudits[0]), levels=2)
    assert before is None and qs.parameter_drift_time == 0 * pulse.ns


def test_a_pickled_drifting_system_continues_where_it_was_and_the_twin_stays_calm():
    twin = _target()
    qs = twin.replace(parameter_drift=_frequency_only(1e5, 1 * pulse.s), parameter_drift_seed=9)
    qs.advance(3 * pulse.s)
    program = rabi.lower(qs.qudits[0], 1.0)
    copy, copied_program = pickle.loads(pickle.dumps((qs, program)))
    assert copied_program.quantum_system is copy and copy == qs
    assert copy.true_qudits == qs.true_qudits
    assert copy.parameter_drift_time == qs.parameter_drift_time
    copy.advance(1 * pulse.s)
    qs.advance(1 * pulse.s)
    assert copy.true_qudits == qs.true_qudits
    # The twin pickles as a system without drift, even where the drifting
    # system is known.
    calm = pickle.loads(pickle.dumps(twin))
    assert calm.parameter_drift is None
    calm_program = pickle.loads(pickle.dumps(rabi.lower(twin.qudits[0], 1.0)))
    assert calm_program.quantum_system.parameter_drift is None


# ── The outcome probabilities against simulation ────────────────────────────


def test_the_outcome_probabilities_match_simulating_the_true_values():
    dev = _target()
    q = dev.qudits[0]
    detuned = dev.replace(
        qudits={0: dev.physical_qudits[0].replace(t1_us=8.0, frequency_hz=5e9 + 1e6)}
    )
    compiled_t1 = pulse.compile(t1_shot, quantum_system=dev)
    for wait_cycles in (0.0, 4000.0, 16000.0):
        result = pulse.simulate(
            compiled_t1.lower(q, wait_cycles),
            quantum_system=detuned.replace(
                qudits={0: detuned.physical_qudits[0].replace(frequency_hz=5e9)}
            ),
            dt_ns=1.0,
            levels=2,
        )
        p1 = {b.levels: b.probability for b in result.branches}.get((1,), 0.0)
        # The probability counts the wait; the pi pulse and readout add 40 ns.
        elapsed = wait_cycles / dev.clock_ghz + 40
        assert p1 == pytest.approx(cal.survival_probability(elapsed, 8.0), abs=0.03)
    # On resonance the amplitude scale stretches the rotation exactly.
    resonant = _target(t1_us=0.0)
    scale = 1.05
    overdriven = resonant.replace(
        qudits={
            0: resonant.physical_qudits[0].replace(
                drive=pulse.quantum_systems.MicrowaveDrive(
                    amplitude_scale_rad_per_ns=scale * resonant.drive_amplitude_scale(0)
                )
            )
        }
    )
    compiled_rabi = pulse.compile(rabi, quantum_system=dev)
    for angle in (0.5, 1.0):
        result = pulse.simulate(compiled_rabi.lower(q, angle), quantum_system=overdriven, levels=2)
        assert result.density_matrix[1, 1].real == pytest.approx(
            cal.rabi_probability(math.pi * angle, scale), abs=1e-6
        )
    # Ramsey: the simulated sequence gives (1 + cos(2 pi d T - phase)) / 2,
    # which is the estimators' convention at phase pi - phase.
    offset = _target(t1_us=0.0)
    offset = offset.replace(qudits={0: offset.physical_qudits[0].replace(frequency_hz=5e9 + 1e6)})
    compiled_ramsey = pulse.compile(ramsey, quantum_system=dev)
    separation_ns = (400 + 40) / dev.clock_ghz
    for phase in (0.0, 1.0, 2.5):
        result = pulse.simulate(compiled_ramsey.lower(q, phase), quantum_system=offset, levels=2)
        expected = cal.ramsey_probability(separation_ns, math.pi - phase, 1e6)
        assert result.density_matrix[1, 1].real == pytest.approx(expected, abs=0.03)


def test_the_simulated_system_must_define_the_programs_qudits():
    dev = _target()
    lowered = pulse.compile(rabi, quantum_system=dev).lower(dev.qudits[0], 1.0)
    empty = pulse.quantum_system(name="none", qudits={})
    with pytest.raises(ValueError, match="does not define qudits"):
        pulse.simulate(lowered, quantum_system=empty, levels=2)
    with pytest.raises(TypeError, match="quantum_system must be a QuantumSystem"):
        pulse.simulate(lowered, quantum_system="transmon_surface17", levels=2)
