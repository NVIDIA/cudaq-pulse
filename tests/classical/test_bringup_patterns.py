# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patterns for calibrating while the QPU runs: N-way switches on an index,
an on-device random source, hidden and exposed latency, and the converged
pattern."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import emulation, random_index
from cudaq_pulse.targets import DeviceDescription, DispersiveReadout, PhysicalQudit

PI_AMP = math.pi / 20


def _device(**readout):
    qudit = PhysicalQudit(
        index=0,
        frequency_hz=5e9,
        anharmonicity_hz=-2e8,
        t1_us=0.0,
        t2_star_us=0.0,
        readout=DispersiveReadout(**readout),
    )
    return pulse.device(DeviceDescription(name="sim", qudits={0: qudit}))


@pulse.classical(max_latency_ns=100)
def pick(bits: pulse.Bits) -> int:
    return 0


@pulse.kernel
def chain_switch(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    i = pick.submit([m]).result()
    if i == 0:
        drive(d, square(40, PI_AMP * 0.25))
    elif i == 1:
        drive(d, square(80, PI_AMP * 0.25))
    elif i == 2:
        drive(d, square(120, PI_AMP * 0.25))
    else:
        drive(d, square(160, PI_AMP * 0.25))


@pulse.kernel
def balanced_switch(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    i = pick.submit([m]).result()
    if i < 2:
        if i == 0:
            drive(d, square(40, PI_AMP * 0.25))
        else:
            drive(d, square(80, PI_AMP * 0.25))
    else:
        if i == 2:
            drive(d, square(120, PI_AMP * 0.25))
        else:
            drive(d, square(160, PI_AMP * 0.25))


@pytest.mark.parametrize("kernel", [chain_switch, balanced_switch], ids=["chain", "balanced"])
def test_an_n_way_switch_takes_the_arm_of_its_index(kernel):
    dev = _device()
    lowered = pulse.compile(kernel, device=dev).lower(dev.qudits[0])
    (call,) = lowered.classical_report()
    # Every arm ends at the same time, after the longest one.
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert {e.start_cycles + e.duration_cycles for e in drives} == {call.due_cycles + 160}
    for index in range(4):
        result = pulse.simulate(lowered, handlers={"pick": lambda bits, i=index: i}, levels=2)
        angle = math.pi * 0.25 * (index + 1)
        assert result.density_matrix[1, 1].real == pytest.approx(math.sin(angle / 2) ** 2)


gate = random_index(4, name="gate", max_latency_ns=20, link=0)


@pulse.kernel
def random_rotation(q):
    d = get_drive_line(q)
    i = gate.submit().result()
    if i < 2:
        if i == 0:
            drive(d, square(40, PI_AMP * 0.25))
        else:
            drive(d, square(40, PI_AMP * 0.5))
    else:
        if i == 2:
            drive(d, square(40, PI_AMP * 0.75))
        else:
            drive(d, square(40, PI_AMP))


def _expected_p1():
    return np.mean([math.sin(math.pi * 0.25 * (k + 1) / 2) ** 2 for k in range(4)])


def test_a_random_source_follows_every_value_without_shots():
    dev = _device()
    lowered = pulse.compile(random_rotation, device=dev).lower(dev.qudits[0])
    result = pulse.simulate(lowered, levels=2)
    assert sorted(branch.calls[0].answer for branch in result.branches) == [0, 1, 2, 3]
    assert result.density_matrix[1, 1].real == pytest.approx(_expected_p1())
    assert result.density_matrix.trace().real == pytest.approx(1.0)


def test_a_random_source_draws_per_shot():
    dev = _device()
    lowered = pulse.compile(random_rotation, device=dev).lower(dev.qudits[0])
    result = pulse.simulate(lowered, shots_count=4000, seed=2, levels=2)
    assert result.records is not None and len(result.records) == 4000
    # The ensemble state is exact; the shots split about evenly.
    assert result.density_matrix[1, 1].real == pytest.approx(_expected_p1())
    a = pulse.simulate(lowered, shots_count=100, seed=9, levels=2)
    b = pulse.simulate(lowered, shots_count=100, seed=9, levels=2)
    assert np.array_equal(a.records, b.records)


def test_a_random_source_is_marked_in_the_manifest_and_emulated_locally():
    dev = _device()
    lowered = pulse.compile(random_rotation, device=dev).lower(dev.qudits[0])
    (entry,) = emulation.manifest(lowered)["functions"]
    assert entry["random"] == 4
    run = emulation.emulate(lowered, shots_count=2000, seed=4)
    counts = np.bincount([shot[0].answer for shot in run.calls], minlength=4)
    assert counts.sum() == 2000 and counts.min() > 400
    assert run.late_fraction == 0.0
    with pytest.raises(ValueError, match="at least 2"):
        random_index(1)


@pytest.mark.parametrize("depletion_ns", [0.0, 150.0, 300.0, 500.0])
def test_latency_within_the_depletion_time_is_hidden(depletion_ns):
    dev = _device(depletion_ns=depletion_ns)

    @pulse.classical(max_latency_ns=300)
    def decide(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.kernel
    def feedback(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(200, 0.05))
        if decide.submit([m]).result()[0]:
            drive(d, square(40, 0.1))

    lowered = pulse.compile(feedback, device=dev).lower(dev.qudits[0])
    (call,) = lowered.classical_report()
    clock = dev.clock_ghz
    budget = call.due_cycles - call.submit_cycles
    expected = max(0, budget - math.ceil(depletion_ns * clock))
    assert call.exposed_cycles == expected
    assert call.hidden_cycles == budget - expected
    if depletion_ns * clock >= budget:
        assert call.exposed_cycles == 0


def test_depletion_defaults_from_kappa():
    qudit = PhysicalQudit(
        index=0,
        frequency_hz=5e9,
        anharmonicity_hz=-2e8,
        t1_us=0.0,
        t2_star_us=0.0,
        readout=DispersiveReadout(linewidth_hz=1e6),
    )
    assert qudit.depletion_ns == pytest.approx(math.log(100) / (2 * math.pi * 1e-3))
    assert _device().description.qudits[0].depletion_ns == 0.0


# The converged pattern: the trip count is fixed; the estimator carries a
# done flag in its state and keeps answering with its final point, and the
# kernel gates its probe pulses on the flag.


@pulse.classical(max_latency_ns=50)
def count_up(bits: pulse.Bits, x: float, done: float) -> tuple[float, float]:
    if done > 0.5:
        return x, 1.0
    x = x + 1.0
    return x, 1.0 if x >= 2.0 else 0.0


@pulse.kernel
def converging(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    x = 0.0
    done = 0.0
    for _ in range(5):
        if done < 0.5:
            drive(d, square(40, PI_AMP * 0.5))
        sync(d, r)
        m = readout(r, square(100, 0.05))
        x, done = count_up.submit([m], x, done).result()


def test_the_converged_pattern_stops_probing_and_keeps_its_trip_count():
    dev = _device()
    lowered = pulse.compile(converging, device=dev).lower(dev.qudits[0])
    assert len(lowered.classical_report()) == 5
    result = pulse.simulate(lowered, initial_state=[1, 0], levels=2)
    for branch in result.branches:
        assert [c.answer for c in branch.calls] == [
            (1.0, 0.0),
            (2.0, 1.0),
            (2.0, 1.0),
            (2.0, 1.0),
            (2.0, 1.0),
        ]
    # Two pi/2 probes, then none: after the second measurement nothing
    # rotates the qubit, so every later outcome repeats it.
    assert sum(branch.probability for branch in result.branches) == pytest.approx(1.0)
    for branch in result.branches:
        assert len(branch.levels) == 5
        assert branch.levels[2] == branch.levels[3] == branch.levels[4] == branch.levels[1]
    first = {branch.levels[0]: 0.0 for branch in result.branches}
    for branch in result.branches:
        first[branch.levels[0]] += branch.probability
    assert first == pytest.approx({0: 0.5, 1: 0.5})
