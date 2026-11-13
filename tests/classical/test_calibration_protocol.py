# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The calibration protocol of examples/21 as programs: its steps simulated
end to end with their estimators on the controller link, and the
controller's limits on the whole chain. benchmarks/classical/
calibration_protocol.py runs the same steps over many more seeds."""

from __future__ import annotations

import dataclasses
import runpy
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import controller
from cudaq_pulse.targets import transmon_17q

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "21_calibration_protocol.py"
if not _EXAMPLE.exists():
    pytest.skip("needs the source tree's examples/", allow_module_level=True)
P = runpy.run_path(str(_EXAMPLE), run_name="calibration_protocol")
PI_AMP, CLOCK = P["nominal_pi_amplitude"], P["clock_ghz"]
amplitude, tracked_t1, TRAIN = P["amplitude"], P["tracked_t1"], P["pulses_per_train"]


@pulse.kernel
def _pi_step(q):
    amplitude(q, PI_AMP, TRAIN, PI_AMP)


@pulse.kernel
def _adaptive_t1(q):
    tracked_t1(q, PI_AMP, 3.0, 450_000.0 * CLOCK, 0.11, 0.14, 1_600_000)


_calibration_chain = P["calibration_chain"]


@pulse.kernel
def _two_chains(q, p, t1_guess_cycles: float):
    _calibration_chain(q, t1_guess_cycles)
    _calibration_chain(p, t1_guess_cycles)


def _device(target):
    dev = pulse.device(target)
    return dev, dev.qudits[0]


@pytest.mark.slow
def test_the_pi_train_corrects_a_drive_amplitude_error():
    target = transmon_17q()
    dev, q = _device(target)
    lowered = pulse.compile(_pi_step, device=dev, control_system=P["control_system"]).lower(q)
    plant = P["chain_plant"](target, amplitude_scale=1.02)
    result = pulse.simulate(
        lowered, plant=plant, shots=2, seed=3, branches="sampled", readout="model"
    )
    for branch in result.branches:
        corrected, relative_sigma = branch.calls[-1].answer
        # 21 pulses amplify a 2% error to 0.84 rad; 50 shots per point.
        assert corrected == pytest.approx(PI_AMP / 1.02, rel=0.01)
        assert relative_sigma < 0.01


@pytest.mark.slow
def test_adaptive_t1_tracks_the_papers_device_on_the_controller():
    base = transmon_17q()
    qubit = base.qubits[0]
    device_qubit = qubit.replace(
        frequency_hz=4.13e9,
        anharmonicity_hz=-210e6,
        t1_us=160.0,
        t2_star_us=100.0,
        thermal_population=0.0,
        readout=qubit.readout.replace(error_1to0=0.11, error_0to1=0.14),
    )
    target = dataclasses.replace(base, qubits={**base.qubits, 0: device_qubit})
    dev, q = _device(target)
    lowered = pulse.compile(_adaptive_t1, device=dev, control_system=P["control_system"]).lower(q)
    shots = 32
    result = pulse.simulate(lowered, shots=shots, seed=4, branches="sampled", readout="model")
    estimates = []
    for branch in result.branches:
        assert len(branch.answers("adaptive_t1")) == P["tracker_shots"]
        compiled = {c.compiled_name for c in branch.calls if c.function == "adaptive_t1"}
        assert compiled == {"adaptive_t1__1"}
        shape, rate, _ = branch.calls[-1].answer
        estimates += [rate / shape / CLOCK / 1e3] * round(branch.probability * shots)
    # The paper: 159 us, 68% interval [131, 202] us; the benchmark's 400
    # seeds give a median near 151 us at a true 160 us.
    assert 125 < np.median(estimates) < 190


def test_the_binary_search_narrows_on_the_controller():
    target = transmon_17q()
    dev, q = _device(target)
    lowered = pulse.compile(
        P["binary_search"], device=dev, control_system=P["control_system"]
    ).lower(q, 15)
    for seed, offset in enumerate((-200e3, 120e3)):
        result = pulse.simulate(
            lowered,
            plant=P["search_plant"](target, offset),
            shots=1,
            seed=seed,
            branches="sampled",
            readout="model",
        )
        mean, sigma, _, _ = result.branches[0].calls[-1].answer
        assert 330e3 < sigma < 400e3
        assert abs(mean - offset) < 4 * sigma


def _chain():
    target = transmon_17q()
    dev, q = _device(target)
    return (
        pulse.compile(P["calibration_chain"], device=dev, control_system=P["control_system"]),
        dev,
        q,
    )


def test_the_chain_fits_the_declared_controller():
    compiled, _, q = _chain()
    lowered = compiled.lower(q, 20_000.0 * CLOCK)
    report = controller.resources(lowered)
    assert report.violations == ()
    assert all(row.on_controller and row.cost.runs_on_processor for row in report.functions)
    by_name = {row.compiled_name: row for row in report.functions}
    assert by_name["adaptive_t1__1"].calls_per_shot == P["tracker_shots"]
    assert by_name["adaptive_t1__1"].function == "adaptive_t1"
    assert report.calls_per_shot == 7 + P["tracker_shots"]
    assert all(row.compute_ns < row.max_latency_ns for row in report.functions)


def test_a_slower_controller_rejects_the_chain_at_compile_time():
    _, dev, q = _chain()
    processor = P["processor"]
    slow = dataclasses.replace(processor, cycles_per_op=processor.cycles_per_op.replace(exp=400))
    system = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[slow])], links=P["control_system"].links
    )
    slow = pulse.compile(P["calibration_chain"], device=dev, control_system=system)
    with pytest.raises(controller.ControllerLimitError, match="adaptive_t1__1 computes for about"):
        slow.lower(q, 20_000.0 * CLOCK)


def test_two_qubits_calibrate_side_by_side():
    chain, dev, _ = _chain()
    q, p = dev.qudits[:2]
    one = chain.lower(q, 20_000.0 * CLOCK)
    two = pulse.compile(_two_chains, device=dev, control_system=P["control_system"]).lower(
        q, p, 20_000.0 * CLOCK
    )
    # Each chain waits only on its own answers, so two take about as long as
    # one: only their requests on the shared link go out one after the other.
    alone, side_by_side = (max(e.end for e in low.schedule) for low in (one, two))
    assert alone <= side_by_side < alone * 1.01
    assert controller.resources(two).violations == ()
