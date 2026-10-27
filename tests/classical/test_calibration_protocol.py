# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.targets import transmon_krinner_17q

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "21_calibration_protocol.py"
P = runpy.run_path(str(_EXAMPLE), run_name="calibration_protocol")
PI_AMP, CLOCK = P["PI_AMP"], P["CLOCK_GHZ"]
amplitude, tracked_t1, TRAIN = P["amplitude"], P["tracked_t1"], P["TRAIN"]


@pulse.kernel
def _pi_step(q):
    amplitude(q, PI_AMP, TRAIN, PI_AMP)


@pulse.kernel
def _adaptive_t1(q):
    tracked_t1(q, PI_AMP, 3.0, 450_000.0 * CLOCK, 0.11, 0.14, 1_600_000)


def _device(target):
    dev = pulse.device({"target": target})
    return dev, dev.qudit_handles[0]


def test_the_pi_train_corrects_a_drive_amplitude_error():
    target = transmon_krinner_17q()
    dev, q = _device(target)
    lowered = pulse.compile(_pi_step, device=dev, machine=P["MACHINE"]).lower(q)
    plant = P["chain_plant"](target, amplitude_scale=1.02)
    result = pulse.simulate(lowered, plant=plant, shots=2, seed=3, paths="sampled", readout="model")
    for calls in result.calls:
        corrected, relative_sigma = calls[-1].answer
        # 21 pulses amplify a 2% error to 0.84 rad; 50 shots per point.
        assert corrected == pytest.approx(PI_AMP / 1.02, rel=0.01)
        assert relative_sigma < 0.01


def test_adaptive_t1_tracks_the_papers_device_on_the_controller():
    base = transmon_krinner_17q()
    qubit = base.qubits[0]
    device_qubit = dataclasses.replace(
        qubit,
        frequency_hz=4.13e9,
        anharmonicity_hz=-210e6,
        t1_us=160.0,
        t2_star_us=100.0,
        thermal_population=0.0,
        readout_params={**qubit.readout_params, "error_1to0": 0.11, "error_0to1": 0.14},
    )
    target = dataclasses.replace(base, qubits={**base.qubits, 0: device_qubit})
    dev, q = _device(target)
    lowered = pulse.compile(_adaptive_t1, device=dev, machine=P["MACHINE"]).lower(q)
    shots = 32
    result = pulse.simulate(lowered, shots=shots, seed=4, paths="sampled", readout="model")
    estimates = []
    for calls, (_, share) in zip(result.calls, result.branches):
        assert sum(c.function == "adaptive_t1" for c in calls) == P["ATT_SHOTS"]
        shape, rate, _ = calls[-1].answer
        estimates += [rate / shape / CLOCK / 1e3] * round(share * shots)
    # The paper: 159 us, 68% interval [131, 202] us; the benchmark's 400
    # seeds give a median near 151 us at a true 160 us.
    assert 125 < np.median(estimates) < 190


def test_the_binary_search_narrows_on_the_controller():
    target = transmon_krinner_17q()
    dev, q = _device(target)
    lowered = pulse.compile(P["binary_search"], device=dev, machine=P["MACHINE"]).lower(q, 15)
    for seed, offset in enumerate((-200e3, 120e3)):
        result = pulse.simulate(
            lowered,
            plant=P["search_plant"](target, offset),
            shots=1,
            seed=seed,
            paths="sampled",
            readout="model",
        )
        mean, sigma, _, _ = result.calls[0][-1].answer
        assert 330e3 < sigma < 400e3
        assert abs(mean - offset) < 4 * sigma


def _chain():
    target = transmon_krinner_17q()
    dev, q = _device(target)
    return pulse.compile(P["calibration_chain"], device=dev, machine=P["MACHINE"]), dev, q


def test_the_chain_fits_the_declared_controller():
    compiled, _, q = _chain()
    lowered = compiled.lower(q, 20_000.0 * CLOCK)
    report = controller.resources(lowered)
    assert report.violations == ()
    assert all(row.on_controller and row.cost.in_controller_form for row in report.functions)
    by_name = {row.function: row for row in report.functions}
    assert by_name["adaptive_t1"].calls_per_shot == P["ATT_SHOTS"]
    assert report.calls_per_shot == 7 + P["ATT_SHOTS"]
    assert all(row.compute_ns < row.max_latency_ns for row in report.functions)


def test_a_slower_controller_rejects_the_chain_at_compile_time():
    _, dev, q = _chain()
    limits = dataclasses.replace(P["LIMITS"], op_cycles={**P["LIMITS"].op_cycles, "exp": 400})
    machine = MachineModel(links={"controller": P["MACHINE"].link("controller")}, controller=limits)
    slow = pulse.compile(P["calibration_chain"], device=dev, machine=machine)
    with pytest.raises(controller.ControllerLimitError, match="adaptive_t1 computes for about"):
        slow.lower(q, 20_000.0 * CLOCK)
