# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Closed loops between shots: no recompiling, a parameter store, an
estimate log, and tracking a drifting plant against its truth."""

from __future__ import annotations

import math
import time

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
from cudaq_pulse.runtime.drift import DriftingPlant, TelegraphT1
from cudaq_pulse.targets import Qubit, TargetDescription

PI_AMP = math.pi / 20


def _target():
    qubit = Qubit(index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=20.0, t2_star_us=0.0)
    return TargetDescription(name="sim", qubits={0: qubit})


def test_the_store_versions_every_change_and_the_log_round_trips():
    store = ParameterStore({"wait_ns": 1000.0}, amplitude=0.1)
    assert store.values("amplitude", "wait_ns") == (0.1, 1000.0)
    assert store.version == 2
    store.set("amplitude", 0.12, iteration=7)
    assert store["amplitude"] == 0.12 and store.version == 3
    assert store.history[-1].iteration == 7
    with pytest.raises(KeyError, match="no parameter 'phase'"):
        store["phase"]
    with pytest.raises(ValueError, match="finite"):
        store.set("amplitude", math.nan)
    log = EstimateLog()
    log.record(0, "t1_us", 18.0, uncertainty=2.0, truth=20.0)
    log.record(1, "t1_us", 21.0, truth=20.0)
    log.record(1, "amplitude", 0.12)
    assert log.tracking_error("t1_us") == pytest.approx(math.sqrt((4 + 1) / 2))
    assert EstimateLog.from_json(log.to_json()).entries == log.entries
    with pytest.raises(ValueError, match="no estimates"):
        log.tracking_error("amplitude")


STORE = ParameterStore(wait_cycles=4000.0, amplitude=PI_AMP)
READ_WAIT = STORE.reader("wait_cycles", bounds=(0.0, 80000.0))


@pulse.kernel
def t1_shot(q, amplitude: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(40, amplitude))
    wait(d, READ_WAIT.submit().result(), bounds=(0, 80000))
    sync(d, r)
    readout(r, square(400, 0.05))


def test_ten_thousand_iterations_run_without_compiling_again():
    dev = pulse.device({"target": _target()})
    q = dev.qudit_handles[0]
    compiled = pulse.compile(t1_shot, device=dev)
    start = time.perf_counter()
    for i in range(10_000):
        compiled.lower(q, STORE["amplitude"])
        STORE.set("amplitude", PI_AMP * (1 + 0.001 * (i % 7)), iteration=i)
    per_iteration_us = (time.perf_counter() - start) / 10_000 * 1e6
    assert compiled.cache_info().compiled == 1
    assert per_iteration_us < 200  # a cache hit and a bind, no compiling
    # The wait is read when the program runs: simulating after changing the
    # store gives the new wait's decay, from the same compiled kernel.
    STORE.set("amplitude", PI_AMP)
    for wait_cycles in (4000.0, 20000.0):
        STORE.set("wait_cycles", wait_cycles)
        result = pulse.simulate(compiled.lower(q, STORE["amplitude"]), dt_ns=1.0, levels=2)
        elapsed_ns = wait_cycles / dev.clock_ghz + 20 + 200
        assert dict(result.branches)[(1,)] == pytest.approx(math.exp(-elapsed_ns / 20e3), abs=5e-3)
    assert compiled.cache_info().compiled == 1


def test_tracking_a_switching_t1_against_its_truth():
    # The between-shot loop of adaptive T1 tracking on the surrogate: one
    # shot per iteration, the wait from the posterior, the posterior reset
    # to a weak prior every 200 shots so it follows a switching T1.
    plant = DriftingPlant(_target(), t1=TelegraphT1(5.0, 30.0, rate_hz=0.5), seed=7)
    store = ParameterStore(alpha=1.0, beta=10_000.0)
    log = EstimateLog()
    for i in range(10_000):
        alpha, beta = store.values("alpha", "beta")
        wait_ns = cal.t1_wait(alpha, beta)
        survived = plant.sample(plant.p_survive(wait_ns))
        alpha, beta = cal.t1_update(alpha, beta, bool(survived), wait_ns)
        if (i + 1) % 200 == 0:
            log.record(i, "t1_us", beta / alpha / 1e3, truth=plant.state.t1_us)
            alpha, beta = 2.0, 2.0 * beta / alpha
        store.update({"alpha": alpha, "beta": beta}, iteration=i)
        plant.advance(1e-3)
    estimates = log.series("t1_us")
    assert len(estimates) == 50
    # Classify each block as the short or the long T1; mostly right, except
    # in blocks where the plant switched.
    right = sum((e.value > 12.0) == (e.truth > 12.0) for e in estimates)
    assert right >= 40
    assert log.tracking_error("t1_us") < 10.0
