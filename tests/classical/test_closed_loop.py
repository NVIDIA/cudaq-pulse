# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed loops between shots: no recompiling, a parameter store, an
estimate log, and tracking a drifting quantum system against its truth."""

from __future__ import annotations

import math
import time

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
from cudaq_pulse.quantum_systems import PhysicalQudit
from cudaq_pulse.quantum_systems.drift import ParameterDrift, TelegraphT1

PI_AMP = math.pi / 20


def _target(t1_us=20.0):
    qudit = PhysicalQudit(
        index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=t1_us, t2_star_us=0.0
    )
    return pulse.quantum_system(name="sim", qudits={0: qudit})


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
    dev = _target()
    q = dev.qudits[0]
    compiled = pulse.compile(t1_shot, quantum_system=dev)
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
        assert {b.levels: b.probability for b in result.branches}[(1,)] == pytest.approx(
            math.exp(-elapsed_ns / 20e3), abs=5e-3
        )
    assert compiled.cache_info().compiled == 1


def test_tracking_a_switching_t1_against_its_truth():
    # The between-shot loop of adaptive T1 tracking on the surrogate: one
    # shot per iteration, the wait from the posterior, the posterior reset
    # to a weak prior every 200 shots so it follows a switching T1.
    # T1 switches between 30 us and 5 us, staying about 2 s in each.
    t1 = TelegraphT1(low_fraction=1 / 6, dwell_time=2 * pulse.s)
    qs = _target(t1_us=30.0).replace(
        parameter_drift=ParameterDrift(frequency=None, t1=t1, amplitude=None),
        parameter_drift_seed=7,
    )
    shots = np.random.default_rng(7)
    store = ParameterStore(alpha=1.0, beta=10_000.0)
    log = EstimateLog()
    for i in range(10_000):
        alpha, beta = store.values("alpha", "beta")
        wait_ns = cal.t1_wait(alpha, beta)
        survived = shots.random() < cal.survival_probability(wait_ns, qs.true_qudits[0].t1_us)
        alpha, beta = cal.t1_update(alpha, beta, bool(survived), wait_ns)
        if (i + 1) % 200 == 0:
            log.record(i, "t1_us", beta / alpha / 1e3, truth=qs.true_qudits[0].t1_us)
            alpha, beta = 2.0, 2.0 * beta / alpha
        store.update({"alpha": alpha, "beta": beta}, iteration=i)
        qs.advance(1 * pulse.ms)
    estimates = log.series("t1_us")
    assert len(estimates) == 50
    # Classify each block as the short or the long T1; mostly right, except
    # in blocks where T1 switched.
    right = sum((e.value > 12.0) == (e.truth > 12.0) for e in estimates)
    assert right >= 40
    assert log.tracking_error("t1_us") < 10.0
