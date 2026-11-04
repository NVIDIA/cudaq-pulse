# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU simulations running at once each evaluate their own drive envelopes.

Each run owns its waveform callbacks, so runs in flight together neither see
each other's envelopes nor share a callback capacity: together these runs need
more envelope callbacks than any one run may hold.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import MicrowaveDrive, Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import on

pytestmark = pytest.mark.gpu

_RUNS = 8
_PULSES = 160


def _target():
    return TargetDescription(
        name="concurrent-runs",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )


def _pulse_train(amplitude):
    @pulse.kernel
    def train(q0):
        d0 = get_drive_line(q0)
        for i in range(_PULSES):
            drive(d0, square(4, amplitude + 1e-6 * (i % 3)))

    return train


def _lowered(index):
    return on(_target(), _pulse_train(0.001 * (index + 1)), Q[0])


def test_concurrent_gpu_runs_match_their_sequential_results():
    lowered = [_lowered(i) for i in range(_RUNS)]
    expected = [pulse.simulate(k).density_matrix for k in lowered]
    with ThreadPoolExecutor(max_workers=_RUNS) as pool:
        got = list(pool.map(lambda k: pulse.simulate(k, backend="gpu").density_matrix, lowered))
    for index, (cpu, gpu) in enumerate(zip(expected, got, strict=True)):
        np.testing.assert_allclose(gpu, cpu, atol=1e-6, err_msg=f"run {index}")
    populations = [rho[1, 1].real for rho in got]
    assert populations == sorted(populations)
    assert not math.isclose(populations[0], populations[-1])
