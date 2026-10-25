# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration test: full pipeline from @pulse.kernel to GPU state."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests._handles import Q
from tests.runtime._lowering import pulse_mlir

gpu = pytest.mark.gpu


def test_full_pipeline_mlir():
    """Verify the full pipeline produces valid MLIR from @pulse.kernel."""

    @pulse.kernel
    def my_kernel(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        sync(d0, d1)
        drive(d1, wf, t1)

    ir = (my_kernel, Q[0], Q[1])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9, 1: 5.1e9})

    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "return" in mlir


@gpu
def test_full_pipeline_gpu():
    """Public compile -> MLIR lowering -> JIT -> GPU path."""

    @pulse.kernel
    def my_kernel(q0):
        d0, t0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf, t0)

    compiled = my_kernel.lower(Q[0])
    results = compiled.run()
    state = results[0].to_numpy()
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-6)
    assert abs(state[1]) ** 2 > 0.999
