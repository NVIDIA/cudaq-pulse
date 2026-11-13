# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: full pipeline from @pulse.kernel to GPU state."""

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets import DeviceDescription, MicrowaveDrive, PhysicalQudit

from tests._handles import Q
from tests.runtime._lowering import pulse_mlir

gpu = pytest.mark.gpu


def test_full_pipeline_mlir():
    """Verify the full pipeline produces valid MLIR from @pulse.kernel."""

    @pulse.kernel
    def my_kernel(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf)
        sync(d0, d1)
        drive(d1, wf)

    ir = (my_kernel, Q[0], Q[1])
    mlir = pulse_mlir(ir, clock_ghz=2.0, qudit_freq_hz={0: 5.0e9, 1: 5.1e9})

    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "return" in mlir


@gpu
def test_full_pipeline_gpu():
    """Public device -> lower -> simulate on the GPU: MLIR lowering, JIT, cuDensityMat."""
    target = DeviceDescription(
        name="pipeline-1q",
        qudits={
            0: PhysicalQudit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )
    q0 = pulse.device(target).qudits[0]

    @pulse.kernel
    def my_kernel(q0):
        d0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf)

    result = pulse.simulate(my_kernel.lower(q0), levels=2, backend="gpu")
    rho = result.density_matrix
    assert rho.shape == (2, 2)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-6)
    assert rho[1, 1].real > 0.999
