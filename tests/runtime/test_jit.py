# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.targets.base import Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import evolve, on


@pulse.kernel
def _jit_test_kernel(q0):
    d0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf)


@pytest.mark.gpu
def test_jit_executes_target_aware_program():
    """The direct JIT path executes scheduled, target-aware pulse MLIR."""
    ir = (_jit_test_kernel, Q[0])
    target = TargetDescription(
        name="jit-test",
        qubits={
            0: Qubit(
                index=0, frequency_hz=5.0e9, anharmonicity_hz=-200.0e6, t1_us=0.0, t2_star_us=0.0
            )
        },
    )
    result = evolve(on(target, *ir), t_start=0.0, t_end=20.0, step_count=200)
    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-8)


def test_jit_import():
    """JIT module should be importable."""
    from cudaq_pulse.runtime import _jit as jit

    assert hasattr(jit, "JITCompiler")


def _qubit_target(t1_us=50.0, t2_us=30.0):
    return TargetDescription(
        name="test",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=t2_us,
            )
        },
    )


def test_simulate_annotates_the_module_with_the_target():
    lowered = on(_qubit_target(), _jit_test_kernel, Q[0])
    pulse.simulate(lowered, dt_ns=0.25)
    mlir = lowered.mlir()
    assert "pulse.t1_times" in mlir
    assert 'qop.integrator = "rk4"' in mlir


@pytest.mark.parametrize("integrator", ["rk1", "rk2", "rk4", "magnus", "crank_nicolson"])
def test_simulate_emits_the_selected_integrator(integrator):
    """Every supported integrator name lowers to its dialect attribute."""
    lowered = on(_qubit_target(), _jit_test_kernel, Q[0])
    pulse.simulate(lowered, dt_ns=0.25, integrator=integrator)
    assert f'qop.integrator = "{integrator}"' in lowered.mlir()


def test_simulate_rejects_an_unknown_integrator():
    lowered = on(_qubit_target(), _jit_test_kernel, Q[0])
    with pytest.raises(ValueError, match="Unknown integrator"):
        pulse.simulate(lowered, integrator="magnus_cf4")


def test_simulate_lowers_multilevel_transmons():
    from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline

    lowered = on(_qubit_target(0.0, 0.0), _jit_test_kernel, Q[0])
    result = pulse.simulate(lowered, dt_ns=0.25, levels=3)
    assert result.density_matrix.shape == (3, 3)
    mlir = lowered.mlir()
    assert "pulse.levels = array<i64: 3>" in mlir
    qop = MLIRPipeline().run_pulse_to_qop(mlir)
    assert "qop.mode_extents = array<i64: 3>" in qop and "tensor<3x3x2xf64>" in qop
