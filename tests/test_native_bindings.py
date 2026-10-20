# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Every native binding returns its result to Python."""

import cudaq_pulse as pulse
from cudaq_pulse._native import _cudaq_pulse_native as native


@pulse.kernel
def _x(q, amplitude: float):
    d, t = get_drive_line(q)
    drive(d, square(40, amplitude), t)


def _public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def test_module_builder_exposes_only_the_packed_decoder():
    assert _public(native.PulseModuleBuilder) == {"build_from_packed"}


def test_pulse_module_methods_return_results():
    compiled = pulse.compile(_x, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9})
    module = compiled.module
    assert module.is_parametric()
    assert module.param_names() == ["amplitude"]
    assert "pulse.drive" in module.run_passes(["pulse-verify"])
    concrete = module.specialize([0.1], [], "alap")
    assert not concrete.is_parametric()
    assert "start_vtu = 0" in concrete.schedule("asap")
    assert "llvm.func" in concrete.run_full_lowering()


def test_pipeline_methods_return_results():
    text = pulse.compile(_x, [pulse.qudit_ref(), 0.1], qubit_freq_hz={0: 5e9}).mlir
    pipeline = native.MLIRPipeline()
    assert "pulse.drive" in pipeline.parse_and_print(text)
    assert "pulse.drive" in pipeline.parse_to_module(text).print()
    qop = pipeline.run_pulse_to_qop(text)
    assert "qop." in qop
    cudm = pipeline.run_qop_to_cudm(qop)
    assert "llvm.func" in pipeline.run_cudm_to_llvm(cudm) or "llvm." in cudm
    assert "llvm.func" in pipeline.run_full_pipeline(text)
    assert "define" in pipeline.run_full_pipeline_to_llvm_ir(text)
