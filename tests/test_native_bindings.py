# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Every native binding returns its result to Python."""

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native

from tests._handles import Q


@pulse.kernel
def _x(q, amplitude: float):
    d, t = get_drive_line(q)
    drive(d, square(40, amplitude), t)


def _public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def test_module_builder_exposes_only_the_packed_decoder_and_the_pickle_loader():
    assert _public(native.PulseModuleBuilder) == {"build_from_packed", "load"}


def test_pulse_module_methods():
    # The cached specialization keeps ``amplitude`` a function argument.
    compiled = _x._dispatch.lookup((Q[0], 0.0)).lowered
    module = compiled._module
    assert module.param_names() == ["amplitude"]
    assert module.run_passes(["pulse-verify"]) is None
    assert "pulse.drive" in module.print()
    concrete = module.specialize([0.1], [], "alap")
    assert "%arg0" not in concrete.print()
    assert concrete.schedule("asap") is None
    assert "start_cycles = 0" in concrete.print()
    assert "llvm.func" in concrete.run_full_lowering()


def test_pipeline_methods_return_results():
    text = _x.lower(Q[0], 0.1).mlir()
    pipeline = native.MLIRPipeline()
    assert "pulse.drive" in pipeline.parse_to_module(text).print()
    qop = pipeline.run_pulse_to_qop(text)
    assert "qop." in qop
    assert "cudm." in pipeline.run_qop_to_cudm(qop)


def test_pipeline_exposes_only_the_live_entry_points():
    assert _public(native.MLIRPipeline) == {
        "jit_llvm_dialect",
        "jit_pulse",
        "parse_to_module",
        "run_pulse_to_qop",
        "run_qop_to_cudm",
    }


def test_pass_failures_carry_the_diagnostics():
    text = (
        "func.func @main() {\n"
        "  %q = pulse.qudit_alloc : !pulse.qref\n"
        "  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)\n"
        "  %c = arith.constant 40 : i64\n"
        "  %a = arith.constant 0.1 : f64\n"
        "  %w = pulse.square %c, %a, %a : i64, f64, f64 -> !pulse.waveform\n"
        "  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone"
        " -> !pulse.drive_line, !pulse.tone\n"
        "  %d2, %t2 = pulse.drive %d, %w, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone"
        " -> !pulse.drive_line, !pulse.tone\n"
        "  return\n"
        "}\n"
    )
    module = native.MLIRPipeline().parse_to_module(text)
    with pytest.raises(RuntimeError, match="linear pulse value is consumed more than once"):
        module.run_passes(["pulse-verify"])
    with pytest.raises(RuntimeError, match="expected"):
        native.MLIRPipeline().parse_to_module("func.func @main() {")


def test_a_pass_by_name_takes_its_options():
    from tests.workloads import w4_syndrome

    text = pulse.compile(w4_syndrome.build).lower(*Q[: w4_syndrome.NUM_QUBITS]).mlir()
    pipeline = native.MLIRPipeline()
    scheduled = pipeline.parse_to_module(text)
    scheduled.schedule("alap_rcp", 1, 1, 3, 2)
    by_name = pipeline.parse_to_module(text)
    options = "max-drives=1 max-readouts=1 readout-latency=3 switch-penalty=2 alap=true"
    by_name.run_passes([f"pulse-schedule-rcp{{{options}}}"])
    assert by_name.print() == scheduled.print() != pipeline.parse_to_module(text).print()


def test_compiled_module_is_named_after_the_kernel():
    compiled = _x.lower(Q[0], 0.1)
    assert compiled.mlir().startswith("module @_x")
    assert "pulse.qubit_freq_hz = array<f64: 5.000000e+09>" in compiled.mlir()
