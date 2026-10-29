# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""compile() builds the same program, regions included, as the kernel IR.

The comparison is on the module the packed program decodes to, before
scheduling, which may peel loop iterations.
"""

from __future__ import annotations

import re

import pytest
from cudaq_pulse._native._cudaq_pulse_native import PulseModuleBuilder
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests._ir import python_ir
from tests.kernel.ir_corpus import CASES, UNROLL_THRESHOLD

_FREQ = {i: 5e9 + 1e6 * i for i in range(64)}
_RENAMED = {
    "pulse.wf_add": "pulse.add",
    "pulse.wf_sub": "pulse.sub",
    "pulse.wf_mul": "pulse.mul",
    "pulse.wf_scale": "pulse.scale",
    "pulse.wf_neg": "pulse.neg",
}
# Ops that only one side spells out: qudit binding, the region ends that
# MLIR leaves implicit, and the conversions MLIR needs.
_IR_ONLY = {"pulse.qudit_arg", "pulse.qudit_alloc", "scf.for_end", "scf.if_end"}
_MLIR_ONLY = {"pulse.qudit_alloc", "pulse.duration_from_int", "pulse.is_set"}
_OP = re.compile(r"(?<![!\w])((?:pulse|scf)\.[a-z_]+)")


def _ir_sequence(ir):
    kinds = [_RENAMED.get(op.kind, op.kind) for op in ir.ops if op.kind not in _IR_ONLY]
    return ["else" if kind == "scf.else" else kind for kind in kinds]


def _mlir_sequence(text):
    sequence = []
    for line in text[text.index("func.func") :].splitlines()[1:]:
        if line.strip() == "} else {":
            sequence.append("else")
            continue
        match = _OP.search(line)
        if match and match.group(1) not in _MLIR_ONLY:
            sequence.append(match.group(1))
    return sequence


def _traces(case):
    try:
        return python_ir(case.kernel, *case.args(), unroll_threshold=UNROLL_THRESHOLD)
    except Exception:  # noqa: BLE001
        return None


_PROGRAMS = [c for c in CASES if _traces(c) is not None]


@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c.name)
def test_compile_builds_the_kernel_ir(case):
    ir = python_ir(case.kernel, *case.args(), unroll_threshold=UNROLL_THRESHOLD)
    builder = PackedIRBuilder(name="k", qubit_freq_hz=_FREQ)
    builder.unroll_threshold = UNROLL_THRESHOLD
    _trace_kernel_with_builder(case.kernel.__wrapped__, builder, case.args())
    params = [p for p in case.args() if isinstance(p, Parameter)]
    module = PulseModuleBuilder().build_from_packed(
        builder.get_buffer(),
        2.0,
        builder.n_qubits,
        builder.get_freq_array(),
        [p.name for p in params],
        [p.dtype for p in params],
    )
    assert builder.op_count == ir.op_count()
    assert _mlir_sequence(module.print()) == _ir_sequence(ir)
