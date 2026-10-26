# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Python syntax inside kernels: what traces, and how the rest is rejected."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError

from tests._handles import Q
from tests._ir import python_ir


def _kinds(kernel, *args) -> list[str]:
    return [op.kind for op in python_ir(kernel, *args).ops]


def _compiles(kernel, *args) -> None:
    """Lowers through the MLIR verifier, which checks linear use of lines."""
    pulse.compile(kernel).lower(*args)


# ── Star-argument and ** calls ───────────────────────────────────────


def test_sync_star_lines_from_the_friction_log():
    @pulse.kernel
    def k(qubits):
        lines = []
        for q in qubits:
            d, t = get_drive_line(q)
            lines.append(d)
        sync(*lines)
        for d in lines:
            wait(d, 8)

    ops = python_ir(k, Q[0:3]).ops
    (sync_op,) = [op for op in ops if op.kind == "pulse.sync"]
    assert len(sync_op.operands) == 3
    # Each wait consumes the synced line, not the one sync consumed.
    waits = [op for op in ops if op.kind == "pulse.wait"]
    assert [w.operands[0] for w in waits] == list(sync_op.results)
    _compiles(k, Q[0:3])


def test_star_arguments_mix_with_positional_and_keyword_arguments():
    def helper(d, t, n, amp=0.1):
        drive(d, square(n, amp), t)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        args = (d, t)
        opts = {"amp": 0.3}
        helper(*args, 16, **opts)
        helper(d, t, *[8], **{"amp": 0.2})
        drive(d, gaussian(*(40, 0.1), **{"sigma": 10.0}), t)

    ops = python_ir(k, Q[0]).ops
    squares = [
        (op.attrs["duration"], op.attrs["amplitude"]) for op in ops if op.kind == "pulse.square"
    ]
    assert squares == [(16, 0.3), (8, 0.2)]
    assert [op.attrs["sigma"] for op in ops if op.kind == "pulse.gaussian"] == [10.0]
    _compiles(k, Q[0])


def test_star_calls_to_python_functions_run_at_trace_time():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        widths = [8, 24, 16]
        drive(d, square(max(*widths), 0.1), t)

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert sq.attrs["duration"] == 24


def test_tuple_displays_with_a_star_and_unary_plus():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        widths = (*[8, 16], 24)
        drive(d, square(widths[-1], +0.25), t)

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert (sq.attrs["duration"], sq.attrs["amplitude"]) == (24, 0.25)


def test_star_unpacking_a_line_is_rejected():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        sync(*d)

    with pytest.raises(CompilationError, match="cannot unpack a pulse drive_line value with"):
        python_ir(k, Q[0])


def test_double_star_needs_a_mapping_and_distinct_keys():
    @pulse.kernel
    def not_a_mapping(q):
        d, t = get_drive_line(q)
        drive(d, square(**[1, 2]), t)

    @pulse.kernel
    def repeated(q):
        d, t = get_drive_line(q)
        drive(d, square(duration=8, **{"duration": 8, "amplitude": 0.1}), t)

    with pytest.raises(CompilationError, match="argument after \\*\\* must be a mapping, not list"):
        python_ir(not_a_mapping, Q[0])
    with pytest.raises(CompilationError, match="multiple values for keyword argument 'duration'"):
        python_ir(repeated, Q[0])
