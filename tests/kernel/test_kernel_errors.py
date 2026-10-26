# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Mistakes in a kernel are CompilationErrors that say what to write."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError

from tests._handles import Q
from tests._ir import python_ir

# ── Operands are checked when the op is emitted ──────────────────────


def test_swapped_line_and_tone_from_the_friction_log():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(t, gaussian(40, 0.1, 10.0), d)

    message = (
        r"drive\(\): argument 1 must be a drive line, got a tone; "
        r"the line comes first and the tone last: drive\(line, waveform, tone\)"
    )
    with pytest.raises(CompilationError, match=message):
        python_ir(k, Q[0])
    with pytest.raises(CompilationError, match=message):
        pulse.compile(k).lower(Q[0])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("drive(d, 0.5, t)", r"drive\(\): argument 2 must be a waveform, got float 0.5; pass a"),
        ("drive(r, square(8, 0.1), t)", r"argument 1 must be a drive line, got a readout line"),
        ("readout(d, square(8, 0.1), t)", r"argument 1 must be a readout line, got a drive line"),
        ("wait(t, 8)", r"wait\(\): argument 1 must be a drive line or readout line, got a tone"),
        ("shift_phase(d, 0.1)", r"shift_phase\(\): argument 1 must be a tone, got a drive line"),
        ("get_drive_line(3)", r"argument 1 must be a qudit, got int 3; pass one of the kernel"),
        ("wf_add(square(8, 0.1), 0.2)", r"wf_add\(\): argument 2 must be a waveform, got float"),
    ],
)
def test_wrong_operands_name_the_argument(body, message):
    namespace: dict = {}
    exec(
        f"def k(q):\n    d, t = get_drive_line(q)\n    r, rt = get_readout_line(q)\n    {body}\n",
        namespace,
    )
    kernel = pulse.kernel(namespace["k"])
    with pytest.raises(CompilationError, match=message):
        pulse.compile(kernel).lower(Q[0])


def test_correct_operands_still_compile():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        wait(d, 8)
        wait(r, 8)
        drive(d, wf_scale(square(8, 0.1), 0.5), t)
        readout(r, square(8, 0.1), rt)

    pulse.compile(k).lower(Q[0])


# ── sync() takes lines, or lists and tuples of them ──────────────────


def test_sync_flattens_lists_and_tuples_from_the_friction_log():
    @pulse.kernel
    def k(q0, q1, q2):
        d0, _ = get_drive_line(q0)
        d1, _ = get_drive_line(q1)
        d2, t2 = get_drive_line(q2)
        sync([d0, d1])
        sync((d0, d1), d2)
        lines = [d0, [d1, d2]]
        sync(lines)
        drive(d2, square(8, 0.1), t2)

    syncs = [op for op in python_ir(k, Q[0], Q[1], Q[2]).ops if op.kind == "pulse.sync"]
    assert [len(op.operands) for op in syncs] == [2, 3, 3]
    pulse.compile(k).lower(Q[0], Q[1], Q[2])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("sync(d0, t0)", r"sync\(\): argument 2 must be a line, got a tone"),
        ("sync([d0, 3])", r"sync\(\): argument 1\[1\] must be a line, got int 3"),
        ("sync(get_drive_line(q0), d1)", r"argument 1\[1\] must be a line, got a tone; pass the"),
        ("sync(d0)", r"sync\(\) needs at least 2 lines, got 1"),
        ("sync([])", r"sync\(\) needs at least 2 lines, got 0"),
        ("sync([d0], d0)", r"sync\(\): a line is passed more than once"),
    ],
)
def test_sync_rejects_anything_but_lines(body, message):
    namespace: dict = {}
    exec(
        "def k(q0, q1):\n"
        "    d0, t0 = get_drive_line(q0)\n"
        "    d1, t1 = get_drive_line(q1)\n"
        f"    {body}\n",
        namespace,
    )
    with pytest.raises(CompilationError, match=message):
        python_ir(pulse.kernel(namespace["k"]), Q[0], Q[1])
