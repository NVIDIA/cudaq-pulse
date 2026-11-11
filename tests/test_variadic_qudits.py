# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels with a ``*qudits`` parameter: a register of the call's qudits."""

from __future__ import annotations

import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import InvalidTypeError, NoQuantumSystemError, Qudit, Register

from tests._handles import LADDER, Q


def _mlir(kernel, *args, **kwargs):
    """The kernel's module for a call, with the kernel's name taken out."""
    return kernel.lower(*args, **kwargs).mlir().replace(f"@{kernel.__name__}", "@kernel")


@pulse.kernel
def spread(tau: float, *qudits: Qudit, amp=0.2):
    for q in qudits:
        d = get_drive_line(q)
        drive(d, square(16, amp))
        wait(d, tau)


@pulse.kernel
def spread_register(tau: float, qudits: Register, amp=0.2):
    for q in qudits:
        d = get_drive_line(q)
        drive(d, square(16, amp))
        wait(d, tau)


@pulse.kernel
def fixed(tau: float, *qudits, amp=0.2):
    for q in qudits:
        d = get_drive_line(q)
        drive(d, square(16, amp))
        wait(d, tau)


@pulse.kernel
def fixed_register(tau: float, qudits, amp=0.2):
    for q in qudits:
        d = get_drive_line(q)
        drive(d, square(16, amp))
        wait(d, tau)


def test_annotated_qudits_lower_as_a_runtime_register():
    assert spread.runtime_parameters == ("tau", "qudits")
    assert spread.static_parameters == ("amp",)
    for qudits in (Q[0:1], Q[0:3], Q[[4, 2]]):
        assert _mlir(spread, 8.0, *qudits) == _mlir(spread_register, 8.0, qudits)
        assert spread.lower(8.0, *qudits).schedule == spread_register.lower(8.0, qudits).schedule


def test_unannotated_qudits_lower_as_a_static_register():
    assert fixed.runtime_parameters == ("tau",)
    assert fixed.static_parameters == ("qudits", "amp")
    for qudits in (Q[0:1], Q[0:3], Q[[4, 2]]):
        assert _mlir(fixed, 8.0, *qudits) == _mlir(fixed_register, 8.0, qudits)


def test_parameters_are_listed_in_signature_order():
    @pulse.kernel
    def ramsey(tau: float, *qudits: Qudit, amp: float, steps: int):
        for q in qudits:
            d = get_drive_line(q)
            for _ in range(steps):
                drive(d, square(16, amp))
                wait(d, tau)

    assert ramsey.runtime_parameters == ("tau", "qudits", "amp")
    assert ramsey.static_parameters == ("steps",)


def test_arity_is_part_of_the_specialization():
    @pulse.kernel
    def k(*qudits: Qudit):
        for q in qudits:
            drive(get_drive_line(q), square(16, 0.1))

    k.lower(Q[0], Q[1])
    k.lower(Q[5], Q[3])
    assert k.cache_info().compiled == 1
    k.lower(Q[0], Q[1], Q[2])
    assert k.cache_info().compiled == 2
    k.lower(Q[7], Q[8], Q[9])
    assert k.cache_info().compiled == 2

    @pulse.kernel
    def s(*qudits):
        for q in qudits:
            drive(get_drive_line(q), square(16, 0.1))

    s.lower(Q[0], Q[1])
    s.lower(Q[5], Q[3])
    assert s.cache_info().compiled == 2


def test_qudits_index_slice_iterate_and_measure_like_a_register():
    @pulse.kernel
    def k(*qudits: Qudit):
        first, *rest = qudits
        drive(get_drive_line(qudits[-1]), square(len(qudits) * 8, 0.1))
        for q in qudits[1::2]:
            drive(get_drive_line(q), square(16, 0.1))
        drive(get_drive_line(first), square(8 * len(rest), 0.1))

    @pulse.kernel
    def r(qudits: Register):
        first, *rest = qudits
        drive(get_drive_line(qudits[-1]), square(len(qudits) * 8, 0.1))
        for q in qudits[1::2]:
            drive(get_drive_line(q), square(16, 0.1))
        drive(get_drive_line(first), square(8 * len(rest), 0.1))

    assert _mlir(k, *Q[0:4]) == _mlir(r, Q[0:4])


def test_keyword_only_and_positional_only_parameters_bind():
    @pulse.kernel
    def k(q0: Qudit, /, *qudits: Qudit, amp: float = 0.1, width=16):
        for q in (q0, *qudits):
            drive(get_drive_line(q), square(width, amp))

    @pulse.kernel
    def r(q0: Qudit, qudits: Register, amp: float = 0.1, width=16):
        for q in (q0, *qudits):
            drive(get_drive_line(q), square(width, amp))

    assert _mlir(k, Q[0], Q[1], Q[2], width=32) == _mlir(r, Q[0], Q[1:3], width=32)
    assert _mlir(k, Q[0], amp=0.2) == _mlir(r, Q[0], Q[1:1], amp=0.2)
    with pytest.raises(TypeError, match="positional-only"):
        k.lower(q0=Q[0])


def test_qudits_pass_on_to_other_kernels():
    @pulse.kernel
    def outer(tau: float, *qudits: Qudit):
        spread(tau, *qudits, amp=0.3)
        fixed(tau, *qudits[1:])
        spread_register(tau, qudits)
        spread(tau, qudits[0], qudits[0])

    @pulse.kernel
    def unrolled(tau: float, a: Qudit, b: Qudit, c: Qudit):
        spread_register(tau, [a, b, c], 0.3)
        spread_register(tau, [b, c])
        spread_register(tau, [a, b, c])
        spread_register(tau, [a, a])

    assert outer.lower(4.0, *Q[0:3]).schedule == unrolled.lower(4.0, *Q[0:3]).schedule


def test_empty_qudits_are_a_register_of_none():
    @pulse.kernel
    def k(q0: Qudit, *rest: Qudit):
        drive(get_drive_line(q0), square(16, 0.1))
        for q in rest:
            drive(get_drive_line(q), square(16, 0.1))
        return len(rest)

    assert k(Q[0]) == 0
    assert _mlir(k, Q[0]) == _mlir(k, Q[0], *Q[1:1])
    with pytest.raises(NoQuantumSystemError, match="without qudit handles"):
        spread.lower(4.0)
    assert pulse.compile(spread, quantum_system=LADDER).lower(4.0).metrics.op_count == 0


def test_qudits_must_be_qudit_handles():
    with pytest.raises(InvalidTypeError, match=r"\*qudits takes qudit handles, got Register"):
        spread.lower(4.0, Q[0:2])
    with pytest.raises(InvalidTypeError, match=r"got float at position 1; pass other values"):
        fixed.lower(4.0, Q[0], 0.5)
    with pytest.raises(InvalidTypeError, match="got qudit 1 twice in 'qudits'"):
        spread.lower(4.0, Q[1], Q[2], Q[1])


def test_only_qudits_may_be_variadic():
    def options(q, **opts):
        pass

    def amplitudes(q, *amps: float):
        pass

    def registers(*regs: Register):
        pass

    def lists(*qudits: list[Qudit]):
        pass

    with pytest.raises(TypeError, match=r"cannot take \*\*opts"):
        pulse.kernel(options)
    for fn, name in ((amplitudes, "amps"), (registers, "regs"), (lists, "qudits")):
        with pytest.raises(TypeError, match=rf"\*{name} takes qudits: annotate it Qudit"):
            pulse.kernel(fn)


def test_signature_and_pickling_keep_the_star():
    assert str(spread.__signature__) == "(tau: 'float', *qudits: 'Qudit', amp=0.2)"
    again = pickle.loads(pickle.dumps(spread))
    assert _mlir(again, 4.0, *Q[0:2]) == _mlir(spread, 4.0, *Q[0:2])
