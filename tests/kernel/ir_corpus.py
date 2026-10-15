# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Kernel corpus for the cross-version IR golden test.

Each case traces a kernel and renders the result as canonical text that
contains no object ids or paths, so the same source must produce byte-for-byte
the same dump on every supported interpreter. The kernels deliberately use
the constructs whose bytecode differs between CPython versions: loops,
short-circuit conditions at the end of loop bodies, conditional expressions,
chained comparisons, keyword calls, closures and helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter, PythonIRBuilder
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests.workloads import (
    w1_bell,
    w2_cnot_cr,
    w3_qaoa4,
    w4_syndrome,
    w5_dd_cpmg8,
    w6_vqe_hea,
)

# ── Canonical dumps ──────────────────────────────────────────────────


def _canon(v: Any) -> str:
    if isinstance(v, (float, complex)):
        return repr(v)
    if isinstance(v, np.ndarray):
        return "array" + _canon(v.tolist())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_canon(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_canon(v[k])}" for k in sorted(v)) + "}"
    return repr(v)


def dump_python(ir: PythonIRBuilder) -> list[str]:
    lines = [f"func @{ir.name}"]
    for op in ir.ops:
        res = ", ".join(f"%{r.vid}:{r.vtype}" for r in op.results)
        opr = ", ".join(f"%{o.vid}" for o in op.operands)
        lines.append(f"{res} = {op.kind}({opr}) {_canon(dict(op.attrs))}".lstrip())
    return lines


def dump_packed(b: PackedIRBuilder) -> list[str]:
    return [
        f"packed n_qubits={b.n_qubits} op_count={b.op_count}",
        "words " + " ".join(str(w) for w in b.get_buffer().tolist()),
        "freq " + _canon(b.get_freq_array().tolist()),
        "waveforms " + _canon(b._wf_attrs),
    ]


# ── Corpus kernels ───────────────────────────────────────────────────


def dd_loop(n_iter: int) -> Callable[..., Any]:
    @pulse.kernel
    def dd(q0):
        d0, t0 = get_drive_line(q0)
        x = square(40, 0.047 + 0j)
        for _ in range(n_iter):
            drive(d0, x, t0)

    return dd


@pulse.kernel
def cpmg_with_waits(q0):
    d0, t0 = get_drive_line(q0)
    y = drag(40, 0.2, 10.0, 0.5)
    wait(d0, 50)
    for _ in range(6):
        wait(d0, 100)
        drive(d0, y, t0)
        wait(d0, 100)


def surface_syndrome(d: int) -> Callable[..., Any]:
    """Rotated-lattice syndrome round: one ancilla per plaquette."""
    n_data = d * d
    n_anc = (d - 1) * (d - 1) + 2 * (d - 1)
    n_total = n_data + n_anc

    @pulse.kernel
    def round_(qubits):
        lines = []
        tones = []
        for i in range(n_total):
            line, tone = get_drive_line(qubits[i])
            lines.append(line)
            tones.append(tone)

        h = drag(40, 0.25, 10.0, 0.5)
        cr_p = gaussian(160, 0.12, 40.0)
        cr_m = gaussian(160, -0.12, 40.0)
        for a in range(n_anc):
            anc = n_data + a
            row = a // (d - 1) if a < (d - 1) * (d - 1) else -1
            col = a % (d - 1) if a < (d - 1) * (d - 1) else a - (d - 1) * (d - 1)
            drive(lines[anc], h, tones[anc])
            for dr in range(2):
                for dc in range(2):
                    r = row + dr
                    c = col + dc
                    if 0 <= r and r < d and 0 <= c and c < d:
                        q = r * d + c
                        sync(lines[anc], lines[q])
                        drive(lines[anc], cr_p, tones[q])
                        drive(lines[anc], cr_m, tones[q])

    return round_, n_total


def bicycle_syndrome(l: int, m: int) -> Callable[..., Any]:  # noqa: E741
    """Bivariate-bicycle style check: weight-6 stabilizers on a torus."""
    n_data = 2 * l * m
    n_checks = l * m
    n_total = n_data + 2 * n_checks

    @pulse.kernel
    def checks(qubits):
        lines = []
        tones = []
        for i in range(n_total):
            line, tone = get_drive_line(qubits[i])
            lines.append(line)
            tones.append(tone)

        sx = drag(40, 0.025, 10.0, 0.5)
        cr = gaussian(200, 0.1, 50.0)
        shifts_r = [0, 0, 3, 0, 1, 2]
        shifts_c = [0, 1, 0, 2, 0, 2]
        for kind in range(2):
            for ci in range(n_checks):
                anc = n_data + kind * n_checks + ci
                r0, c0 = ci // m, ci % m
                for s in range(6):
                    q = ((r0 + shifts_r[s]) % l) * m + (c0 + shifts_c[s]) % m
                    q = q + l * m if kind == 1 and q + l * m < n_data else q
                    sync(lines[anc], lines[q])
                    drive(lines[q], sx, tones[q])
                    drive(lines[anc], cr, tones[q])

    return checks, n_total


_SCALE = 0.5


def _pi_pulse(line, tone, amplitude=0.3, sigma=10.0):
    drive(line, gaussian(int(4 * sigma), amplitude * _SCALE, sigma), tone)


def closures_and_helpers(gain: float) -> Callable[..., Any]:
    offsets = (0.0, math.pi / 2)

    @pulse.kernel
    def k(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        for i in range(2):
            t0 = shift_phase(t0, offsets[i] * gain)
            _pi_pulse(d0, t0)
        _pi_pulse(d1, t1, sigma=8.0)
        _pi_pulse(line=d1, tone=t1, amplitude=gain)
        sync(d0, d1)

    return k


@pulse.kernel
def python_expressions(q0):
    d0, t0 = get_drive_line(q0)
    amps = [0.1, 0.2, 0.3, 0.4]
    total = 0
    for i in range(len(amps)):
        if i == 1:
            continue
        if 0 < i < 3 or i == len(amps) - 1:
            total += i
        amp = amps[i] if i % 2 == 0 else -amps[i]
        drive(d0, square(20 + 4 * i, amp), t0)
        if total > 4:
            break
    flag = not (total >= 4 and len(amps) == 4)
    wait(d0, 10 if flag else total * 10)
    w = custom(16, f"shape_{total}_{amps[-1]:.2f}")
    drive(d0, w, t0)
    shift_frequency(t0, frequency=1e6)
    if total:
        return
    drive(d0, w, t0)


@pulse.kernel
def readout_and_algebra():
    q = pulse.qudit_ref()
    d0, t0 = get_drive_line(q)
    r0, rt = get_readout_line(q)
    a = cosine(16, 0.2)
    b = wf_scale(wf_add(a, custom_samples([0.0, 0.1, 0.2, 0.1] * 4)), 1.5)
    drive(d0, wf_neg(b), t0)
    sync(d0, r0)
    readout(r0, square(200, 0.05), rt)


@pulse.kernel
def parametric(q0, amp: float, n: int):
    d0, t0 = get_drive_line(q0)
    for _ in range(3):
        drive(d0, gaussian(40, amp * 0.5, 10.0), t0)
        wait(d0, int(n) * 2)
    shift_phase(t0, -amp)


@pulse.kernel
def measurement_branch(q0):
    r0, t0 = get_readout_line(q0)
    bit = readout(r0, square(10, 0.2), t0)
    if bit:
        wait(r0, 10)


@pulse.kernel
def bad_attribute(q0):
    d0, t0 = get_drive_line(q0)
    drive(d0, math.not_a_function(3), t0)


@pulse.kernel
def while_loop(q0):
    d0, t0 = get_drive_line(q0)
    n = 3
    while n:
        n -= 1


# ── Cases ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    name: str
    kernel: Callable[..., Any]
    args: Callable[[], list[Any]]
    path: str  # "python" (kernel call) or "packed" (compile() trace)


def _qubits(n: int) -> Callable[[], list[Any]]:
    return lambda: [pulse.qudit_ref() for _ in range(n)]


def _qvec(n: int) -> Callable[[], list[Any]]:
    return lambda: [pulse.qvec_ref(n)]


def _cases() -> list[Case]:
    cases = []
    for n in (8, 32, 128, 512):
        cases.append(Case(f"dd_loop_n{n}", dd_loop(n), _qubits(1), "python"))
    cases.append(Case("dd_loop_n8_packed", dd_loop(8), _qubits(1), "packed"))
    cases.append(Case("cpmg_with_waits", cpmg_with_waits, _qubits(1), "python"))
    for mod in (w1_bell, w2_cnot_cr, w3_qaoa4, w4_syndrome, w5_dd_cpmg8, w6_vqe_hea):
        n = mod.NUM_QUBITS
        cases.append(Case(f"workload_{mod.NAME}", mod.build, _qubits(n), "python"))
    for d in (3, 5):
        kernel, n = surface_syndrome(d)
        cases.append(Case(f"surface_d{d}", kernel, _qvec(n), "packed"))
    kernel, n = surface_syndrome(3)
    cases.append(Case("surface_d3_python", kernel, _qvec(n), "python"))
    kernel, n = bicycle_syndrome(3, 3)
    cases.append(Case("bicycle_3x3", kernel, _qvec(n), "packed"))
    cases.append(Case("closures_and_helpers", closures_and_helpers(0.8), _qubits(2), "python"))
    cases.append(Case("python_expressions", python_expressions, _qubits(1), "python"))
    cases.append(Case("python_expressions_packed", python_expressions, _qubits(1), "packed"))
    cases.append(Case("readout_and_algebra", readout_and_algebra, lambda: [], "python"))
    cases.append(Case("readout_and_algebra_packed", readout_and_algebra, lambda: [], "packed"))
    cases.append(
        Case(
            "parametric",
            parametric,
            lambda: [pulse.qudit_ref(), Parameter("amp", 0, "f64"), Parameter("n", 1, "i64")],
            "packed",
        )
    )
    cases.append(Case("measurement_branch", measurement_branch, _qubits(1), "packed"))
    cases.append(Case("bad_attribute", bad_attribute, _qubits(1), "python"))
    cases.append(Case("while_loop", while_loop, _qubits(1), "python"))
    return cases


CASES = _cases()


def trace(case: Case) -> dict[str, Any]:
    """Trace *case*; return its golden record."""
    fn = case.kernel.__wrapped__
    args = case.args()
    try:
        if case.path == "python":
            ir = case.kernel(*args)
            lines = dump_python(ir)
            n_ops = len(ir.ops)
        else:
            builder = PackedIRBuilder(
                name=fn.__name__, clock_ghz=2.0, qubit_freq_hz={i: 5e9 + 1e6 * i for i in range(64)}
            )
            _trace_kernel_with_builder(fn, builder, args)
            lines = dump_packed(builder)
            n_ops = builder.op_count
    except Exception as e:
        record: dict[str, Any] = {"error": f"{type(e).__name__}: {e}"}
        source = getattr(e, "pulse_source", None)
        if source is not None:
            decoded, index = source
            _, line, end_line, col, end_col = decoded.location(index)
            first = decoded.code.co_firstlineno
            record["code"] = decoded.code.co_name
            record["span"] = [line - first, None if end_line is None else end_line - first]
            record["cols"] = [col, end_col]
        return record
    return {"ops": n_ops, "dump": lines}
