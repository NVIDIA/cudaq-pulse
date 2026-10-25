# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""PackedIRBuilder -- fused tracer-to-packed-buffer emitter.

Writes directly into a flat int64 array during kernel tracing,
producing a buffer that ``PulseModuleBuilder.build_from_packed()``
consumes via zero-copy FFI.
"""

from __future__ import annotations

import numbers
import struct
from array import array
from typing import Any

from .ir_builder import (
    UNDEFINED_VID,
    CompilationError,
    IRValue,
    Parameter,
    ParameterExpression,
    is_symbolic,
    parameter_names,
)

_pack_d = struct.Struct("=d")
_unpack_q = struct.Struct("=q")

# OpCodes -- must match the bindings.cpp kOp* constants
_ALLOC_DRIVE = 0
_ALLOC_READOUT = 1
_ALLOC_TONE = 2
_WF_GAUSSIAN = 3
_WF_SQUARE = 4
_WF_DRAG = 5
_WF_COSINE = 6
_WF_TANH_RAMP = 7
_WF_GAUSS_SQUARE = 8
_WF_CUSTOM = 9
_DRIVE = 10
_READOUT = 11
_SYNC = 12
_WAIT = 13
_SHIFT_PHASE = 14
_SET_PHASE = 15
_SHIFT_FREQ = 16
_SET_FREQ = 17
_PARAM = 18
_NUM_CONST = 19
_NUM_BINARY = 20
_NUM_NEG = 21
_NUM_CAST = 22
_WF_CUSTOM_SAMPLES = 23
_WF_ADD = 24
_WF_SUB = 25
_WF_MUL = 26
_WF_SCALE = 27
_WF_NEG = 28
_FOR_BEGIN = 29
_FOR_END = 30
_IF_BEGIN = 31
_ELSE = 32
_IF_END = 33
_BLOCK_BEGIN = 34
_BLOCK_END = 35
_BLOCK_POLICY = {"asap": 0, "alap": 1}

_NUMERIC_TYPE = {"i64": 0, "f64": 1}
_BINARY_OP = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "div": 3,
    "floordiv": 4,
    "mod": 5,
}

_WF_MAP = {
    "gaussian": _WF_GAUSSIAN,
    "square": _WF_SQUARE,
    "drag": _WF_DRAG,
    "cosine": _WF_COSINE,
    "tanh_ramp": _WF_TANH_RAMP,
    "gaussian_square": _WF_GAUSS_SQUARE,
}

_UNSCHEDULED = -1


def _f2i(x: float) -> int:
    return int(_unpack_q.unpack(_pack_d.pack(float(x)))[0])


# The record header stores the payload length in 8 bits. Region records,
# whose payload grows with the values they carry, write 0xFF there and the
# length in the next word.
_MAX_PAYLOAD = 0xFF


def _header(opcode: int, payload_len: int, param_mask: int = 0) -> int:
    return opcode | (payload_len << 8) | (param_mask << 16)


def _complex_parts(val: Any) -> tuple[float, float]:
    """``(re, im)`` of an amplitude: a number, or ``[re]`` / ``[re, im]``."""
    if val.__class__ is float or val.__class__ is int:
        return (float(val), 0.0)
    if isinstance(val, numbers.Complex) and not isinstance(val, numbers.Real):
        return (float(val.real), float(val.imag))
    if isinstance(val, (list, tuple)) or hasattr(val, "__array__"):
        # Array-likes (NumPy included) without importing NumPy.
        items = val.tolist() if hasattr(val, "tolist") else list(val)
        parts = list(_flatten(items))
        if len(parts) not in (1, 2):
            raise CompilationError(
                f"an amplitude sequence must be [re] or [re, im], got {len(parts)} values"
            )
        return (float(parts[0]), float(parts[1]) if len(parts) == 2 else 0.0)
    return (float(val), 0.0)


def _flatten(items: Any) -> Any:
    for item in items if isinstance(items, (list, tuple)) else [items]:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _as_i64(value: Any, description: str) -> int:
    """Convert an integer-valued Python number without silent truncation."""
    if value.__class__ is int:
        return value
    if isinstance(value, bool):
        raise CompilationError(f"{description} must be an integer, got bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return int(float(value))
    raise CompilationError(
        f"{description} must be an integer number of virtual time units, got {value!r}"
    )


class PackedIRBuilder:
    """Fused trace-to-buffer builder.

    Implements the same ``emit()`` interface as ``PythonIRBuilder`` so it
    can be dropped into the ``PulseIREmitter`` unchanged.  Instead of
    accumulating Python objects, it writes packed int64 records into a
    pre-allocated ``array('q')`` buffer, so tracing does not import NumPy.
    """

    # Loops and measurement-conditioned branches become region records.
    supports_regions = True

    def __init__(
        self,
        name: str = "main",
        *,
        clock_ghz: float = 2.0,
        qubit_freq_hz: dict[int, float] | None = None,
        flatten_schedules: bool = False,
    ):
        self.name = name
        # Called kernels with their own policy become schedule blocks.
        self.schedule_blocks = not flatten_schedules
        self.schedule_block_count = 0
        self.clock_ghz = clock_ghz
        self._freq_hz = qubit_freq_hz or {}
        self._buf = array("q", bytes(8 * 8192))
        self._cur = 0
        self._next_id = 0
        self._next_qubit = 0
        self._op_count = 0
        self._qref_to_qubit: dict[int, int] = {}
        self._param_vids: dict[int, int] = {}  # Parameter.index → vid
        self._param_names: dict[int, str] = {}  # Parameter.index → name
        # Key by the expression object itself, not id(expression): temporary
        # expressions may be collected and Python may reuse their integer IDs.
        self._numeric_vids: dict[tuple[ParameterExpression, str], int] = {}
        # Open regions, innermost last: the yields seen so far in each.
        self._regions: list[dict[str, tuple[int, ...]]] = []
        # Measurement vid -> readout ordinal, for readouts outside any region.
        self.readouts: dict[int, int] = {}
        self._n_readouts = 0
        # Set by the emitter when the trace evaluated an impure host call.
        self.impure = False
        # The compilation's loop-unroll threshold; None for the default.
        self.unroll_threshold: int | None = None
        # What the traced kernel returned.
        self.returned: Any = None
        # Float arguments that reached an integer (a duration, a cast to
        # int): the schedule needs their values, so they are specialized on.
        self.structural: set[str] = set()

    def _mk(self, vtype: str, name: str = "") -> IRValue:
        v = IRValue(self._next_id, vtype, name)
        self._next_id += 1
        return v

    def _ensure(self, n: int) -> None:
        while self._cur + n >= len(self._buf):
            self._buf.frombytes(bytes(8 * len(self._buf)))

    def _w(self, val: int) -> None:
        self._buf[self._cur] = val
        self._cur += 1

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        a = attrs or {}
        results = tuple(self._mk(rt) for rt in result_types)

        # --- Qubit alloc (not encoded, just tracked) ---
        if kind in ("pulse.qudit_arg", "pulse.qudit_alloc"):
            idx = a.get("index", self._next_qubit)
            for r in results:
                self._qref_to_qubit[r.vid] = idx
            self._next_qubit = max(self._next_qubit, idx + 1)
            return results

        # Like PythonIRBuilder.op_count(), count pulse-program ops but not qudit binding.
        self._op_count += 1

        # --- get_drive_line → ALLOC_DRIVE ---
        if kind == "pulse.get_drive_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid, 0) if qref_vid is not None else 0
            self._ensure(4)
            self._w(_header(_ALLOC_DRIVE, 3))
            self._w(qubit)
            self._w(results[0].vid)  # line_vid
            self._w(results[1].vid)  # tone_vid
            return results

        # --- get_readout_line → ALLOC_READOUT ---
        if kind == "pulse.get_readout_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid, 0) if qref_vid is not None else 0
            self._ensure(4)
            self._w(_header(_ALLOC_READOUT, 3))
            self._w(qubit)
            self._w(results[0].vid)
            self._w(results[1].vid)
            return results

        # --- Waveform constructors ---
        wf_name = kind.removeprefix("pulse.")
        wf_code = _WF_MAP.get(wf_name)
        if wf_code is not None or wf_name in ("custom", "custom_samples"):
            rv = results[0].vid
            dur_raw = a.get("duration", 0)
            dur = dur_raw if is_symbolic(dur_raw) else _as_i64(dur_raw, f"{wf_name} duration")

            if wf_code == _WF_GAUSSIAN:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals: list[Any] = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_GAUSSIAN, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                if is_symbolic(amp_raw):
                    vals = [dur, amp_raw, 0.0]
                else:
                    re, im = _complex_parts(amp_raw)
                    vals = [dur, re, im]
                self._emit_wf(_WF_SQUARE, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_DRAG:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                beta_raw = a.get("beta", 0.0)
                vals = [dur, amp_raw, sig_raw, beta_raw]
                self._emit_wf(_WF_DRAG, rv, vals, ["i64", "f64", "f64", "f64"])
            elif wf_code == _WF_COSINE:
                amp_raw = a.get("amplitude", 0.0)
                vals = [dur, amp_raw]
                self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64"])
            elif wf_code == _WF_TANH_RAMP:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_TANH_RAMP, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_GAUSS_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                width_raw = a.get("width", 0)
                # The Python API specifies flat-top width while the dialect
                # stores the duration of each rise/fall edge.
                if not is_symbolic(width_raw):
                    width_raw = _as_i64(width_raw, "gaussian_square flat-top width")
                if not is_symbolic(dur) and not is_symbolic(width_raw):
                    if width_raw < 0 or width_raw >= dur:
                        raise CompilationError(
                            "gaussian_square width must satisfy "
                            f"0 <= width < duration, got {width_raw} and {dur}"
                        )
                    if (dur - width_raw) % 2:
                        raise CompilationError(
                            "gaussian_square requires duration - width to be "
                            "even so its two edges have equal integer length"
                        )
                rf_raw = (dur - width_raw) // 2
                vals = [dur, amp_raw, sig_raw, rf_raw]
                self._emit_wf(_WF_GAUSS_SQUARE, rv, vals, ["i64", "f64", "f64", "i64"])
            elif wf_name == "custom_samples":
                self._emit_custom_samples(rv, a.get("samples", ()))
            else:
                self._emit_custom(rv, dur, a.get("name", "custom"))
            return results

        # --- drive ---
        if kind == "pulse.drive":
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            self._ensure(8)
            self._w(_header(_DRIVE, 7))
            self._w(lv)
            self._w(wv)
            self._w(tv)
            self._w(rlv)
            self._w(rtv)
            self._w(_UNSCHEDULED)  # start_vtu (set by C++ scheduler)
            self._w(_UNSCHEDULED)  # duration_vtu (set by C++ scheduler)
            return results

        # --- readout ---
        if kind == "pulse.readout":
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            mv = results[2].vid
            self._ensure(7)
            self._w(_header(_READOUT, 6))
            self._w(lv)
            self._w(wv)
            self._w(tv)
            self._w(rlv)
            self._w(rtv)
            self._w(mv)
            if not self._regions:
                self.readouts[mv] = self._n_readouts
            self._n_readouts += 1
            return results

        # --- sync ---
        if kind == "pulse.sync":
            n = len(operands)
            payload_len = 1 + 3 * n
            if payload_len > _MAX_PAYLOAD:
                raise CompilationError(
                    f"sync() over {n} lines exceeds the packed limit of "
                    f"{(_MAX_PAYLOAD - 1) // 3} lines per sync"
                )
            self._ensure(1 + payload_len)
            self._w(_header(_SYNC, payload_len))
            self._w(n)
            line_types = {"drive_line": 0, "readout_line": 1}
            for j in range(n):
                in_vid = operands[j].vid
                out_vid = results[j].vid if j < len(results) else in_vid
                vtype = line_types.get(results[j].vtype if j < len(results) else "drive_line", 0)
                self._w(in_vid)
                self._w(out_vid)
                self._w(vtype)
            return results

        # --- wait ---
        if kind == "pulse.wait":
            lv = operands[0].vid
            rlv = results[0].vid
            dur_raw = a.get("duration", 0)
            if is_symbolic(dur_raw):
                pvid = self._get_numeric_vid(dur_raw, "i64")
                self._ensure(4)
                self._w(_header(_WAIT, 3, 1 << 2))
                self._w(lv)
                self._w(rlv)
                self._w(pvid)
            else:
                dur = _as_i64(dur_raw, "wait duration")
                self._ensure(4)
                self._w(_header(_WAIT, 3))
                self._w(lv)
                self._w(rlv)
                self._w(dur)
            return results

        # --- shift_phase ---
        if kind == "pulse.shift_phase":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("phase_rad", 0.0)
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
                self._ensure(4)
                self._w(_header(_SHIFT_PHASE, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                delta = float(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_PHASE, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(delta))
            return results

        # --- set_phase ---
        if kind == "pulse.set_phase":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("phase_rad", 0.0)
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
                self._ensure(4)
                self._w(_header(_SET_PHASE, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                phase = float(val_raw)
                self._ensure(4)
                self._w(_header(_SET_PHASE, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(phase))
            return results

        # --- shift_frequency ---
        if kind == "pulse.shift_frequency":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("freq_hz", 0.0)
            if isinstance(val_raw, IRValue):
                raise CompilationError(
                    "a frequency must be a number; time-varying frequencies "
                    "(a waveform as a frequency) are reserved and not supported yet"
                )
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
                self._ensure(4)
                self._w(_header(_SHIFT_FREQ, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                freq = float(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_FREQ, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(freq))
            return results

        # --- set_frequency ---
        if kind == "pulse.set_frequency":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("freq_hz", 0.0)
            if isinstance(val_raw, IRValue):
                raise CompilationError(
                    "a frequency must be a number; time-varying frequencies "
                    "(a waveform as a frequency) are reserved and not supported yet"
                )
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
                self._ensure(4)
                self._w(_header(_SET_FREQ, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                freq = float(val_raw)
                self._ensure(4)
                self._w(_header(_SET_FREQ, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(freq))
            return results

        if kind.startswith("scf."):
            return self._region(kind, operands, results, a)

        algebra_opcodes = {
            "pulse.wf_add": _WF_ADD,
            "pulse.wf_sub": _WF_SUB,
            "pulse.wf_mul": _WF_MUL,
            "pulse.wf_neg": _WF_NEG,
        }
        if kind in algebra_opcodes:
            rv = results[0].vid
            self._ensure(2 + len(operands))
            self._w(_header(algebra_opcodes[kind], 1 + len(operands)))
            self._w(rv)
            for operand in operands:
                self._w(operand.vid)
            return results

        if kind == "pulse.wf_scale":
            rv = results[0].vid
            scale_vid = self._get_numeric_vid(a["scale"], "f64")
            self._ensure(4)
            self._w(_header(_WF_SCALE, 3))
            self._w(rv)
            self._w(operands[0].vid)
            self._w(scale_vid)
            return results

        raise CompilationError(f"unsupported packed pulse op: {kind}")

    def _record(self, opcode: int, payload: list[int]) -> None:
        extended = len(payload) >= _MAX_PAYLOAD
        self._ensure(len(payload) + 2)
        self._w(_header(opcode, _MAX_PAYLOAD if extended else len(payload)))
        if extended:
            self._w(len(payload))
        for word in payload:
            self._w(word)

    def _region(
        self,
        kind: str,
        operands: tuple[IRValue, ...],
        results: tuple[IRValue, ...],
        attrs: dict[str, Any],
    ) -> tuple[IRValue, ...]:
        if kind == "scf.yield":
            # Yields are part of the record that closes their region.
            self._op_count -= 1
            region = self._regions[-1]
            region["else" if "else" in region else "yield"] = tuple(v.vid for v in operands)
            return results
        if kind == "scf.for":
            lb = _as_i64(attrs.get("lb", 0), "loop lower bound")
            ub = _as_i64(attrs.get("ub", 0), "loop upper bound")
            step = _as_i64(attrs.get("step", 1), "loop step")
            self._record(_FOR_BEGIN, [lb, ub, step])
            self._regions.append({})
        elif kind == "scf.for_end":
            region = self._regions.pop()
            inits = [v.vid if v.vid != UNDEFINED_VID else -1 for v in operands]
            yields = list(region.get("yield", ()))
            self._record(_FOR_END, [len(results), *inits, *yields, *(r.vid for r in results)])
        elif kind == "scf.if":
            self._record(_IF_BEGIN, [operands[0].vid])
            self._regions.append({})
        elif kind == "scf.else":
            self._record(_ELSE, [])
            self._regions[-1]["else"] = ()
        elif kind == "scf.if_end":
            region = self._regions.pop()
            self._record(
                _IF_END,
                [
                    len(results),
                    *region.get("yield", ()),
                    *region.get("else", ()),
                    *(r.vid for r in results),
                ],
            )
        else:
            raise CompilationError(f"unsupported structured control flow op {kind!r}")
        return results

    def begin_schedule_block(self, policy: str, callee: str) -> None:
        """Open a block whose ops are scheduled on their own with *policy*.

        The extension makes the lines, tones and measurements the block uses
        from outside its operands, and the ones it leaves its results.
        """
        encoded = callee.encode("utf-8")
        padded = encoded + b"\0" * ((8 - len(encoded) % 8) % 8)
        words = [_unpack_q.unpack(padded[i : i + 8])[0] for i in range(0, len(padded), 8)]
        self._record(_BLOCK_BEGIN, [_BLOCK_POLICY[policy], len(encoded), *words])
        self.schedule_block_count += 1

    def end_schedule_block(self) -> None:
        self._record(_BLOCK_END, [])

    def mark(self) -> int:
        return self._cur

    def insert(self, index: int, kind: str, operands: tuple[IRValue, ...]) -> None:
        """The emitter inserts only the then-arm yield, once the arm is traced."""
        if kind != "scf.yield" or not self._regions:
            raise CompilationError(f"cannot insert {kind!r} into a packed program")
        self._regions[-1]["yield"] = tuple(v.vid for v in operands)

    def checkpoint(self) -> tuple[Any, ...]:
        return (
            self._cur,
            self._next_id,
            self._next_qubit,
            self._op_count,
            [dict(region) for region in self._regions],
            self._n_readouts,
        )

    def rollback(self, checkpoint: tuple[Any, ...]) -> None:
        (
            self._cur,
            self._next_id,
            self._next_qubit,
            self._op_count,
            regions,
            self._n_readouts,
        ) = checkpoint
        self._regions = regions
        live = self._next_id
        for vid in [vid for vid in self.readouts if vid >= live]:
            del self.readouts[vid]
        for vid in [vid for vid in self._qref_to_qubit if vid >= live]:
            del self._qref_to_qubit[vid]
        for index in [i for i, vid in self._param_vids.items() if vid >= live]:
            del self._param_vids[index]
            self._param_names.pop(index, None)
        for key in [k for k, vid in self._numeric_vids.items() if vid >= live]:
            del self._numeric_vids[key]

    def _get_parameter_base_vid(self, param: Parameter, expected_dtype: str) -> int:
        """Get a block-argument reference and infer its storage type."""
        if param.dtype == "unknown":
            param.dtype = expected_dtype
        if param.index in self._param_vids:
            return self._param_vids[param.index]
        vid = self._next_id
        self._next_id += 1
        self._param_vids[param.index] = vid
        self._param_names[param.index] = param.name
        self._ensure(3)
        self._w(_header(_PARAM, 2))
        self._w(vid)
        self._w(param.index)
        return vid

    def _get_numeric_vid(self, value: Any, expected_dtype: str) -> int:
        """Materialize a symbolic or literal number as an SSA value record."""
        if expected_dtype not in _NUMERIC_TYPE:
            raise CompilationError(f"unsupported numeric type {expected_dtype}")
        if is_symbolic(value) and (
            expected_dtype == "i64"
            or (
                isinstance(value, ParameterExpression)
                and value.op == "cast"
                and value.dtype == "i64"
            )
        ):
            self.structural.update(parameter_names(value))

        if isinstance(value, Parameter):
            base = self._get_parameter_base_vid(value, expected_dtype)
            if value.dtype == expected_dtype:
                return base
            return self._emit_numeric_cast(base, value.dtype, expected_dtype)

        if isinstance(value, ParameterExpression):
            cache_key = (value, expected_dtype)
            if cache_key in self._numeric_vids:
                return self._numeric_vids[cache_key]

            if value.op == "cast":
                target_dtype = value.dtype
                source_dtype = getattr(value.lhs, "dtype", "unknown")
                if source_dtype == "unknown":
                    source_dtype = "f64" if target_dtype == "i64" else "i64"
                source = self._get_numeric_vid(value.lhs, source_dtype)
                result = self._emit_numeric_cast(source, source_dtype, target_dtype)
                if target_dtype != expected_dtype:
                    result = self._emit_numeric_cast(result, target_dtype, expected_dtype)
                self._numeric_vids[cache_key] = result
                return result

            dtype = "f64" if value.op == "div" else expected_dtype
            if value.op in ("floordiv", "mod"):
                dtype = "i64"
            if value.op == "pow":
                raise CompilationError(
                    "symbolic exponentiation is not supported; specialize "
                    "that value before compilation"
                )
            if value.op == "neg":
                operand = self._get_numeric_vid(value.lhs, dtype)
                result = self._next_id
                self._next_id += 1
                self._ensure(4)
                self._w(_header(_NUM_NEG, 3))
                self._w(result)
                self._w(_NUMERIC_TYPE[dtype])
                self._w(operand)
            else:
                opcode = _BINARY_OP.get(value.op)
                if opcode is None:
                    raise CompilationError(f"unsupported symbolic operation {value.op!r}")
                lhs = self._get_numeric_vid(value.lhs, dtype)
                rhs = self._get_numeric_vid(value.rhs, dtype)
                result = self._next_id
                self._next_id += 1
                self._ensure(6)
                self._w(_header(_NUM_BINARY, 5))
                self._w(result)
                self._w(_NUMERIC_TYPE[dtype])
                self._w(opcode)
                self._w(lhs)
                self._w(rhs)
            if dtype != expected_dtype:
                result = self._emit_numeric_cast(result, dtype, expected_dtype)
            self._numeric_vids[cache_key] = result
            return result

        result = self._next_id
        self._next_id += 1
        self._ensure(4)
        self._w(_header(_NUM_CONST, 3))
        self._w(result)
        self._w(_NUMERIC_TYPE[expected_dtype])
        self._w(
            _as_i64(value, "integer expression") if expected_dtype == "i64" else _f2i(float(value))
        )
        return result

    def _emit_numeric_cast(self, operand: int, source_dtype: str, target_dtype: str) -> int:
        if source_dtype == target_dtype:
            return operand
        result = self._next_id
        self._next_id += 1
        self._ensure(5)
        self._w(_header(_NUM_CAST, 4))
        self._w(result)
        self._w(_NUMERIC_TYPE[source_dtype])
        self._w(_NUMERIC_TYPE[target_dtype])
        self._w(operand)
        return result

    def _emit_wf(self, wf_code: int, rv: int, vals: list[Any], dtypes: list[str]) -> None:
        """Emit a waveform record, handling mixed Parameter/concrete values.

        For each value slot, if the value is a Parameter, emit a PARAM record
        first and encode the param vid. A param_mask bit flags parametric slots
        so the C++ decoder can distinguish vids from literals.
        """
        encoded: list[int] = []
        param_mask = 0
        for i, (v, dtype) in enumerate(zip(vals, dtypes)):
            if is_symbolic(v):
                pvid = self._get_numeric_vid(v, dtype)
                encoded.append(pvid)
                param_mask |= 1 << (i + 1)  # +1 because slot 0 is the rv
            elif dtype == "i64":
                encoded.append(_as_i64(v, "integer operand"))
            else:
                encoded.append(_f2i(float(v)))
        payload_len = 1 + len(encoded)  # rv + values
        self._ensure(1 + payload_len)
        self._w(_header(wf_code, payload_len, param_mask))
        self._w(rv)
        for e in encoded:
            self._w(e)

    def _emit_custom(self, rv: int, duration: Any, callback: Any) -> None:
        name = callback if isinstance(callback, str) else getattr(callback, "__name__", "custom")
        encoded = name.encode("utf-8")
        if not encoded:
            raise CompilationError("custom waveform callback name is empty")
        padded = encoded + b"\0" * ((8 - len(encoded) % 8) % 8)
        words = [struct.unpack("=q", padded[i : i + 8])[0] for i in range(0, len(padded), 8)]
        if 3 + len(words) > 255:
            raise CompilationError("custom waveform callback name is too long")
        param_mask = 0
        if is_symbolic(duration):
            duration_word = self._get_numeric_vid(duration, "i64")
            param_mask |= 1 << 1
        else:
            duration_word = _as_i64(duration, "custom waveform duration")
        self._ensure(4 + len(words))
        self._w(_header(_WF_CUSTOM, 3 + len(words), param_mask))
        self._w(rv)
        self._w(duration_word)
        self._w(len(encoded))
        for word in words:
            self._w(word)

    def _emit_custom_samples(self, rv: int, samples: Any) -> None:
        try:
            sample_values = list(samples)
        except TypeError as exc:
            raise CompilationError("custom_samples() requires a finite sequence") from exc
        if len(sample_values) > _MAX_PAYLOAD - 2:
            raise CompilationError(
                f"custom_samples() supports at most {_MAX_PAYLOAD - 2} samples per waveform"
            )
        if not sample_values:
            raise CompilationError("custom_samples() requires at least one sample")
        encoded: list[int] = []
        for sample in sample_values:
            value = complex(sample)
            if value.imag != 0.0:
                raise CompilationError(
                    "complex custom samples are not supported yet; provide "
                    "real-valued envelope samples"
                )
            encoded.append(_f2i(value.real))
        self._ensure(3 + len(encoded))
        self._w(_header(_WF_CUSTOM_SAMPLES, 2 + len(encoded)))
        self._w(rv)
        self._w(len(encoded))
        for word in encoded:
            self._w(word)

    @property
    def param_names(self) -> list[str]:
        """Names of the parameters referenced so far, in index order."""
        return [self._param_names[i] for i in sorted(self._param_names)]

    def get_buffer(self) -> array[int]:
        """Return the trimmed packed buffer (int64 words)."""
        return self._buf[: self._cur]

    def get_freq_array(self) -> array[float]:
        """Return qubit frequencies as float64 values indexed by qubit."""
        arr = array("d", bytes(8 * self._next_qubit))
        for q, f in self._freq_hz.items():
            if q < self._next_qubit:
                arr[q] = f
        return arr

    @property
    def n_qubits(self) -> int:
        return self._next_qubit

    @property
    def qubits(self) -> tuple[int, ...]:
        """Indices of the allocated qudits."""
        return tuple(sorted(set(self._qref_to_qubit.values())))

    @property
    def op_count(self) -> int:
        return self._op_count
