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
import sys
from array import array
from collections.abc import Callable
from typing import Any

from ..classical import _LIST_NAME, _RESERVED_KEYS, _unsized_names
from .ir_builder import (
    MAX_CYCLES,
    UNDEFINED_VID,
    AnswerValue,
    CompilationError,
    IRValue,
    Parameter,
    ParameterExpression,
    Selection,
    arithmetic_text,
    contains_answer,
    is_arithmetic,
    is_symbolic,
    parameter_names,
    without_answers,
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
_CLASSICAL_DECLARE = 36
_CLASSICAL_SUBMIT = 37
_CLASSICAL_WAIT = 38
_CLASSICAL_COMPARE = 39
_VARIABLE_WAIT = 40
_SELECT = 41
_MAX_CYCLES = 42
_LATENCY_MODE = {"bounded": 0, "timeout": 1, "blocking": 2}
_COMPARE = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}

_DURATION_FROM_ANSWER = (
    "a duration cannot depend on the answer of a classical call: the answer "
    "arrives while the program runs, after its timeline is fixed; branch on "
    "the answer to choose between durations instead"
)

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
_WF_NAMES = {code: name for name, code in _WF_MAP.items()}

# The name of each waveform and of its slots, as the kernel writes them.
_WF_FIELDS = {
    _WF_GAUSSIAN: ("gaussian", ("duration", "amplitude", "sigma")),
    _WF_DRAG: ("drag", ("duration", "amplitude", "sigma", "beta")),
    _WF_COSINE: ("cosine", ("duration", "amplitude")),
    _WF_TANH_RAMP: ("tanh_ramp", ("duration", "amplitude", "sigma")),
    _WF_GAUSS_SQUARE: ("gaussian_square", ("duration", "amplitude", "sigma", "width")),
}

_UNSCHEDULED = -1
_ZERO = (0,)


def _real(value: Any, description: str) -> float:
    """*value* as a float, or the error naming *description* and what it got."""
    try:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        raise CompilationError(f"{description} must be a real number, got {value!r}") from None


def _tables(machine: Any) -> dict[int, dict[str, tuple[int, str]]]:
    """Per link id, the keys *machine*'s controllers fix."""
    if machine is None or (machine.controller is None and machine._system is None):
        return {}
    from ..compiler import _dispatch_tables

    return _dispatch_tables(machine)


def _link_without_machine(link: Any) -> int:
    if isinstance(link, str):
        raise CompilationError(
            f"link {link!r} is a name; compile with a MachineModel whose links name it"
        )
    return int(link)


def _fixed_key(function: Any, table: dict[str, tuple[int, str]]) -> tuple[int, str]:
    """The key a controller's firmware gives *function*: a function on its
    links must be listed, and has one layout."""
    name = function.base_name
    found = table.get(name)
    if found is None:
        from .._errors import _closest

        served = sorted(table)
        close = _closest(name, served)
        hint = f" (did you mean {close!r}?)" if close else ""
        raise CompilationError(
            f"{name} runs on {table[served[0]][1]}, whose dispatch_keys have no {name!r}{hint};"
            f" its firmware answers {', '.join(served)}. Add it to the controller's"
            " dispatch_keys, or call it on another link"
        )
    key, label = found
    template = function._template
    if template is not None:
        unsized = _unsized_names(template)[0]
        kind = _LIST_NAME[template._shape.plan[template._shape.names.index(unsized)]]
        raise CompilationError(
            f"{name} has the fixed dispatch key {key} from {label}.dispatch_keys[{name!r}], so"
            f" it has one layout, but its parameter {unsized!r} is {kind} with no length. Give"
            f" it a fixed size: {unsized}: pulse.Length[{kind}, 8]"
        )
    return key, label


def _collision(first: Any, second: Any, key: int) -> str:
    """Two functions of one program whose names hash to one key on one link
    (a controller that fixes keys lists every function on its links)."""
    link = f"{second.link!r}" if isinstance(second.link, str) else str(second.link)
    return (
        f"{first.name} and {second.name} hash to the same dispatch key {key:#x} on link"
        f" {link}; rename one of them"
    )


def _f2i(x: float) -> int:
    """The bits of *x* as a double. Any real number converts as float()
    converts it; a string, which float() would parse, raises struct.error."""
    return int(_unpack_q.unpack(_pack_d.pack(x))[0])


# The record header stores the payload length in 8 bits. Region records,
# whose payload grows with the values they carry, write 0xFF there and the
# length in the next word.
_MAX_PAYLOAD = 0xFF
# One custom_samples record holds its result, its count and the samples, in
# a payload shorter than the 0xFF that marks an extended length.
MAX_CUSTOM_SAMPLES = _MAX_PAYLOAD - 3


def _header(opcode: int, payload_len: int, param_mask: int = 0) -> int:
    return opcode | (payload_len << 8) | (param_mask << 16)


_DRIVE_HEADER = _header(_DRIVE, 7)
_WAIT_HEADER = _header(_WAIT, 3)
_LINE_TYPE_CODES = {"drive_line": 0, "readout_line": 1}
_ALGEBRA_OPCODES = {
    "pulse.wf_add": _WF_ADD,
    "pulse.wf_sub": _WF_SUB,
    "pulse.wf_mul": _WF_MUL,
    "pulse.wf_neg": _WF_NEG,
}


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
        try:
            return (parts[0].__float__(), parts[1].__float__() if len(parts) == 2 else 0.0)
        except AttributeError:
            return (_number(parts[0]), _number(parts[1]) if len(parts) == 2 else 0.0)
    try:
        return (val.__float__(), 0.0)
    except AttributeError:
        return (_number(val), 0.0)


def _number(value: Any) -> float:
    """*value*, a number without ``__float__`` (one with ``__index__``), as a
    float; a TypeError for a string, which float() would parse, as for any
    other value that is not a number. The caller's error says what it got."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError
    return float(value)


def _sample_number(sample: Any) -> complex:
    """*sample*, a ``custom_samples()`` value without ``__float__``, as
    complex() reads it; a string, which complex() would parse, is an error."""
    if not isinstance(sample, (str, bytes, bytearray)):
        try:
            return complex(sample)
        except (TypeError, ValueError):
            pass
    raise CompilationError(f"custom_samples() samples must be real numbers, got {sample!r}")


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
        f"{description} must be an integer number of clock cycles, got {value!r}"
    )


def _wait_bounds(bounds: Any) -> tuple[int, int]:
    """``(lowest, highest)`` of ``wait(..., bounds=)`` in cycles; highest -1
    for none."""
    if bounds is None:
        return 0, -1
    if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
        raise CompilationError(f"bounds= takes (lowest, highest) in clock cycles, got {bounds!r}")
    lo, hi = bounds
    if is_symbolic(lo) or is_symbolic(hi):
        raise CompilationError("the bounds of a wait must be known while the kernel compiles")
    lowest = _as_i64(lo, "the lowest length of a wait")
    highest = -1 if hi is None else _as_i64(hi, "the highest length of a wait")
    if lowest < 0 or (hi is not None and highest < lowest):
        raise CompilationError(f"bounds= must satisfy 0 <= lowest <= highest, got ({lo!r}, {hi!r})")
    return lowest, highest


class PackedIRBuilder:
    """Fused trace-to-buffer builder.

    Implements the same ``emit()`` interface as ``PythonIRBuilder`` so it
    can be dropped into the ``PulseIREmitter`` unchanged.  Instead of
    accumulating Python objects, it writes packed int64 records into a
    pre-allocated ``array('q')`` buffer, so tracing does not import NumPy.
    """

    # Loops and measurement-conditioned branches become region records.
    supports_regions = True
    # The grid of a (role, qudit) line, in program cycles; compile() sets it
    # when a controller runs a coarser clock. None: every line is 1.
    line_grid: Callable[[str, int], int] | None = None
    # The samples every line plays per program cycle, as (n, d): n every d
    # cycles; compile() sets it when that is not one. custom() samples at
    # it, and a custom_samples array must be made for it.
    sample_rate: tuple[int, int] | None = None
    # A carrier's default transition and the tone frequency of each
    # transition its line drives, by (qudit, is_readout); compile() sets it
    # from the target's model. None: line.tones is refused.
    line_tones: (
        Callable[[int, bool], tuple[tuple[int, int], dict[tuple[int, int], float]]] | None
    ) = None
    # The sample rate of a (role, qudit) line, when the lines play different
    # ones (a runtime qudit's line: none is known); compile() sets it for a
    # control system of several controllers. Waveforms are then sampled for
    # the line that plays them.
    sample_rate_of: Callable[[str, int], tuple[int, int] | None] | None = None
    # The machine model, which names links and whose controllers may fix
    # dispatch keys; compile() sets it. None: links are ids.
    machine: Any = None
    # Per link id, the keys the machine's controllers fix, by function name,
    # each with what fixes it; read from the machine at the first call.
    dispatch_tables: dict[int, dict[str, tuple[int, str]]] | None = None
    # Per entry of classical, its link id, dispatch key and the controller
    # that fixes the key (None for a hashed or built-in key); made at the
    # first call.
    routes: list[tuple[int, int, str | None]] | None = None
    # Per (link id, dispatch key), the entry of classical sent to it; made
    # at the first call.
    _dispatched: dict[tuple[int, int], int] | None = None
    # Per template, its versions in this program by key (list lengths);
    # made at the first call of a template.
    _versions: dict[Any, dict[Any, Any]] | None = None
    # Where every line can meet, in program cycles: a wait whose length
    # comes from an answer moves every later op by it, so its bounds must be
    # multiples of this.
    wait_grid = 1
    # When lines move on different grids, the grid each variable wait's
    # length must be a whole number of steps of (compile() sets it).
    wait_grids: list[int] | None = None
    # Of the float arguments that reach an integer, those named here record
    # the instructions being traced then, innermost first, in structural_at.
    locate_structural: frozenset[str] = frozenset()
    structural_at: dict[str, list[tuple[Any, int]]]
    # Whether the trace emitted a select: lowering then chooses per target
    # how each op that reads one runs.
    selects = False

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
        # Repeating one zero is twice as fast as converting a zeroed bytes.
        self._buf = array("q", _ZERO) * 8192
        # Where _ensure next grows the buffer: its end, or sooner when
        # max_instructions bounds the ops still allowed, to check them.
        self._cap = len(self._buf)
        self._cur = 0
        self._next_id = 0
        self._next_qubit = 0
        self._op_count = 0
        self._qref_to_qubit: dict[int, int] = {}
        self._line_qubit: dict[int, int] = {}  # line vid → qudit
        self._param_vids: dict[int, int] = {}  # Parameter.index → vid
        self._param_names: dict[int, str] = {}  # Parameter.index → name
        # Key by the expression object itself, not id(expression): temporary
        # expressions may be collected and Python may reuse their integer IDs.
        self._numeric_vids: dict[tuple[ParameterExpression | Selection, str], int] = {}
        # Open regions, innermost last: the yields seen so far in each, and
        # under "cached" the size of _numeric_vids when it opened.
        self._regions: list[dict[str, Any]] = []
        # The size of _numeric_vids when each open schedule block opened.
        self._block_marks: list[int] = []
        # Measurement vid -> readout ordinal, for readouts outside any region.
        self.readouts: dict[int, int] = {}
        self._n_readouts = 0
        # Set by the emitter when the trace evaluated an impure host call.
        self.impure = False
        # For the guard, kept by the emitter: the dict items the trace read,
        # (id(dict), key) -> (dict, key, value), and the ids of the dicts it
        # iterated or tested with `in`, whose every item matters.
        self.dict_items: dict[tuple[int, Any], tuple[dict[Any, Any], Any, Any]] = {}
        self.dict_whole: set[int] = set()
        # The compilation's loop-unroll threshold; None for the default.
        self.unroll_threshold: int | None = None
        # The controller's ControllerLimits.max_instructions; set it with
        # limit_instructions().
        self.max_instructions: int | None = None
        # Unrolled loop id -> (iterations done, op count) at the last check.
        self._limit_checks: dict[int, tuple[int, int]] = {}
        # What the traced kernel returned.
        self.returned: Any = None
        # Float arguments that reached an integer (a duration, a cast to
        # int): the schedule needs their values, so they are specialized on.
        self.structural: set[str] = set()
        # (length, lowest, highest or -1) of each wait whose length the
        # module computes from float arguments, checked at call time.
        self.variable_waits: list[tuple[Any, ...]] = []
        # Classical functions declared so far, in record order, and the
        # regions open now (ids), which bound where an answer is defined.
        self.classical: list[Any] = []
        self._classical_index: dict[int, int] = {}
        self._scope: list[int] = []
        # Loops and branch arms open now, as (id, is_loop): where a future
        # submitted in one may be waited on.
        self._flow: list[tuple[int, bool]] = []
        self._next_scope = 0
        self._answer_integers = False

    def _ensure(self, n: int) -> None:
        if self._cur + n >= self._cap:
            while self._cur + n >= len(self._buf):
                self._buf.frombytes(bytes(8 * len(self._buf)))
            self._cap = len(self._buf)
            if self.max_instructions is not None:
                self._check_instructions(n, self.max_instructions)

    def _check_instructions(self, n: int, limit: int) -> None:
        # Every op writes at least one word after counting itself, so the
        # op past the limit reaches the cap set here, or an earlier one.
        if self._op_count > limit:
            from ._emitter import instruction_limit_error

            raise instruction_limit_error(self._op_count, limit, self._limit_checks)
        from ._emitter import note_unrolled_progress

        note_unrolled_progress(self._op_count, self._limit_checks)
        self._cap = min(self._cap, self._cur + n + limit - self._op_count + 1)

    def limit_instructions(self, limit: int | None) -> None:
        """Stop tracing with a ``CompilationError`` once the program holds
        more than *limit* ops (``None``: no limit)."""
        self.max_instructions = limit
        self._cap = 0
        self._ensure(0)

    def limit_cycles(self, cycles: int) -> None:
        """Make a program longer than *cycles* (the controller's longest
        program) an error when the module is built."""
        self._record(_MAX_CYCLES, [cycles])

    def _w(self, val: int) -> None:
        self._buf[self._cur] = val
        self._cur += 1

    def _put(self, words: tuple[int, ...] | list[int]) -> None:
        """Write the record *words* at the cursor."""
        n = len(words)
        self._ensure(n)
        cur = self._cur
        self._buf[cur : cur + n] = array("q", words)
        self._cur = cur + n

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        a = attrs or {}
        vid = self._next_id
        count = len(result_types)
        results: tuple[IRValue, ...]
        if count == 2:
            results = (IRValue(vid, result_types[0]), IRValue(vid + 1, result_types[1]))
        elif count == 1:
            results = (IRValue(vid, result_types[0]),)
        else:
            results = tuple(IRValue(vid + i, vtype) for i, vtype in enumerate(result_types))
        self._next_id = vid + count

        # --- drive, the most common op ---
        if kind == "pulse.drive":
            self._op_count += 1
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            self._ensure(8)
            cur = self._cur
            # start_cycles and duration_cycles are set by the C++ scheduler.
            self._buf[cur : cur + 8] = array(
                "q", (_DRIVE_HEADER, lv, wv, tv, rlv, rtv, _UNSCHEDULED, _UNSCHEDULED)
            )
            self._cur = cur + 8
            return results

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
            self._line_qubit[results[0].vid] = qubit
            self._put((_header(_ALLOC_DRIVE, 3), qubit, results[0].vid, results[1].vid))
            return results

        # --- get_readout_line → ALLOC_READOUT ---
        if kind == "pulse.get_readout_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid, 0) if qref_vid is not None else 0
            self._line_qubit[results[0].vid] = qubit
            self._put((_header(_ALLOC_READOUT, 3), qubit, results[0].vid, results[1].vid))
            return results

        # --- Waveform constructors ---
        wf_name = kind.removeprefix("pulse.")
        wf_code = _WF_MAP.get(wf_name)
        if wf_code is not None or wf_name == "custom_samples":
            rv = results[0].vid
            dur = a.get("duration", 0)
            if dur.__class__ is not int or dur < 1 or dur > MAX_CYCLES:
                dur = self._duration(wf_name, a)

            if wf_code == _WF_GAUSSIAN:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals: list[Any] = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_GAUSSIAN, rv, vals, ["i64", "f64", "t64"])
            elif wf_code == _WF_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                if is_symbolic(amp_raw):
                    vals = [dur, amp_raw, 0.0]
                else:
                    try:
                        re, im = _complex_parts(amp_raw)
                    except (TypeError, ValueError):
                        raise CompilationError(
                            f"square amplitude must be a number or [re, im], got {amp_raw!r}"
                        ) from None
                    vals = [dur, re, im]
                self._emit_wf(_WF_SQUARE, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_DRAG:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                beta_raw = a.get("beta", 0.0)
                vals = [dur, amp_raw, sig_raw, beta_raw]
                self._emit_wf(_WF_DRAG, rv, vals, ["i64", "f64", "t64", "t64"])
            elif wf_code == _WF_COSINE:
                amp_raw = a.get("amplitude", 0.0)
                vals = [dur, amp_raw]
                self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64"])
            elif wf_code == _WF_TANH_RAMP:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_TANH_RAMP, rv, vals, ["i64", "f64", "t64"])
            elif wf_code == _WF_GAUSS_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                width_raw = a.get("width", 0)
                # The Python API specifies flat-top width while the dialect
                # stores the duration of each rise/fall edge.
                if width_raw.__class__ is not int or width_raw < 0 or width_raw > MAX_CYCLES:
                    self._times(wf_name, a)
                    width_raw = a.get("width", 0)
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
                self._emit_wf(_WF_GAUSS_SQUARE, rv, vals, ["i64", "f64", "t64", "i64"])
            else:
                self._emit_custom_samples(rv, a.get("samples", ()), a.get("rate"))
            return results

        # --- readout ---
        if kind == "pulse.readout":
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            mv = results[2].vid
            self._put((_header(_READOUT, 6), lv, wv, tv, rlv, rtv, mv))
            if not self._regions:
                self.readouts[mv] = self._n_readouts
            self._n_readouts += 1
            return results

        # --- sync ---
        if kind == "pulse.sync":
            n = len(operands)
            payload_len = 1 + 3 * n
            if payload_len >= _MAX_PAYLOAD:
                words = [_header(_SYNC, _MAX_PAYLOAD), payload_len, n]
            else:
                words = [_header(_SYNC, payload_len), n]
            for j in range(n):
                in_vid = operands[j].vid
                out_vid = results[j].vid if j < len(results) else in_vid
                vtype = _LINE_TYPE_CODES.get(
                    results[j].vtype if j < len(results) else "drive_line", 0
                )
                words += (in_vid, out_vid, vtype)
            self._put(words)
            return results

        # --- wait ---
        if kind == "pulse.wait":
            lv = operands[0].vid
            rlv = results[0].vid
            dur_raw = a.get("duration", 0)
            bounds = a.get("bounds")
            if bounds is None and dur_raw.__class__ is int and 0 <= dur_raw <= MAX_CYCLES:
                self._put((_WAIT_HEADER, lv, rlv, dur_raw))
                return results
            if "duration" in a:
                self._times("wait", a)
                dur_raw = a["duration"]
                bounds = a.get("bounds")
            symbolic = is_symbolic(dur_raw)
            if bounds is not None or symbolic:
                lowest, highest = (0, -1) if bounds is None else _wait_bounds(bounds)
                if symbolic and self.classical and contains_answer(dur_raw):
                    if self.wait_grid > 1:
                        self._check_answer_bounds(operands[0], lowest, highest)
                    self._variable_wait(
                        lv, rlv, dur_raw, lowest, highest, answer=True, grid=self.wait_grid
                    )
                    return results
                if symbolic and is_arithmetic(dur_raw):
                    grid = None
                    if self.wait_grids is not None:
                        assert self.line_grid is not None
                        grid = self.line_grid(*self._role_and_qudit(operands[0]))
                    self._variable_wait(lv, rlv, dur_raw, lowest, highest, grid=grid)
                    return results
                if not symbolic:
                    dur = _as_i64(dur_raw, "wait duration")
                    if dur < lowest or (highest >= 0 and dur > highest):
                        top = highest if highest >= 0 else None
                        raise CompilationError(
                            f"wait duration {dur} is outside its bounds {(lowest, top)!r}"
                        )
            if symbolic:
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
                self._put((_header(_SHIFT_PHASE, 3, 1 << 2), tv, rtv, pvid))
            else:
                try:
                    bits = _f2i(val_raw)
                except struct.error:
                    bits = _f2i(_real(val_raw, "shift_phase phase"))
                self._put((_header(_SHIFT_PHASE, 3), tv, rtv, bits))
            return results

        # --- set_phase ---
        if kind == "pulse.set_phase":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("phase_rad", 0.0)
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
                self._put((_header(_SET_PHASE, 3, 1 << 2), tv, rtv, pvid))
            else:
                try:
                    bits = _f2i(val_raw)
                except struct.error:
                    bits = _f2i(_real(val_raw, "set_phase phase"))
                self._put((_header(_SET_PHASE, 3), tv, rtv, bits))
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
                self._put((_header(_SHIFT_FREQ, 3, 1 << 2), tv, rtv, pvid))
            else:
                try:
                    bits = _f2i(val_raw)
                except struct.error:
                    bits = _f2i(_real(val_raw, "shift_frequency frequency"))
                self._put((_header(_SHIFT_FREQ, 3), tv, rtv, bits))
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
                self._put((_header(_SET_FREQ, 3, 1 << 2), tv, rtv, pvid))
            else:
                try:
                    bits = _f2i(val_raw)
                except struct.error:
                    bits = _f2i(_real(val_raw, "set_frequency frequency"))
                self._put((_header(_SET_FREQ, 3), tv, rtv, bits))
            return results

        if kind.startswith("scf."):
            return self._region(kind, operands, results, a)

        algebra = _ALGEBRA_OPCODES.get(kind)
        if algebra is not None:
            rv = results[0].vid
            self._put(
                [_header(algebra, 1 + len(operands)), rv, *(operand.vid for operand in operands)]
            )
            return results

        if kind == "pulse.wf_scale":
            rv = results[0].vid
            scale = a["scale"]
            if scale.__class__ is float and scale - scale != 0.0:
                raise CompilationError(f"wf_scale scale must be finite, got {scale!r}")
            scale_vid = self._get_numeric_vid(scale, "f64")
            self._put((_header(_WF_SCALE, 3), rv, operands[0].vid, scale_vid))
            return results

        if kind == "pulse.tone":
            self._put(
                (
                    _header(_ALLOC_TONE, 3),
                    results[0].vid,
                    _f2i(a["frequency_hz"]),
                    _f2i(a["phase_rad"]),
                )
            )
            return results

        raise CompilationError(f"unsupported packed pulse op: {kind}")

    def _record(self, opcode: int, payload: list[int]) -> None:
        n = len(payload)
        if n >= _MAX_PAYLOAD:
            self._put([_header(opcode, _MAX_PAYLOAD), n, *payload])
        else:
            self._put([_header(opcode, n), *payload])

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
            self._regions.append({"cached": len(self._numeric_vids)})
            self._open_scope(loop=True)
        elif kind == "scf.for_end":
            self._close_scope()
            region = self._regions.pop()
            if len(self._numeric_vids) > region["cached"]:
                self._forget(region["cached"])
            inits = [v.vid if v.vid != UNDEFINED_VID else -1 for v in operands]
            yields = list(region.get("yield", ()))
            self._record(_FOR_END, [len(results), *inits, *yields, *(r.vid for r in results)])
        elif kind == "scf.if":
            self._record(_IF_BEGIN, [operands[0].vid])
            self._regions.append({"cached": len(self._numeric_vids)})
            self._open_scope(loop=False)
        elif kind == "scf.else":
            self._record(_ELSE, [])
            self._regions[-1]["else"] = ()
            if len(self._numeric_vids) > self._regions[-1]["cached"]:
                self._forget(self._regions[-1]["cached"])
            self._close_scope()
            self._open_scope(loop=False)
        elif kind == "scf.if_end":
            self._close_scope()
            region = self._regions.pop()
            if len(self._numeric_vids) > region["cached"]:
                self._forget(region["cached"])
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

    def _forget(self, cached: int) -> None:
        """Keep the first *cached* numbers materialized: the later ones are
        defined in a region that has closed, which later uses must not reach
        into. Every entry added since has a new id, and they are the newest."""
        numeric = self._numeric_vids
        while len(numeric) > cached:
            numeric.popitem()

    def begin_schedule_block(
        self, policy: str, callee: str, max_duration_cycles: int | None = None
    ) -> None:
        """Open a block whose ops are scheduled on their own with *policy*,
        and that scheduling rejects if it can take more than
        *max_duration_cycles*.

        The extension makes the lines, tones and measurements the block uses
        from outside its operands, and the ones it leaves its results.
        """
        encoded = callee.encode("utf-8")
        padded = encoded + b"\0" * ((8 - len(encoded) % 8) % 8)
        words = [_unpack_q.unpack(padded[i : i + 8])[0] for i in range(0, len(padded), 8)]
        if max_duration_cycles is not None:
            words.append(max_duration_cycles)
        self._record(_BLOCK_BEGIN, [_BLOCK_POLICY[policy], len(encoded), *words])
        self.schedule_block_count += 1
        self._block_marks.append(len(self._numeric_vids))
        self._open_scope()

    def end_schedule_block(self) -> None:
        self._close_scope()
        self._record(_BLOCK_END, [])
        cached = self._block_marks.pop()
        if len(self._numeric_vids) > cached:
            self._forget(cached)

    def _open_scope(self, loop: bool | None = None) -> None:
        """Open a region: a loop body, a branch arm, or (``None``) a block."""
        self._scope.append(self._next_scope)
        if loop is not None:
            self._flow.append((self._next_scope, loop))
        self._next_scope += 1

    def _close_scope(self) -> None:
        closed = self._scope.pop()
        if self._flow and self._flow[-1][0] == closed:
            self._flow.pop()

    @property
    def flow(self) -> tuple[tuple[int, bool], ...]:
        """The loops and branch arms open now, outermost first."""
        return tuple(self._flow)

    def check_wait(self, flow: tuple[tuple[int, bool], ...], name: str) -> None:
        """A future submitted under *flow* is waited on here: in the same
        loop iteration, possibly in a branch arm inside it."""
        here = self._flow
        if tuple(here[: len(flow)]) != flow or any(loop for _, loop in here[len(flow) :]):
            raise CompilationError(
                f"the answer of {name} is taken in another loop iteration or outside "
                "the branch arm that submitted it; take it in the same loop body, "
                "after the submission"
            )

    @property
    def scope(self) -> tuple[int, ...]:
        """The regions open now, outermost first."""
        return tuple(self._scope)

    def check_scope(self, scope: tuple[int, ...], what: str) -> None:
        """Reject a value taken in a region that has since closed."""
        here = self._scope
        if len(scope) > len(here) or tuple(here[: len(scope)]) != scope:
            raise CompilationError(
                f"{what} is used outside the loop, branch or kernel with its own "
                "schedule= where it was taken; take it where it is used, or return "
                "the future and take the answer after the kernel returns"
            )

    # ── Classical calls ───────────────────────────────────────────────────

    def classical_function(self, function: Any) -> int:
        """The record index of classical function *function*, declared once."""
        index = self._classical_index.get(id(function))
        if index is not None:
            return index
        if function._plan is not None:  # a template reached with no arguments
            raise CompilationError(
                f"{function.name}.submit() takes {len(function._plan)} arguments, got 0"
            )
        for other in self.classical:
            if other.name == function.name:
                raise CompilationError(
                    f"two different classical functions are named {function.name!r}"
                )
        index = len(self.classical)
        machine = self.machine
        if machine is None:
            link = _link_without_machine(function.link)
        else:
            link = machine.link_id(function.link)
        # Its key: its controller's, else its own; one function per key on a
        # link.
        key = function._dispatch_key
        fixed_by = None
        own = function._key_route is None
        tables = self.dispatch_tables
        if tables is None:
            tables = self.dispatch_tables = _tables(machine)
        if tables and own:
            table = tables.get(link)
            if table is not None:
                key, fixed_by = _fixed_key(function, table)
        if own and key in _RESERVED_KEYS:
            raise CompilationError(
                f"{function.name} is dispatch key {key:#x}, which {_RESERVED_KEYS[key]} uses"
                " for its decoder messages; rename it"
            )
        dispatched = self._dispatched
        if dispatched is None:
            dispatched = self._dispatched = {}
        else:
            other = dispatched.get((link, key))
            if other is not None:
                first = self.classical[other]
                if own or first._key_route is None:
                    raise CompilationError(_collision(first, function, key))
        mode = _LATENCY_MODE[function.latency_mode]
        flags = mode | (4 if function.submit_policy == "lazy" else 0)
        # Its own key is its name's hash, unless a built-in's.
        flags |= 8 if key != function._dispatch_key or not own else 0
        flags |= 16 if function.timeout_ns is not None else 0
        flags |= 32 if function.deadline_ns is not None else 0
        flags |= 64 if function.reaction_deadline_ns is not None else 0
        name = function.name.encode("utf-8")
        args = ",".join(a.descriptor for a in function.args).encode("ascii")
        results = ",".join(r.descriptor for r in function.results).encode("ascii")
        text = name + args + results
        padded = text + b"\0" * ((8 - len(text) % 8) % 8)
        words = [_unpack_q.unpack(padded[i : i + 8])[0] for i in range(0, len(padded), 8)]
        self._record(
            _CLASSICAL_DECLARE,
            [
                index,
                key,
                link,
                flags,
                _f2i(function.min_latency_ns),
                _f2i(function.nominal_latency_ns),
                _f2i(function.max_latency_ns),
                _f2i(function.timeout_ns or 0.0),
                _f2i(function.deadline_ns or 0.0),
                len(name),
                len(args),
                len(results),
                *words,
                *(
                    ()
                    if function.reaction_deadline_ns is None
                    else (_f2i(function.reaction_deadline_ns),)
                ),
            ],
        )
        self.classical.append(function)
        routes = self.routes
        if routes is None:
            routes = self.routes = []
        routes.append((link, key, fixed_by))
        dispatched[(link, key)] = index
        self._classical_index[id(function)] = index
        return index

    def _forget_classical(self, n_classical: int) -> None:
        """Drop the classical functions declared after the first
        *n_classical*, with their keys and versions."""
        dispatched = self._dispatched
        assert dispatched is not None and self.routes is not None
        for sent, entry in list(dispatched.items()):
            if entry >= n_classical:
                del dispatched[sent]
        del self.routes[n_classical:]
        for function in self.classical[n_classical:]:
            del self._classical_index[id(function)]
            template = function._template
            if template is not None and self._versions:
                self._versions.get(template, {}).pop(function._key, None)
        del self.classical[n_classical:]

    def classical_submit(self, index: int, operands: list[int]) -> int:
        """Record a submission; returns the future's id."""
        future = self._next_id
        self._next_id += 1
        self._op_count += 1
        self._record(_CLASSICAL_SUBMIT, [index, future, len(operands), *operands])
        return future

    def classical_wait(
        self, index: int, future: int, dtypes: list[str], timeout_ns: float | None
    ) -> list[AnswerValue]:
        """Record a wait; returns its answers, one per SSA result."""
        vids = list(range(self._next_id, self._next_id + len(dtypes)))
        self._next_id += len(dtypes)
        self._op_count += 1
        has_timeout = timeout_ns is not None
        self._record(
            _CLASSICAL_WAIT,
            [index, future, int(has_timeout), _f2i(timeout_ns or 0.0), len(vids), *vids],
        )
        scope = self.scope
        return [AnswerValue(v, t, scope) for v, t in zip(vids, dtypes)]

    def classical_compare(self, predicate: str, lhs: Any, rhs: Any) -> int:
        """Record a comparison of answers and numbers; returns the i1 id."""

        def floating(v: Any) -> bool:
            if isinstance(v, AnswerValue):
                return v.dtype == "f64"
            if isinstance(v, float):
                return True
            return is_symbolic(v) and getattr(v, "dtype", "f64") != "i64"

        is_float = floating(lhs) or floating(rhs)
        dtype = "f64" if is_float else "i64"
        self._answer_integers = True
        try:
            a = self._get_numeric_vid(lhs, dtype)
            b = self._get_numeric_vid(rhs, dtype)
        finally:
            self._answer_integers = False
        result = self._next_id
        self._next_id += 1
        self._record(_CLASSICAL_COMPARE, [result, _COMPARE[predicate], int(is_float), a, b])
        return result

    def compare_bits(self, predicate: str, lhs: int, rhs: int) -> int:
        """Record ``==`` or ``!=`` of the i1 values *lhs* and *rhs*; returns the i1 id."""
        result = self._next_id
        self._next_id += 1
        self._record(_CLASSICAL_COMPARE, [result, _COMPARE[predicate], 0, lhs, rhs])
        return result

    def bit_constant(self, value: bool) -> int:
        """The id of an i1 that is *value*."""
        return self.classical_compare("==" if value else "!=", 0, 0)

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
            len(self.variable_waits),
            len(self._block_marks),
            len(self.classical),
            list(self._scope),
            list(self._flow),
            self._next_scope,
        )

    def rollback(self, checkpoint: tuple[Any, ...]) -> None:
        (
            self._cur,
            self._next_id,
            self._next_qubit,
            self._op_count,
            regions,
            self._n_readouts,
            waits,
            blocks,
            n_classical,
            scope,
            flow,
            self._next_scope,
        ) = checkpoint
        self._regions = regions
        if self.max_instructions is not None:
            self._cap = 0
        del self.variable_waits[waits:]
        if self.wait_grids is not None:
            del self.wait_grids[waits:]
        del self._block_marks[blocks:]
        self._scope = scope
        self._flow = flow
        if len(self.classical) > n_classical:
            self._forget_classical(n_classical)
        live = self._next_id
        for vid in [vid for vid in self.readouts if vid >= live]:
            del self.readouts[vid]
        for vid in [vid for vid in self._qref_to_qubit if vid >= live]:
            del self._qref_to_qubit[vid]
        for vid in [vid for vid in self._line_qubit if vid >= live]:
            del self._line_qubit[vid]
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
        self._put((_header(_PARAM, 2), vid, param.index))
        return vid

    def _get_numeric_vid(self, value: Any, expected_dtype: str, specializes: bool = True) -> int:
        """Materialize a symbolic or literal number as an SSA value record.

        An integer computed from float arguments makes them static, unless
        *specializes* is false: the schedule then takes it at call time."""
        if expected_dtype not in _NUMERIC_TYPE:
            raise CompilationError(f"unsupported numeric type {expected_dtype}")
        if self.classical:
            if expected_dtype == "i64" and not self._answer_integers and contains_answer(value):
                raise CompilationError(_DURATION_FROM_ANSWER)
            if isinstance(value, AnswerValue):
                self.check_scope(value.scope, "an answer of a classical call")
                if value.dtype == expected_dtype:
                    return value.vid
                if value.dtype == "i1":
                    raise CompilationError(
                        "an answer bit is a branch condition, not a number; use it as `if bit:`"
                    )
                return self._emit_numeric_cast(value.vid, value.dtype, expected_dtype)
        if (
            specializes
            and is_symbolic(value)
            and (
                expected_dtype == "i64"
                or (
                    isinstance(value, ParameterExpression)
                    and value.op == "cast"
                    and value.dtype == "i64"
                )
            )
        ):
            names = parameter_names(value)
            located = self.locate_structural
            if located and not located.isdisjoint(names) and not self.structural.issuperset(names):
                from ._emitter import tracing_frames

                frames = tracing_frames()
                at = self.__dict__.setdefault("structural_at", {})
                for name in names:
                    at.setdefault(name, frames)
            self.structural.update(names)

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
                # A cast that emitted nothing is not cached: every entry is
                # then newer than those before it, which _forget relies on.
                if result != source:
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
                self._put((_header(_NUM_NEG, 3), result, _NUMERIC_TYPE[dtype], operand))
            else:
                opcode = _BINARY_OP.get(value.op)
                if opcode is None:
                    raise CompilationError(f"unsupported symbolic operation {value.op!r}")
                lhs = self._get_numeric_vid(value.lhs, dtype)
                rhs = self._get_numeric_vid(value.rhs, dtype)
                result = self._next_id
                self._next_id += 1
                self._put((_header(_NUM_BINARY, 5), result, _NUMERIC_TYPE[dtype], opcode, lhs, rhs))
            if dtype != expected_dtype:
                result = self._emit_numeric_cast(result, dtype, expected_dtype)
            self._numeric_vids[cache_key] = result
            return result

        cls = value.__class__
        if cls is not float and cls is not int and isinstance(value, Selection):
            return self.select(value, expected_dtype, specializes)

        result = self._next_id
        self._next_id += 1
        try:
            word = _as_i64(value, "integer expression") if expected_dtype == "i64" else _f2i(value)
        except struct.error:
            word = _f2i(_real(value, "a pulse op operand"))
        self._put((_header(_NUM_CONST, 3), result, _NUMERIC_TYPE[expected_dtype], word))
        return result

    def select(self, value: Selection, dtype: str, specializes: bool = True) -> int:
        """The id of *value* as a *dtype* ("i64", "f64" or "waveform"): one
        select between its two values, each made once where it is first
        needed. Lowering keeps it, or emits each op that reads it in both
        arms of an if on its condition."""
        key = (value, dtype)
        vid = self._numeric_vids.get(key)
        if vid is not None:
            return vid
        then = self._selected(value.then, dtype, specializes)
        other = self._selected(value.other, dtype, specializes)
        vid = self._next_id
        self._next_id += 1
        self._put((_header(_SELECT, 4), vid, value.cond.vid, then, other))
        self._numeric_vids[key] = vid
        self.selects = True
        return vid

    def _selected(self, value: Any, dtype: str, specializes: bool) -> int:
        if isinstance(value, Selection):
            return self.select(value, dtype, specializes)
        if dtype == "waveform":
            return int(value.vid)
        return self._get_numeric_vid(value, dtype, specializes)

    def _check_answer_bounds(self, line: IRValue, lowest: int, highest: int) -> None:
        """Reject the bounds of a wait whose length comes from an answer when
        they are not on ``wait_grid``: the run clamps the length to them, and
        every later op on every line moves by it."""
        grid = self.wait_grid
        off = [b for b in (lowest, highest) if b >= 0 and b % grid]
        if not off:
            return
        role, qudit = self._role_and_qudit(line)
        top = "None" if highest < 0 else str(highest)

        def nearest(b: int) -> str:
            return f"{b // grid * grid} or {-(-b // grid) * grid}"

        raise CompilationError(
            f"wait() on line {role[0]}{qudit}: its length comes from a classical answer, "
            "so every later operation on every line moves by it, and together the lines "
            f"move in timesteps of {grid} cycles; its bounds ({lowest}, {top}) aren't "
            "whole numbers of timesteps: " + "; ".join(f"for {b}, use {nearest(b)}" for b in off)
        )

    def _role_and_qudit(self, line: IRValue) -> tuple[str, int]:
        """The role of *line* and its qudit (-1 for a runtime qudit's)."""
        qudit = self._line_qubit.get(line.vid)
        if qudit is None:
            # A later version of the line: its channel names the qudit.
            held = getattr(line.chan, "held", None)
            qudit = -1 if held is None else self._qref_to_qubit.get(held.key[0], -1)
        return line.vtype.removesuffix("_line"), qudit

    def _variable_wait(
        self,
        line: int,
        result: int,
        duration: Any,
        lowest: int,
        highest: int,
        answer: bool = False,
        grid: int | None = None,
    ) -> None:
        """A wait whose length is computed at run time, rounded to the nearest
        cycle (ties up). From float arguments it does not specialize on them,
        and each call checks it against the bounds; from the answer of a
        classical call it is a variable region, clamped to the bounds. A
        length from both does not specialize on its float arguments either:
        each call checks that its float part (its answers taken as 0) is
        finite and at most the highest length, and the run clamps it whole."""
        rounded = ParameterExpression(
            "cast", ParameterExpression("add", duration, 0.5), dtype="i64"
        )
        if answer:
            floats = without_answers(duration)
            at_call = bool(parameter_names(duration)) and is_arithmetic(floats)
            self._answer_integers = True
            try:
                cycles = self._get_numeric_vid(rounded, "i64", specializes=not at_call)
            finally:
                self._answer_integers = False
            if at_call:
                text = f"{arithmetic_text(duration)} (its answers at 0)"
                self.variable_waits.append((floats, None, highest, text))
                if self.wait_grids is not None:
                    self.wait_grids.append(grid or 1)
        else:
            cycles = self._get_numeric_vid(rounded, "i64", specializes=False)
            self.variable_waits.append((duration, lowest, highest))
            if self.wait_grids is not None:
                self.wait_grids.append(grid or 1)
        self._ensure(7)
        self._w(_header(_VARIABLE_WAIT, 6))
        self._w(line)
        self._w(result)
        self._w(cycles)
        self._w(lowest)
        self._w(highest)
        self._w(int(answer))

    def _emit_numeric_cast(self, operand: int, source_dtype: str, target_dtype: str) -> int:
        if source_dtype == target_dtype:
            return operand
        result = self._next_id
        self._next_id += 1
        self._put(
            (
                _header(_NUM_CAST, 4),
                result,
                _NUMERIC_TYPE[source_dtype],
                _NUMERIC_TYPE[target_dtype],
                operand,
            )
        )
        return result

    def _times(self, fname: str, attrs: dict[str, Any]) -> None:
        """Convert the times in *attrs*, a *fname* op's, to program cycles,
        and reject plain floats and out-of-range lengths in its durations."""
        # No time exists before _units is imported, so neither does one
        # inside a symbolic wait.
        if (
            fname == "wait"
            and attrs.get("bounds") is None
            and is_symbolic(attrs["duration"])
            and "cudaq_pulse._units" not in sys.modules
        ):
            return
        from ._durations import in_program_cycles

        in_program_cycles(fname, attrs, self.clock_ghz)

    def _duration(self, wf_name: str, attrs: dict[str, Any]) -> Any:
        """The duration of a waveform whose duration is not a whole number
        of cycles in range, after its times are converted."""
        if wf_name != "custom_samples":
            self._times(wf_name, attrs)
        dur_raw = attrs.get("duration", 0)
        return dur_raw if is_symbolic(dur_raw) else _as_i64(dur_raw, f"{wf_name} duration")

    def _shape_time(self, wf_code: int, slot: int, value: Any) -> Any:
        """*value*, a waveform's shape time (sigma or beta) in operand *slot*,
        in program cycles."""
        from ._durations import cycles

        what = ("duration", "amplitude", "sigma", "beta")[slot]
        return cycles(_WF_NAMES[wf_code], what, value, False, "", self.clock_ghz)

    def _emit_wf(self, wf_code: int, rv: int, vals: list[Any], dtypes: list[str]) -> None:
        """Emit a waveform record, handling mixed Parameter/concrete values.

        For each value slot, if the value is a Parameter, emit a PARAM record
        first and encode the param vid. A param_mask bit flags parametric slots
        so the C++ decoder can distinguish vids from literals.
        """
        encoded: list[int] = []
        param_mask = 0
        try:
            for i, (v, dtype) in enumerate(zip(vals, dtypes)):
                # A float or a whole number in an integer slot as it is; "t64" is
                # a shape time ("f64"), which may be a time to convert.
                cls = v.__class__
                if cls is float:
                    encoded.append(_f2i(v))
                    continue
                if cls is int and dtype == "i64":
                    encoded.append(v)
                    continue
                if dtype == "t64":
                    if cls is not int:
                        v = self._shape_time(wf_code, i, v)
                    dtype = "f64"
                if is_symbolic(v):
                    pvid = self._get_numeric_vid(v, dtype)
                    encoded.append(pvid)
                    param_mask |= 1 << (i + 1)  # +1 because slot 0 is the rv
                elif dtype == "i64":
                    encoded.append(_as_i64(v, "integer operand"))
                else:
                    encoded.append(_f2i(v))
        except (TypeError, ValueError, struct.error):
            op, fields = _WF_FIELDS[wf_code]
            for v, dtype, field in zip(vals, dtypes, fields):
                if dtype != "i64" and not is_symbolic(v):
                    _real(v, f"{op} {field}")
            raise
        payload_len = 1 + len(encoded)  # rv + values
        self._put([_header(wf_code, payload_len, param_mask), rv, *encoded])

    def _emit_custom_samples(
        self, rv: int, samples: Any, rate: tuple[int, int] | None = None
    ) -> None:
        """A custom_samples record; with *rate*, n samples every d cycles."""
        try:
            sample_values = list(samples)
        except TypeError as exc:
            raise CompilationError("custom_samples() requires a finite sequence") from exc
        most = MAX_CUSTOM_SAMPLES if rate is None else MAX_CUSTOM_SAMPLES - 2
        if len(sample_values) > most:
            raise CompilationError(f"custom_samples() supports at most {most} samples per waveform")
        if not sample_values:
            raise CompilationError("custom_samples() requires at least one sample")
        encoded: list[int] = []
        for sample in sample_values:
            try:
                value: complex = sample.__float__()
            except (AttributeError, TypeError, ValueError):
                value = _sample_number(sample)
            if value.imag != 0.0:
                raise CompilationError(
                    "complex custom samples are not supported yet; provide "
                    "real-valued envelope samples"
                )
            encoded.append(_f2i(value.real))
        extra = 0 if rate is None else 2
        words = [_header(_WF_CUSTOM_SAMPLES, 2 + len(encoded) + extra), rv, len(encoded), *encoded]
        if rate is not None:
            words += (rate[0], rate[1])
        self._put(words)

    @property
    def param_names(self) -> list[str]:
        """Names of the parameters referenced so far, in index order."""
        return [self._param_names[i] for i in sorted(self._param_names)]

    def get_buffer(self) -> array[int]:
        """Return the trimmed packed buffer (int64 words)."""
        return self._buf[: self._cur]

    def get_freq_array(self) -> array[float]:
        """Return qubit frequencies as float64 values indexed by qubit."""
        n = self._next_qubit
        freq = self._freq_hz
        arr = array("d", bytes(8 * n))
        # The program's qudits, not all of the target's.
        for q in range(n):
            if q in freq:
                arr[q] = freq[q]
        return arr

    @property
    def n_qubits(self) -> int:
        return self._next_qubit

    @property
    def qubits(self) -> tuple[int, ...]:
        """Indices of the allocated qudits (runtime qudit k: -1 - k)."""
        return tuple(sorted(set(self._qref_to_qubit.values())))

    @property
    def used_qubits(self) -> tuple[int, ...]:
        """Indices of the qudits the program gets a drive or readout line on
        (runtime qudit k: -1 - k)."""
        return tuple(sorted(set(self._line_qubit.values())))

    @property
    def used_slots(self) -> tuple[int, ...]:
        """The runtime qudits (slots) the program gets a line on."""
        return tuple(sorted(-1 - q for q in set(self._line_qubit.values()) if q < 0))

    @property
    def op_count(self) -> int:
        return self._op_count

    def finder(self, word: int) -> _RecordFinder:
        """A builder configured like this one that, given the same trace,
        records the instructions that write the record at *word*."""
        found = _RecordFinder(
            name=self.name,
            clock_ghz=self.clock_ghz,
            qubit_freq_hz=self._freq_hz,
            flatten_schedules=not self.schedule_blocks,
        )
        found.word = word
        found.unroll_threshold = self.unroll_threshold
        found.machine = self.machine
        found.locate_structural = self.locate_structural
        return found


class _Stop(BaseException):
    """Ends a ``_RecordFinder`` trace; a BaseException, so that no handler
    for errors of the traced code catches it."""


# Code from these directories is the compiler's own or the standard
# library's, which the trace may run again; any other is the kernel's host
# code, which it must not.
_TRUSTED_DIRS: tuple[str, ...] | None = None
# Builtins whose call from a kernel acts on the world outside the trace, by
# id: next() advances an iterator the kernel may not own.
_ACTING_BUILTINS = frozenset(
    map(id, (print, input, open, breakpoint, exec, eval, setattr, delattr, __import__, next))
)


def _trusted_dirs() -> tuple[str, ...]:
    global _TRUSTED_DIRS
    if _TRUSTED_DIRS is None:
        import os
        import sysconfig

        paths = sysconfig.get_paths()
        found = [os.path.dirname(os.path.dirname(__file__)), paths["stdlib"], paths["platstdlib"]]
        numpy = sys.modules.get("numpy")
        if numpy is not None and numpy.__file__:
            found.append(os.path.dirname(numpy.__file__))
        dirs = {f(d) for d in found for f in (os.path.abspath, os.path.realpath)}
        _TRUSTED_DIRS = tuple(os.path.join(d, "") for d in dirs)
    return _TRUSTED_DIRS


def _trusted(code: Any, seen: dict[Any, bool]) -> bool:
    trusted = seen.get(code)
    if trusted is None:
        import os

        name = code.co_filename
        trusted = name.startswith(("<frozen ", "<string>")) or os.path.abspath(name).startswith(
            _trusted_dirs()
        )
        seen[code] = trusted
    return trusted


class _RecordFinder(PackedIRBuilder):
    """Traces a kernel again, after the extension rejected the record at
    ``word``, to find where it was written: ``frames`` as ``tracing_frames()``
    gives them. Every record is written right after an ``_ensure`` at its
    first word; a loop traced rolled and rolled back may write there first,
    so the trace stops at the first visit outside such a loop.

    The trace never runs the kernel's host code again: it stops before the
    first call into it, with ``host_frames`` where that call is."""

    word = -1
    frames: list[Any] | None = None
    host_frames: list[Any] | None = None

    @property
    def impure(self) -> bool:
        return False

    @impure.setter
    def impure(self, value: bool) -> None:
        # Set just before a host call that is not pure is evaluated.
        if value:
            self._stop_at_host()

    def _stop_at_host(self) -> None:
        from ._emitter import tracing_frames

        self.host_frames = tracing_frames()
        raise _Stop

    def _ensure(self, n: int) -> None:
        if self._cur == self.word:
            from ._emitter import PulseIREmitter, tracing_frames

            self.frames = tracing_frames()
            rolling = PulseIREmitter._roll.__code__
            frame: Any = sys._getframe(1)
            while frame is not None and frame.f_code is not rolling:
                frame = frame.f_back
            if frame is None:
                raise _Stop
        super()._ensure(n)

    def locate(self, fn: Any, args: Any) -> bool:
        """Trace *fn* again on *args* until the record at ``word``; False if
        the trace could not be watched for host code, and did not run."""
        from types import GeneratorType

        from . import _emitter
        from .bytecode_bridge import _trace_kernel_with_builder

        mon = sys.monitoring
        for tool in (3, 4):
            try:
                mon.use_tool_id(tool, "cudaq_pulse retrace")
                break
            except ValueError:
                continue
        else:
            return False
        seen: dict[Any, bool] = {}
        # Where the emitter calls what the kernel calls.
        on_choice = _emitter.PulseIREmitter._call_on_choice.__code__
        host_sites = {_emitter.PulseIREmitter._dispatch_call.__code__} | {
            c for c in on_choice.co_consts if type(c) is type(on_choice)
        }
        coroutines = 0x20 | 0x80 | 0x100 | 0x200  # generator, coroutine, async

        # The callbacks never return DISABLE: that would outlast this trace
        # for whoever takes the tool id next.
        def on_start(code: Any, offset: int) -> None:
            # A generator is stopped at the call that resumes it: raising in
            # its frame would close it.
            if not code.co_flags & coroutines and not _trusted(code, seen):
                self._stop_at_host()

        def on_call(code: Any, offset: int, fn: Any, arg0: Any) -> None:
            if type(arg0) is GeneratorType and not _trusted(arg0.gi_code, seen):
                self._stop_at_host()
            if id(fn) in _ACTING_BUILTINS and code in host_sites:
                self._stop_at_host()

        events = mon.events
        mon.register_callback(tool, events.PY_START, on_start)
        mon.register_callback(tool, events.CALL, on_call)
        mon.set_events(tool, events.PY_START | events.CALL)
        try:
            _trace_kernel_with_builder(fn, self, args)
        except _Stop:
            pass
        finally:
            mon.set_events(tool, 0)
            mon.register_callback(tool, events.PY_START, None)
            mon.register_callback(tool, events.CALL, None)
            mon.free_tool_id(tool)
        return True
