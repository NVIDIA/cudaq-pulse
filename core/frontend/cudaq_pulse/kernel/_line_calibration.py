# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``line.get_calibration`` while tracing, and the calibrated pulses it gives
played where a kernel passes one as a waveform. Loaded by the first kernel
that uses either."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .._errors import CompilationError, InvalidTypeError, InvalidValueError, NotFoundError

if TYPE_CHECKING:
    from ._emitter import PulseIREmitter, _Held
    from .ir_builder import IRValue


class LineCalibrations:
    """The calibrated pulses of a line's qudit, from the device the kernel
    compiles for; ``get_calibration`` is ``line.get_calibration``."""

    __slots__ = ("_device", "_qudit", "_readout")

    def __init__(self, emitter: PulseIREmitter, held: _Held) -> None:
        builder = emitter.b
        # line_tones is the device's own method.
        device = getattr(getattr(builder, "line_tones", None), "__self__", None)
        qudit = getattr(builder, "_qref_to_qubit", {}).get(held.key[0])
        if device is None or qudit is None:
            raise CompilationError(
                "line.get_calibration needs the device's calibrations; this trace has no device"
            )
        if qudit < 0:
            raise CompilationError(
                "line.get_calibration needs the qudit's carrier, and this qudit is a runtime "
                "parameter (annotated Qudit), whose carrier each call binds; read the "
                "calibration outside the kernel, dev.qudits[i].get_calibration(gate), and pass "
                "its values in, or leave the parameter unannotated"
            )
        self._device = device
        self._qudit = qudit
        self._readout = held.key[1] == "get_readout_line"

    def get_calibration(self, gate: Any, transition: tuple[int, int] = (0, 1)) -> Any:
        try:
            return self._device._line_calibration(self._qudit, self._readout, gate, transition)
        except (InvalidTypeError, InvalidValueError, NotFoundError) as error:
            raise CompilationError(Exception.__str__(error)) from error


def play_calibrated(emitter: PulseIREmitter, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
    """*fname*, whose operands are not all of its types: the op on the
    waveform a calibrated pulse among them plays; else the operand error."""
    for index, value in enumerate(call_args):
        played = calibrated_waveform(emitter, value)
        if played is not None:
            args = list(call_args)
            args[index] = played
            return emitter._emit_pulse_call(fname, args)
    from ._emit_errors import operand_error

    raise operand_error(fname, call_args)


def calibrated_waveform(emitter: PulseIREmitter, pulse: Any) -> IRValue | None:
    """The waveform *pulse* plays, if it is a calibrated pulse (``DragPulse``,
    ``ReadoutPulse``); None for any other value."""
    from ..targets.base import DragPulse, FluxCZ, ReadoutPulse

    cls = pulse.__class__
    if cls is FluxCZ:
        raise CompilationError(
            "a FluxCZ does not play: flux lines are not modeled yet; wait its duration_ns on "
            "the coupled qudits' lines instead"
        )
    if cls is not DragPulse and cls is not ReadoutPulse:
        return None
    from .._units import ns

    emit = emitter._emit_pulse_call
    waveform: IRValue
    if cls is DragPulse:
        duration, sigma, amplitude, beta, phase = pulse._values
        if amplitude is None:
            raise CompilationError(
                f"{pulse!r} has no amplitude, so it does not play; line.get_calibration gives "
                "a calibration's pulse the amplitude of its rotation"
            )
        (waveform,) = emit("drag", [duration * ns, amplitude, sigma * ns, beta * ns])
        if phase:
            (factor,) = emit("square", [duration * ns, complex(math.cos(phase), math.sin(phase))])
            (waveform,) = emit("wf_mul", [waveform, factor])
        return waveform
    duration, amplitude, sigma, _, _ = pulse._values
    if duration is None:
        raise CompilationError(f"{pulse!r} has no duration_ns, so it does not play")
    amplitude = 1.0 if amplitude is None else amplitude
    if sigma:
        width = max(duration - 4 * sigma, 0.0)
        (waveform,) = emit("gaussian_square", [duration * ns, amplitude, sigma * ns, width * ns])
    else:
        (waveform,) = emit("square", [duration * ns, amplitude])
    return waveform
