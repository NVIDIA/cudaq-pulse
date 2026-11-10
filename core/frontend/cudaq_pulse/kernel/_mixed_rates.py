# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tracer for a program whose lines play different sample rates, loaded
only for one: it records the rates each waveform's samples were made for,
and samples a waveform again for the rate of the line that plays it."""

from __future__ import annotations

from typing import Any

from ._durations import for_line, track_samples
from ._emitter import _MAKES_WAVEFORM, _OP_NAMES, _WAVEFORMS, PulseIREmitter, _current
from .ir_builder import IRValue

_LINES = ("drive_line", "readout_line")
# What _emit_pulse_call reached the choice path with: it tracks its own.
_CHOSEN = object()


class MixedRates(PulseIREmitter):
    """A ``PulseIREmitter`` for lines of different sample rates."""

    __slots__ = ("_sampled", "_made")

    # Of the waveform op being emitted: ("custom", duration, name, rate) or
    # ("op", operands, attributes), else None or _CHOSEN.
    _made: Any

    def __init__(
        self,
        builder: Any,
        fn_globals: dict[str, Any] | None = None,
        fn_closures: dict[str, Any] | None = None,
        caller: PulseIREmitter | None = None,
    ):
        super().__init__(builder, fn_globals, fn_closures, caller)
        # Waveform vid -> (the rates its samples were made for, how to make
        # it again: ("custom", duration, envelope, result types), ("op", op
        # name, operands, result types, attributes), or None when it can't).
        self._sampled: dict[int, Any]
        if caller is None:
            self._sampled = {}
        else:
            assert isinstance(caller, MixedRates)
            self._sampled = caller._sampled
        self._made = None

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
        if fname in ("drive", "readout") and len(call_args) > 1:
            line, wave = call_args[0], call_args[1]
            if line.__class__ is IRValue and line.vtype in _LINES and wave.__class__ is IRValue:
                again = for_line(self, fname, (_current(line), wave))[1]
                if again is not wave:
                    call_args = [line, again, *call_args[2:]]
        self._made = None
        results = super()._emit_pulse_call(fname, call_args)
        if fname in _MAKES_WAVEFORM:
            made, self._made = self._made, None
            if made is not _CHOSEN:
                self._track(fname, made, results[0].vid)
        return results

    def _track(self, fname: str, made: Any, vid: int) -> None:
        out_types = ("waveform",)
        if fname == "custom":
            _, duration, name, rate = made
            track_samples(self, fname, None, (), out_types, {"rate": rate}, vid, (duration, name))
        elif fname in _WAVEFORMS or made is None:
            self._sampled.pop(vid, None)
        else:
            _, operands, attrs = made
            track_samples(self, fname, _OP_NAMES[fname], operands, out_types, attrs, vid)

    def _custom_samples(self, duration: Any, attrs: dict[str, Any]) -> tuple[dict[str, Any], int]:
        made, length = super()._custom_samples(duration, attrs)
        if self._made is not _CHOSEN:
            self._made = ("custom", duration, attrs.get("name"), made.get("rate"))
        return made, length

    def _waveform_duration(
        self, fname: str, operands: tuple[IRValue, ...], attrs: dict[str, Any]
    ) -> int | None:
        duration = super()._waveform_duration(fname, operands, attrs)
        if self._made is not _CHOSEN:
            self._made = ("op", operands, attrs)
        return duration

    def _emit_chosen(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        self._made = _CHOSEN
        return super()._emit_chosen(*args, **kwargs)
