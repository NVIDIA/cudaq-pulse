# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Kernel DSL operations for pulse programming.

These functions form the "language extension" vocabulary used as bare
names inside ``@pulse.kernel`` decorated functions.  During bytecode
tracing the decorator intercepts calls to these names and lowers them
to MLIR ops.  If called outside a kernel context they raise
``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NoReturn

from ..kernel.ir_builder import Parameter

__all__ = [
    "cosine",
    "custom",
    "custom_samples",
    "drag",
    "drive",
    "gaussian",
    "gaussian_square",
    "get_drive_line",
    "get_readout_line",
    "readout",
    "set_frequency",
    "set_phase",
    "shift_frequency",
    "shift_phase",
    "square",
    "sync",
    "tanh_ramp",
    "wait",
    "wf_add",
    "wf_mul",
    "wf_neg",
    "wf_scale",
    "wf_sub",
]

Numeric = int | float | Parameter

# ── Opaque type aliases (for IDE hover-docs, not enforced at runtime) ────


class Waveform:
    """Opaque waveform handle returned by envelope constructors."""


class Line:
    """Opaque drive or readout line handle from ``get_drive_line()``."""


class Tone:
    """Opaque tone handle for phase and frequency operations."""


class MeasurementResult:
    """Opaque measurement result from ``readout()``."""


def _require_context(op_name: str) -> NoReturn:
    raise RuntimeError(f"{op_name}() must be called inside a @cudaq_pulse.kernel function")


# ── Channel access ───────────────────────────────────────────────────────


def get_drive_line(qubit: Any) -> tuple[Line, Tone]:
    """Obtain the drive line and tone for a qubit.

    A qubit has one drive channel for the whole program: later calls, also
    from called kernels, return its current line and tone, so pulses on it
    play in sequence.

    Args:
        qubit: The qudit whose drive channel to use.

    Returns:
        A ``(line, tone)`` tuple used by ``drive()`` and phase/frequency ops.
    """
    return _require_context("get_drive_line")


def get_readout_line(qubit: Any) -> tuple[Line, Tone]:
    """Obtain the readout line and tone for a qubit.

    Like the drive channel, a qubit has one readout channel for the whole
    program.

    Args:
        qubit: The qudit whose readout channel to use.

    Returns:
        A ``(line, tone)`` tuple used by ``readout()``.
    """
    return _require_context("get_readout_line")


# ── Scheduling ops ──────────────────────────────────────────────────────


def drive(line: Line, waveform: Waveform, tone: Tone) -> None:
    """Play a waveform on a drive line.

    Args:
        line: Drive line from ``get_drive_line()``.
        waveform: Waveform envelope to play (e.g. from ``gaussian()``).
        tone: Tone handle from ``get_drive_line()``.
    """
    _require_context("drive")


def readout(line: Line, waveform: Waveform, tone: Tone) -> MeasurementResult:
    """Acquire a measurement through a readout line.

    Args:
        line: Readout line from ``get_readout_line()``.
        waveform: Readout waveform envelope.
        tone: Tone handle from ``get_readout_line()``.

    Returns:
        Measurement result handle.
    """
    _require_context("readout")


def wait(target: Line, duration: Numeric, bounds: tuple[int, int | None] | None = None) -> None:
    """Insert an idle delay on a line.

    Args:
        target: Drive or readout line.
        duration: Wait duration in clock cycles. A literal must be a whole
            number of cycles. It may be computed from ``float`` kernel
            parameters with ``+``, ``-``, ``*`` and ``/``; the kernel then
            compiles once, and each call rounds the length to the nearest
            cycle (ties up) and schedules with it. When it is computed from
            the answer of a classical call, the wait is a variable region:
            every line syncs before it, and everything after it shifts with
            its length, which is known only when the program runs.
        bounds: ``(lowest, highest)`` in clock cycles; ``highest`` may be
            ``None``. A call whose parameter-derived length falls outside
            them raises ``ValueError``; a length from an answer is rounded
            and clamped to them. Without bounds the length must not be
            negative. Scheduling without the length puts the wait at
            ``lowest``.
    """
    _require_context("wait")


def sync(*targets: Line) -> None:
    """Synchronize multiple lines to a common time point.

    All lines are padded to the latest time among them before
    subsequent operations proceed.

    Args:
        targets: Two or more drive/readout lines to synchronize.
    """
    _require_context("sync")


# ── Phase / frequency ops ───────────────────────────────────────────────


def shift_phase(tone: Tone, phase: Numeric) -> Tone:
    """Advance a tone's frame phase: a virtual Z rotation by *phase*.

    Following McKay et al., PRA 96, 022330 (2017), a pulse played after it
    turns about an axis rotated by ``-phase``.

    Args:
        tone: Tone handle (second element of ``get_drive_line()``).
        phase: Phase increment in radians.
    """
    _require_context("shift_phase")


def set_phase(tone: Tone, phase: Numeric) -> Tone:
    """Set the frame phase of a tone, with the sign of ``shift_phase``.

    Args:
        tone: Tone handle.
        phase: Absolute phase in radians.
    """
    _require_context("set_phase")


def shift_frequency(tone: Tone, frequency: Numeric) -> Tone:
    """Add a relative frequency offset to a tone.

    Args:
        tone: Tone handle.
        frequency: Frequency offset in Hz.
    """
    _require_context("shift_frequency")


def set_frequency(tone: Tone, frequency: Numeric) -> Tone:
    """Set the absolute frequency of a tone.

    Args:
        tone: Tone handle.
        frequency: Absolute frequency in Hz.
    """
    _require_context("set_frequency")


# ── Waveform constructors ───────────────────────────────────────────────


def gaussian(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a Gaussian envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Standard deviation in clock cycles.

    Returns:
        Waveform value for use with ``drive()`` or waveform arithmetic.
    """
    return _require_context("gaussian")


def square(duration: Numeric, amplitude: Numeric | complex | Sequence[float]) -> Waveform:
    """Create a flat-top (square) envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Constant amplitude in ``[-1, 1]``: a real number, a
            complex number ``re + im*1j``, or ``[re, im]``.

    Returns:
        Waveform value.
    """
    return _require_context("square")


def drag(
    duration: Numeric,
    amplitude: Numeric,
    sigma: Numeric,
    beta: Numeric,
) -> Waveform:
    """Create a DRAG (Derivative Removal by Adiabatic Gate) waveform.

    The envelope is ``G(t) + i beta dG/dt`` with ``G`` the Gaussian (Motzoi
    et al., PRL 103, 110501 (2009)). Leakage to the second excited level of
    a transmon with anharmonicity ``alpha`` is smallest near
    ``beta = -1 / alpha``, with ``alpha`` in radians per clock cycle.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Gaussian standard deviation in clock cycles.
        beta: DRAG coefficient in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("drag")


def cosine(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a raised-cosine envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.

    Returns:
        Waveform value.
    """
    return _require_context("cosine")


def tanh_ramp(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a hyperbolic-tangent ramp waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Rise/fall steepness in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("tanh_ramp")


def gaussian_square(
    duration: Numeric,
    amplitude: Numeric,
    sigma: Numeric,
    width: Numeric,
) -> Waveform:
    """Create a Gaussian-square (flat-top Gaussian) waveform.

    A square pulse with Gaussian rise and fall edges.

    Args:
        duration: Total pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Gaussian edge standard deviation in clock cycles.
        width: Flat-top width in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("gaussian_square")


def custom(duration: int, envelope_fn: Callable[[float], float]) -> Waveform:
    """Create a waveform by sampling a Python function while the kernel is traced.

    *envelope_fn* is called once per clock cycle, at the middle of each cycle:
    ``t = 0.5, 1.5, ..., duration - 0.5``, in clock cycles from the start of
    the pulse. Each value is held for its cycle, like ``custom_samples`` of
    the values (which is what the kernel's IR records), so the waveform
    compiles, schedules and simulates like the built-in envelopes. A function
    that matches a built-in envelope, such as
    ``lambda t: a * math.exp(-((t - duration / 2) ** 2) / (2 * sigma**2))``
    for ``gaussian(duration, a, sigma)``, gives the same waveform up to that
    per-cycle sampling.

    The duration must be known while tracing (an ``int`` argument or a
    constant) and at most 252 clock cycles; play a longer envelope as
    consecutive drives. The function must return real numbers that do not
    depend on kernel parameters: scale the waveform by a parameter with
    ``wf_scale(custom(duration, f), amplitude)``. Define the function outside
    the kernel: a lambda or ``def`` in a kernel body raises
    ``CompilationError``.

    Args:
        duration: Pulse duration in clock cycles.
        envelope_fn: A function ``f(t) -> float`` of the time in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("custom")


def custom_samples(samples: Sequence[float]) -> Waveform:
    """Create a waveform from pre-computed sample data.

    Args:
        samples: Non-empty array-like of real envelope sample values.

    Returns:
        Waveform value.
    """
    return _require_context("custom_samples")


# ── Waveform arithmetic ─────────────────────────────────────────────────


def wf_add(left: Waveform, right: Waveform) -> Waveform:
    """Add two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform (must have same duration as *left*).

    Returns:
        Combined waveform ``left + right``.
    """
    return _require_context("wf_add")


def wf_sub(left: Waveform, right: Waveform) -> Waveform:
    """Subtract two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Difference waveform ``left - right``.
    """
    return _require_context("wf_sub")


def wf_mul(left: Waveform, right: Waveform) -> Waveform:
    """Multiply two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Product waveform ``left * right``.
    """
    return _require_context("wf_mul")


def wf_scale(waveform: Waveform, scalar: Numeric, /) -> Waveform:
    """Scale a waveform by a constant factor.

    Args:
        waveform: Waveform to scale.
        scalar: Real scaling factor.

    Returns:
        Scaled waveform ``scalar * waveform``.
    """
    return _require_context("wf_scale")


def wf_neg(waveform: Waveform) -> Waveform:
    """Negate a waveform (flip sign of all samples).

    Args:
        waveform: Waveform to negate.

    Returns:
        Negated waveform ``-waveform``.
    """
    return _require_context("wf_neg")
