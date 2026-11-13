# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A drifting device to calibrate against, with its true parameters recorded.

``DriftingPlant`` wraps one qubit of a target description and moves its
parameters as time advances:

- ``TelegraphT1``: T1 switches between two values at random, at a rate, as
  when a two-level system near the qubit switches.
- ``OrnsteinUhlenbeck``: a stationary Gaussian process with a correlation
  time, for the qubit's frequency offset (Hz) and its drive amplitude scale
  (relative, around 1).

``plant.description()`` is the target description at the current time; simulate
a compiled program on it with ``pulse.simulate(lowered, plant=...)``, which
keeps the program as compiled. ``plant.truth`` records every state the plant
has been in, for checking what a tracker estimated against what was true.

For loops of many iterations the surrogates give the outcome probabilities
of the standard experiments directly, without simulating: ``p_survive`` (T1:
excite, wait, measure), ``p_ramsey`` and ``p_rabi``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._errors import InvalidValueError

__all__ = ["DriftingPlant", "OrnsteinUhlenbeck", "PlantState", "TelegraphT1"]


@dataclass(frozen=True)
class TelegraphT1:
    """T1 switching between ``low_us`` and ``high_us``, at ``rate_hz`` each
    way."""

    low_us: float
    high_us: float
    rate_hz: float


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """A stationary process with standard deviation ``sigma`` and correlation
    time ``correlation_s``."""

    sigma: float
    correlation_s: float

    def step(self, x: float, dt_s: float, rng: np.random.Generator) -> float:
        """The exact update over *dt_s*."""
        decay = math.exp(-dt_s / self.correlation_s)
        return x * decay + self.sigma * math.sqrt(1 - decay * decay) * float(rng.standard_normal())


@dataclass(frozen=True)
class PlantState:
    """The true parameters of the plant's qubit at ``time_s``."""

    time_s: float
    t1_us: float
    frequency_offset_hz: float
    amplitude_scale: float


class DriftingPlant:
    """One qubit of *target* whose T1, frequency and drive amplitude drift."""

    def __init__(
        self,
        target: Any,
        qudit: int = 0,
        *,
        t1: TelegraphT1 | None = None,
        frequency: OrnsteinUhlenbeck | None = None,
        amplitude: OrnsteinUhlenbeck | None = None,
        seed: int | None = None,
    ) -> None:
        if qudit not in target.qudits:
            raise InvalidValueError(f"the target has no qubit {qudit}")
        self._target = target
        self.qudit = qudit
        self.t1 = t1
        self.frequency = frequency
        self.amplitude = amplitude
        self._rng = np.random.default_rng(seed)
        base = target.qudits[qudit]
        self._high = t1 is None or bool(self._rng.integers(2))
        self.state = PlantState(
            0.0,
            float(base.t1_us) if t1 is None else (t1.high_us if self._high else t1.low_us),
            0.0 if frequency is None else frequency.sigma * float(self._rng.standard_normal()),
            1.0
            + (0.0 if amplitude is None else amplitude.sigma * float(self._rng.standard_normal())),
        )
        self.truth: list[PlantState] = [self.state]

    def advance(self, dt_s: float) -> PlantState:
        """Move time on by *dt_s* seconds; returns the new state."""
        if dt_s < 0:
            raise InvalidValueError("time only moves forward")
        s = self.state
        t1_us = s.t1_us
        if self.t1 is not None:
            # Switches in dt_s: Poisson at rate_hz; an odd number flips it.
            flips = int(self._rng.poisson(self.t1.rate_hz * dt_s))
            if flips % 2:
                self._high = not self._high
            t1_us = self.t1.high_us if self._high else self.t1.low_us
        offset = s.frequency_offset_hz
        if self.frequency is not None:
            offset = self.frequency.step(offset, dt_s, self._rng)
        scale = s.amplitude_scale
        if self.amplitude is not None:
            scale = 1.0 + self.amplitude.step(scale - 1.0, dt_s, self._rng)
        self.state = PlantState(s.time_s + dt_s, t1_us, offset, scale)
        self.truth.append(self.state)
        return self.state

    def description(self) -> Any:
        """The target description with the qubit's current true parameters."""
        from ..targets.base import MicrowaveDrive

        base = self._target.qudits[self.qudit]
        scale = self._target.drive_amplitude_scale(self.qudit) * self.state.amplitude_scale
        drive = base.drive if base.drive is not None else MicrowaveDrive()
        qudit = base.replace(
            t1_us=self.state.t1_us,
            frequency_hz=base.frequency_hz + self.state.frequency_offset_hz,
            drive=drive.replace(amplitude_scale_rad_per_ns=scale),
        )
        qudits = dict(self._target.qudits)
        qudits[self.qudit] = qudit
        return dataclasses.replace(self._target, qudits=qudits)

    def truth_trace(self) -> dict[str, list[float]]:
        """Every recorded state, as columns (JSON-ready)."""
        return {
            "time_s": [s.time_s for s in self.truth],
            "t1_us": [s.t1_us for s in self.truth],
            "frequency_offset_hz": [s.frequency_offset_hz for s in self.truth],
            "amplitude_scale": [s.amplitude_scale for s in self.truth],
        }

    # ── Surrogates ──────────────────────────────────────────────────────────

    def p_survive(self, wait_ns: float) -> float:
        """P(still excited) after exciting and waiting *wait_ns*."""
        t1_ns = self.state.t1_us * 1e3
        return math.exp(-wait_ns / t1_ns) if t1_ns > 0 else 1.0

    def p_ramsey(self, wait_ns: float, phase: float, detuning_hz: float = 0.0) -> float:
        """P(1) of a Ramsey shot: pi/2, a wait, then pi/2 advanced by *phase*,
        with the drive detuned by *detuning_hz* from the compiled frequency;
        the plant's frequency offset adds to it. Ideal pulses, no decay."""
        total = detuning_hz + self.state.frequency_offset_hz
        return 0.5 * (1 - math.cos(2 * math.pi * total * wait_ns * 1e-9 + phase))

    def p_rabi(self, angle: float) -> float:
        """P(1) after a resonant rotation meant to be *angle*, which the
        amplitude scale stretches."""
        return math.sin(self.state.amplitude_scale * angle / 2) ** 2

    def sample(self, p: float) -> int:
        """One outcome with P(1) = *p*, from the plant's generator."""
        return int(self._rng.random() < p)
