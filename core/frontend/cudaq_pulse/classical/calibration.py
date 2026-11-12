# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimators for calibrating while the QPU runs.

Each estimator is a plain function whose state goes in and comes back out,
so the same code is the body of a classical function a kernel calls between
measurements, runs between shots, and is tested on its own. Nothing is kept
between calls.

- ``t1_update`` and ``t1_wait``: a gamma posterior over the decay rate
  1/T1, updated from one shot of "excite, wait, measure", and the next wait.
- ``frequency_update`` and ``ramsey_setting``: a Gaussian posterior over a
  qubit's detuning, updated from one Ramsey shot, and the wait and phase of
  the next shot (a binary search: each shot says which side of the estimate
  the detuning is on).
- ``survival_probability``, ``ramsey_probability`` and ``rabi_probability``:
  the outcome probabilities of those experiments with ideal pulses, for
  drawing shots without simulating them (``rng.random() < p``).
- ``three_point_decay``: the decay time of ``A exp(-t / T) + B`` from three
  equally spaced points.
- ``three_point_phase``: the phase error ``e`` of ``a - b cos(theta + e)``
  from three equally spaced settings.
- ``golden_section_step``: one step of a golden-section search for a
  minimum; ``GoldenSection`` holds its state.
- ``nelder_mead_step``: one step of a Nelder-Mead simplex search;
  ``NelderMead`` holds its state.

The two searches are ask-and-tell: a step takes the cost measured at the
point the last step asked for and returns the next point, and a flag that is
true once the search has converged, for the converged pattern (see the user
guide).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._errors import InvalidValueError
from .._units import Time, as_nanoseconds

if TYPE_CHECKING:
    from .._op_tables import OpCounts

__all__ = [
    "GoldenSection",
    "NelderMead",
    "adaptive_t1_update",
    "adaptive_t1_wait",
    "binary_search_phase",
    "binary_search_update",
    "binary_search_wait",
    "count_ones",
    "decay_time",
    "frequency_update",
    "golden_section_step",
    "nelder_mead_step",
    "pi_train_amplitude",
    "rabi_probability",
    "ramsey_detuning",
    "ramsey_probability",
    "ramsey_setting",
    "sparse_phase",
    "survival_probability",
    "t1_update",
    "t1_wait",
    "three_point_decay",
    "three_point_phase",
]


# ── T1: a gamma posterior over the decay rate ───────────────────────────────


def t1_update(
    alpha: float, beta: float, survived: bool, wait_ns: float | Time
) -> tuple[float, float]:
    """The posterior Gamma(alpha, beta) over Gamma = 1/T1 (per ns) after one
    shot that excited the qubit, waited *wait_ns* (a time, or a number of
    ns) and found it still excited (*survived*) or not.

    Survival multiplies the prior by exp(-Gamma t), which keeps it a gamma
    distribution: beta grows by t. A decay multiplies it by
    1 - exp(-Gamma t); the posterior is then the difference of two gamma
    distributions, and the result is the gamma distribution with its mean and
    variance.
    """
    if alpha <= 0 or beta <= 0:
        raise InvalidValueError(f"alpha and beta must be positive, got {alpha}, {beta}")
    t = max(float(wait_ns), 0.0)
    if survived:
        return alpha, beta + t
    if t == 0.0:
        return alpha, beta
    # Moments of Gamma(alpha, b): E[G] = alpha / b, E[G^2] = alpha (alpha + 1) / b^2.
    c = (beta / (beta + t)) ** alpha
    if c >= 1.0:
        return alpha, beta
    m1 = (alpha / beta - c * alpha / (beta + t)) / (1.0 - c)
    m2 = (alpha * (alpha + 1) / beta**2 - c * alpha * (alpha + 1) / (beta + t) ** 2) / (1.0 - c)
    variance = m2 - m1 * m1
    if not variance > 0:
        return alpha + 1.0, beta
    return m1 * m1 / variance, m1 / variance


def t1_wait(alpha: float, beta: float, scale: float = 1.0) -> float:
    """The next wait: *scale* times the estimated T1 (beta / alpha, in the
    units of beta)."""
    return scale * beta / alpha


# ── Frequency: a Gaussian posterior over the detuning ───────────────────────


def frequency_update(
    mean_hz: float, sigma_hz: float, outcome: int, wait_ns: float | Time, phase: float
) -> tuple[float, float]:
    """The Gaussian posterior over a detuning after one Ramsey shot.

    The shot waits *wait_ns* (a time, or a number of ns) between two pi/2
    pulses, the second advanced by
    *phase*, and reports 1 with probability (1 - cos(2 pi D t + phase)) / 2
    for detuning D. The posterior is the Gaussian with the exact mean and
    variance of prior times likelihood.
    """
    if sigma_hz <= 0:
        raise InvalidValueError(f"sigma_hz must be positive, got {sigma_hz}")
    wait_ns = as_nanoseconds(wait_ns)
    k = 2 * math.pi * wait_ns * 1e-9
    m, s2 = k * mean_hz, (k * sigma_hz) ** 2
    sign = -1.0 if outcome else 1.0
    damp = math.exp(-s2 / 2)
    # E[e^{i(x + phase)}], E[x e^{i(x + phase)}] and E[x^2 e^{i(x + phase)}]
    # for x ~ N(m, s2).
    if k == 0:
        return mean_hz, sigma_hz
    c, s = math.cos(m + phase) * damp, math.sin(m + phase) * damp
    # (m + i s2) e^{i(m + phase)} and ((m + i s2)^2 + s2) e^{i(m + phase)},
    # real parts.
    first = m * c - s2 * s
    second = (m * m - s2 * s2 + s2) * c - 2 * m * s2 * s
    z = 0.5 * (1 + sign * c)
    if not z > 1e-300:
        return mean_hz, sigma_hz
    ex = 0.5 * (m + sign * first) / z
    ex2 = 0.5 * (m * m + s2 + sign * second) / z
    variance = ex2 - ex * ex
    if not (math.isfinite(ex) and math.isfinite(variance) and variance > 0):
        return mean_hz, sigma_hz
    return ex / k, math.sqrt(variance) / k


def ramsey_setting(
    mean_hz: float,
    sigma_hz: float,
    scale: float = 0.1,
    max_wait_ns: float | Time | None = None,
) -> tuple[float, float]:
    """The wait (ns) and phase of the next Ramsey shot.

    The phase puts the estimated detuning on the fringe's steepest point, so
    the outcome says which side of the estimate the detuning is on. The wait
    is *scale* / sigma. Longer waits resolve the detuning more finely but
    wash out the fringe over the prior: 0.155 narrows the posterior most per
    shot (about 20%), but then one unlucky early shot can leave it
    confidently wrong; the default 0.1 narrows it by about 14% per shot and
    stays on the detuning in all but a few runs in a thousand. *max_wait_ns*
    caps the wait (a fraction of T2*, past which the fringe is gone; a time,
    or a number of ns); once it binds, every shot still narrows the
    posterior, more slowly.
    """
    max_wait_ns = as_nanoseconds(max_wait_ns)
    wait_ns = scale * 1e9 / sigma_hz
    if max_wait_ns is not None:
        wait_ns = min(wait_ns, max_wait_ns)
    phase = (-2 * math.pi * mean_hz * wait_ns * 1e-9 + math.pi / 2) % (2 * math.pi)
    return wait_ns, phase


# ── Outcome probabilities ───────────────────────────────────────────────────


def survival_probability(wait_ns: float | Time, t1_us: float) -> float:
    """P(still excited) after exciting a qudit of *t1_us* and waiting
    *wait_ns* (a time, or a number of ns): exp(-wait / T1), the likelihood
    ``t1_update`` uses. A T1 of 0 is no decay."""
    t1_ns = t1_us * 1e3
    return math.exp(-float(wait_ns) / t1_ns) if t1_ns > 0 else 1.0


def ramsey_probability(wait_ns: float | Time, phase: float, detuning_hz: float) -> float:
    """P(1) of a Ramsey shot that waits *wait_ns* (a time, or a number of
    ns) between two pi/2 pulses, the second advanced by *phase*, on a qudit
    *detuning_hz* off the drive: (1 - cos(2 pi D t + phase)) / 2, the
    likelihood ``frequency_update`` uses. Ideal pulses, no decay."""
    return 0.5 * (1 - math.cos(math.tau * detuning_hz * float(wait_ns) * 1e-9 + phase))


def rabi_probability(angle: float, amplitude_scale: float) -> float:
    """P(1) after a resonant rotation meant to be *angle* (rad), on a qudit
    whose drive is *amplitude_scale* times the calibrated one: sin(scale x
    angle / 2) ** 2."""
    return math.sin(amplitude_scale * angle / 2) ** 2


# ── Three-point estimators ──────────────────────────────────────────────────


def three_point_decay(p0: float, p1: float, p2: float, spacing: float | Time) -> float:
    """T in ``p(t) = A exp(-t / T) + B`` from p at t, t + spacing, t + 2 spacing.

    The ratio (p1 - p2) / (p0 - p1) is exp(-spacing / T) whatever A and B
    are. Points that do not decay give ``inf``. T is in the unit of
    *spacing*: ns when it is a time.
    """
    spacing = as_nanoseconds(spacing)
    if p0 == p1:
        return math.inf
    ratio = (p1 - p2) / (p0 - p1)
    if not 0 < ratio < 1:
        return math.inf
    return -spacing / math.log(ratio)


def three_point_phase(
    p_minus: float, p_zero: float, p_plus: float, theta: float, delta: float
) -> float:
    """The phase error e in ``p = a - b cos(theta + e)``, from p measured at
    theta - delta, theta and theta + delta (radians), whatever a and b are.

    p+ - p- = 2 b sin(delta) sin(theta + e) and
    p+ + p- - 2 p0 = 2 b (1 - cos(delta)) cos(theta + e).
    """
    difference = p_plus - p_minus
    curvature = p_plus + p_minus - 2 * p_zero
    angle = math.atan2(difference * (1 - math.cos(delta)), curvature * math.sin(delta))
    error = angle - theta
    return (error + math.pi) % (2 * math.pi) - math.pi


# ── Golden-section search ───────────────────────────────────────────────────

_INVPHI = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class GoldenSection:
    """The state of a golden-section search on [lo, hi].

    ``c`` and ``d`` are the interior points, ``fc`` and ``fd`` their costs
    (NaN until measured). ``as_floats``/``from_floats`` pack it into the 7
    values of a ``list[float]`` argument.
    """

    lo: float
    hi: float
    c: float
    d: float
    fc: float
    fd: float
    tolerance: float

    @classmethod
    def start(cls, lo: float, hi: float, tolerance: float) -> GoldenSection:
        c = hi - _INVPHI * (hi - lo)
        d = lo + _INVPHI * (hi - lo)
        return cls(lo, hi, c, d, math.nan, math.nan, tolerance)

    def next_point(self) -> float:
        """The point whose cost the search needs next."""
        return self.c if math.isnan(self.fc) else self.d

    def converged(self) -> bool:
        return self.hi - self.lo <= self.tolerance

    def best(self) -> float:
        """The current estimate of the minimum."""
        return (self.lo + self.hi) / 2

    def as_floats(self) -> list[float]:
        return [self.lo, self.hi, self.c, self.d, self.fc, self.fd, self.tolerance]

    @classmethod
    def from_floats(cls, values: Sequence[float]) -> GoldenSection:
        return cls(*(float(v) for v in values[:7]))


def golden_section_step(state: GoldenSection, cost: float) -> tuple[GoldenSection, float, bool]:
    """Tell the search *cost* at ``state.next_point()``; returns the new
    state, the point to measure next and whether the search has converged.
    """
    if state.converged():
        return state, state.best(), True
    if math.isnan(state.fc):
        state = GoldenSection(state.lo, state.hi, state.c, state.d, cost, state.fd, state.tolerance)
    else:
        state = GoldenSection(state.lo, state.hi, state.c, state.d, state.fc, cost, state.tolerance)
    if math.isnan(state.fd):
        return state, state.d, False
    lo, hi, c, d, fc, fd = state.lo, state.hi, state.c, state.d, state.fc, state.fd
    if fc < fd:
        hi, d, fd = d, c, fc
        c = hi - _INVPHI * (hi - lo)
        state = GoldenSection(lo, hi, c, d, math.nan, fd, state.tolerance)
    else:
        lo, c, fc = c, d, fd
        d = lo + _INVPHI * (hi - lo)
        state = GoldenSection(lo, hi, c, d, fc, math.nan, state.tolerance)
    done = state.converged()
    return state, state.best() if done else state.next_point(), done


# ── Nelder-Mead simplex search ──────────────────────────────────────────────

# The stages of a step: what the point asked for last is.
_INIT, _REFLECT, _EXPAND, _CONTRACT, _SHRINK = range(5)


@dataclass(frozen=True)
class NelderMead:
    """The state of a Nelder-Mead search in ``dim`` dimensions.

    ``simplex`` holds dim + 1 vertices and ``costs`` their costs (NaN until
    measured). ``stage`` says what the point asked for last is: a vertex of
    the start (``index``), the reflection, the expansion, the contraction,
    or a vertex of a shrink. ``trial`` and ``trial_cost`` keep the
    reflection while its expansion or contraction is measured.
    ``as_floats``/``from_floats`` pack it into a ``list[float]`` argument of
    ``NelderMead.size(dim)`` values.
    """

    simplex: tuple[tuple[float, ...], ...]
    costs: tuple[float, ...]
    stage: int
    index: int
    trial: tuple[float, ...]
    trial_cost: float
    tolerance: float

    @classmethod
    def start(cls, x0: Sequence[float], step: float, tolerance: float) -> NelderMead:
        dim = len(x0)
        vertices = [tuple(float(v) for v in x0)]
        for i in range(dim):
            v = list(vertices[0])
            v[i] += step
            vertices.append(tuple(v))
        return cls(
            tuple(vertices),
            (math.nan,) * (dim + 1),
            _INIT,
            0,
            tuple(vertices[0]),
            math.nan,
            tolerance,
        )

    @property
    def dim(self) -> int:
        return len(self.simplex) - 1

    @staticmethod
    def size(dim: int) -> int:
        """How many floats ``as_floats`` gives for *dim* dimensions."""
        return (dim + 1) * dim + (dim + 1) + 3 + dim + 1

    def next_point(self) -> tuple[float, ...]:
        if self.stage in (_INIT, _SHRINK):
            return self.simplex[self.index]
        return self.trial

    def converged(self) -> bool:
        """Every vertex is within ``tolerance`` of the best, in every
        coordinate."""
        if any(math.isnan(c) for c in self.costs):
            return False
        best = self.best()
        return all(
            abs(a - b) <= self.tolerance
            for v in self.simplex
            for a, b in zip(v, best, strict=False)
        )

    def best(self) -> tuple[float, ...]:
        i = min(range(len(self.costs)), key=lambda k: (math.isnan(self.costs[k]), self.costs[k]))
        return self.simplex[i]

    def as_floats(self) -> list[float]:
        flat = [x for v in self.simplex for x in v]
        return [
            *flat,
            *self.costs,
            float(self.stage),
            float(self.index),
            self.tolerance,
            *self.trial,
            self.trial_cost,
        ]

    @classmethod
    def from_floats(cls, values: Sequence[float], dim: int) -> NelderMead:
        v = [float(x) for x in values]
        n = dim + 1
        simplex = tuple(tuple(v[i * dim : (i + 1) * dim]) for i in range(n))
        at = n * dim
        costs = tuple(v[at : at + n])
        at += n
        stage, index, tolerance = int(v[at]), int(v[at + 1]), v[at + 2]
        at += 3
        trial = tuple(v[at : at + dim])
        return cls(simplex, costs, stage, index, trial, v[at + dim], tolerance)


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(sum(c) / len(points) for c in zip(*points, strict=False))


def _along(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    """a + t (b - a)."""
    return tuple(x + t * (y - x) for x, y in zip(a, b, strict=False))


def _ordered(state: NelderMead) -> NelderMead:
    order = sorted(range(len(state.costs)), key=lambda k: state.costs[k])
    return NelderMead(
        tuple(state.simplex[k] for k in order),
        tuple(state.costs[k] for k in order),
        state.stage,
        state.index,
        state.trial,
        state.trial_cost,
        state.tolerance,
    )


def _reflect(state: NelderMead) -> NelderMead:
    state = _ordered(state)
    centroid = _centroid(state.simplex[:-1])
    trial = _along(state.simplex[-1], centroid, 2.0)
    return NelderMead(state.simplex, state.costs, _REFLECT, 0, trial, math.nan, state.tolerance)


def _replace_worst(state: NelderMead, point: Sequence[float], cost: float) -> NelderMead:
    return NelderMead(
        (*state.simplex[:-1], tuple(point)),
        (*state.costs[:-1], cost),
        state.stage,
        0,
        state.trial,
        math.nan,
        state.tolerance,
    )


def nelder_mead_step(state: NelderMead, cost: float) -> tuple[NelderMead, tuple[float, ...], bool]:
    """Tell the search *cost* at ``state.next_point()``; returns the new
    state, the point to measure next and whether the search has converged.

    Standard coefficients: reflection 1, expansion 2, contraction 1/2,
    shrink 1/2.
    """
    if state.stage in (_INIT, _SHRINK):
        costs = list(state.costs)
        costs[state.index] = cost
        state = NelderMead(
            state.simplex,
            tuple(costs),
            state.stage,
            state.index + 1,
            state.trial,
            state.trial_cost,
            state.tolerance,
        )
        if state.index <= state.dim:
            return state, state.next_point(), False
        state = _reflect(state)
    elif state.stage == _REFLECT:
        best, second_worst, worst = state.costs[0], state.costs[-2], state.costs[-1]
        if cost < best:
            centroid = _centroid(state.simplex[:-1])
            expansion = _along(state.simplex[-1], centroid, 3.0)
            state = NelderMead(
                state.simplex, state.costs, _EXPAND, 0, expansion, cost, state.tolerance
            )
            # The reflection is kept in trial_cost; the expansion point is
            # asked for, and the reflection recomputed from the centroid.
            return state, expansion, False
        if cost < second_worst:
            state = _reflect(_replace_worst(state, state.trial, cost))
        else:
            centroid = _centroid(state.simplex[:-1])
            if cost < worst:
                contraction = _along(centroid, state.trial, 0.5)
            else:
                contraction = _along(centroid, state.simplex[-1], 0.5)
            threshold = min(cost, worst)
            state = NelderMead(
                state.simplex, state.costs, _CONTRACT, 0, contraction, threshold, state.tolerance
            )
            return state, contraction, False
    elif state.stage == _EXPAND:
        centroid = _centroid(state.simplex[:-1])
        reflection = _along(state.simplex[-1], centroid, 2.0)
        if cost < state.trial_cost:
            state = _reflect(_replace_worst(state, state.trial, cost))
        else:
            state = _reflect(_replace_worst(state, reflection, state.trial_cost))
    else:  # contraction
        if cost < state.trial_cost:
            state = _reflect(_replace_worst(state, state.trial, cost))
        else:
            anchor = state.simplex[0]
            shrunk = (anchor, *(_along(anchor, v, 0.5) for v in state.simplex[1:]))
            state = NelderMead(
                shrunk,
                (state.costs[0],) + (math.nan,) * state.dim,
                _SHRINK,
                1,
                state.trial,
                math.nan,
                state.tolerance,
            )
            return state, state.next_point(), False
    done = state.converged()
    return state, state.best() if done else state.next_point(), done


# ── The millisecond calibration protocol, as a processor runs it ────────────
#
# The estimators below are those of on-controller calibration: the analytical
# decay and sparse phase estimators of M. A. Marciniak et al.,
# "Millisecond-Scale Calibration and Benchmarking of Superconducting Qubits",
# arXiv:2602.11912 (2026); the adaptive T1 tracker of F. Berritta et al.,
# "Real-time adaptive tracking of fluctuating relaxation rates in
# superconducting qubits", arXiv:2506.09576 (2025); and the frequency binary
# search of F. Berritta et al., "Efficient Qubit Calibration by Binary-Search
# Hamiltonian Tracking", arXiv:2501.05386 (2025). They are written so that
# a controller's processor can run them as they are
# (``Processor.usage`` reports it): a fixed number of scalars
# of state, closed-form arithmetic (+ - * /, comparisons, sqrt, exp, log,
# atan2, cos), branches and loops of a trip count fixed when the program
# compiles, and no allocation or exceptions. An estimate that cannot be formed
# is reported through a result, not raised.


def count_ones(bits: Sequence[int]) -> int:
    """How many of *bits* are 1. Measurement results arrive bit-packed, so
    a controller counts them a 32-bit word at a time."""
    total = 0
    for bit in bits:
        total += bit
    return total


def _count_ones_cost(length: int) -> OpCounts:
    from .._op_tables import OpCounts

    words = -(-length // 32)
    return OpCounts(popcount=words, add=words)


count_ones.controller_cost = _count_ones_cost  # type: ignore[attr-defined]


def adaptive_t1_update(
    shape: float,
    rate: float,
    excited: int,
    wait: float | Time,
    error_1to0: float,
    error_0to1: float,
) -> tuple[float, float]:
    """The gamma posterior over Gamma = 1/T1 after one "excite, wait *wait*,
    measure" shot that read 1 (*excited*) or 0, through a readout that reads
    ``|1>`` as 0 with probability *error_1to0* and ``|0>`` as 1 with *error_0to1*.

    The prior is Gamma(shape k, rate theta), in the time unit of *wait*
    (ns when it is a time); the
    likelihood of reading 1 is ``error_0to1 + (1 - error_1to0 - error_0to1)
    exp(-Gamma wait)``. The posterior's exact mean and variance give the new
    gamma distribution (moment matching), as in Berritta et al.; with
    r = theta / (theta + wait), E[Gamma^n exp(-Gamma wait)] is
    k (k + 1) ... (k + n - 1) r^k / (theta + wait)^n.
    """
    wait = as_nanoseconds(wait)
    contrast = 1.0 - error_1to0 - error_0to1
    inverse = 1.0 / (rate + wait)
    decay = math.exp(shape * math.log(rate * inverse))
    first = shape * inverse * decay
    second = (shape + 1.0) * inverse * first
    mean_prior = shape / rate
    square_prior = mean_prior * (shape + 1.0) / rate
    if excited:
        base = error_0to1
        weight = contrast
    else:
        base = 1.0 - error_0to1
        weight = -contrast
    norm = base + weight * decay
    mean = (base * mean_prior + weight * first) / norm
    variance = (base * square_prior + weight * second) / norm - mean * mean
    variance = max(variance, mean * mean * 1e-12)
    return mean * mean / variance, mean / variance


def adaptive_t1_wait(shape: float, rate: float, scale: float = 0.51) -> float:
    """The next wait: *scale* times the estimated T1 (rate / shape). The
    adaptive T1 tracker uses a fixed scale of 0.51."""
    return scale * rate / shape


def binary_search_update(
    mean_hz: float,
    sigma_hz: float,
    outcome: int,
    wait_ns: float | Time,
    offset: float = -0.02,
    contrast: float = 0.6,
    coherence_ns: float | Time = 10_000.0,
) -> tuple[float, float]:
    """The Gaussian posterior over a detuning after one shot of the
    frequency binary search: a Ramsey shot of wait *wait_ns* set (by its
    phase, ``binary_search_phase``) to the fringe's steepest point at the
    estimate, reading *outcome* +1 or -1.

    The likelihood of +1 is ``1/2 + (offset + contrast exp(-t / T)
    sin(2 pi (D - mean) t)) / 2`` with T = *coherence_ns*, and the posterior
    is the Gaussian of its exact moments (Berritta et al. 2025); a +1 moves
    the estimate up. The wait and T are times, or numbers of ns.
    """
    wait_ns = as_nanoseconds(wait_ns)
    coherence_ns = as_nanoseconds(coherence_ns)
    t = wait_ns * 1e-9
    variance = sigma_hz * sigma_hz
    damp = math.exp(-wait_ns / coherence_ns - 2.0 * math.pi * math.pi * variance * t * t)
    norm = 1.0 + outcome * offset
    mean = mean_hz + 2.0 * math.pi * outcome * contrast * variance * t * damp / norm
    shrink = 2.0 * math.pi * contrast * variance * t * damp / norm
    variance = max(variance - shrink * shrink, 1.0)
    return mean, math.sqrt(variance)


def binary_search_wait(sigma_hz: float, coherence_ns: float | Time = 10_000.0) -> float:
    """The wait (ns) of the next binary-search shot, from the posterior
    width: ``(sqrt(16 pi^2 s^2 + 1/T^2) - 1/T) / (8 pi^2 s^2)``, for T a
    time or a number of ns."""
    coherence_ns = as_nanoseconds(coherence_ns)
    variance = sigma_hz * sigma_hz
    inverse_t = 1e9 / coherence_ns
    root = math.sqrt(16.0 * math.pi * math.pi * variance + inverse_t * inverse_t)
    wait_s = (root - inverse_t) / (8.0 * math.pi * math.pi * variance)
    return wait_s * 1e9


def binary_search_phase(mean_hz: float, wait_ns: float | Time) -> float:
    """The phase of a Ramsey shot (as for ``frequency_update``: it reports 1
    with probability (1 - cos(2 pi D t + phase)) / 2) that puts the estimate
    *mean_hz* on the fringe's steepest point, so that a detuning above it
    makes 1 more likely: the Ramsey of detuning 1 / (4 t) + mean. The wait
    is a time, or a number of ns."""
    wait_ns = as_nanoseconds(wait_ns)
    turn = 2.0 * math.pi
    return (0.5 * math.pi - turn * mean_hz * wait_ns * 1e-9) % turn


def decay_time(
    p0: float, p1: float, p3: float, spacing: float | Time, shots_count: int
) -> tuple[float, float, bool]:
    """The analytical decay estimator: the decay time T of
    ``A exp(-t / T) + C`` from its values at t0, t0 + spacing and
    t0 + 3 spacing, whatever A and C are (Marciniak et al., Eq. 1).

    With c = (p3 - p0) / (p1 - p0) = x^2 + x + 1 and x = exp(-spacing / T),
    x = sqrt(c - 3/4) - 1/2. Returns T, its standard deviation from the
    binomial noise of *shots_count* shots per point (error propagation), and
    whether an estimate was formed (1 < c < 3). T is in the unit of
    *spacing*: ns when it is a time.
    """
    spacing = as_nanoseconds(spacing)
    d1 = p1 - p0
    d3 = p3 - p0
    if d1 == 0.0:
        return 0.0, 0.0, False
    c = d3 / d1
    if not (1.0 < c < 3.0):
        return 0.0, 0.0, False
    root = math.sqrt(c - 0.75)
    x = root - 0.5
    log_x = math.log(x)
    t = -spacing / log_x
    inverse = 1.0 / d1
    dc0 = (p3 - p1) * inverse * inverse
    dc1 = -d3 * inverse * inverse
    dc3 = inverse
    variance_c = (
        dc0 * dc0 * p0 * (1.0 - p0) + dc1 * dc1 * p1 * (1.0 - p1) + dc3 * dc3 * p3 * (1.0 - p3)
    ) / shots_count
    slope = spacing / (x * log_x * log_x) * 0.5 / root
    return t, abs(slope) * math.sqrt(variance_c), True


def sparse_phase(
    p_minus: float, p_zero: float, p_plus: float, shots_count: int
) -> tuple[float, float]:
    """The sparse phase estimator: the phase theta0 of ``A cos(theta) + C``
    from its values at theta0 - pi/2, theta0 and theta0 + pi/2, whatever A
    and C are: atan2(p- - p+, 2 (p0 - (p- + p+) / 2)) (Marciniak et al.,
    Eq. 3; theta0 + pi for A < 0). Returns it and its standard deviation
    from the binomial noise of *shots_count* shots per point."""
    y = p_minus - p_plus
    x = 2.0 * p_zero - p_minus - p_plus
    theta = math.atan2(y, x)
    v_minus = p_minus * (1.0 - p_minus) / shots_count
    v_zero = p_zero * (1.0 - p_zero) / shots_count
    v_plus = p_plus * (1.0 - p_plus) / shots_count
    r2 = x * x + y * y
    if r2 == 0.0:
        return theta, math.pi
    variance = (
        x * x * (v_minus + v_plus)
        + y * y * (4.0 * v_zero + v_minus + v_plus)
        - 2.0 * x * y * (v_plus - v_minus)
    ) / (r2 * r2)
    return theta, math.sqrt(variance)


def pi_train_amplitude(amplitude: float, theta: float, turns: int) -> float:
    """The corrected amplitude of a pulse from the sparse phase estimate
    *theta* of a train of its pulses that rotates by *turns* (odd) times pi
    when the amplitude is right, sampled at amplitude scales 1 - 1/(2 turns),
    1 and 1 + 1/(2 turns) and read out as P(1).

    P(1) = (1 - cos R) / 2 for the train's rotation R, so theta = R + pi
    (mod 2 pi) and is 0 when R = turns pi. A pi pulse makes a train of
    *turns* pulses; a pi/2 pulse a train of 2 *turns* (Marciniak et al.,
    Eq. 4, applying the pi/2 pulse twice).
    """
    target = turns * math.pi
    return amplitude * target / (target + theta)


def ramsey_detuning(mean_hz: float, theta: float, wait_ns: float | Time) -> float:
    """The detuning from the sparse phase estimate *theta* of Ramsey shots
    of wait *wait_ns* at phases phi0 - pi/2, phi0 and phi0 + pi/2, with
    phi0 = -2 pi mean t (the convention of ``frequency_update``: P(1) =
    (1 - cos(2 pi D t + phase)) / 2): D = mean + (theta - pi) / (2 pi t),
    the phase taken in (-pi, pi]. The wait is a time, or a number of ns."""
    wait_ns = as_nanoseconds(wait_ns)
    error = theta - math.pi
    if error <= -math.pi:
        error += 2.0 * math.pi
    return mean_hz + error / (2.0 * math.pi * wait_ns * 1e-9)
