# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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

__all__ = [
    "GoldenSection",
    "NelderMead",
    "frequency_update",
    "golden_section_step",
    "nelder_mead_step",
    "ramsey_setting",
    "t1_update",
    "t1_wait",
    "three_point_decay",
    "three_point_phase",
]


# ── T1: a gamma posterior over the decay rate ───────────────────────────────


def t1_update(alpha: float, beta: float, survived: bool, wait_ns: float) -> tuple[float, float]:
    """The posterior Gamma(alpha, beta) over Gamma = 1/T1 (per ns) after one
    shot that excited the qubit, waited *wait_ns* and found it still excited
    (*survived*) or not.

    Survival multiplies the prior by exp(-Gamma t), which keeps it a gamma
    distribution: beta grows by t. A decay multiplies it by
    1 - exp(-Gamma t); the posterior is then the difference of two gamma
    distributions, and the result is the gamma distribution with its mean and
    variance.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha and beta must be positive, got {alpha}, {beta}")
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
    mean_hz: float, sigma_hz: float, outcome: int, wait_ns: float, phase: float
) -> tuple[float, float]:
    """The Gaussian posterior over a detuning after one Ramsey shot.

    The shot waits *wait_ns* between two pi/2 pulses, the second advanced by
    *phase*, and reports 1 with probability (1 - cos(2 pi D t + phase)) / 2
    for detuning D. The posterior is the Gaussian with the exact mean and
    variance of prior times likelihood.
    """
    if sigma_hz <= 0:
        raise ValueError(f"sigma_hz must be positive, got {sigma_hz}")
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
    mean_hz: float, sigma_hz: float, scale: float = 0.1, max_wait_ns: float | None = None
) -> tuple[float, float]:
    """The wait (ns) and phase of the next Ramsey shot.

    The phase puts the estimated detuning on the fringe's steepest point, so
    the outcome says which side of the estimate the detuning is on. The wait
    is *scale* / sigma. Longer waits resolve the detuning more finely but
    wash out the fringe over the prior: 0.155 narrows the posterior most per
    shot (about 20%), but then one unlucky early shot can leave it
    confidently wrong; the default 0.1 narrows it by about 14% per shot and
    stays on the detuning in all but a few runs in a thousand. *max_wait_ns*
    caps the wait (a fraction of T2*, past which the fringe is gone); once it
    binds, every shot still narrows the posterior, more slowly.
    """
    wait_ns = scale * 1e9 / sigma_hz
    if max_wait_ns is not None:
        wait_ns = min(wait_ns, max_wait_ns)
    phase = (-2 * math.pi * mean_hz * wait_ns * 1e-9 + math.pi / 2) % (2 * math.pi)
    return wait_ns, phase


# ── Three-point estimators ──────────────────────────────────────────────────


def three_point_decay(p0: float, p1: float, p2: float, spacing: float) -> float:
    """T in ``p(t) = A exp(-t / T) + B`` from p at t, t + spacing, t + 2 spacing.

    The ratio (p1 - p2) / (p0 - p1) is exp(-spacing / T) whatever A and B
    are. Points that do not decay give ``inf``.
    """
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
    (NaN until measured). ``as_floats``/``from_floats`` pack it for a
    ``Floats(7)`` argument.
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
    ``as_floats``/``from_floats`` pack it for a ``Floats`` argument of
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
        return all(abs(a - b) <= self.tolerance for v in self.simplex for a, b in zip(v, best))

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
    return tuple(sum(c) / len(points) for c in zip(*points))


def _along(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    """a + t (b - a)."""
    return tuple(x + t * (y - x) for x, y in zip(a, b))


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
