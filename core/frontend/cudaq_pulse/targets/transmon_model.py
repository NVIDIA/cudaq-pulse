# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Flux-tunable transmons and their charge coupling, from circuit values.

A transmon is a Cooper-pair box diagonalized in the charge basis,

    H = 4 E_C n^2 - E_J(Phi) cos(phi),
    E_J(Phi) = E_J,max sqrt(cos^2(pi Phi) + d^2 sin^2(pi Phi)),

with Phi in flux quanta and d the asymmetry of its SQUID (Koch et al., PRA
76, 042319 (2007); Hellings et al., arXiv:2503.04610, eq. B2). The offset
charge is omitted: for E_J / E_C > 70 its dispersion is below 1 kHz. Two
transmons couple through g_c n_a n_b, with no rotating-wave approximation;
their 0-1 exchange is J_qq = g_c ``<0|n_a|1> <0|n_b|1>``.

Energies are in Hz (E / h). Circuit values (E_C, E_J,max, d, g_c) are
fitted, so they come from a calibration, never from a target's published
values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._errors import InvalidValueError

__all__ = ["ChargeCoupledPair", "TransmonCircuit", "anchored_charge_coupling_hz"]

# Charge states -25..25: converged to < 1 Hz for E_J / E_C < 300.
CHARGE_CUTOFF = 25

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class TransmonCircuit:
    """An asymmetric-SQUID transmon; ``asymmetry`` is d = (E_J2 - E_J1) / E_JSigma.

    Flux is in flux quanta; f01 falls monotonically from Phi = 0 to Phi = 1/2,
    its two sweet spots.
    """

    ec_hz: float
    ej_max_hz: float
    asymmetry: float
    label: str = ""

    def ej_hz(self, flux: float) -> float:
        x = math.pi * flux
        return self.ej_max_hz * math.sqrt(math.cos(x) ** 2 + self.asymmetry**2 * math.sin(x) ** 2)

    def eigensystem(
        self, flux: float, n_levels: int = 5
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Energies above the ground state and the charge matrix ``<i|n|j>``.

        Signs are fixed so that ``<i|n|i+1> > 0``, which keeps n_ij continuous in
        flux.
        """
        n = np.arange(-CHARGE_CUTOFF, CHARGE_CUTOFF + 1, dtype=float)
        off = np.full(n.size - 1, -0.5 * self.ej_hz(flux))
        h = np.diag(4.0 * self.ec_hz * n**2) + np.diag(off, 1) + np.diag(off, -1)
        energies, vectors = np.linalg.eigh(h)
        energies, vectors = energies[:n_levels], vectors[:, :n_levels].copy()
        for i in range(1, n_levels):
            if vectors[:, i - 1] @ (n * vectors[:, i]) < 0:
                vectors[:, i] *= -1
        charge: npt.NDArray[np.float64] = vectors.T @ (n[:, None] * vectors)
        return energies - energies[0], charge

    def f01_hz(self, flux: float) -> float:
        return float(self.eigensystem(flux, 2)[0][1])

    def f12_hz(self, flux: float) -> float:
        e = self.eigensystem(flux, 3)[0]
        return float(e[2] - e[1])

    def anharmonicity_hz(self, flux: float) -> float:
        e = self.eigensystem(flux, 3)[0]
        return float(e[2] - 2.0 * e[1])

    def charge_matrix(self, flux: float, n_levels: int = 5) -> npt.NDArray[np.float64]:
        return self.eigensystem(flux, n_levels)[1]

    def flux_for_f01(self, f01_hz: float, tol: float = 1e-9) -> float:
        """The flux in [0, 1/2] at which f01 is ``f01_hz``."""
        return self._invert(self.f01_hz, f01_hz, tol)

    def flux_for_f12(self, f12_hz: float, tol: float = 1e-9) -> float:
        """The flux in [0, 1/2] at which the 1-2 transition is ``f12_hz``."""
        return self._invert(self.f12_hz, f12_hz, tol)

    def flux_curvature_hz(self, flux: float, step: float = 1e-3) -> float:
        """d^2 f01 / dPhi^2, the second-order flux-noise sensitivity (Hz per
        flux quantum squared)."""
        f = self.f01_hz
        return (f(flux + step) - 2.0 * f(flux) + f(flux - step)) / step**2

    def _invert(self, frequency: Callable[[float], float], target_hz: float, tol: float) -> float:
        lo, hi = 0.0, 0.5
        f_lo, f_hi = frequency(lo), frequency(hi)
        # A sweet spot's own frequency, up to rounding, is in range.
        for flux, edge in ((lo, f_lo), (hi, f_hi)):
            if abs(target_hz - edge) < 1.0:
                return flux
        if not min(f_lo, f_hi) <= target_hz <= max(f_lo, f_hi):
            raise InvalidValueError(
                f"{self.label or 'transmon'}: {target_hz:.6g} Hz is outside the tuning range "
                f"[{min(f_lo, f_hi):.6g}, {max(f_lo, f_hi):.6g}] Hz"
            )
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if (frequency(mid) - target_hz) * (f_lo - target_hz) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ChargeCoupledPair:
    """Two transmons coupled by g_c n_a n_b. Product states are ``|i_a i_b>``.

    With the data (lower-frequency) transmon as ``a`` and the auxiliary as
    ``b``, ``|02>`` is the auxiliary's second excited state: the level the CZ
    brings into resonance with ``|11>``.
    """

    a: TransmonCircuit
    b: TransmonCircuit
    g_c_hz: float
    n_levels: int = 5

    def hamiltonian(self, flux_a: float, flux_b: float) -> npt.NDArray[np.float64]:
        ea, na = self.a.eigensystem(flux_a, self.n_levels)
        eb, nb = self.b.eigensystem(flux_b, self.n_levels)
        eye = np.eye(self.n_levels)
        h: npt.NDArray[np.float64] = (
            np.kron(np.diag(ea), eye) + np.kron(eye, np.diag(eb)) + self.g_c_hz * np.kron(na, nb)
        )
        return h

    def dressed_spectrum(self, flux_a: float, flux_b: float) -> dict[tuple[int, int], float]:
        """The dressed energy of each product state ``|i j>``, by largest overlap."""
        energies, vectors = np.linalg.eigh(self.hamiltonian(flux_a, flux_b))
        owner = np.argmax(np.abs(vectors) ** 2, axis=1)
        n = self.n_levels
        return {(i, j): float(energies[owner[i * n + j]]) for i in range(n) for j in range(n)}

    def j_qq_hz(self, flux_a: float, flux_b: float) -> float:
        """The 0-1 exchange g_c ``<0|n_a|1> <0|n_b|1>``."""
        na = self.a.charge_matrix(flux_a, 2)
        nb = self.b.charge_matrix(flux_b, 2)
        return float(self.g_c_hz * na[0, 1] * nb[0, 1])

    def exchange_hz(self, f01_a_hz: float, f01_b_hz: float) -> float:
        """J_qq with the transmons tuned to these 0-1 frequencies: J(w_a, w_b)."""
        return self.j_qq_hz(self.a.flux_for_f01(f01_a_hz), self.b.flux_for_f01(f01_b_hz))

    def idle_zz_hz(self, flux_a: float, flux_b: float) -> float:
        """Static ZZ, E_11 - E_10 - E_01 + E_00 of the dressed levels."""
        e = self.dressed_spectrum(flux_a, flux_b)
        return e[(1, 1)] - e[(1, 0)] - e[(0, 1)] + e[(0, 0)]

    def gap_11_02_hz(self, flux_a: float, flux_b: float) -> float:
        """E_11 - E_02 of the dressed levels; its smallest magnitude over flux
        is 2 J_2, with J_2 the ``|11>-|02>`` coupling (about sqrt(2) J_qq)."""
        e = self.dressed_spectrum(flux_a, flux_b)
        return e[(1, 1)] - e[(0, 2)]

    def splitting_11_02_hz(self, flux_a: float, flux_b: float) -> float:
        """The splitting of the two levels that share ``|11>`` and ``|02>`` most,
        which stays defined through their avoided crossing."""
        energies, vectors = np.linalg.eigh(self.hamiltonian(flux_a, flux_b))
        n = self.n_levels
        weight = np.abs(vectors[1 * n + 1]) ** 2 + np.abs(vectors[0 * n + 2]) ** 2
        k, m = np.argsort(weight)[-2:]
        return float(abs(energies[k] - energies[m]))

    def j2_hz(self, interaction_frequency_hz: float, span: float = 0.02) -> float:
        """J_2, half the smallest ``|11>-|02>`` splitting, with ``a`` at
        ``interaction_frequency_hz`` and ``b`` swept through f12 = that
        frequency (within ``span`` flux quanta).

        If ``b``'s 1-2 transition cannot reach that frequency, ``b`` stays
        at the nearer sweet spot and ``a`` is swept through the crossing.
        """
        f12_range = sorted((self.b.f12_hz(0.0), self.b.f12_hz(0.5)))
        if f12_range[0] <= interaction_frequency_hz <= f12_range[1]:
            flux_a = self.a.flux_for_f01(interaction_frequency_hz)
            center = self.b.flux_for_f12(interaction_frequency_hz)

            def split(flux: float) -> float:
                return self.splitting_11_02_hz(flux_a, flux)

        else:
            nearer = (
                0.0
                if abs(self.b.f12_hz(0.0) - interaction_frequency_hz)
                < abs(self.b.f12_hz(0.5) - interaction_frequency_hz)
                else 0.5
            )
            center = self.a.flux_for_f01(self.b.f12_hz(nearer))

            def split(flux: float) -> float:
                return self.splitting_11_02_hz(flux, nearer)

        lo, hi = max(0.0, center - span), min(0.5, center + span)
        x1, x2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        s1, s2 = split(x1), split(x2)
        while hi - lo > 1e-10:
            if s1 < s2:
                hi, x2, s2 = x2, x1, s1
                x1 = hi - _GOLDEN * (hi - lo)
                s1 = split(x1)
            else:
                lo, x1, s1 = x1, x2, s2
                x2 = lo + _GOLDEN * (hi - lo)
                s2 = split(x2)
        return 0.5 * min(s1, s2)


def anchored_charge_coupling_hz(
    data: TransmonCircuit,
    auxiliary: TransmonCircuit,
    g_11_20_hz: float,
    interaction_frequency_hz: float,
    n_levels: int = 5,
) -> float:
    """The g_c for which the pair's ``|11>-|20>`` coupling J_2 is ``g_11_20_hz``
    at its interaction point: the data transmon at
    ``interaction_frequency_hz`` and the auxiliary's 1-2 transition there.

    J_2 is nearly linear in g_c, so a few fixed-point steps converge to
    machine precision.
    """
    g_c = g_11_20_hz / math.sqrt(2.0)
    for _ in range(8):
        j2 = ChargeCoupledPair(data, auxiliary, g_c, n_levels).j2_hz(interaction_frequency_hz)
        step = g_11_20_hz / j2
        g_c *= step
        if abs(step - 1.0) < 1e-12:
            break
    return g_c
