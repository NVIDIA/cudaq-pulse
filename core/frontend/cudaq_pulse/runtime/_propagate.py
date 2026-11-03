# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Propagators of the CPU reference integrator.

A segment's Lindbladian is a static part plus drive terms whose coefficients
come from envelope callbacks. Over a stretch of steps where every coefficient
is constant (an idle wait, or a square pulse in its own frame) the
Lindbladian is constant, and the stretch is propagated exactly by the
exponential of its superoperator, at a cost independent of its length. Other
stretches step with RK4 as before; on a small register the RK4 map of a
stretch is built once, on every basis matrix at the same time, and reused
wherever the same stretch recurs (the same pulse in another shot). RK4 is
linear in the state, so the reused map gives what stepping would.

Superoperators act on the row-major flattening of the density matrix:
``vec(A rho B) = (A kron B^T) vec(rho)``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# A register of at most this many levels gets the RK4 map of a driven
# stretch built and kept; its superoperator has (levels)^2 rows.
MAX_MAPPED_DIM = 4
# Kept maps (each (levels)^4 complex numbers): at most 16 MB.
_CACHE_ENTRIES = 4096

_cache: dict[Any, npt.NDArray[np.complex128]] = {}
_cache_lock = threading.Lock()

# Coefficients of the degree-13 Pade approximant (Higham 2005).
_PADE13 = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)
_THETA13 = 5.371920351148152


def expm(a: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """The matrix exponential, by scaling and squaring with a degree-13 Pade
    approximant."""
    norm = float(np.abs(a).sum(axis=0).max()) if a.size else 0.0
    squarings = max(0, math.ceil(math.log2(norm / _THETA13))) if norm > _THETA13 else 0
    a = a / (2.0**squarings)
    b = _PADE13
    ident = np.eye(a.shape[0], dtype=a.dtype)
    a2 = a @ a
    a4 = a2 @ a2
    a6 = a4 @ a2
    odd = a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
    u = a @ (odd + b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * ident)
    even = a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
    v = even + b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * ident
    r: npt.NDArray[np.complex128] = np.linalg.solve(v - u, v + u)
    for _ in range(squarings):
        r = r @ r
    return r


def kron(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """``np.kron`` of two matrices: the same products, without its handling
    of arrays of any rank, which costs more than they do."""
    rows, columns = a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]
    out: npt.NDArray[Any] = (a[:, None, :, None] * b[None, :, None, :]).reshape(rows, columns)
    return out


def cached(key: Any) -> npt.NDArray[np.complex128] | None:
    with _cache_lock:
        return _cache.get(key)


def keep(key: Any, value: npt.NDArray[np.complex128]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_ENTRIES:
            for old in list(_cache)[: _CACHE_ENTRIES // 2]:
                del _cache[old]
        _cache[key] = value


def clear_cache() -> None:
    """Forget every kept propagator."""
    with _cache_lock:
        _cache.clear()


def support(params: Sequence[float]) -> tuple[float, float]:
    """The interval [start, end) outside which a drive callback with these
    parameters is zero (see WaveformCallback.h: start, duration, ..., and for
    a train its levels' periods and counts after the 10-entry header)."""
    start, duration = float(params[0]), float(params[1])
    end = start + max(duration, 0.0)
    if len(params) > 10:
        levels = int(params[10]) if math.isfinite(params[10]) else 0
        if 0 < levels and len(params) >= 11 + 3 * levels:
            for level in range(levels):
                period = float(params[11 + 3 * level])
                count = float(params[12 + 3 * level])
                if period > 0 and count >= 1:
                    end += (count - 1) * period
    return start, end


def dissipator(
    jumps: Sequence[npt.NDArray[np.complex128]], dim: int, key: bytes
) -> npt.NDArray[np.complex128]:
    """The superoperator of sum_c (c rho c^+ - {c^+ c, rho} / 2); *key*
    identifies the jumps, for keeping it."""
    kept = cached(("dissipator", dim, key))
    if kept is not None:
        return kept
    ident = np.eye(dim, dtype=np.complex128)
    out = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
    for c in jumps:
        out += kron(c, c.conj())
        decay = c.conj().T @ c
        out -= 0.5 * (kron(decay, ident) + kron(ident, decay.T))
    keep(("dissipator", dim, key), out)
    return out


def liouvillian(
    hamiltonian: npt.NDArray[np.complex128], dissipative: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """The superoperator of -i [H, rho] plus *dissipative*."""
    dim = hamiltonian.shape[0]
    ident = np.eye(dim, dtype=np.complex128)
    out: npt.NDArray[np.complex128] = -1j * (kron(hamiltonian, ident) - kron(ident, hamiltonian.T))
    return out + dissipative


def apply(
    propagator: npt.NDArray[np.complex128], rho: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    dim = rho.shape[0]
    out: npt.NDArray[np.complex128] = (propagator @ rho.reshape(-1)).reshape(dim, dim)
    return out


def commutators(operators: Any) -> Any:
    """The superoperators of -i [op, rho], one per operator (stacked)."""
    dim = operators.shape[-1]
    ident = np.eye(dim, dtype=np.complex128)
    return np.stack([-1j * (kron(op, ident) - kron(ident, op.T)) for op in operators]).reshape(
        len(operators), dim * dim, dim * dim
    )


# Steps whose maps are built at once, bounding memory to a few MB.
_CHUNK = 512


def rk4_map(
    static: npt.NDArray[np.complex128], terms: Any, values: Any, h: float
) -> npt.NDArray[np.complex128]:
    """The superoperator of RK4 steps of length *h* under the Lindbladian
    ``static + sum_k values[k] terms[k]`` (superoperators), *values* holding
    each term's coefficient at the start, middle and end of each step
    (terms x steps x 3).

    RK4 is linear in the state, so each step is the map
    I + h/6 (k1 + 2 k2 + 2 k3 + k4) of the generators at its three times;
    the maps of all steps are built at once and multiplied pairwise."""
    size = static.shape[0]
    ident = np.eye(size, dtype=np.complex128)
    count = values.shape[1]
    total: npt.NDArray[np.complex128] | None = None
    for lo in range(0, count, _CHUNK):
        part = values[:, lo : lo + _CHUNK, :]
        n = part.shape[1]
        flat = part.reshape(len(terms), n * 3).T
        generators = (static + np.tensordot(flat, terms, axes=1)).reshape(n, 3, size, size)
        l0, lm, l1 = generators[:, 0], generators[:, 1], generators[:, 2]
        k2 = lm @ (ident + (0.5 * h) * l0)
        k3 = lm @ (ident + (0.5 * h) * k2)
        k4 = l1 @ (ident + h * k3)
        maps = ident + (h / 6.0) * (l0 + 2.0 * k2 + 2.0 * k3 + k4)
        # Later steps act on the left.
        while len(maps) > 1:
            half = len(maps) // 2
            paired = maps[1 : 2 * half : 2] @ maps[0 : 2 * half : 2]
            maps = np.concatenate([paired, maps[2 * half :]]) if len(maps) % 2 else paired
        total = maps[0] if total is None else maps[0] @ total
    assert total is not None
    return total
