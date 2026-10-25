# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Device calibrations: fitted circuit values, kept apart from published targets.

A calibration file is JSON whose leaves are ``{value, units, kind, source,
uncertainty, cooldown[, method]}`` objects: ``kind`` is ``published``,
``digitized`` or ``fitted``, ``source`` cites where the value comes from and
``cooldown`` names one of the file's ``cooldowns``. ``Calibration.load``
reads one, a shipped name (``"krinner17"``) or a path, for one cooldown.

A calibration gives each flux-tunable qubit its circuit (E_C, E_J,max and
SQUID asymmetry d, so flux in flux quanta converts to frequency and back)
and each coupled pair its charge coupling g_c, from which the exchange
J(w_a, w_d) and the idle ZZ follow at any operating point.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .transmon_model import ChargeCoupledPair, TransmonCircuit

_SHIPPED = Path(__file__).resolve().parent / "calibrations"
# The flux at which a qubit idles, by the sweet spot it idles at.
_SWEET_SPOT_FLUX = {"upper": 0.0, "lower": 0.5}


class Calibration:
    """One cooldown of a calibration file; see the module docstring.

    Qubits and pairs are named by their labels (``"D4"``, ``"X2-D4"`` with
    the auxiliary first).
    """

    def __init__(self, data: Mapping[str, Any], *, name: str, cooldown: str | None = None) -> None:
        schema = data.get("_schema", {})
        cooldown = schema.get("default_cooldown") if cooldown is None else cooldown
        cooldowns = data.get("cooldowns", {})
        if cooldown not in cooldowns:
            raise ValueError(
                f"calibration {name!r} has no cooldown {cooldown!r}; "
                f"it has {', '.join(sorted(cooldowns))}"
            )
        self._data = copy.deepcopy(dict(data))
        self.name = name
        self.cooldown: str = cooldown
        self.target: str | None = schema.get("target")

    @classmethod
    def load(
        cls, source: str | os.PathLike[str] = "krinner17", *, cooldown: str | None = None
    ) -> Calibration:
        """A shipped calibration by name, or a calibration file by path."""
        path = Path(source)
        name = os.fspath(source)
        if not path.suffix and (_SHIPPED / f"{name}.json").is_file():
            path = _SHIPPED / f"{name}.json"
        elif not path.is_file():
            shipped = sorted(p.stem for p in _SHIPPED.glob("*.json"))
            raise FileNotFoundError(
                f"no calibration {name!r}: not a file, and the shipped ones are "
                + ", ".join(shipped)
            )
        else:
            name = path.stem
        return cls(json.loads(path.read_text(encoding="utf-8")), name=name, cooldown=cooldown)

    @property
    def cooldowns(self) -> Mapping[str, str]:
        """Each cooldown of the file and what it is."""
        return dict(self._data["cooldowns"])

    def entry(self, *path: str) -> Mapping[str, Any]:
        """The leaf at *path* with its provenance, e.g. ``entry("qubits", "D4", "ec")``."""
        node: Any = self._data
        for key in path:
            node = node[key]
        if not (isinstance(node, Mapping) and "value" in node):
            raise KeyError(f"{'/'.join(path)} is not a calibrated value")
        return copy.deepcopy(dict(node))

    def value(self, *path: str) -> Any:
        """The value of the leaf at *path*."""
        return self.entry(*path)["value"]

    def __repr__(self) -> str:
        return f"<Calibration {self.name} ({self.cooldown})>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Calibration) and other.fingerprint == self.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    @cached_property
    def fingerprint(self) -> str:
        import hashlib

        content = json.dumps(self._data, sort_keys=True).encode() + self.cooldown.encode()
        return hashlib.blake2b(content, digest_size=16).hexdigest()

    # ── Circuits and couplings ──────────────────────────────────────────

    @property
    def flux_qubits(self) -> tuple[str, ...]:
        """The qubits whose circuit, including d, is calibrated."""
        return tuple(
            label for label, q in self._data["qubits"].items() if q.get("flux_quanta_capable")
        )

    def circuit(self, qubit: str) -> TransmonCircuit:
        """The circuit of *qubit*, for flux in flux quanta."""
        from .transmon_model import TransmonCircuit

        if qubit not in self.flux_qubits:
            raise KeyError(f"calibration {self.name!r} has no circuit for qubit {qubit!r}")
        q = self._data["qubits"][qubit]
        return TransmonCircuit(
            q["ec"]["value"], q["ej_max"]["value"], q["asymmetry_d"]["value"], qubit
        )

    def idle_flux(self, qubit: str) -> float:
        """The flux at which *qubit* idles: its calibrated sweet spot."""
        return _SWEET_SPOT_FLUX[self._data["qubits"][qubit]["sweet_spot_idle"]]

    @property
    def pairs(self) -> tuple[str, ...]:
        """The coupled pairs with a charge coupling, auxiliary first."""
        return tuple(self._couplings()["per_pair"])

    def charge_coupling_hz(self, pair: str) -> float:
        """g_c of *pair* (``"X2-D4"``)."""
        try:
            value: float = self._couplings()["per_pair"][pair]["value"]
        except KeyError:
            raise KeyError(f"calibration {self.name!r} has no coupling for pair {pair!r}") from None
        return value

    def pair(self, pair: str, n_levels: int = 5) -> ChargeCoupledPair:
        """The charge-coupled pair, data transmon first (so ``|02>`` is the
        auxiliary's second excited state)."""
        from .transmon_model import ChargeCoupledPair

        auxiliary, data = pair.split("-")
        return ChargeCoupledPair(
            self.circuit(data), self.circuit(auxiliary), self.charge_coupling_hz(pair), n_levels
        )

    def exchange_hz(
        self, pair: str, auxiliary_hz: float | None = None, data_hz: float | None = None
    ) -> float:
        """The 0-1 exchange J_qq of *pair* with its qubits at these 0-1
        frequencies; at their idle points by default."""
        auxiliary, data = pair.split("-")
        coupled = self.pair(pair)
        flux_a = (
            self.idle_flux(auxiliary)
            if auxiliary_hz is None
            else coupled.b.flux_for_f01(auxiliary_hz)
        )
        flux_d = self.idle_flux(data) if data_hz is None else coupled.a.flux_for_f01(data_hz)
        return coupled.j_qq_hz(flux_d, flux_a)

    def idle_zz_hz(self, pair: str) -> float:
        """The static ZZ of *pair* at its idle points, from the dressed
        levels of the charge-coupled pair."""
        auxiliary, data = pair.split("-")
        return self.pair(pair).idle_zz_hz(self.idle_flux(data), self.idle_flux(auxiliary))

    # ── Flux quanta ─────────────────────────────────────────────────────

    def detuning_for_flux(self, qubit: str, flux: float) -> float:
        """The 0-1 detuning (Hz) from the idle point that a flux of *flux*
        flux quanta tunes *qubit* to."""
        circuit = self.circuit(qubit)
        return circuit.f01_hz(flux) - circuit.f01_hz(self.idle_flux(qubit))

    def flux_for_detuning(self, qubit: str, detuning_hz: float) -> float:
        """The flux (flux quanta, in [0, 1/2]) that detunes *qubit* by
        *detuning_hz* from its idle point."""
        circuit = self.circuit(qubit)
        return circuit.flux_for_f01(circuit.f01_hz(self.idle_flux(qubit)) + detuning_hz)

    # ── Targets ─────────────────────────────────────────────────────────

    def apply(self, target: Any) -> Any:
        """*target* with each calibrated pair's coupling set to its exchange
        at the qubits' frequencies in the target.

        A pair whose qubits sit at the idle frequencies its entry records
        takes the exchange stored with it, computed when the file was built.
        """
        labels = {q.index: q.label for q in target.qubits.values()}
        couplings = []
        for c in target.couplings:
            name = f"{labels.get(c.qubit_a)}-{labels.get(c.qubit_b)}"
            if name in self.pairs:
                at = (target.qubits[c.qubit_a].frequency_hz, target.qubits[c.qubit_b].frequency_hz)
                entry = self._couplings()["per_pair"][name]
                stored = entry.get("idle_frequencies_hz")
                if stored is not None and all(abs(f - g) < 1e3 for f, g in zip(at, stored)):
                    strength = entry["idle_exchange_hz"]
                else:
                    strength = self.exchange_hz(name, *at)
                c = replace(c, coupling_strength_hz=strength)
            couplings.append(c)
        return replace(target, couplings=couplings)

    def _couplings(self) -> Mapping[str, Any]:
        couplings: Mapping[str, Any] = self._data["coupling"]["charge_coupling_g_c"]
        return couplings


_LOADED: dict[tuple[str, str | None], Calibration] = {}


def shipped_calibration(name: str, cooldown: str | None = None) -> Calibration:
    """A shipped calibration, loaded once."""
    key = (name, cooldown)
    if key not in _LOADED:
        _LOADED[key] = Calibration.load(name, cooldown=cooldown)
    return _LOADED[key]
