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
reads one, a shipped name (``"transmon_17q"``) or a path, for one cooldown.

A calibration gives each flux-tunable qubit its circuit (E_C, E_J,max and
SQUID asymmetry d, so flux in flux quanta converts to frequency and back)
and each coupled pair its charge coupling g_c, from which the exchange
J(w_a, w_d) and the idle ZZ follow at any operating point.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError, NotFoundError

if TYPE_CHECKING:
    from .transmon_model import ChargeCoupledPair, TransmonCircuit

# A path string, not a pathlib.Path: pathlib (with urllib.parse) would add a
# tenth to the time ``import cudaq_pulse`` takes.
_SHIPPED = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibrations")


def _shipped_file(name: str) -> str:
    return os.path.join(_SHIPPED, name + ".json")


def _read(path: str) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


# The flux at which a qubit idles, by the sweet spot it idles at.
_SWEET_SPOT_FLUX = {"upper": 0.0, "lower": 0.5}


class Calibration:
    """One cooldown of a calibration file; see the module docstring.

    Qubits and pairs are named by their labels (``"D4"``, ``"X2-D4"`` with
    the auxiliary first).
    """

    def __init__(self, data: Mapping[str, Any], *, name: str, cooldown: str | None = None) -> None:
        self._name(data, name, cooldown)
        self._data = copy.deepcopy(dict(data))

    def _name(self, data: Mapping[str, Any], name: str, cooldown: str | None) -> None:
        schema = data.get("_schema", {})
        cooldown = schema.get("default_cooldown") if cooldown is None else cooldown
        cooldowns = data.get("cooldowns", {})
        if cooldown not in cooldowns:
            raise InvalidValueError(
                f"calibration {name!r} has no cooldown {cooldown!r}; "
                f"it has {', '.join(sorted(cooldowns))}"
            )
        self.name = name
        self.cooldown: str = cooldown
        # The digest of the shipped file this was loaded from, if it was.
        self._shipped: str | None = None

    @cached_property
    def _data(self) -> dict[str, Any]:
        # Set by __init__; a shipped calibration that was unpickled reads its
        # file when its values are first used.
        data: dict[str, Any] = _read(_shipped_file(self.name))
        return data

    @cached_property
    def target(self) -> str | None:
        """The target the calibration was made for, if its file names one."""
        target: str | None = self._data.get("_schema", {}).get("target")
        return target

    @classmethod
    def load(
        cls, source: str | os.PathLike[str] = "transmon_17q", *, cooldown: str | None = None
    ) -> Calibration:
        """A shipped calibration by name, or a calibration file by path."""
        path = name = os.fspath(source)
        stem, suffix = os.path.splitext(os.path.basename(path))
        digest = None
        if not suffix and os.path.isfile(_shipped_file(name)):
            from .._source_stamp import CALIBRATIONS

            path, digest = _shipped_file(name), CALIBRATIONS.get(name)
        elif not os.path.isfile(path):
            shipped = sorted(f[:-5] for f in os.listdir(_SHIPPED) if f.endswith(".json"))
            raise NotFoundError(
                f"no calibration {name!r}: not a file, and the shipped ones are "
                + ", ".join(shipped)
            )
        else:
            name = stem
        data = _read(path)
        loaded = cls.__new__(cls)
        loaded._name(data, name, cooldown)
        # Not copied: parsed here, it is held by this calibration alone, which
        # only reads it and hands out copies (entry, value, cooldowns).
        loaded._data = data
        loaded._shipped = digest
        return loaded

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
            raise NotFoundError(f"{'/'.join(path)} is not a calibrated value")
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

    # Read-only: a copy is the calibration itself.
    def __copy__(self) -> Calibration:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Calibration:
        return self

    def __reduce_ex__(self, protocol: Any) -> Any:
        # A shipped calibration pickles as its name, not its values: the
        # package that unpickles it ships it, checked by the file's digest.
        if self._shipped is None or type(self) is not Calibration:
            return super().__reduce_ex__(protocol)
        from .._build_config import PACKAGE_VERSION
        from .._pickling import rebuilt_shipped_calibration

        return (
            rebuilt_shipped_calibration,
            (self.name, self.cooldown, self._shipped, PACKAGE_VERSION),
        )

    @cached_property
    def fingerprint(self) -> str:
        import hashlib

        content = json.dumps(self._data, sort_keys=True).encode() + self.cooldown.encode()
        return hashlib.blake2b(content, digest_size=16).hexdigest()

    @property
    def _identity(self) -> tuple[str, ...]:
        """What a device's fingerprint takes of it: a shipped calibration's
        name, cooldown and the file's digest, recorded when the package was
        built; any other's ``fingerprint``, a hash of its contents."""
        if self._shipped is not None:
            return ("shipped", self.name, self.cooldown, self._shipped)
        return ("contents", self.fingerprint)

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
            raise NotFoundError(f"calibration {self.name!r} has no circuit for qubit {qubit!r}")
        q = self._data["qubits"][qubit]
        return TransmonCircuit(
            q["ec"]["value"], q["ej_max"]["value"], q["asymmetry_d"]["value"], qubit
        )

    def idle_flux(self, qubit: str) -> float:
        """The flux at which *qubit* idles: its calibrated sweet spot."""
        return _SWEET_SPOT_FLUX[self._data["qubits"][qubit]["sweet_spot_idle"]]

    @property
    def pairs(self) -> tuple[str, ...]:
        """The coupled pairs, auxiliary first."""
        return tuple(self._couplings()["idle_exchange"])

    def charge_coupling_hz(self, pair: str) -> float:
        """g_c of *pair* (``"X2-D4"``): the device-wide charge coupling."""
        if pair not in self._couplings()["idle_exchange"]:
            raise NotFoundError(f"calibration {self.name!r} has no coupling for pair {pair!r}")
        couplings = self._couplings()
        value: float = couplings[couplings["used"]]["value"]
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
        idle_exchange = self._idle_exchange
        couplings = []
        for c in target.couplings:
            name = f"{labels.get(c.qubit_a)}-{labels.get(c.qubit_b)}"
            found = idle_exchange.get(name)
            if found is not None:
                at = (target.qubits[c.qubit_a].frequency_hz, target.qubits[c.qubit_b].frequency_hz)
                value, stored = found
                if stored is not None and all(abs(f - g) < 1e3 for f, g in zip(at, stored)):
                    strength = value
                else:
                    strength = self.exchange_hz(name, *at)
                c = replace(c, coupling_strength_hz=strength)
            couplings.append(c)
        return replace(target, couplings=couplings)

    @cached_property
    def _idle_exchange(self) -> Mapping[str, tuple[float, Any]]:
        """Each pair's stored idle exchange and the idle frequencies it was
        computed at (None if not recorded); preset for an indexed shipped
        calibration (see ``shipped_calibration``)."""
        return {
            pair: (entry["value"], entry.get("idle_frequencies_hz"))
            for pair, entry in self._couplings()["idle_exchange"].items()
        }

    def _couplings(self) -> Mapping[str, Any]:
        couplings: Mapping[str, Any] = self._data["coupling"]["charge_coupling_g_c"]
        return couplings


_LOADED: dict[tuple[str, str | None], Calibration] = {}
_LOADED_LOCK = threading.Lock()


def shipped_calibration(name: str, cooldown: str | None = None) -> Calibration:
    """A shipped calibration, loaded once: its default cooldown asked for by
    name or as ``None`` is one calibration. One in ``_shipped_calibrations``
    is not read until a value beyond what a device takes is used."""
    key = (name, cooldown)
    loaded = _LOADED.get(key)
    if loaded is None:
        with _LOADED_LOCK:
            loaded = _LOADED.get(key)
            if loaded is None:
                loaded = _indexed(name, cooldown) or Calibration.load(name, cooldown=cooldown)
                loaded = _LOADED.setdefault((name, loaded.cooldown), loaded)
                _LOADED[key] = loaded
    return loaded


def _indexed(name: str, cooldown: str | None) -> Calibration | None:
    """The shipped calibration *name* (*cooldown*), unread, from
    ``_shipped_calibrations`` if its entry is of the file this package ships;
    None otherwise, and for a cooldown the file does not have."""
    from .._source_stamp import CALIBRATIONS
    from ._shipped_calibrations import SHIPPED

    entry = SHIPPED.get(name)
    if entry is None or CALIBRATIONS.get(name) != entry[0]:
        return None
    digest, default, cooldowns, idle_exchange = entry
    cooldown = default if cooldown is None else cooldown
    if cooldown not in cooldowns:
        return None
    unread = Calibration.__new__(Calibration)
    unread.name, unread.cooldown, unread._shipped = name, cooldown, digest
    unread._idle_exchange = idle_exchange
    return unread


def _shipped_unread(name: str, cooldown: str, digest: str) -> Calibration | None:
    """The shipped calibration *name* (*cooldown*), if this package ships the
    file of that *digest*, without reading the file until its values are used."""
    from .._source_stamp import CALIBRATIONS

    if CALIBRATIONS.get(name) != digest:
        return None
    found = _LOADED.get((name, cooldown))
    if found is None:
        unread = Calibration.__new__(Calibration)
        unread.name, unread.cooldown, unread._shipped = name, cooldown, digest
        with _LOADED_LOCK:
            found = _LOADED.setdefault((name, cooldown), unread)
    return found
