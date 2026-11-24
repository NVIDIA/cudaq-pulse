# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tuning a qudit's frequency and turning a coupling on.

What a quantum system declares of its baseband controls, by qudit and by
pair (``pulse.quantum_system(..., tunings={q: ...}, controls={(a, d): ...},
interactions={(a, d): ...})``), and answers for (``qs.tuning(q)``,
``qs.control(a, d)``, ``qs.interaction(a, d)``). A tuning (a
``FluxTuning``) says how an amplitude, an offset from the qudit's idle
point in the tuning's ``unit``, moves its frequencies. A control is a
coupling's own, which a coupling line reaches: an ``ExchangeControl`` is
the pair's control, its exchange tabulated against its amplitude. An
interaction is what a coupling with no line of its own does: a
``ChargeCoupling`` is a pair coupled through charge whose interaction is
made by tuning the two qudits, each by its own tuning. A ``response``
(``ExponentialSettling``, ``HighPass``) says how a programmed amplitude
arrives at the device. They are read without compiling anything.

The module loads only when a quantum system declares or is asked for one
of them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError, Problem
from .base import _Component, _Field, _finite

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np
    import numpy.typing as npt

__all__ = [
    "ChargeCoupling",
    "CouplingControl",
    "CouplingInteraction",
    "CouplingPoint",
    "ExchangeControl",
    "ExponentialSettling",
    "FluxTuning",
    "HighPass",
    "Response",
    "Tuning",
]


class Tuning(_Component):
    """How a qudit's own frequency is tuned: the base of the tuning
    modalities, such as ``FluxTuning``. An amplitude is in the component's
    ``unit``, an offset from the idle point the qudit idles at, and
    ``frequency_at``, ``anharmonicity_at`` and ``amplitude_for`` map it to
    the qudit's frequencies and back."""

    __slots__ = ()
    if TYPE_CHECKING:
        unit: ClassVar[str]

        def frequency_at(self, amplitude: Any) -> Any: ...
        def anharmonicity_at(self, amplitude: Any) -> Any: ...
        def amplitude_for(self, frequency_hz: Any) -> Any: ...

    def _problems(self, where: str) -> list[Problem]:
        """The problems with the tuning, the one at *where*."""
        return []


class Response(_Component):
    """How the wiring between a controller and the device passes a
    programmed amplitude on: the base of the responses a ``Tuning`` or a
    ``CouplingControl`` declares, such as ``ExponentialSettling`` and
    ``HighPass``. The samples of a program stay as written; a simulation
    applies the response, and a backend or an instrument compensates it."""

    __slots__ = ()


class CouplingControl(_Component):
    """The control of a coupling, which a coupling line reaches: the base of
    the coupling controls, such as ``ExchangeControl`` (the pair's own
    control sets its exchange)."""

    __slots__ = ()
    if TYPE_CHECKING:

        def at(self, amplitude: Any) -> CouplingPoint:
            """The coupling at *amplitude* of the control."""
            ...

    def _problems(self, qudits: tuple[int, ...], where: str) -> list[Problem]:
        """The problems with the control of the coupling of *qudits*, at *where*."""
        return []


class CouplingInteraction(_Component):
    """How two qudits interact through a coupling that has no line of its
    own: the base of the interactions, such as ``ChargeCoupling`` (a fixed
    charge coupling, turned on by tuning its qudits)."""

    __slots__ = ()
    if TYPE_CHECKING:

        def at(self, qudits: Sequence[Any], amplitude: Mapping[Any, Any]) -> CouplingPoint:
            """The coupling of *qudits* with each tuned qudit at its *amplitude*."""
            ...

    def _problems(self, qudits: tuple[int, ...], where: str) -> list[Problem]:
        """The problems with the interaction of the coupling of *qudits*, at *where*."""
        return []

    def _through(self, tunings: tuple[Tuning, ...]) -> CouplingInteraction:
        """The interaction as the quantum system answers with it: made
        through *tunings*, its two qudits' in the order asked."""
        return self


# Charge states -25..25, as transmon_model.TransmonCircuit diagonalizes them.
_CHARGE_CUTOFF = 25
# Operating points diagonalized at once: a batch of 51 x 51 matrices.
_BATCH = 512


class ExponentialSettling(Response):
    """A step that settles exponentially: a step of 1 in the programmed
    amplitude arrives at the device as ``1 + amplitude * exp(-t /
    tau_ns)``, an overshoot (``amplitude`` above 0) or a slow rise (below
    0) that dies out in ``tau_ns``. ``amplitude`` is a fraction of the
    step, between -1 and 1."""

    __slots__ = ()
    _fields = ("amplitude", "tau_ns")
    _kind = "exponential_settling"

    def __init__(self, *, amplitude: float, tau_ns: float) -> None:
        object.__setattr__(self, "_values", (amplitude, tau_ns))

    if TYPE_CHECKING:

        @property
        def amplitude(self) -> float:
            """The size of the settling part, as a fraction of the step."""
            ...

        @property
        def tau_ns(self) -> float:
            """Its time constant, in ns."""
            ...

    else:
        amplitude, tau_ns = map(_Field, range(2))

    def _problems(self, where: str) -> list[Problem]:
        amplitude, tau = self._values
        problems = []
        if not (_finite(amplitude) and -1.0 < amplitude < 1.0):
            problems.append(
                Problem(
                    f"{amplitude!r} is not a fraction of the step between -1 and 1",
                    path=f"{where}.amplitude",
                    value=amplitude,
                )
            )
        problems += _tau_problems(tau, where)
        return problems


class HighPass(Response):
    """A line that does not pass a constant, such as one behind a bias tee:
    a step of 1 in the programmed amplitude arrives at the device as
    ``exp(-t / tau_ns)``, so a long pulse droops and leaves a tail of the
    other sign."""

    __slots__ = ()
    _fields = ("tau_ns",)
    _kind = "high_pass"

    def __init__(self, *, tau_ns: float) -> None:
        object.__setattr__(self, "_values", (tau_ns,))

    if TYPE_CHECKING:

        @property
        def tau_ns(self) -> float:
            """Its time constant, in ns."""
            ...

    else:
        (tau_ns,) = map(_Field, range(1))

    def _problems(self, where: str) -> list[Problem]:
        return _tau_problems(self._values[0], where)


def _tau_problems(tau: Any, where: str) -> list[Problem]:
    if _finite(tau) and tau > 0:
        return []
    return [Problem(f"{tau!r} is not a finite time above 0 ns", path=f"{where}.tau_ns", value=tau)]


_RESPONSES: dict[str, type[Response]] = {
    ExponentialSettling._kind: ExponentialSettling,
    HighPass._kind: HighPass,
}


def _responses(given: Any) -> tuple[Any, ...]:
    """``response=`` as a component keeps it: one response, or a sequence of
    them applied one after another, as a tuple."""
    if given is None:
        return ()
    if isinstance(given, Response) or isinstance(given, (str, bytes, Mapping)):
        return (given,)
    try:
        return tuple(given)
    except TypeError:
        return (given,)


def _response_problems(responses: tuple[Any, ...], where: str) -> list[Problem]:
    problems: list[Problem] = []
    for j, response in enumerate(responses):
        path = f"{where}.response[{j}]"
        if type(response) not in (ExponentialSettling, HighPass):
            problems.append(
                Problem(
                    f"must be an ExponentialSettling or a HighPass, got {type(response).__name__}",
                    path=path,
                    value=response,
                )
            )
        else:
            problems += response._problems(path)
    return problems


def _component_data(kind: str, component: Any) -> dict[str, Any]:
    """A tuning or a control as a system file holds it: its fields, lists
    for tuples, without a ``ChargeCoupling``'s tunings (its qudits' own,
    which the file gives with them), and its responses where it has some."""
    data: dict[str, Any] = {"kind": kind}
    for field, value in zip(component._fields, component._values, strict=True):
        if field == "response":
            if value:
                data[field] = _response_data(value)
        elif field != "tunings":
            data[field] = list(value) if isinstance(value, tuple) else value
    return data


def _response_data(responses: tuple[Any, ...]) -> list[dict[str, Any]]:
    """A component's responses as a system file holds them."""
    return [{"kind": r._kind, **dict(zip(r._fields, r._values, strict=True))} for r in responses]


def _response_from(data: Any) -> tuple[Any, ...]:
    """The responses a system file's ``response`` list gives; an
    ``InvalidValueError`` names an entry that is not one."""
    if not isinstance(data, list):
        raise InvalidValueError(f"response must be a list of responses, got {type(data).__name__}")
    found = []
    for j, entry in enumerate(data):
        kind = entry.get("kind") if isinstance(entry, Mapping) else None
        made = _RESPONSES.get(kind) if isinstance(kind, str) else None
        if made is None:
            raise InvalidValueError(
                f"response[{j}] is not a response; give its kind, one of {', '.join(_RESPONSES)}"
            )
        unknown = sorted(set(entry) - {"kind", *made._fields})
        if unknown:
            raise InvalidValueError(
                f"response[{j}] has no field {unknown[0]!r}; a {kind} has {', '.join(made._fields)}"
            )
        found.append(made(**{k: v for k, v in entry.items() if k != "kind"}))
    return tuple(found)


class FluxTuning(Tuning):
    """A qudit tuned by the flux through its SQUID: an asymmetric-SQUID
    transmon of charging energy ``ec_hz``, largest Josephson energy
    ``ej_max_hz`` and SQUID asymmetry ``asymmetry`` (d = (E_J2 - E_J1) /
    (E_J1 + E_J2)), idling at the flux ``idle_flux`` in flux quanta, from 0
    to 1/2, its sweet spots (the circuit is periodic and even in flux, so
    every idle point has its equal there).

    An amplitude is a flux offset from the idle point, in flux quanta
    (``unit`` ``"flux_quantum"``), and signed: the frequency is even about a
    sweet spot, so ``a`` and ``-a`` give the same one there, which is why a
    net-zero pulse returns the qudit to its frequency. The levels are the
    Cooper-pair box's, H = 4 E_C n^2 - E_J(Phi) cos(phi) with E_J(Phi) =
    E_J,max sqrt(cos^2(pi Phi) + d^2 sin^2(pi Phi)), diagonalized in the
    charge basis.

    ``response`` is how the wiring between the controller and the qudit
    passes a programmed amplitude on: an ``ExponentialSettling``, a
    ``HighPass``, or a sequence of them applied one after another; none by
    default, the amplitude arriving as programmed.

    ``source`` says where the values come from: a citation, or the
    calibration entries they were read from (``Calibration.entry`` gives each
    with its kind, source and method).
    """

    __slots__ = ()
    _fields = ("ec_hz", "ej_max_hz", "asymmetry", "idle_flux", "source", "response")
    unit = "flux_quantum"

    def __init__(
        self,
        *,
        ec_hz: float,
        ej_max_hz: float,
        asymmetry: float,
        idle_flux: float,
        source: str = "",
        response: Response | Sequence[Response] = (),
    ) -> None:
        object.__setattr__(
            self,
            "_values",
            (ec_hz, ej_max_hz, asymmetry, idle_flux, source, _responses(response)),
        )

    if TYPE_CHECKING:

        @property
        def ec_hz(self) -> float:
            """The charging energy E_C / h."""
            ...

        @property
        def ej_max_hz(self) -> float:
            """The Josephson energy E_J,max / h of the SQUID at zero flux."""
            ...

        @property
        def asymmetry(self) -> float:
            """The SQUID asymmetry d."""
            ...

        @property
        def idle_flux(self) -> float:
            """The flux (flux quanta) the qudit idles at."""
            ...

        @property
        def source(self) -> str:
            """Where the values come from."""
            ...

        @property
        def response(self) -> tuple[Response, ...]:
            """How the wiring passes a programmed amplitude on, in order."""
            ...

    else:
        ec_hz, ej_max_hz, asymmetry, idle_flux, source, response = map(_Field, range(6))

    def frequency_at(self, amplitude: Any) -> Any:
        """The 0-1 frequency (Hz) at the flux offset *amplitude* from the
        idle point: a float, or an array for an array of amplitudes."""
        energies, _ = self._spectrum(amplitude, 2)
        return _scalar(energies[..., 1], amplitude)

    def anharmonicity_at(self, amplitude: Any) -> Any:
        """The anharmonicity E_12 - E_01 (Hz) at *amplitude*."""
        energies, _ = self._spectrum(amplitude, 3)
        return _scalar(energies[..., 2] - 2.0 * energies[..., 1], amplitude)

    def amplitude_for(self, frequency_hz: Any) -> Any:
        """The flux offset from the idle point that puts the 0-1 frequency at
        *frequency_hz*, on the branch between the two sweet spots (fluxes 0
        and 1/2) the idle point is on; its negative gives the same frequency
        when the qudit idles at a sweet spot. A frequency the qudit cannot
        reach is an ``InvalidValueError``."""
        import numpy as np

        from .transmon_model import TransmonCircuit

        circuit = TransmonCircuit(self._values[0], self._values[1], self._values[2])
        idle = float(self._values[3])
        given = np.asarray(frequency_hz, dtype=float)
        found = np.array([circuit.flux_for_f01(float(f)) - idle for f in given.reshape(-1)])
        return _scalar(found.reshape(given.shape), frequency_hz)

    def _spectrum(
        self, amplitude: Any, count: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The first *count* energies above the ground state (Hz) and the
        charge matrix ``<i|n|j>`` at each amplitude, with the signs that make
        ``<i|n|i+1>`` positive, as ``TransmonCircuit.eigensystem`` gives them."""
        import numpy as np

        ec, ej_max, d, idle = (float(v) for v in self._values[:4])
        flux = idle + np.asarray(amplitude, dtype=float)
        flat = flux.reshape(-1)
        n = np.arange(-_CHARGE_CUTOFF, _CHARGE_CUTOFF + 1, dtype=float)
        size = n.size
        rows = np.arange(size)
        energies = np.empty((flat.size, count))
        charge = np.empty((flat.size, count, count))
        for start in range(0, flat.size, _BATCH):
            x = math.pi * flat[start : start + _BATCH]
            ej = ej_max * np.sqrt(np.cos(x) ** 2 + d**2 * np.sin(x) ** 2)
            h = np.zeros((x.size, size, size))
            h[:, rows, rows] = 4.0 * ec * n**2
            h[:, rows[:-1], rows[1:]] = -0.5 * ej[:, None]
            h[:, rows[1:], rows[:-1]] = -0.5 * ej[:, None]
            values, vectors = np.linalg.eigh(h)
            vectors = vectors[:, :, :count]
            for i in range(1, count):
                overlap = np.einsum("pk,k,pk->p", vectors[:, :, i - 1], n, vectors[:, :, i])
                vectors[:, :, i] *= np.where(overlap < 0, -1.0, 1.0)[:, None]
            energies[start : start + x.size] = values[:, :count] - values[:, :1]
            charge[start : start + x.size] = np.einsum("pki,k,pkj->pij", vectors, n, vectors)
        return (
            energies.reshape(flux.shape + (count,)),
            charge.reshape(flux.shape + (count, count)),
        )

    def _problems(self, where: str) -> list[Problem]:
        problems = []
        for name, value, ok in (
            ("ec_hz", self._values[0], _finite(self._values[0]) and self._values[0] > 0),
            ("ej_max_hz", self._values[1], _finite(self._values[1]) and self._values[1] > 0),
            ("asymmetry", self._values[2], _finite(self._values[2]) and 0 <= self._values[2] <= 1),
            (
                "idle_flux",
                self._values[3],
                _finite(self._values[3]) and 0.0 <= self._values[3] <= 0.5,
            ),
        ):
            if not ok:
                what = "a finite asymmetry in [0, 1]" if name == "asymmetry" else None
                problems.append(
                    Problem(
                        f"{value!r} is not {what or _WHAT[name]}",
                        path=f"{where}.{name}",
                        value=value,
                    )
                )
        if self._values[5]:
            problems += _response_problems(self._values[5], where)
        return problems


_WHAT = {
    "ec_hz": "a finite energy above 0 Hz",
    "ej_max_hz": "a finite energy above 0 Hz",
    "idle_flux": "a flux between the sweet spots, 0 to 0.5 flux quanta",
}


class ChargeCoupling(CouplingInteraction):
    """A pair coupled through charge, g_c n_a n_b, whose interaction is made
    by tuning the two qudits: the coupling has no control and no line of
    its own, and a coupling line reaches it through either qudit's tuning,
    in that tuning's unit (flux quanta for a ``FluxTuning``).

    ``charge_coupling_hz`` is g_c, which is all a quantum system is given:
    ``interactions={(a, d): ChargeCoupling(charge_coupling_hz=3e6)}``. The
    two qudits' tunings are the system's, and the interaction it answers
    with (``qs.interaction(a, d)``) has them as ``tunings``, in the order
    asked. In the rotating-wave approximation the levels i, i + 1 of qudit
    a exchange with j + 1, j of qudit b at g_c ``<i|n_a|i+1> <j+1|n_b|j>``,
    the charge matrix elements at the qudits' operating points; the 0-1
    exchange J_qq = g_c ``<0|n_a|1> <0|n_b|1>`` at the idle points is the
    coupling's ``strength_hz``. ``source`` says where g_c comes from.
    """

    __slots__ = ()
    _fields = ("charge_coupling_hz", "tunings", "source")

    def __init__(
        self, *, charge_coupling_hz: float, tunings: Sequence[Tuning] = (), source: str = ""
    ) -> None:
        object.__setattr__(self, "_values", (charge_coupling_hz, tuple(tunings), source))

    if TYPE_CHECKING:

        @property
        def charge_coupling_hz(self) -> float:
            """The charge coupling g_c / h."""
            ...

        @property
        def tunings(self) -> tuple[Tuning, ...]:
            """The two qudits' tunings, in the order asked of the quantum
            system; empty on the interaction a system is given."""
            ...

        @property
        def source(self) -> str:
            """Where g_c comes from."""
            ...

    else:
        charge_coupling_hz, tunings, source = map(_Field, range(3))

    @property
    def unit(self) -> str:
        """The unit of an amplitude of either qudit: its tuning's."""
        tunings = self._values[1]
        return tunings[0].unit if tunings else ""

    def _problems(self, qudits: tuple[int, ...], where: str) -> list[Problem]:
        g = self._values[0]
        problems = []
        if not (_finite(g) and g > 0):
            problems.append(
                Problem(
                    f"{g!r} is not a finite coupling above 0 Hz",
                    path=f"{where}.charge_coupling_hz",
                    value=g,
                )
            )
        if self._values[1]:
            problems.append(
                Problem(
                    "gives tunings; the quantum system has its qudits' tunings",
                    path=f"{where}.tunings",
                    value=self._values[1],
                    fix="leave tunings out, and give the qudits' with tunings={q: ...}",
                )
            )
        return problems

    def _through(self, tunings: tuple[Tuning, ...]) -> ChargeCoupling:
        g, _, source = self._values
        return ChargeCoupling(charge_coupling_hz=g, tunings=tunings, source=source)

    def at(self, qudits: Sequence[Any], amplitude: Mapping[Any, Any]) -> CouplingPoint:
        """The coupling of *qudits* (the pair's indices or ``Qudit`` handles,
        in the order of ``tunings``) with the tunings of its qudits at
        *amplitude*, a mapping from a qudit to its amplitude (a float or an
        array; a qudit left out idles), of the interaction a quantum system
        answers with (``qs.interaction(a, d).at((a, d), {d: 0.1})``)."""
        if len(self._values[1]) != 2:
            raise InvalidValueError(
                "this ChargeCoupling has no tunings: ask the quantum system for the "
                "interaction, as qs.interaction(a, d), which has its qudits' tunings"
            )
        pair = tuple(map(_carrier, qudits))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise InvalidValueError(
                "ChargeCoupling.at takes the coupling's two qudits, in the order of its "
                f"tunings, as (a, b); got {tuple(qudits)!r}"
            )
        a, b = pair
        given = {_carrier(k): v for k, v in amplitude.items()}
        other = sorted(set(given) - {a, b}, key=repr)
        if other:
            raise InvalidValueError(
                f"{other[0]!r} is not a qudit of the coupling of qudits {a} and {b}"
            )
        return self._point(a, b, given.get(a, 0.0), given.get(b, 0.0))

    def _point(self, a: int, b: int, amplitude_a: Any, amplitude_b: Any) -> CouplingPoint:
        import numpy as np

        tuning_a, tuning_b = self._values[1]
        at_a, at_b = np.broadcast_arrays(
            np.asarray(amplitude_a, dtype=float), np.asarray(amplitude_b, dtype=float)
        )
        energies_a, n_a = tuning_a._spectrum(at_a, 3)
        energies_b, n_b = tuning_b._spectrum(at_b, 3)
        g = float(self._values[0])
        scalar = at_a.ndim == 0

        def value(x: Any) -> Any:
            return float(x) if scalar else x

        return CouplingPoint(
            frequencies_hz={a: value(energies_a[..., 1]), b: value(energies_b[..., 1])},
            anharmonicities_hz={
                a: value(energies_a[..., 2] - 2.0 * energies_a[..., 1]),
                b: value(energies_b[..., 2] - 2.0 * energies_b[..., 1]),
            },
            exchange_hz=value(g * n_a[..., 0, 1] * n_b[..., 0, 1]),
            exchanges_hz={
                ((0, 1), (1, 0)): value(g * n_a[..., 0, 1] * n_b[..., 0, 1]),
                ((1, 1), (2, 0)): value(g * n_a[..., 1, 2] * n_b[..., 0, 1]),
                ((1, 1), (0, 2)): value(g * n_a[..., 0, 1] * n_b[..., 1, 2]),
            },
        )


class ExchangeControl(CouplingControl):
    """The pair's own control, such as a tunable coupler's flux or the
    voltage of a gate between two qudits: the 0-1 exchange J / 2 pi (Hz) of
    the coupled qudits at each of ``amplitudes`` (in ``unit``, offsets from
    the idle point, increasing, and spanning 0) is ``exchange_hz``, and
    linear between them. It moves no qudit's frequency. The exchange at 0 is
    the idle coupling, the coupling's ``strength_hz``. ``source`` says where
    the table comes from, and ``response`` how the wiring passes a
    programmed amplitude on (as a ``FluxTuning``'s).
    """

    __slots__ = ()
    _fields = ("unit", "amplitudes", "exchange_hz", "source", "response")

    def __init__(
        self,
        *,
        unit: str,
        amplitudes: Sequence[float],
        exchange_hz: Sequence[float],
        source: str = "",
        response: Response | Sequence[Response] = (),
    ) -> None:
        object.__setattr__(
            self,
            "_values",
            (unit, tuple(amplitudes), tuple(exchange_hz), source, _responses(response)),
        )

    if TYPE_CHECKING:

        @property
        def unit(self) -> str:
            """The unit of an amplitude, as ``"volt"`` or ``"flux_quantum"``."""
            ...

        @property
        def amplitudes(self) -> tuple[float, ...]:
            """The amplitudes the exchange is given at, increasing."""
            ...

        @property
        def exchange_hz(self) -> tuple[float, ...]:
            """The exchange J / 2 pi at each amplitude."""
            ...

        @property
        def source(self) -> str:
            """Where the table comes from."""
            ...

        @property
        def response(self) -> tuple[Response, ...]:
            """How the wiring passes a programmed amplitude on, in order."""
            ...

    else:
        unit, amplitudes, exchange_hz, source, response = map(_Field, range(5))

    def _problems(self, qudits: tuple[int, ...], where: str) -> list[Problem]:
        unit, amplitudes, exchange, _, responses = self._values
        problems = []
        if responses:
            problems += _response_problems(responses, where)
        if not (isinstance(unit, str) and unit):
            problems.append(
                Problem(
                    f"{unit!r} is not the name of a unit",
                    path=f"{where}.unit",
                    value=unit,
                    fix='name the unit of an amplitude, as "volt"',
                )
            )
        if len(amplitudes) < 2 or len(amplitudes) != len(exchange):
            problems.append(
                Problem(
                    f"gives {len(amplitudes)} amplitudes and {len(exchange)} exchanges",
                    path=f"{where}.exchange_hz",
                    fix="give the exchange at each amplitude, at two amplitudes or more",
                )
            )
            return problems
        if not all(map(_finite, amplitudes)) or any(
            lo >= hi for lo, hi in zip(amplitudes, amplitudes[1:], strict=False)
        ):
            problems.append(
                Problem(
                    "must be finite and increasing",
                    path=f"{where}.amplitudes",
                    value=amplitudes,
                )
            )
        elif not amplitudes[0] <= 0.0 <= amplitudes[-1]:
            problems.append(
                Problem(
                    "does not span the idle point, amplitude 0",
                    path=f"{where}.amplitudes",
                    value=amplitudes,
                    fix="give the exchange at amplitudes on both sides of 0, or at 0",
                )
            )
        if not all(map(_finite, exchange)):
            problems.append(
                Problem("must be finite (Hz)", path=f"{where}.exchange_hz", value=exchange)
            )
        return problems

    def at(self, amplitude: Any) -> CouplingPoint:
        """The coupling at *amplitude* of this control (a float or an array,
        in the control's unit, from its idle point): its 0-1 exchange."""
        import numpy as np

        _, amplitudes, exchange = self._values[:3]
        at = np.asarray(amplitude, dtype=float)
        if at.size and (at.min() < amplitudes[0] or at.max() > amplitudes[-1]):
            raise InvalidValueError(
                f"the control is tabulated from {amplitudes[0]} to {amplitudes[-1]} "
                f"{self._values[0]}; {amplitude!r} is outside it"
            )
        j = np.interp(at, amplitudes, exchange)
        j = float(j) if at.ndim == 0 else j
        return CouplingPoint(
            frequencies_hz={},
            anharmonicities_hz={},
            exchange_hz=j,
            exchanges_hz={((0, 1), (1, 0)): j},
        )


@dataclass(frozen=True)
class CouplingPoint:
    """A coupling at an amplitude of its control or of its qudits' tunings,
    as ``ExchangeControl.at`` and ``ChargeCoupling.at`` read it: each tuned
    qudit's 0-1 frequency and anharmonicity (Hz) by qudit
    index, the 0-1 exchange J / 2 pi (Hz), and the exchange of each pair of
    levels the control gives, keyed ``((i_a, i_b), (j_a, j_b))`` by the
    coupled qudits' levels in the coupling's order: ``((1, 1), (2, 0))`` is
    ``|11> <-> |20>``. Floats, or arrays the shape of the amplitude. An
    ``ExchangeControl`` moves no qudit, so its point has no frequencies."""

    frequencies_hz: Mapping[int, Any]
    anharmonicities_hz: Mapping[int, Any]
    exchange_hz: Any
    exchanges_hz: Mapping[tuple[tuple[int, int], tuple[int, int]], Any]


def _scalar(values: Any, given: Any) -> Any:
    """*values*, a float where the amplitude *given* is a number."""
    import numpy as np

    return float(values) if np.ndim(given) == 0 else values


def _carrier(qudit: Any) -> Any:
    """A qudit, as its index: a ``Qudit`` handle names its carrier."""
    from .._device import Qudit

    return qudit.index if qudit.__class__ is Qudit else qudit
