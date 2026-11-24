# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a quantum system says of the controls of its qudits and couplings:
the tuning of a qudit and the interaction of a coupling, as declared on the
``PhysicalQudit`` or the ``Coupling``, else as the system's calibration
gives them.

What a calibration gives is not part of a qudit or a coupling: it is found
here, when asked for, from the calibration the system holds, by the qudit's
label. A system with no calibration of its own (a variant made with
``replace``, a system built from another's qudits) is asked the same way:
a qudit of it that has the values of a qudit a shipped calibration builds
has that calibration's tuning. The values compared are those a fingerprint
takes of a qudit, so two systems of one fingerprint give the same answers.
"""

from __future__ import annotations

import threading
import weakref
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from .._errors import NotFoundError

if TYPE_CHECKING:
    from .base import Coupling, CouplingInteraction, Tuning, _Description
    from .calibration import Calibration

# What a fingerprint takes of a qudit and of a coupling.
_QUDIT_VALUES = attrgetter(
    "index",
    "frequency_hz",
    "anharmonicity_hz",
    "t1_us",
    "t2_star_us",
    "t2_echo_us",
    "thermal_population",
    "drive",
    "readout",
    "_calibrations",
)
_COUPLING_VALUES = attrgetter("qudits", "strength_hz", "_calibrations")


def _same(values: Any, shipped: Any, given: Any) -> bool:
    if shipped is given:
        return True
    a, b = values(shipped), values(given)
    return bool(a == b and a[-1]._content == b[-1]._content)


def _shipped() -> list[tuple[Calibration, _Description]]:
    """Each shipped calibration that gives tunings, with the description it builds."""
    from ._shipped_calibrations import SHIPPED_TUNINGS
    from .calibration import shipped_calibration
    from .transmon import _calibrated_surface17

    return [(shipped_calibration(name), _calibrated_surface17(name)) for name in SHIPPED_TUNINGS]


# What was made of each calibration, kept apart from the calibration (its
# pickle and its values are what they were) and by the calibration itself,
# not by its value (which would read all of it): its tuning values, the
# tuning of each label, and the interactions made for each pair of labels.
_MADE: dict[int, _Made] = {}
_MADE_LOCK = threading.Lock()
# The interactions kept per calibration before they are made again.
_INTERACTIONS_KEPT = 1024


class _Made:
    __slots__ = ("values", "tunings", "interactions")

    def __init__(self, calibration: Calibration) -> None:
        used, g_c, circuits = calibration._tunings()
        self.values = (used, g_c, dict(circuits))
        self.tunings: dict[str, Any] = {}
        self.interactions: dict[tuple[tuple[str, str], int, int], Any] = {}


def _made(calibration: Calibration) -> _Made:
    """What is kept of *calibration*; called under ``_MADE_LOCK``."""
    key = id(calibration)
    made = _MADE.get(key)
    if made is None:
        made = _MADE[key] = _Made(calibration)
        # Forgotten with the calibration, whose id another may then take.
        weakref.finalize(calibration, _MADE.pop, key, None)
    return made


def _derived_tuning(calibration: Calibration, label: str) -> Tuning | None:
    """The ``FluxTuning`` of the qudit labelled *label*, from its circuit in
    *calibration*, or None for a qudit whose circuit the calibration does
    not give. Made once per calibration and label."""
    with _MADE_LOCK:
        made = _made(calibration)
        if label not in made.tunings:
            circuit = made.values[2].get(label)
            if circuit is None:
                made.tunings[label] = None
            else:
                from .tuning import FluxTuning

                ec, ej_max, d, idle = circuit
                made.tunings[label] = FluxTuning(
                    ec_hz=ec,
                    ej_max_hz=ej_max,
                    asymmetry=d,
                    idle_flux=idle,
                    source=f"{calibration.name} calibration, qudits.{label}: ec, ej_max, "
                    "asymmetry_d, sweet_spot_idle",
                )
        found: Tuning | None = made.tunings[label]
        return found


def _derived_interaction(
    calibration: Calibration, labels: tuple[str, str], tunings: tuple[Tuning, Tuning]
) -> CouplingInteraction | None:
    """The ``ChargeCoupling`` of the qudits labelled *labels*, made through
    *tunings* (the ones the system answers with, declared or derived) with
    the g_c of *calibration*; None if the calibration gives no g_c. Made
    once per pair and pair of tunings."""
    with _MADE_LOCK:
        made = _made(calibration)
        used, g_c, _ = made.values
        if g_c is None:
            return None
        # By the tunings themselves (the interaction keeps them, so the key
        # stays theirs): systems of one calibration that answer with
        # different tunings each keep their own.
        key = (labels, id(tunings[0]), id(tunings[1]))
        kept = made.interactions.get(key)
        if kept is None:
            from .tuning import ChargeCoupling

            if len(made.interactions) >= _INTERACTIONS_KEPT:
                made.interactions.clear()

            kept = made.interactions[key] = ChargeCoupling(
                charge_coupling_hz=g_c,
                tunings=tunings,
                source=f"{calibration.name} calibration, coupling.charge_coupling_g_c.{used}",
            )
        found: CouplingInteraction = kept
        return found


def _origin(calibration: Calibration | None, qudit: Any) -> tuple[Calibration, str] | None:
    """The calibration that gives *qudit* a tuning and the label it has
    there: the system's *calibration*, or with none a shipped one that
    builds a qudit of these values."""
    if calibration is not None:
        return (calibration, qudit.label)
    for shipped, description in _shipped():
        built = description.qudits.get(qudit.index)
        if built is not None and _same(_QUDIT_VALUES, built, qudit):
            return (shipped, built.label)
    return None


def tuning_of(
    description: _Description, calibration: Calibration | None, index: int
) -> Tuning | None:
    """The tuning of qudit *index* of *description*: declared, else what
    *calibration* (the system's) gives it, else None."""
    qudit = description.qudits.get(index)
    if qudit is None:
        raise NotFoundError(f"qudit {index!r} is not in quantum system {description.name!r}")
    declared: Tuning | None = qudit.tuning
    if declared is not None:
        return declared
    origin = _origin(calibration, qudit)
    if origin is None:
        return None
    return _derived_tuning(*origin)


def interaction_of(
    description: _Description,
    calibration: Calibration | None,
    coupling: Coupling,
    first: int | None = None,
) -> CouplingInteraction | None:
    """The interaction of *coupling* of *description*: declared, else, for a
    coupling with no control, the charge coupling made through the tunings
    the system answers with for its two qudits (declared or derived), with
    the g_c of the calibration that gives the qudits; else None. With
    *first*, the index of one of its qudits, as that qudit and then the
    other see it (its tunings in that order)."""
    declared = coupling.interaction
    turned = first is not None and first != coupling.qudits[0]
    if declared is not None or coupling.control is not None:
        return declared._reversed() if turned and declared is not None else declared
    qudits = description.qudits
    a, b = (qudits[q] for q in (coupling.qudits[::-1] if turned else coupling.qudits))
    one, other = _origin(calibration, a), _origin(calibration, b)
    known = one or other
    if known is None or (one and other and one[0] is not other[0]):
        return None
    source = known[0]
    if calibration is None:
        # With no calibration of its own, only a coupling the shipped one builds.
        built = next(d for c, d in _shipped() if c is source)
        if not any(_same(_COUPLING_VALUES, c, coupling) for c in built.couplings):
            return None
    through_a = tuning_of(description, calibration, a.index)
    through_b = tuning_of(description, calibration, b.index)
    if through_a is None or through_b is None:
        return None
    from .tuning import FluxTuning

    # A charge coupling is read through transmon circuits.
    if not (isinstance(through_a, FluxTuning) and isinstance(through_b, FluxTuning)):
        return None
    return _derived_interaction(source, (a.label, b.label), (through_a, through_b))
