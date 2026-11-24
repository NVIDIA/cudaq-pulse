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
    found: Tuning | None = origin[0]._tuning(origin[1])
    return found


def interaction_of(
    description: _Description, calibration: Calibration | None, coupling: Coupling
) -> CouplingInteraction | None:
    """The interaction of *coupling* of *description*: declared, else, for a
    coupling with no control whose two qudits a calibration gives tunings,
    the charge coupling that calibration gives, else None."""
    declared = coupling.interaction
    if declared is not None or coupling.control is not None:
        return declared
    qudits = description.qudits
    a, b = (qudits[q] for q in coupling.qudits)
    if a.tuning is not None or b.tuning is not None:
        return None
    first, second = _origin(calibration, a), _origin(calibration, b)
    if first is None or second is None or first[0] is not second[0]:
        return None
    source = first[0]
    if calibration is None:
        # With no calibration of its own, only a coupling the shipped one builds.
        built = next(d for c, d in _shipped() if c is source)
        if not any(_same(_COUPLING_VALUES, c, coupling) for c in built.couplings):
            return None
    if source._tuning(first[1]) is None or source._tuning(second[1]) is None:
        return None
    found: CouplingInteraction = source._interaction(first[1], second[1])
    return found
