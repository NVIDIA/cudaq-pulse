# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A qudit's drive and readout: components of their modality."""

from __future__ import annotations

import copy
import math
import pickle
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.runtime._call_tree import ReadoutModel
from cudaq_pulse.runtime._calls import IQModel
from cudaq_pulse.targets import (
    DispersiveReadout,
    Drive,
    MicrowaveDrive,
    Qubit,
    Readout,
    TargetDescription,
    transmon_17q,
)


class _Fluorescence(Readout):
    """A readout no model knows."""

    __slots__ = ()
    _fields = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_values", ())


class _Optical(Drive):
    """A drive no model knows."""

    __slots__ = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_values", ())


def _target(**members):
    return TargetDescription(name="c", qubits={0: Qubit(0, 5e9, -2e8, 10.0, 5.0, **members)})


def test_a_component_is_a_read_only_value():
    a = DispersiveReadout(frequency_hz=7e9, linewidth_hz=1e6)
    b = DispersiveReadout(linewidth_hz=1e6, frequency_hz=7e9)
    assert a == b and hash(a) == hash(b)
    assert a != DispersiveReadout(frequency_hz=7e9) and a != MicrowaveDrive()
    assert repr(a) == "DispersiveReadout(frequency_hz=7000000000.0, linewidth_hz=1000000.0)"
    assert (a.frequency_hz, a.photon_number, a.error_1to0, a.depletion_ns) == (7e9, 4.0, 0.0, None)
    with pytest.raises(AttributeError, match="is read-only"):
        a.linewidth_hz = 2e6  # type: ignore[misc]
    with pytest.raises(AttributeError):
        a.kappa = 2e6  # type: ignore[attr-defined]
    changed = a.replace(linewidth_hz=2e6)
    assert changed.linewidth_hz == 2e6 and changed.frequency_hz == 7e9 and a.linewidth_hz == 1e6
    with pytest.raises(TypeError, match="kapa_hz"):
        a.replace(kapa_hz=2e6)
    with pytest.raises(TypeError):
        DispersiveReadout(7e9)  # type: ignore[call-arg]
    for back in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        assert back == a and type(back) is DispersiveReadout
    if sys.version_info >= (3, 13):
        assert copy.replace(a, photon_number=2.0).photon_number == 2.0  # type: ignore[attr-defined]
    drive = MicrowaveDrive(x_amplitude=0.2, x_duration_ns=40.0, x_sigma_ns=10.0)
    assert pickle.loads(pickle.dumps(drive)) == drive
    assert drive.amplitude_scale_rad_per_ns is None


def test_the_depletion_comes_from_the_readout():
    explicit = Qubit(0, 5e9, -2e8, 1, 1, readout=DispersiveReadout(depletion_ns=250.0))
    from_kappa = Qubit(0, 5e9, -2e8, 1, 1, readout=DispersiveReadout(linewidth_hz=1e6))
    assert explicit.depletion_ns == 250.0
    assert from_kappa.depletion_ns == pytest.approx(math.log(100) / (2 * math.pi * 1e-3))
    assert Qubit(0, 5e9, -2e8, 1, 1).depletion_ns == 0.0
    assert Qubit(0, 5e9, -2e8, 1, 1, readout=_Fluorescence()).depletion_ns == 0.0


def test_the_shipped_target_describes_its_drive_and_readout():
    qubit = transmon_17q().qubits[0]
    assert isinstance(qubit.drive, MicrowaveDrive) and isinstance(qubit.readout, DispersiveReadout)
    assert qubit.drive.x_duration_ns == 40.0 and qubit.drive.x_sigma_ns == 10.0
    assert qubit.readout.frequency_hz > 6e9 and qubit.readout.linewidth_hz > 0


def test_the_drive_strength_comes_from_the_drive():
    pi = MicrowaveDrive(x_amplitude=0.5, x_duration_ns=40.0, x_sigma_ns=10.0)
    area = 10.0 * math.sqrt(2 * math.pi) * math.erf(40.0 / (2 * math.sqrt(2) * 10.0))
    assert _target(drive=pi).drive_amplitude_scale(0) == pytest.approx(math.pi / (0.5 * area))
    scaled = _target(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=2.0))
    assert scaled.drive_amplitude_scale(0) == 2.0
    assert _target().drive_amplitude_scale(0) == 1.0
    with pytest.raises(pulse.InvalidValueError, match="must be positive"):
        _target(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.0)).drive_amplitude_scale(0)
    with pytest.raises(pulse.UnsupportedError, match="no drive model for _Optical"):
        _target(drive=_Optical()).drive_amplitude_scale(0)


def test_a_readout_no_model_knows_is_refused():
    target = _target(readout=_Fluorescence())
    with pytest.raises(pulse.UnsupportedError, match="no readout model for _Fluorescence"):
        IQModel(target, (0,), decay=False)
    with pytest.raises(pulse.UnsupportedError, match="no readout model for _Fluorescence"):
        ReadoutModel(target, (0,), (2,), decay=False)
    # Without a readout, ideal reports, and no IQ points.
    assert ReadoutModel(_target(), (0,), (2,), decay=True).reports(0, 1) == [(1, 1.0)]
    with pytest.raises(
        pulse.UnsupportedError, match="qudit 0 has no readout: IQ points are synthesized"
    ):
        IQModel(_target(), (0,), decay=False)


def test_the_iq_model_needs_the_dispersive_shift_and_the_linewidth():
    given = DispersiveReadout(frequency_hz=7e9, dispersive_shift_hz=1e6, linewidth_hz=2e6)
    IQModel(_target(readout=given), (0,), decay=False)
    message = r"qudit 0 need its DispersiveReadout's linewidth_hz, which it does not give"
    with pytest.raises(pulse.UnsupportedError, match=message):
        IQModel(_target(readout=given.replace(linewidth_hz=None)), (0,), decay=False)
    both = r"dispersive_shift_hz and linewidth_hz"
    with pytest.raises(pulse.UnsupportedError, match=both):
        IQModel(_target(readout=DispersiveReadout(frequency_hz=7e9)), (0,), decay=False)
