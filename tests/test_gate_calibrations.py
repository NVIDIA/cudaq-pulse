# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate calibrations: pulse.gates, get_calibration on lines, qudits and
couplings, and the pulses they derive where a device publishes none."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.gates import CZ, READOUT, X90, Y90, Gate, X, Y
from cudaq_pulse.targets import (
    Coupling,
    DispersiveReadout,
    DragPulse,
    FluxCZ,
    MicrowaveDrive,
    Qubit,
    ReadoutPulse,
    TargetDescription,
)

_X = DragPulse(duration_ns=40.0, sigma_ns=10.0)
_AREA_NS = 10.0 * math.sqrt(2 * math.pi) * math.erf(40.0 / (2 * math.sqrt(2) * 10.0))
_READOUT = ReadoutPulse(duration_ns=300.0, sigma_ns=10.0, integration_ns=400.0)


def _qudit(index=0, **fields):
    fields.setdefault("calibrations", {X: _X, READOUT: _READOUT})
    return Qubit(index, 5.0e9 + 1e8 * index, -2.0e8, 50.0, 30.0, **fields)


def _device(**fields):
    qudits = {0: _qudit(0, **fields), 1: _qudit(1)}
    cz = FluxCZ(duration_ns=60.0, buffer_ns=10.0, interaction_frequency_hz=4.4e9)
    couplings = [Coupling((0, 1), 5.0e6, calibrations={CZ: cz})]
    target = TargetDescription(name="gates", qubits=qudits, couplings=couplings)
    return pulse.device(target)


def _mlir(lowered, name):
    return lowered.mlir().replace(name, "kernel")


# ── pulse.gates ─────────────────────────────────────────────────────────────


def test_the_gates_are_one_name_each():
    assert [g.name for g in Gate] == ["X", "X90", "Y", "Y90", "CZ", "READOUT"]
    assert [str(g) for g in Gate] == ["x", "x90", "y", "y90", "cz", "readout"]
    assert (pulse.gates.X, pulse.gates.READOUT) == (Gate.X, Gate.READOUT)
    assert not hasattr(pulse.gates, "SX") and not hasattr(pulse.gates, "MEASURE")


def test_the_gate_values_are_the_ones_targets_check():
    from cudaq_pulse.targets.base import _GATES

    assert set(_GATES) == {str(g) for g in Gate}


def test_neither_import_nor_a_device_loads_the_gates():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cudaq_pulse as p; p.device('transmon_17q'); "
            "print('cudaq_pulse.gates' in sys.modules); p.gates; "
            "print('cudaq_pulse.gates' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["False", "True"]


# ── Derivations ─────────────────────────────────────────────────────────────


def test_an_unpublished_amplitude_makes_the_rotation():
    qudit = _qudit()
    assert qudit.get_calibration(X) == _X.replace(amplitude=math.pi / _AREA_NS)
    assert qudit.get_calibration(X90) == _X.replace(amplitude=math.pi / 2 / _AREA_NS)
    scaled = _qudit(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=2.0))
    assert scaled.get_calibration(X).amplitude == pytest.approx(math.pi / (2.0 * _AREA_NS))


def test_x90_y_and_y90_derive_from_x():
    x = _X.replace(amplitude=0.3, beta_ns=0.5)
    qudit = _qudit(calibrations={X: x})
    assert qudit.get_calibration(X) == x
    assert qudit.get_calibration(X90) == x.replace(amplitude=0.15)
    assert qudit.get_calibration(Y) == x.replace(phase_rad=math.pi / 2)
    assert qudit.get_calibration(Y90) == x.replace(amplitude=0.15, phase_rad=math.pi / 2)


def test_a_published_calibration_is_used_as_published():
    x90 = DragPulse(duration_ns=20.0, sigma_ns=5.0, amplitude=0.4)
    qudit = _qudit(calibrations={X: _X, X90: x90})
    assert qudit.get_calibration(X90) == x90
    assert qudit.get_calibration(Y90) == x90.replace(phase_rad=math.pi / 2)


def test_what_is_neither_published_nor_derived_names_the_qudit():
    bare = _qudit(label="D1", calibrations={})
    with pytest.raises(pulse.NotFoundError) as raised:
        bare.get_calibration(READOUT)
    assert str(raised.value).strip("'\"") == (
        "qudit 0 (D1) publishes no pulse.gates.READOUT calibration; it publishes none"
    )
    with pytest.raises(pulse.NotFoundError, match="publishes no pulse.gates.X calibration"):
        bare.get_calibration(Y90)
    with pytest.raises(pulse.InvalidValueError, match="CZ is calibrated on a coupling"):
        _qudit().get_calibration(CZ)
    with pytest.raises(pulse.InvalidValueError, match="did you mean pulse.gates.X90"):
        _qudit().get_calibration("x91")
    with pytest.raises(pulse.InvalidTypeError, match="a gate is one of"):
        _qudit().get_calibration(1)


def test_another_transition_must_be_published():
    x12 = DragPulse(duration_ns=40.0, sigma_ns=10.0, amplitude=0.2)
    qudit = _qudit(calibrations={X: _X, (X, (1, 2)): x12})
    assert qudit.get_calibration(X, transition=(1, 2)) == x12
    assert qudit.get_calibration(X90, transition=[1, 2]) == x12.replace(amplitude=0.1)
    assert qudit.calibrations == {"x": _X, ("x", (1, 2)): x12}
    with pytest.raises(pulse.NotFoundError) as raised:
        _qudit().get_calibration(X, transition=(1, 2))
    assert "publishes no pulse.gates.X (1, 2) calibration; it publishes pulse.gates.X, " in str(
        raised.value
    )
    for wrong in ((1, 1), (2, 1), (0,), "01", (0, 1.0)):
        with pytest.raises(pulse.InvalidValueError, match="a transition is"):
            _qudit().get_calibration(X, transition=wrong)


def test_a_qudit_checks_its_calibrations():
    with pytest.raises(
        pulse.InvalidTypeError, match="X calibration of qudit 0 must be a DragPulse"
    ):
        _qudit(calibrations={X: _READOUT})
    with pytest.raises(pulse.InvalidValueError, match="CZ is a coupling's calibration"):
        _qudit(calibrations={CZ: FluxCZ(duration_ns=60.0)})
    with pytest.raises(pulse.InvalidTypeError, match="map gates to calibrations"):
        _qudit(calibrations=[X])
    with pytest.raises(TypeError, match="read-only"):
        _qudit().calibrations[X] = _X  # type: ignore[index]


# ── On a device ─────────────────────────────────────────────────────────────


def test_a_qudit_handle_and_a_coupling_give_the_device_calibrations():
    dev = pulse.device("transmon_17q")
    q = dev.qudits[4]
    assert q.get_calibration(X) == dev.target.qubits[4].get_calibration(X)
    assert q.get_calibration(X).amplitude == pytest.approx(math.pi / _AREA_NS)
    assert q.get_calibration(READOUT) == ReadoutPulse(
        duration_ns=300.0, sigma_ns=10.0, integration_ns=400.0
    )
    a, d = dev.qudit("X1"), dev.qudit("D2")
    cz = dev.coupling(a, d).get_calibration(CZ)
    assert (cz.duration_ns, cz.buffer_ns) == (98.0, 15.0)
    assert cz.interaction_frequency_hz == pytest.approx(4.693e9)
    assert dev.coupling(d, a) is dev.coupling(a, d)
    assert dev.coupling(a, d).strength_hz == dev.target.coupling(a.index, d.index).strength_hz


def test_a_coupling_of_a_device_takes_its_handles():
    dev = pulse.device("transmon_17q")
    with pytest.raises(pulse.NotFoundError, match="are not coupled"):
        dev.coupling(dev.qudits[0], dev.qudits[1])
    with pytest.raises(pulse.InvalidTypeError, match="takes two qudit handles"):
        dev.coupling(0, 1)  # type: ignore[arg-type]
    other = _device()
    with pytest.raises(pulse.DeviceMismatchError, match="not of 'transmon_17q'"):
        dev.coupling(other.qudits[0], dev.qudits[0])


def test_the_fingerprint_holds_the_calibrations():
    prints = {
        _device().fingerprint,
        _device(calibrations={X: _X}).fingerprint,
        _device(calibrations={X: _X.replace(amplitude=0.1)}).fingerprint,
        _device(calibrations={X: _X, (X, (1, 2)): _X}).fingerprint,
    }
    assert len(prints) == 4


# ── In a kernel ─────────────────────────────────────────────────────────────


def test_a_line_gives_its_qudits_calibrated_pulses_and_they_play():
    dev = _device()
    amplitude = math.pi / _AREA_NS

    @pulse.kernel
    def calibrated(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        x = d.get_calibration(pulse.gates.X)
        drive(d, x)
        drive(d, x.replace(amplitude=x.amplitude / 2))
        sync(d, r)
        readout(r, r.get_calibration(pulse.gates.READOUT))

    @pulse.kernel
    def written(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, drag(40 * pulse.ns, amplitude, 10 * pulse.ns, 0 * pulse.ns))
        drive(d, drag(40 * pulse.ns, amplitude / 2, 10 * pulse.ns, 0 * pulse.ns))
        sync(d, r)
        readout(r, gaussian_square(300 * pulse.ns, 1.0, 10 * pulse.ns, 260 * pulse.ns))

    q = dev.qudits[0]
    assert _mlir(calibrated.lower(q), "calibrated") == _mlir(written.lower(q), "written")


def test_a_y_pulse_plays_x_at_ninety_degrees():
    dev = _device()

    @pulse.kernel
    def y(q):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.Y))

    text = y.lower(dev.qudits[0]).mlir()
    assert "pulse.drag" in text and "pulse.square" in text and "pulse.mul" in text


def test_a_line_has_the_calibrations_of_its_kind():
    dev = _device()

    @pulse.kernel
    def wrong_line(q):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.READOUT))

    with pytest.raises(pulse.CompilationError, match="it is the readout line's"):
        wrong_line.lower(dev.qudits[0])

    @pulse.kernel
    def cz_on_a_line(q):
        r = get_readout_line(q)
        r.get_calibration(pulse.gates.CZ)

    with pytest.raises(pulse.CompilationError, match=r"dev.coupling\(a, b\)"):
        cz_on_a_line.lower(dev.qudits[0])


def test_a_runtime_qudit_has_no_calibration_while_compiling():
    dev = _device()

    @pulse.kernel
    def slotted(q: pulse.Qudit):
        d = get_drive_line(q)
        drive(d, d.get_calibration(pulse.gates.X))

    with pytest.raises(pulse.CompilationError, match="runtime parameter"):
        slotted.lower(dev.qudits[0])


def test_what_does_not_play_says_why():
    dev = _device()
    cz = dev.coupling(dev.qudits[0], dev.qudits[1]).get_calibration(CZ)

    @pulse.kernel
    def flux(q):
        drive(get_drive_line(q), cz)

    with pytest.raises(pulse.CompilationError, match="a FluxCZ does not play"):
        flux.lower(dev.qudits[0])

    @pulse.kernel
    def unpublished(q):
        drive(get_drive_line(q), _X)

    with pytest.raises(pulse.CompilationError, match="has no amplitude, so it does not play"):
        unpublished.lower(dev.qudits[0])

    @pulse.kernel
    def not_a_waveform(q):
        drive(get_drive_line(q), 3.0)

    with pytest.raises(pulse.CompilationError):
        not_a_waveform.lower(dev.qudits[0])


# ── Metadata ────────────────────────────────────────────────────────────────


def test_metadata_is_the_users_own_and_no_part_of_the_device():
    tagged = {"fridge": "B", "fit": {"r2": 0.99, "points": [1, 2, 3]}, "note": None}
    dev = _device(metadata=tagged)
    qudit = dev.target.qubits[0]
    assert qudit.metadata == tagged and _qudit().metadata == {}
    with pytest.raises(TypeError):
        qudit.metadata["fridge"] = "C"  # type: ignore[index]
    assert dev == _device() and dev.fingerprint == _device().fingerprint
    for wrong in ({"when": object()}, {1: "one"}, ["a"]):
        with pytest.raises(pulse.InvalidTypeError, match="metadata of qudit 0"):
            _qudit(metadata=wrong)


# ── System files ────────────────────────────────────────────────────────────


def test_calibrations_and_metadata_round_trip_through_a_system_file():
    x12 = DragPulse(duration_ns=40.0, sigma_ns=10.0, amplitude=0.2)
    dev = _device(
        calibrations={X: _X, (X, (1, 2)): x12, READOUT: _READOUT},
        metadata={"fridge": "B"},
        readout=DispersiveReadout(frequency_hz=7e9, dispersive_shift_hz=1e6, linewidth_hz=2e6),
    )
    data = dev.as_dict()
    qudit = data["quantum_system"]["qudits"][0]
    assert qudit["calibrations"] == [
        {"gate": "x", "duration_ns": 40.0, "sigma_ns": 10.0, "amplitude": None, "beta_ns": 0.0,
         "phase_rad": 0.0},
        {"gate": "x", "transition": [1, 2], "duration_ns": 40.0, "sigma_ns": 10.0,
         "amplitude": 0.2, "beta_ns": 0.0, "phase_rad": 0.0},
        {"gate": "readout", "duration_ns": 300.0, "amplitude": None, "sigma_ns": 10.0,
         "integration_ns": 400.0, "photon_number": 4.0},
    ]  # fmt: skip
    assert qudit["metadata"] == {"fridge": "B"}
    assert "calibrations" in data["quantum_system"]["couplings"][0]
    back = pulse.Device.from_dict(json.loads(json.dumps(data)))
    assert back == dev and back.target.qubits[0].metadata == {"fridge": "B"}
    assert back.target.qubits[0].get_calibration(X, (1, 2)) == x12


def test_a_system_file_names_a_calibration_of_the_wrong_kind(tmp_path):
    data = _device().as_dict()
    data["quantum_system"]["qudits"][0]["calibrations"] = [
        {"gate": "cz", "duration_ns": 60.0},
        {"gate": "x", "duration": 40.0, "sigma_ns": 10.0},
        {"duration_ns": 40.0},
    ]
    path = tmp_path / "qpu.json"
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(pulse.DeviceDefinitionError) as raised:
        pulse.Device.load(path)
    found = [(p.path, p.message) for p in raised.value.problems]
    where = "quantum_system.qudits[0].calibrations"
    assert found[:3] == [
        (f"{where}[0].gate", "is not a gate of a qudit"),
        (f"{where}[1].duration", "is not a known key"),
        (f"{where}[1]", "DragPulse.__init__() missing 1 required keyword-only argument: "
         "'duration_ns'"),
    ]  # fmt: skip
    assert (f"{where}[2].gate", "is missing") in found
