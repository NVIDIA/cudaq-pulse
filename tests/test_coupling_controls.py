# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qudit tunings and coupling controls: the components a quantum system
declares for its coupling lines, their read-only maps, and that a system
without them is what it was."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import (
    ChargeCoupling,
    Coupling,
    CouplingControl,
    CouplingInteraction,
    ExchangeControl,
    ExponentialSettling,
    FluxTuning,
    HighPass,
    PhysicalQudit,
    Response,
    Tuning,
    transmon_surface17,
)
from cudaq_pulse.quantum_systems.calibration import shipped_calibration
from cudaq_pulse.quantum_systems.transmon_model import TransmonCircuit

D4, X2 = 3, 10


@pytest.fixture(scope="module")
def qs():
    return pulse.quantum_system("transmon_surface17")


def _pair(control=None, tunings=(None, None), interaction=None):
    return pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, f, 50.0, 30.0, anharmonicity_hz=-2.0e8, tuning=t)
            for i, (f, t) in enumerate(zip([4.8e9, 6.9e9], tunings, strict=True))
        },
        couplings=[Coupling((0, 1), strength_hz=5.7e6, control=control, interaction=interaction)],
    )


def test_every_shipped_qudit_is_tuned_by_its_calibrated_circuit(qs):
    cal = shipped_calibration("transmon_surface17")
    for q in qs.physical_qudits.values():
        tuning = q.tuning
        assert isinstance(tuning, FluxTuning) and isinstance(tuning, Tuning)
        assert tuning.unit == "flux_quantum"
        assert tuning.source.startswith(f"transmon_surface17 calibration, qudits.{q.label}:")
        assert cal.value("qudits", q.label, "ec") == tuning.ec_hz
        assert tuning.idle_flux == cal.idle_flux(q.label)
        # The circuit is fitted to the published idle frequency.
        assert q.frequency_at(0.0) == pytest.approx(q.frequency_hz, abs=1e3)
        assert q.anharmonicity_at(0.0) == pytest.approx(q.anharmonicity_hz, abs=1e3)


def test_frequency_at_is_the_circuit_and_the_calibration_conversion(qs):
    cal = shipped_calibration("transmon_surface17")
    d4 = qs.physical_qudits[D4]
    circuit = TransmonCircuit(d4.tuning.ec_hz, d4.tuning.ej_max_hz, d4.tuning.asymmetry)
    for amplitude in (-0.3, -0.1, 0.0):
        flux = 0.5 + amplitude
        assert d4.frequency_at(amplitude) == pytest.approx(circuit.f01_hz(flux), rel=1e-12)
        assert d4.anharmonicity_at(amplitude) == pytest.approx(
            circuit.anharmonicity_hz(flux), rel=1e-9
        )
        assert d4.frequency_at(amplitude) - d4.frequency_at(0.0) == pytest.approx(
            cal.detuning_for_flux("D4", flux), rel=1e-9
        )


def test_amplitudes_are_signed_and_even_about_the_sweet_spot(qs):
    d4 = qs.physical_qudits[D4]
    amplitudes = np.linspace(-0.3, 0.3, 13)
    frequencies = d4.frequency_at(amplitudes)
    assert frequencies.shape == (13,)
    np.testing.assert_allclose(frequencies, frequencies[::-1], rtol=1e-12)
    # A data qudit idles at its lower sweet spot: any flux tunes it up.
    assert np.all(frequencies >= d4.frequency_at(0.0) - 1.0)
    # amplitude_for inverts on the branch of the idle point.
    found = d4.amplitude_for(frequencies[:7])
    np.testing.assert_allclose(found, amplitudes[:7], atol=1e-7)
    assert d4.amplitude_for(5.276e9) == pytest.approx(-0.30148, abs=1e-4)
    with pytest.raises(pulse.InvalidValueError, match="outside the tuning range"):
        d4.amplitude_for(9e9)


def test_the_coupling_at_its_idle_point_is_its_strength(qs):
    for c in qs.couplings:
        made = c.interaction
        assert isinstance(made, ChargeCoupling) and isinstance(made, CouplingInteraction)
        assert not isinstance(made, CouplingControl) and c.control is None
        assert made.unit == "flux_quantum"
        assert made.tunings == tuple(qs.physical_qudits[q].tuning for q in c.qudits)
        point = c.at({c.qudits[0]: 0.0})
        assert point.exchange_hz == pytest.approx(c.strength_hz, rel=1e-9)
        assert point.frequencies_hz == pytest.approx(
            {q: qs.physical_qudits[q].frequency_hz for q in c.qudits}, abs=1e3
        )


def test_the_cz_operating_point_of_x2_d4(qs):
    # Both qudits tuned: the data qudit to the interaction frequency, the
    # auxiliary's 1-2 transition there, so |11> meets |20>.
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    d4, x2 = qs.physical_qudits[D4], qs.physical_qudits[X2]
    interaction = c.get_calibration(pulse.gates.CZ).interaction_frequency_hz
    amplitudes = np.linspace(0.0, 0.3, 3001)
    f12 = x2.frequency_at(amplitudes) + x2.anharmonicity_at(amplitudes)
    at_aux = float(np.interp(interaction, f12[::-1], amplitudes[::-1]))
    point = c.at({qs.qudits[X2]: at_aux, D4: d4.amplitude_for(interaction)})
    assert point.frequencies_hz[D4] == pytest.approx(interaction, abs=1e3)
    assert point.frequencies_hz[X2] + point.anharmonicities_hz[X2] == pytest.approx(
        interaction, abs=2e5
    )
    assert point.exchange_hz == pytest.approx(6.38e6, abs=0.01e6)
    assert point.exchanges_hz[(1, 1), (2, 0)] == pytest.approx(8.876e6, abs=0.01e6)
    # The same from the calibration's charge-coupled pair (data first there).
    pair = shipped_calibration("transmon_surface17").pair("X2-D4", 3)
    n_d = pair.a.charge_matrix(0.5 + d4.amplitude_for(interaction), 3)
    n_a = pair.b.charge_matrix(at_aux, 3)
    assert point.exchanges_hz[(1, 1), (2, 0)] == pytest.approx(
        pair.g_c_hz * n_a[1, 2] * n_d[0, 1], rel=1e-9
    )
    assert point.exchanges_hz[(1, 1), (0, 2)] == pytest.approx(
        pair.g_c_hz * n_a[0, 1] * n_d[1, 2], rel=1e-9
    )


def test_one_qudit_moves_alone_and_arrays_broadcast(qs):
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    amplitudes = np.array([-0.2, 0.0, 0.2])
    point = c.at({qs.qudits[D4]: amplitudes})
    assert point.frequencies_hz[D4].shape == (3,)
    np.testing.assert_allclose(
        point.frequencies_hz[X2], qs.physical_qudits[X2].frequency_at(0.0), rtol=1e-12
    )
    np.testing.assert_allclose(
        point.frequencies_hz[D4], qs.physical_qudits[D4].frequency_at(amplitudes), rtol=1e-12
    )
    assert point.exchange_hz[0] == pytest.approx(point.exchange_hz[2], rel=1e-12)


def test_coupling_at_names_what_is_wrong(qs):
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    with pytest.raises(
        pulse.InvalidValueError, match=r"made by tuning its qudits: give the amplitude of each"
    ):
        c.at(0.1)
    with pytest.raises(pulse.InvalidValueError, match="4 is not a qudit of the coupling"):
        c.at({4: 0.1})
    with pytest.raises(TypeError):
        c.at(0.1, side=4)
    with pytest.raises(
        pulse.NotFoundError, match="has no control and no interaction: it is always on"
    ):
        pulse.quantum_system("transmon_lattice:2").couplings[0].at(0.0)
    with pytest.raises(pulse.NotFoundError, match=r"qudit 0 \(Q0\) has no tuning"):
        pulse.quantum_system("transmon_lattice:2").physical_qudits[0].frequency_at(0.1)


def test_an_exchange_control_is_the_pairs_own():
    control = ExchangeControl(
        unit="volt", amplitudes=(-0.1, 0.0, 0.2), exchange_hz=(0.0, 5.7e6, 25.7e6)
    )
    pair = _pair(control)
    c = pair.couplings[0]
    assert c.at(0.0).exchange_hz == pytest.approx(5.7e6)
    assert c.at(0.1).exchange_hz == pytest.approx(15.7e6)
    np.testing.assert_allclose(c.at(np.array([-0.05, 0.2])).exchange_hz, [2.85e6, 25.7e6])
    assert c.at(0.1).frequencies_hz == {}
    with pytest.raises(pulse.InvalidValueError, match="turned on by its own control"):
        c.at({0: 0.1})
    with pytest.raises(pulse.InvalidValueError, match="tabulated from -0.1 to 0.2 volt"):
        c.at(0.3)


@pytest.mark.parametrize(
    ("control", "path", "match"),
    [
        (
            ExchangeControl(unit="volt", amplitudes=(0.0, 0.0), exchange_hz=(1.0, 2.0)),
            "Coupling((0, 1)).control.amplitudes",
            "finite and increasing",
        ),
        (
            ExchangeControl(unit="volt", amplitudes=(0.1, 0.2), exchange_hz=(1.0, 2.0)),
            "Coupling((0, 1)).control.amplitudes",
            "does not span the idle point",
        ),
        (
            ExchangeControl(unit="", amplitudes=(0.0, 0.2), exchange_hz=(1.0,)),
            "Coupling((0, 1)).control.unit",
            "not the name of a unit",
        ),
        ("flux", "Coupling((0, 1)).control", "must be a CouplingControl"),
    ],
)
def test_a_control_with_problems_is_refused_at_its_path(control, path, match):
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(control)
    assert any(p.path == path and match in p.message for p in raised.value.problems), str(
        raised.value
    )


def test_a_charge_coupling_is_an_interaction_made_through_its_qudits_tunings(qs):
    d4, x2 = qs.physical_qudits[D4].tuning, qs.physical_qudits[X2].tuning
    made = ChargeCoupling(charge_coupling_hz=3.1e6, tunings=(d4, x2))
    coupling = _pair(None, (d4, x2), made).couplings[0]
    assert coupling.interaction is made and coupling.control is None
    assert "interaction=ChargeCoupling(" in repr(coupling) and "control=" not in repr(coupling)
    # It is not a control: no line reaches it.
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingControl"):
        _pair(made, (d4, x2))
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingInteraction"):
        _pair(None, (d4, x2), ExchangeControl(unit="v", amplitudes=(0.0, 1.0), exchange_hz=(1, 2)))
    bad = ChargeCoupling(charge_coupling_hz=-1.0, tunings=(d4.replace(ec_hz=0.0), x2))
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(None, (d4, x2), bad)
    paths = sorted(p.path for p in raised.value.problems)
    assert paths == [
        "Coupling((0, 1)).interaction.charge_coupling_hz",
        "Coupling((0, 1)).interaction.tunings[0].ec_hz",
    ]


def test_the_published_system_and_untuned_systems_are_unchanged():
    published = transmon_surface17(calibration=None)
    assert all(q.tuning is None for q in published.physical_qudits.values())
    assert all(c.control is None and c.interaction is None for c in published.couplings)
    # The shipped system with its calibration has the fingerprint it has
    # without tunings: they follow from the calibration, which is in it.
    assert pulse.quantum_system("transmon_surface17").fingerprint == (
        "c206f28b5e1c7f2351cc47c17f36449b"
    )
    # Fingerprints of systems without tunings, as before tunings existed.
    assert pulse.quantum_system("transmon_lattice:3").fingerprint == (
        "9814de8af8ac7d098baf74228631cfd4"
    )
    assert published.fingerprint == "acdf23ba20dd32bca283e0e63696d9af"
    assert _pair().fingerprint == "0cc289b040d470cdc776a0bc3d7526ca"
    q = _pair().physical_qudits[0]
    assert "tuning" not in repr(q) and "control" not in repr(_pair().couplings[0])
    data = _pair().as_dict()
    assert "tuning" not in data["qudits"][0] and "control" not in data["couplings"][0]


def test_qudits_pickle_as_before_and_with_their_tuning():
    plain = PhysicalQudit(0, 4.8e9, 50.0, 30.0, anharmonicity_hz=-2.0e8)
    noted = plain.replace(metadata={"fridge": "a"})
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    # A qudit without a tuning has the state it had before tunings existed.
    assert len(plain.__reduce__()[2]) == 11 and len(noted.__reduce__()[2]) == 12
    for q in (plain, noted, plain.replace(tuning=tuning), noted.replace(tuning=tuning)):
        back = pickle.loads(pickle.dumps(q))
        assert back == q and back.tuning == q.tuning and back.metadata == q.metadata


def test_a_wrong_tuning_or_control_is_refused_even_when_falsy():
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a Tuning, got dict"):
        _pair(None, ({}, None))
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingControl"):
        _pair({})
    outside = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=-0.1)
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="between the sweet spots"):
        _pair(None, (outside, None))


def test_the_shipped_tunings_are_made_when_first_asked_for():
    code = (
        "import sys, pickle, cudaq_pulse as pulse\n"
        "loaded = lambda: 'cudaq_pulse.quantum_systems.tuning' in sys.modules\n"
        "qs = pulse.quantum_system('transmon_surface17')\n"
        "qs.fingerprint, qs.as_dict(), pickle.dumps(qs), qs.replace(layout=[3, 10])\n"
        "@pulse.kernel\n"
        "def k(q):\n"
        "    drive(get_drive_line(q), square(20, 0.1))\n"
        "k.lower(qs.qudits[0]).mlir()\n"
        "print(loaded())\n"
        "print(type(qs.physical_qudits[3].tuning).__name__, loaded())\n"
        "print(type(qs.couplings[0].interaction).__name__, qs.couplings[0].control)\n"
        "lattice = pulse.quantum_system('transmon_lattice:4')\n"
        "print(lattice.physical_qudits[0].tuning, lattice.couplings[0].interaction)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == [
        "False", "FluxTuning", "True", "ChargeCoupling", "None", "None", "None",
    ]  # fmt: skip


def test_tunings_change_the_fingerprint():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    tuned = _pair(None, (tuning, None))
    assert tuned.fingerprint != _pair().fingerprint
    assert tuned.fingerprint != _pair(None, (tuning.replace(idle_flux=0.0), None)).fingerprint


def test_tunings_and_controls_pickle_and_round_trip_through_a_system_file(qs):
    variant = qs.replace(layout=[3, 10])
    data = json.loads(
        json.dumps(
            variant.replace(qudits=dict(qs.physical_qudits), couplings=list(qs.couplings)).as_dict()
        )
    )
    assert data["qudits"][D4]["tuning"]["kind"] == "flux"
    made = data["couplings"][0]["interaction"]
    assert made["kind"] == "charge" and "tunings" not in made
    assert "control" not in data["couplings"][0]
    loaded = pulse.QuantumSystem.from_dict(data)
    assert loaded.physical_qudits == qs.physical_qudits
    assert tuple(loaded.couplings) == tuple(qs.couplings)
    exchange = ExchangeControl(unit="volt", amplitudes=(0.0, 1.0), exchange_hz=(1e6, 2e6))
    data = json.loads(json.dumps(_pair(exchange).as_dict()))
    assert data["couplings"][0]["control"] == {
        "kind": "exchange",
        "unit": "volt",
        "amplitudes": [0.0, 1.0],
        "exchange_hz": [1e6, 2e6],
        "source": "",
    }
    assert pulse.QuantumSystem.from_dict(data).couplings[0].control == exchange
    # A system file from before responses existed reads as one without.
    assert "response" not in data["couplings"][0]["control"]
    for value in (qs.physical_qudits[D4], qs.couplings[0], exchange):
        assert pickle.loads(pickle.dumps(value)) == value


def test_a_response_is_declared_checked_and_kept():
    settle = ExponentialSettling(amplitude=0.02, tau_ns=40.0)
    tee = HighPass(tau_ns=19200.0)
    assert isinstance(settle, Response) and isinstance(tee, Response)
    base = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    assert base.response == ()
    # One response, or a sequence applied in order.
    assert base.replace(response=settle).response == (settle,)
    tuning = base.replace(response=[settle, tee])
    assert tuning.response == (settle, tee)
    assert repr(settle) == "ExponentialSettling(amplitude=0.02, tau_ns=40.0)"
    assert "response=" not in repr(base) and "HighPass(tau_ns=19200.0)" in repr(tuning)
    # It does not move the static maps, and it is part of what the system is.
    assert tuning.frequency_at(0.1) == base.frequency_at(0.1)
    tuned = _pair(None, (tuning, None))
    assert tuned.fingerprint != _pair(None, (base, None)).fingerprint
    assert pickle.loads(pickle.dumps(tuning)) == tuning
    data = json.loads(json.dumps(tuned.as_dict()))
    assert data["qudits"][0]["tuning"]["response"] == [
        {"kind": "exponential_settling", "amplitude": 0.02, "tau_ns": 40.0},
        {"kind": "high_pass", "tau_ns": 19200.0},
    ]
    assert pulse.QuantumSystem.from_dict(data).physical_qudits[0].tuning == tuning
    exchange = ExchangeControl(
        unit="volt", amplitudes=(0.0, 1.0), exchange_hz=(1e6, 2e6), response=tee
    )
    data = json.loads(json.dumps(_pair(exchange).as_dict()))
    assert data["couplings"][0]["control"]["response"] == [{"kind": "high_pass", "tau_ns": 19200.0}]
    assert pulse.QuantumSystem.from_dict(data).couplings[0].control == exchange


@pytest.mark.parametrize(
    ("response", "path", "match"),
    [
        (ExponentialSettling(amplitude=1.5, tau_ns=40.0), "response[0].amplitude", "between -1"),
        (ExponentialSettling(amplitude=0.1, tau_ns=0.0), "response[0].tau_ns", "above 0 ns"),
        ([HighPass(tau_ns=10.0), HighPass(tau_ns=-1.0)], "response[1].tau_ns", "above 0 ns"),
        ("exponential", "response[0]", "must be an ExponentialSettling or a HighPass"),
    ],
)
def test_a_response_with_problems_is_refused_at_its_path(response, path, match):
    tuning = FluxTuning(
        ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5, response=response
    )
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(None, (tuning, None))
    assert any(
        p.path == f"qudits[0].tuning.{path}" and match in p.message for p in raised.value.problems
    ), str(raised.value)


def test_a_system_file_names_a_response_that_is_not_one():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    data = json.loads(json.dumps(_pair(None, (tuning, None)).as_dict()))
    data["qudits"][0]["tuning"]["response"] = [{"kind": "lowpass", "tau_ns": 1.0}]
    with pytest.raises(
        pulse.QuantumSystemDefinitionError, match=r"response\[0\] is not a response"
    ):
        pulse.QuantumSystem.from_dict(data)


def test_the_tuning_module_loads_only_with_a_system_that_declares_one():
    code = (
        "import sys, cudaq_pulse as pulse\n"
        "qs = pulse.quantum_system('transmon_lattice:4')\n"
        "@pulse.kernel\n"
        "def k(q):\n"
        "    drive(get_drive_line(q), square(20, 0.1))\n"
        "k.lower(qs.qudits[0]).mlir()\n"
        "print('cudaq_pulse.quantum_systems.tuning' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False"]
