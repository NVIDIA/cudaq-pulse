# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qudit tunings, coupling controls and interactions: the components a
quantum system declares for its coupling lines, by qudit and by pair, their
read-only maps, and that a system without them is what it was."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import Coupling, PhysicalQudit, transmon_surface17
from cudaq_pulse.quantum_systems.calibration import shipped_calibration
from cudaq_pulse.quantum_systems.transmon_model import TransmonCircuit
from cudaq_pulse.quantum_systems.tuning import (
    ChargeCoupling,
    CouplingControl,
    CouplingInteraction,
    ExchangeControl,
    ExponentialSettling,
    FluxTuning,
    HighPass,
    Response,
    Tuning,
)

D4, X2 = 3, 10


@pytest.fixture(scope="module")
def shipped():
    return pulse.quantum_system("transmon_surface17")


@pytest.fixture(scope="module")
def qs(shipped):
    """The shipped system's values with what its calibration gives its
    qudits and couplings declared on the system."""
    return pulse.quantum_system(
        name="declared17",
        qudits=shipped.physical_qudits,
        couplings=shipped.couplings,
        tunings={i: shipped.tuning(i) for i in shipped.physical_qudits},
        interactions={
            c.qudits: ChargeCoupling(
                charge_coupling_hz=shipped.interaction(*c.qudits).charge_coupling_hz
            )
            for c in shipped.couplings
        },
    )


def _pair(control=None, tunings=(None, None), interaction=None, **more):
    components = {}
    if any(t is not None for t in tunings):
        components["tunings"] = {i: t for i, t in enumerate(tunings) if t is not None}
    if control is not None:
        components["controls"] = {(0, 1): control}
    if interaction is not None:
        components["interactions"] = {(0, 1): interaction}
    return pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, f, 50.0, 30.0, anharmonicity_hz=-2.0e8)
            for i, f in enumerate([4.8e9, 6.9e9])
        },
        couplings=[Coupling((0, 1), strength_hz=5.7e6)],
        **components,
        **more,
    )


def test_every_shipped_qudit_is_tuned_by_its_calibrated_circuit(qs):
    cal = shipped_calibration("transmon_surface17")
    for q in qs.physical_qudits.values():
        tuning = qs.tuning(q.index)
        assert isinstance(tuning, FluxTuning) and isinstance(tuning, Tuning)
        assert tuning.unit == "flux_quantum"
        assert tuning.source.startswith(f"transmon_surface17 calibration, qudits.{q.label}:")
        assert cal.value("qudits", q.label, "ec") == tuning.ec_hz
        assert tuning.idle_flux == cal.idle_flux(q.label)
        # The circuit is fitted to the published idle frequency.
        assert tuning.frequency_at(0.0) == pytest.approx(q.frequency_hz, abs=1e3)
        assert tuning.anharmonicity_at(0.0) == pytest.approx(q.anharmonicity_hz, abs=1e3)


def test_frequency_at_is_the_circuit_and_the_calibration_conversion(qs):
    cal = shipped_calibration("transmon_surface17")
    d4 = qs.tuning(D4)
    circuit = TransmonCircuit(d4.ec_hz, d4.ej_max_hz, d4.asymmetry)
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
    d4 = qs.tuning(D4)
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
        made = qs.interaction(*c.qudits)
        assert isinstance(made, ChargeCoupling) and isinstance(made, CouplingInteraction)
        assert not isinstance(made, CouplingControl) and qs.control(*c.qudits) is None
        assert made.unit == "flux_quantum"
        assert made.tunings == tuple(qs.tuning(q) for q in c.qudits)
        point = made.at(c.qudits, {c.qudits[0]: 0.0})
        assert point.exchange_hz == pytest.approx(c.strength_hz, rel=1e-9)
        assert point.frequencies_hz == pytest.approx(
            {q: qs.physical_qudits[q].frequency_hz for q in c.qudits}, abs=1e3
        )


def test_the_cz_operating_point_of_x2_d4(qs):
    # Both qudits tuned: the data qudit to the interaction frequency, the
    # auxiliary's 1-2 transition there, so |11> meets |20>.
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    made = qs.interaction(X2, D4)
    d4, x2 = qs.tuning(D4), qs.tuning(X2)
    interaction = c.get_calibration(pulse.gates.CZ).interaction_frequency_hz
    amplitudes = np.linspace(0.0, 0.3, 3001)
    f12 = x2.frequency_at(amplitudes) + x2.anharmonicity_at(amplitudes)
    at_aux = float(np.interp(interaction, f12[::-1], amplitudes[::-1]))
    point = made.at((X2, D4), {qs.qudits[X2]: at_aux, D4: d4.amplitude_for(interaction)})
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
    made = qs.interaction(X2, D4)
    amplitudes = np.array([-0.2, 0.0, 0.2])
    point = made.at((X2, D4), {qs.qudits[D4]: amplitudes})
    assert point.frequencies_hz[D4].shape == (3,)
    np.testing.assert_allclose(
        point.frequencies_hz[X2], qs.tuning(X2).frequency_at(0.0), rtol=1e-12
    )
    np.testing.assert_allclose(
        point.frequencies_hz[D4], qs.tuning(D4).frequency_at(amplitudes), rtol=1e-12
    )
    assert point.exchange_hz[0] == pytest.approx(point.exchange_hz[2], rel=1e-12)


def test_an_interaction_names_what_is_wrong(qs):
    made = qs.interaction(X2, D4)
    with pytest.raises(pulse.InvalidValueError, match="4 is not a qudit of the coupling"):
        made.at((X2, D4), {4: 0.1})
    # The interaction a system is given has no tunings; the one it answers with has.
    with pytest.raises(pulse.InvalidValueError, match=r"ask the quantum system for the"):
        ChargeCoupling(charge_coupling_hz=3e6).at((X2, D4), {})
    lattice = pulse.quantum_system("transmon_lattice:2")
    assert lattice.interaction(0, 1) is None and lattice.control(0, 1) is None
    assert lattice.tuning(0) is None


def test_an_exchange_control_is_the_pairs_own():
    control = ExchangeControl(
        unit="volt", amplitudes=(-0.1, 0.0, 0.2), exchange_hz=(0.0, 5.7e6, 25.7e6)
    )
    pair = _pair(control)
    c = pair.control(0, 1)
    assert c is control and pair.control(1, 0) is control and pair.interaction(0, 1) is None
    assert c.at(0.0).exchange_hz == pytest.approx(5.7e6)
    assert c.at(0.1).exchange_hz == pytest.approx(15.7e6)
    np.testing.assert_allclose(c.at(np.array([-0.05, 0.2])).exchange_hz, [2.85e6, 25.7e6])
    assert c.at(0.1).frequencies_hz == {}
    with pytest.raises(pulse.InvalidValueError, match="tabulated from -0.1 to 0.2 volt"):
        c.at(0.3)


@pytest.mark.parametrize(
    ("control", "path", "match"),
    [
        (
            ExchangeControl(unit="volt", amplitudes=(0.0, 0.0), exchange_hz=(1.0, 2.0)),
            "controls[(0, 1)].amplitudes",
            "finite and increasing",
        ),
        (
            ExchangeControl(unit="volt", amplitudes=(0.1, 0.2), exchange_hz=(1.0, 2.0)),
            "controls[(0, 1)].amplitudes",
            "does not span the idle point",
        ),
        (
            ExchangeControl(unit="", amplitudes=(0.0, 0.2), exchange_hz=(1.0,)),
            "controls[(0, 1)].unit",
            "not the name of a unit",
        ),
        ("flux", "controls[(0, 1)]", "must be a CouplingControl"),
    ],
)
def test_a_control_with_problems_is_refused_at_its_path(control, path, match):
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(control)
    assert any(p.path == path and match in p.message for p in raised.value.problems), str(
        raised.value
    )


def test_a_charge_coupling_is_an_interaction_made_through_its_qudits_tunings(qs):
    d4, x2 = qs.tuning(D4), qs.tuning(X2)
    given = ChargeCoupling(charge_coupling_hz=3.1e6)
    system = _pair(None, (d4, x2), given)
    # The system answers with the interaction made through its qudits'
    # tunings, in the order asked.
    made = system.interaction(0, 1)
    assert made.charge_coupling_hz == 3.1e6 and made.tunings == (d4, x2)
    assert system.interaction(0, 1) is made and system.control(0, 1) is None
    assert system.interaction(1, 0).tunings == (x2, d4)
    assert "interactions={(0, 1): ChargeCoupling(" in repr(system._description)
    # It is not a control: no line reaches it.
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingControl"):
        _pair(given, (d4, x2))
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingInteraction"):
        _pair(None, (d4, x2), ExchangeControl(unit="v", amplitudes=(0.0, 1.0), exchange_hz=(1, 2)))
    bad = ChargeCoupling(charge_coupling_hz=-1.0, tunings=(d4, x2))
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(None, (d4.replace(ec_hz=0.0), x2), bad)
    paths = sorted(p.path for p in raised.value.problems)
    assert paths == [
        "interactions[(0, 1)].charge_coupling_hz",
        "interactions[(0, 1)].tunings",
        "tunings[0].ec_hz",
    ]


def test_what_is_declared_names_qudits_and_coupled_pairs_of_the_system(qs):
    d4, x2 = qs.tuning(D4), qs.tuning(X2)
    given = ChargeCoupling(charge_coupling_hz=3.1e6)
    exchange = ExchangeControl(unit="v", amplitudes=(0.0, 1.0), exchange_hz=(1.0e6, 2.0e6))

    def problems(**components):
        with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
            _pair().replace(**components)
        return {p.path: p.message for p in raised.value.problems}

    # An interaction is made through both qudits' tunings.
    found = problems(tunings={0: d4}, interactions={(0, 1): given})
    assert found.keys() == {"interactions[(0, 1)]"}
    assert "qudit 1 has none" in found["interactions[(0, 1)]"]
    # A coupling is turned on by a control of its own or through its qudits.
    found = problems(
        tunings={0: d4, 1: x2}, controls={(1, 0): exchange}, interactions={(0, 1): given}
    )
    assert "is turned on by one of them" in found["interactions[(0, 1)]"]
    assert "is not a qudit of the quantum system" in problems(tunings={7: d4})["tunings[7]"]
    assert (
        "is not a pair of coupled qudits"
        in problems(controls={(0, 5): exchange})["controls[(0, 5)]"]
    )
    assert "is not a pair of coupled qudits" in problems(controls={0: exchange})["controls[0]"]
    assert (
        "is given twice"
        in problems(controls={(0, 1): exchange, (1, 0): exchange})["controls[(1, 0)]"]
    )
    assert "maps qudit indices to components, got list" in problems(tunings=[d4])["tunings"]
    # A pair is named in either order, and by handles as by indices.
    plain = _pair()
    one = plain.replace(tunings={plain.qudits[0]: d4}, controls={(1, 0): exchange})
    other = plain.replace(tunings={0: d4}, controls={(plain.qudits[0], plain.qudits[1]): exchange})
    assert one == other and one.fingerprint == other.fingerprint
    assert one.control(0, 1) is exchange and one.tuning(0) is d4 and one.tuning(1) is None
    with pytest.raises(pulse.InvalidTypeError, match="unexpected keyword argument 'tuning'"):
        pulse.quantum_system("transmon_lattice:2", tuning={0: d4})
    # A key is a qudit index: any integer names the same qudit, a bool or a float none.
    numbered = plain.replace(tunings={np.int64(0): d4}, controls={(np.int64(1), 0): exchange})
    assert numbered == one and numbered.fingerprint == one.fingerprint
    json.dumps(numbered.as_dict())
    for wrong in (True, 0.0, "0", None):
        assert (
            "is not a qudit of the quantum system"
            in problems(tunings={wrong: d4})[f"tunings[{wrong!r}]"]
        )
    assert (
        "is not a pair of coupled qudits"
        in problems(controls={(True, 0): exchange})["controls[(True, 0)]"]
    )


def test_a_problem_of_what_a_system_file_declares_is_at_its_entry(qs):
    d4 = qs.tuning(D4)
    exchange = ExchangeControl(unit="v", amplitudes=(0.0, 1.0), exchange_hz=(1.0e6, 2.0e6))
    data = json.loads(json.dumps(_pair(exchange, (None, d4)).as_dict()))

    def paths(change):
        changed = json.loads(json.dumps(data))
        change(changed)
        with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
            pulse.QuantumSystem.from_dict(changed)
        return [p.path for p in raised.value.problems]

    # The tuning of qudit 1 is the first entry of the file's list.
    assert data["tunings"][0]["qudit"] == 1
    assert paths(lambda d: d["tunings"][0].update(ec_hz=0.0)) == ["tunings[0].ec_hz"]
    assert paths(lambda d: d["tunings"][0].update(qudit=7)) == ["tunings[0]"]
    assert paths(lambda d: d["controls"][0].update(amplitudes=[0.0, 0.0])) == [
        "controls[0].amplitudes"
    ]
    for wrong in ([1], True, "1", None):
        assert paths(lambda d, wrong=wrong: d["tunings"][0].update(qudit=wrong)) == [
            "tunings[0].qudit"
        ]
    for wrong in (0, [0], [[0], [1]], [0, 1, 2], {"a": 0}):
        assert paths(lambda d, wrong=wrong: d["controls"][0].update(qudits=wrong)) == [
            "controls[0].qudits"
        ]
    assert paths(lambda d: d["tunings"].append(dict(d["tunings"][0]))) == ["tunings[1]"]
    assert paths(lambda d: d["controls"].append({**d["controls"][0], "qudits": [1, 0]})) == [
        "controls[1]"
    ]
    assert paths(lambda d: d.update(tunings={})) == ["tunings"]


def test_the_published_system_and_untuned_systems_are_unchanged():
    published = transmon_surface17(calibration=None)
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
    # A qudit and a coupling hold no component: they are what they were.
    plain = _pair()
    assert type(plain) is pulse.QuantumSystem
    q, c = plain.physical_qudits[0], plain.couplings[0]
    assert not hasattr(q, "tuning") and not hasattr(c, "control")
    assert not hasattr(c, "interaction")
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    with pytest.raises(TypeError):
        PhysicalQudit(0, 4.8e9, 50.0, 30.0, tuning=tuning)
    with pytest.raises(TypeError):
        Coupling((0, 1), 5.7e6, control=None)
    data = plain.as_dict()
    assert not {"tunings", "controls", "interactions"} & data.keys()
    # An empty declaration is none.
    assert type(plain.replace(tunings={})) is pulse.QuantumSystem
    empty = pulse.quantum_system(
        name="pair",
        qudits=plain.physical_qudits,
        couplings=plain.couplings,
        tunings={},
        controls=None,
    )
    assert type(empty) is pulse.QuantumSystem and empty.fingerprint == plain.fingerprint


def test_a_system_that_declares_is_a_quantum_system_and_one_that_stops_is_plain():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    plain = _pair()
    tuned = _pair(None, (tuning, None))
    assert isinstance(tuned, pulse.QuantumSystem) and type(tuned) is not pulse.QuantumSystem
    assert tuned != plain and tuned == plain.replace(tunings={0: tuning})
    assert hash(tuned) == hash(plain.replace(tunings={0: tuning}))
    assert tuned.physical_qudits == plain.physical_qudits
    assert tuple(tuned.couplings) == tuple(plain.couplings)
    back = pickle.loads(pickle.dumps(tuned))
    assert back == tuned and back.tuning(0) == tuning and type(back) is type(tuned)
    # Its variants keep what it declares, for the qudits that remain.
    assert tuned.replace(layout=[1, 0]).tuning(0) == tuning
    assert tuned.replace(name="other").tuning(0) == tuning
    control = ExchangeControl(unit="v", amplitudes=(0.0, 1.0), exchange_hz=(1.0e6, 2.0e6))
    both = tuned.replace(controls={(0, 1): control})
    assert both.tuning(0) == tuning and both.control(0, 1) == control
    alone = both.replace(qudits={1: plain.physical_qudits[1]}, couplings=[])
    assert type(alone) is pulse.QuantumSystem
    # Taking the components away gives the plain system back.
    again = tuned.replace(tunings={})
    assert type(again) is pulse.QuantumSystem and again.fingerprint == plain.fingerprint
    assert again.tuning(0) is None


def test_a_wrong_tuning_or_control_is_refused_even_when_falsy():
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a Tuning, got dict"):
        _pair(None, ({}, None))
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="must be a CouplingControl"):
        _pair({})
    outside = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=-0.1)
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="between the sweet spots"):
        _pair(None, (outside, None))


def test_the_shipped_system_answers_for_what_its_calibration_gives(shipped):
    # It declares nothing: it is the system it was.
    assert type(shipped) is pulse.QuantumSystem
    assert shipped.true_qudits is shipped.physical_qudits
    d4, x2 = shipped.qudits[D4], shipped.qudits[X2]
    tuning = shipped.tuning(d4)
    assert isinstance(tuning, FluxTuning) and shipped.tuning(D4) is tuning
    assert tuning.source.startswith("transmon_surface17 calibration, qudits.D4:")
    made = shipped.interaction(x2, d4)
    assert isinstance(made, ChargeCoupling) and shipped.interaction(X2, D4) is made
    # Asked the other way, it is the same coupling with its tunings in that order.
    turned = shipped.interaction(D4, X2)
    assert turned.tunings == made.tunings[::-1] and shipped.interaction(d4, x2) is turned
    assert made.tunings == (shipped.tuning(X2), tuning)
    assert shipped.control(x2, d4) is None
    point = made.at((X2, D4), {d4: tuning.amplitude_for(5.276e9)})
    assert point.frequencies_hz[D4] == pytest.approx(5.276e9, abs=1e3)
    with pytest.raises(pulse.NotFoundError, match="not coupled"):
        shipped.interaction(0, 1)
    with pytest.raises(pulse.NotFoundError, match="qudit 99 is not in"):
        shipped.tuning(99)
    # A system with no calibration and other qudits has none.
    lattice = pulse.quantum_system("transmon_lattice:4")
    assert lattice.tuning(0) is None and lattice.interaction(0, 1) is None
    assert transmon_surface17(calibration=None).tuning(D4) is None


def test_the_shipped_tunings_are_made_when_first_asked_for():
    code = (
        "import sys, pickle, cudaq_pulse as pulse\n"
        "loaded = lambda: 'cudaq_pulse.quantum_systems.tuning' in sys.modules\n"
        "qs = pulse.quantum_system('transmon_surface17')\n"
        "qs.fingerprint, qs.as_dict(), pickle.dumps(qs), qs.replace(layout=[3, 10])\n"
        "pickle.dumps(qs.replace(layout=[3, 10]).replace(layout=[10, 3]))\n"
        "built = pulse.quantum_system(name='b', qudits=qs.physical_qudits, couplings=[])\n"
        "pickle.dumps(built.replace(layout=[3, 10]))\n"
        "qs.physical_qudits, qs.couplings, qs.coupling(qs.qudits[10], qs.qudits[3])\n"
        "drifting = pulse.quantum_system('transmon_surface17', parameter_drift=True)\n"
        "@pulse.kernel\n"
        "def k(q):\n"
        "    drive(get_drive_line(q), square(20, 0.1))\n"
        "k.lower(qs.qudits[0]).mlir()\n"
        "print(loaded() or 'cudaq_pulse.quantum_systems.components' in sys.modules)\n"
        "print(type(qs.tuning(3)).__name__, loaded())\n"
        "print(type(drifting.tuning(3)).__name__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "FluxTuning", "True", "FluxTuning"]


def test_a_system_rebuilt_from_anothers_values_gives_the_same_answers(shipped):
    # Built from the shipped qudits and couplings, with no calibration of
    # its own: its qudits are the calibrated ones, so they have their tunings.
    twin = pulse.quantum_system(
        name="twin", qudits=shipped.physical_qudits, couplings=shipped.couplings
    )
    assert twin.calibration is None
    assert twin.tuning(D4) == shipped.tuning(D4) and twin.tuning(D4) is shipped.tuning(D4)
    assert twin.interaction(X2, D4) is shipped.interaction(X2, D4)
    # The system file holds what was declared, nothing, and reads back as the system.
    data = json.loads(json.dumps(twin.as_dict()))
    assert not {"tunings", "controls", "interactions"} & data.keys()
    loaded = pulse.QuantumSystem.from_dict(data)
    assert loaded == twin and loaded.fingerprint == twin.fingerprint
    assert loaded.tuning(D4) == twin.tuning(D4)
    # A qudit that is changed is no longer the calibrated one.
    changed = shipped.physical_qudits[D4].replace(t1_us=40.0)
    variant = shipped.replace(qudits={**shipped.physical_qudits, D4: changed})
    assert variant.tuning(D4) is None and variant.tuning(X2) is shipped.tuning(X2)
    assert variant.interaction(X2, D4) is None
    built = pulse.quantum_system(
        name=shipped.name,
        qudits={**shipped.physical_qudits, D4: changed},
        couplings=shipped.couplings,
    )
    assert variant.fingerprint == built.fingerprint and built.tuning(X2) is shipped.tuning(X2)
    # Declaring it changes what the system is.
    declared = variant.replace(tunings={D4: shipped.tuning(D4)})
    assert declared.tuning(D4) == shipped.tuning(D4)
    assert declared.fingerprint != variant.fingerprint
    # The interaction of the pair is then made through it.
    assert declared.interaction(X2, D4).tunings == (shipped.tuning(X2), shipped.tuning(D4))


def test_a_variant_keeps_its_answers_in_another_process(shipped):
    from cudaq_pulse.controller import CouplingLine

    plain = shipped.replace(layout=[D4, X2])
    wired = shipped.replace(
        control_system=pulse.ControlSystem(
            controllers=[
                pulse.Controller().with_coupling_lines((CouplingLine(D4), CouplingLine(X2)))
            ]
        )
    )
    code = (
        "import sys, pickle, cudaq_pulse as pulse\n"
        "for line in sys.stdin.read().split():\n"
        "    qs = pickle.loads(bytes.fromhex(line))\n"
        "    lines = () if qs.control_system is None else "
        "qs.control_system.controllers[0].coupling_lines\n"
        "    print(type(qs.tuning(3)).__name__, type(qs.interaction(10, 3)).__name__, len(lines))\n"
    )
    sent = "\n".join(pickle.dumps(qs).hex() for qs in (plain, wired))
    out = subprocess.run(
        [sys.executable, "-c", code], input=sent, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["FluxTuning", "ChargeCoupling", "0"] + [
        "FluxTuning",
        "ChargeCoupling",
        "2",
    ]
    # A variant's coupling lines are checked when it is made.
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="qudits 3 and 10 has no control"):
        shipped.replace(
            control_system=pulse.ControlSystem(
                controllers=[pulse.Controller().with_coupling_lines([CouplingLine((D4, X2))])]
            )
        )


def test_tunings_change_the_fingerprint():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    tuned = _pair(None, (tuning, None))
    # Equal components, however their numbers are written, are one system.
    whole = tuning.replace(ec_hz=165000000, asymmetry=0.47, idle_flux=0.5)
    assert whole == tuning and type(whole.ec_hz) is int
    assert _pair(None, (whole, None)).fingerprint == tuned.fingerprint
    assert _pair(None, (whole, None)) == tuned
    assert tuned.fingerprint != _pair().fingerprint
    assert tuned.fingerprint != _pair(None, (tuning.replace(idle_flux=0.0), None)).fingerprint


def test_tunings_and_controls_pickle_and_round_trip_through_a_system_file(qs):
    variant = qs.replace(layout=[3, 10])
    assert variant.tuning(D4) == qs.tuning(D4)
    data = json.loads(json.dumps(qs.as_dict()))
    assert "tuning" not in data["qudits"][D4] and "interaction" not in data["couplings"][0]
    entry = data["tunings"][D4]
    assert entry["qudit"] == D4 and entry["kind"] == "flux"
    assert entry["ec_hz"] == qs.tuning(D4).ec_hz
    made = data["interactions"][0]
    assert made["kind"] == "charge" and "tunings" not in made
    assert tuple(made["qudits"]) in {tuple(sorted(c.qudits)) for c in qs.couplings}
    assert "controls" not in data
    loaded = pulse.QuantumSystem.from_dict(data)
    assert loaded == qs and loaded.fingerprint == qs.fingerprint
    assert loaded.tuning(D4) == qs.tuning(D4)
    assert loaded.interaction(X2, D4) == qs.interaction(X2, D4)
    exchange = ExchangeControl(unit="volt", amplitudes=(0.0, 1.0), exchange_hz=(1e6, 2e6))
    data = json.loads(json.dumps(_pair(exchange).as_dict()))
    assert data["controls"] == [
        {
            "qudits": [0, 1],
            "kind": "exchange",
            "unit": "volt",
            "amplitudes": [0.0, 1.0],
            "exchange_hz": [1e6, 2e6],
            "source": "",
        }
    ]
    assert pulse.QuantumSystem.from_dict(data).control(0, 1) == exchange
    # What a file declares is checked as what a program declares.
    data["controls"][0]["qudits"] = [0, 5]
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="not a pair of coupled qudits"):
        pulse.QuantumSystem.from_dict(data)
    data["controls"][0] = {"qudits": [0, 1], "kind": "capacitive"}
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.from_dict(data)
    assert [p.path for p in raised.value.problems] == ["controls[0].kind"]
    for value in (qs, variant, _pair(exchange), exchange, qs.tuning(D4), qs.interaction(X2, D4)):
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
    assert data["tunings"][0]["response"] == [
        {"kind": "exponential_settling", "amplitude": 0.02, "tau_ns": 40.0},
        {"kind": "high_pass", "tau_ns": 19200.0},
    ]
    assert pulse.QuantumSystem.from_dict(data).tuning(0) == tuning
    exchange = ExchangeControl(
        unit="volt", amplitudes=(0.0, 1.0), exchange_hz=(1e6, 2e6), response=tee
    )
    data = json.loads(json.dumps(_pair(exchange).as_dict()))
    assert data["controls"][0]["response"] == [{"kind": "high_pass", "tau_ns": 19200.0}]
    assert pulse.QuantumSystem.from_dict(data).control(0, 1) == exchange


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
        p.path == f"tunings[0].{path}" and match in p.message for p in raised.value.problems
    ), str(raised.value)


def test_a_system_file_names_a_response_that_is_not_one():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    data = json.loads(json.dumps(_pair(None, (tuning, None)).as_dict()))
    data["tunings"][0]["response"] = [{"kind": "lowpass", "tau_ns": 1.0}]
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


def test_an_interaction_is_made_through_the_tunings_the_system_answers_with(shipped):
    # One qudit declares its tuning (here, with the wiring's response); the
    # other's comes from the calibration: the interaction is made of both.
    settled = shipped.tuning(D4).replace(response=ExponentialSettling(amplitude=0.02, tau_ns=40.0))
    variant = shipped.replace(tunings={D4: settled})
    made = variant.interaction(X2, D4)
    assert isinstance(made, ChargeCoupling)
    assert made.tunings == (shipped.tuning(X2), settled)
    assert made.charge_coupling_hz == shipped.interaction(X2, D4).charge_coupling_hz
    assert variant.interaction(X2, D4) is made
    # The tunings are in the order asked, so at() reads them in that order.
    turned = variant.interaction(D4, X2)
    assert turned.tunings == (settled, shipped.tuning(X2))
    at = shipped.tuning(D4).amplitude_for(5.276e9)
    one = shipped.interaction(X2, D4).at((X2, D4), {D4: at})
    other = shipped.interaction(D4, X2).at((D4, X2), {D4: at})
    assert one.exchange_hz == pytest.approx(other.exchange_hz)
    assert one.frequencies_hz[D4] == pytest.approx(other.frequencies_hz[D4])
    for pair in ((D4,), (D4, D4), (X2, D4, D4)):
        with pytest.raises(pulse.InvalidValueError, match="takes the coupling's two qudits"):
            shipped.interaction(X2, D4).at(pair, {})


def test_the_answers_take_handles_of_the_system_or_indices(shipped):
    other = pulse.quantum_system("transmon_lattice:2")
    for ask in (
        lambda: shipped.tuning(other.qudits[0]),
        lambda: shipped.interaction(other.qudits[0], D4),
        lambda: shipped.control(X2, other.qudits[1]),
    ):
        with pytest.raises(pulse.QuantumSystemMismatchError, match="not of 'transmon_surface17'"):
            ask()
    with pytest.raises(pulse.InvalidTypeError, match="takes qudit handles"):
        shipped.tuning(True)
    assert shipped.tuning(shipped.qudits[0]) is shipped.tuning(shipped.qudits[0].index)
    assert shipped.tuning(np.int64(D4)) is shipped.tuning(D4)
    # Asking does not read the calibration's values to find what it kept.
    assert shipped.interaction(np.int64(X2), D4) is shipped.interaction(X2, D4)


def test_a_calibration_without_circuits_gives_no_tunings_and_pickles_as_it_was(shipped):
    cal = shipped_calibration("transmon_surface17")
    data = json.loads(json.dumps(cal._data))
    del data["coupling"]["charge_coupling_g_c"]["used"]
    for entry in data["qudits"].values():
        entry.pop("ec", None)
    bare = type(cal)(data, name="bare")
    qs = pulse.quantum_system("transmon_surface17", calibration=bare)
    before = pickle.dumps(bare)
    assert qs.tuning(D4) is None and qs.interaction(X2, D4) is None
    # What is made of a calibration is kept apart from it.
    custom = type(cal)(json.loads(json.dumps(cal._data)), name="custom")
    held = pulse.quantum_system("transmon_surface17", calibration=custom)
    before_custom = pickle.dumps(custom)
    # Each calibration answers for itself, however equal their values.
    assert held.tuning(D4).source.startswith("custom calibration, qudits.D4:")
    assert held.tuning(D4) == shipped.tuning(D4).replace(source=held.tuning(D4).source)
    assert pickle.dumps(custom) == before_custom and pickle.dumps(bare) == before


def _circuits_changed(name="mine"):
    """The shipped calibration's data with every circuit's E_C changed: its
    qudits' values are the shipped ones, its tunings are not."""
    cal = shipped_calibration("transmon_surface17")
    data = json.loads(json.dumps(cal._data))
    for entry in data["qudits"].values():
        if "ec" in entry:
            entry["ec"]["value"] *= 1.01
    return type(cal)(data, name=name)


def test_a_variant_answers_from_the_calibration_of_the_system_it_was_made_from(shipped):
    from cudaq_pulse.controller import CouplingLine

    mine = _circuits_changed()
    qs = pulse.quantum_system("transmon_surface17", calibration=mine)
    own = qs.tuning(D4)
    assert own.ec_hz == pytest.approx(1.01 * shipped.tuning(D4).ec_hz)
    # A variant has no calibration of its own: its answers are those of the
    # system it was made from, not the shipped calibration's, though the
    # qudits' values are the shipped ones.
    variant = qs.replace(layout=[D4, X2])
    assert variant.calibration is None
    assert variant.tuning(D4) is own and variant.interaction(X2, D4) is qs.interaction(X2, D4)
    # So does a variant of the variant, and one given coupling lines, whose
    # check needs the tuning.
    again = variant.replace(layout=[X2, D4])
    assert again.tuning(D4) is own
    wired = qs.replace(
        control_system=pulse.ControlSystem(
            controllers=[pulse.Controller().with_coupling_lines([CouplingLine(D4)])]
        )
    )
    assert wired.tuning(D4) is own
    # A qudit changed in any value is no longer the calibrated one.
    changed = qs.physical_qudits[D4].replace(t1_us=40.0)
    assert qs.replace(qudits={**qs.physical_qudits, D4: changed}).tuning(D4) is None
    # In another process too.
    code = (
        "import sys, pickle, cudaq_pulse as pulse\n"
        "qs = pickle.loads(bytes.fromhex(sys.stdin.read()))\n"
        "print(qs.calibration is None, qs.tuning(3).ec_hz, qs.tuning(3).source.split()[0])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=pickle.dumps(variant).hex(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["True", repr(own.ec_hz), "mine"]
    # A variant given coupling lines is checked against its own sources when
    # another process rebuilds it, and one that was never asked carries them.
    unasked = qs.replace(
        control_system=pulse.ControlSystem(
            controllers=[pulse.Controller().with_coupling_lines([CouplingLine(X2)])]
        )
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=pickle.dumps(unasked).hex(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["True", repr(own.ec_hz), "mine"]


def test_a_changed_qudit_has_no_derived_tuning_whether_or_not_a_calibration_is_held(shipped):
    changed = shipped.physical_qudits[D4].replace(t1_us=40.0)
    held = pulse.quantum_system(
        name="held",
        qudits={**shipped.physical_qudits, D4: changed},
        couplings=shipped.couplings,
        calibration="transmon_surface17",
    )
    assert held.calibration is not None
    assert held.tuning(D4) is None and held.interaction(X2, D4) is None
    assert held.tuning(X2) == shipped.tuning(X2)
    assert held.replace(layout=[D4, X2]).tuning(D4) is None


def test_building_a_system_stores_no_more_than_it_did(shipped):
    # The reference a variant keeps is paid for by two caches that are class
    # defaults until they are filled.
    assert "_arrays" not in vars(shipped) or vars(shipped)["_arrays"] is not None
    fresh = pulse.quantum_system("transmon_lattice:2")
    assert "_arrays" not in vars(fresh) and "_tables" not in vars(fresh)
    assert fresh._arrays is None and fresh._tables == (-1, None, None)
    variant = fresh.replace(layout=[1, 0])
    # A system that was built names itself; a variant, and a variant of a
    # variant, the built system.
    assert vars(fresh)["_made_from"] is fresh and vars(variant)["_made_from"] is fresh
    assert vars(variant.replace(layout=[0, 1]))["_made_from"] is fresh


def test_a_variant_of_a_variant_lets_the_variant_between_go(shipped):
    import gc
    import weakref

    v1 = shipped.replace(layout=[D4, X2])
    held = weakref.ref(v1)
    v2 = v1.replace(layout=[X2, D4])
    del v1
    gc.collect()
    # Never asked, never used: v2 holds the system they were made from only.
    assert held() is None
    assert v2.tuning(D4) is shipped.tuning(D4)
    # A variant made from one that was asked answers as it does.
    custom = pulse.quantum_system("transmon_surface17", calibration=_circuits_changed())
    asked = custom.replace(layout=[D4, X2])
    own = asked.tuning(D4)
    assert asked.replace(layout=[X2, D4]).tuning(D4) is own


def test_a_drifting_system_keeps_what_it_declares(qs):
    d4, x2 = qs.tuning(D4), qs.tuning(X2)
    system = _pair(None, (d4, x2), ChargeCoupling(charge_coupling_hz=3.1e6))
    drifting = system.replace(parameter_drift=True, parameter_drift_seed=3)
    assert drifting.tuning(0) == d4 and drifting.interaction(0, 1).tunings == (d4, x2)
    drifting.advance(2 * pulse.s)
    assert drifting.tuning(1) == x2
    built = _pair(None, (d4, x2), parameter_drift=True, parameter_drift_seed=3)
    assert built.tuning(0) == d4
    # Its variants, its system file and its pickle keep them too.
    moved = drifting.replace(layout=[1, 0])
    assert moved.tuning(0) == d4 and moved == system.replace(layout=[1, 0])
    assert moved.fingerprint != _pair().replace(layout=[1, 0]).fingerprint
    assert drifting.as_dict()["tunings"] == system.as_dict()["tunings"]
    back = pickle.loads(pickle.dumps(drifting))
    assert type(back) is type(drifting) and back.tuning(0) == d4
    assert back.replace(layout=[1, 0]) == moved
    assert drifting.replace(tunings={}, interactions={}).tuning(0) is None
    again = drifting.replace(parameter_drift=True, parameter_drift_seed=4)
    assert type(again) is type(drifting) and again.tuning(1) == x2
    # A change and drift in one call: the drifting system is the changed one's.
    renamed = system.replace(name="other", parameter_drift=True, parameter_drift_seed=3)
    assert type(renamed) is type(drifting) and renamed == system.replace(name="other")
    assert renamed.fingerprint == system.replace(name="other").fingerprint
    # A system that declares nothing yet takes its components first.
    with pytest.raises(pulse.InvalidTypeError, match="give the components first"):
        _pair().replace(tunings={0: d4}, parameter_drift=True)
    with pytest.raises(pulse.InvalidValueError, match="seeds parameter_drift"):
        system.replace(parameter_drift_seed=3)
    assert system.replace(parameter_drift=False) is system


def test_a_custom_calibration_is_compared_against_the_registered_system_it_names(shipped):
    changed = shipped.physical_qudits[D4].replace(t1_us=40.0)

    def held(calibration):
        return pulse.quantum_system(
            name="held",
            qudits={**shipped.physical_qudits, D4: changed},
            couplings=shipped.couplings,
            calibration=calibration,
        )

    # Named after a registered quantum system: a qudit has the calibration's
    # tuning when it has the values the calibration gives that system's qudit.
    named = held(_circuits_changed("transmon_surface17"))
    assert named.tuning(D4) is None and named.interaction(X2, D4) is None
    own = named.tuning(X2)
    assert own.ec_hz == pytest.approx(1.01 * shipped.tuning(X2).ec_hz)
    assert named.replace(layout=[D4, X2]).tuning(X2) is own
    # Named after none: the qudits of the system that holds it, by label.
    other = held(_circuits_changed("mine"))
    assert other.tuning(D4).ec_hz == pytest.approx(1.01 * shipped.tuning(D4).ec_hz)
    assert other.tuning(D4).source.startswith("mine calibration, qudits.D4:")
    assert other.interaction(X2, D4) is not None


def test_variants_of_equal_values_keep_their_own_answers_through_a_pickle(shipped):
    custom = pulse.quantum_system("transmon_surface17", calibration=_circuits_changed())
    a = shipped.replace(layout=[D4, X2])
    b = custom.replace(layout=[D4, X2])
    # The same qudit values, other circuits.
    assert a.tuning(D4) != b.tuning(D4)
    from cudaq_pulse import _pickling

    blobs = pickle.dumps(a), pickle.dumps(b)
    kept = dict(_pickling._devices)
    _pickling._devices.clear()
    try:
        back_a, back_b = map(pickle.loads, blobs)
        again = pickle.loads(blobs[1])
    finally:
        _pickling._devices.clear()
        _pickling._devices.update(kept)
    assert back_a is not back_b and again is back_b
    assert back_a.tuning(D4) == a.tuning(D4) and back_b.tuning(D4) == b.tuning(D4)
    # A variant of the shipped system has the fingerprint it always had.
    built = pulse.quantum_system(
        name=shipped.name,
        qudits=shipped.physical_qudits,
        couplings=shipped.couplings,
        layout=[D4, X2],
    )
    assert a.fingerprint == built.fingerprint


def test_a_long_chain_of_variants_answers_from_the_system_they_were_made_from(shipped):
    variant = shipped
    for _ in range(1500):
        variant = variant.replace(layout=[D4, X2])
    # Asked, the last one answers from the system they were all made from
    # and holds none of the others.
    assert vars(variant)["_made_from"] is shipped
    assert variant.tuning(D4) is shipped.tuning(D4)
    assert isinstance(vars(variant)["_made_from"], list)
    again = shipped.replace(layout=[D4, X2]).replace(layout=[X2, D4])
    assert again.qudits[0].index == X2
