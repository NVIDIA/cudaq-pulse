# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped transmon_surface17 calibration and loading calibrations onto quantum systems."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import _transmon_surface17 as _k
from cudaq_pulse.quantum_systems import transmon_surface17
from cudaq_pulse.quantum_systems.calibration import _SHIPPED, Calibration
from cudaq_pulse.quantum_systems.transmon_model import anchored_charge_coupling_hz

KINDS = {"published", "digitized", "fitted"}
_SHIPPED_SURFACE17 = Path(_SHIPPED, "transmon_surface17.json")


@pytest.fixture(scope="module")
def cal():
    return Calibration.load("transmon_surface17")


def _leaves(node, path=()):
    if isinstance(node, dict):
        if "value" in node and "kind" in node:
            yield path, node
            return
        for k, v in node.items():
            yield from _leaves(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, (*path, str(i)))


def test_every_value_carries_its_kind_source_and_measurement_run():
    data = json.loads(_SHIPPED_SURFACE17.read_text())
    leaves = list(_leaves(data))
    assert len(leaves) > 200
    for path, leaf in leaves:
        name = "/".join(path)
        assert leaf["kind"] in KINDS, name
        assert leaf["source"], name
        assert leaf["measurement_run"] in data["measurement_runs"], name
        assert "units" in leaf, name


def test_the_shipped_file_counts_its_values_by_kind():
    # The counts a release's review quotes: a changed file changes them here.
    data = json.loads(_SHIPPED_SURFACE17.read_text())
    kinds = Counter(leaf["kind"] for _, leaf in _leaves(data))
    assert kinds == {"published": 89, "digitized": 89, "fitted": 421}


def test_the_shipped_file_holds_no_local_paths_or_working_notes():
    text = _SHIPPED_SURFACE17.read_text()
    assert not re.search(
        r"\.py\b|/(?:Users|home)/|\bTODO\b|\bdraft\b|unpublished", text, re.IGNORECASE
    )


def test_measurement_runs_and_provenance(cal):
    assert cal.measurement_run == "2022-main" and "2025-flux-tuning" in cal.measurement_runs()
    ec = cal.entry("qudits", "D4", "ec")
    assert ec["kind"] == "fitted" and ec["units"] == "Hz" and "Tab. I" in ec["source"]
    assert cal.value("coupling", "device_default", "g_11_20") == 7.0e6
    with pytest.raises(ValueError, match="no measurement run"):
        Calibration.load("transmon_surface17", measurement_run="nope")
    with pytest.raises(pulse.NotFoundError, match="transmon_surface17"):
        Calibration.load("missing")


def test_a_loaded_calibration_keeps_the_values_it_parsed():
    # Calibration.load keeps the dict it parsed without copying it, and the
    # shipped one is shared by every quantum system: no use of it may change it.
    shipped = pulse.quantum_system("transmon_surface17").calibration
    parsed = json.loads(_SHIPPED_SURFACE17.read_text())
    for cal in (
        shipped,
        Calibration.load("transmon_surface17", measurement_run="2025-flux-tuning"),
    ):
        pair = cal.pairs[0]
        qudit = cal.flux_qudits[0]
        cal.measurement_runs()["2022-main"] = "changed"
        cal.entry("qudits", qudit, "ec")["value"] = 0.0
        cal.entry("coupling", "device_default", "g_11_20")["uncertainty"] = None
        cal.value("coupling", "device_default", "g_11_20")
        cal.circuit(qudit), cal.idle_flux(qudit), cal.charge_coupling_hz(pair)
        cal.pair(pair), cal.exchange_hz(pair), cal.idle_zz_hz(pair)
        cal.flux_for_detuning(qudit, cal.detuning_for_flux(qudit, 0.1))
        cal._apply(transmon_surface17()._description), cal.fingerprint, cal.quantum_system
        assert cal._data == parsed
    assert shipped._data is not parsed


def test_every_qudit_has_a_circuit_that_reproduces_its_idle_frequency(cal):
    assert len(cal.flux_qudits) == 17
    for q in _k.QUDITS:
        circuit = cal.circuit(q.label)
        assert circuit.f01_hz(cal.idle_flux(q.label)) == pytest.approx(q.idle_frequency_hz, abs=1e3)
        assert circuit.anharmonicity_hz(cal.idle_flux(q.label)) == pytest.approx(
            q.anharmonicity_hz, abs=1e3
        )


def test_the_device_coupling_reproduces_the_measured_x2_d4_coupling(cal):
    """One charge coupling for the quantum system, fitted so that X2-D4's
    |11> <-> |20> coupling is Hellings et al.'s 7.4 MHz at 4.4 GHz."""
    g_c = cal.value("coupling", "charge_coupling_g_c", "calibration_A")
    assert {cal.charge_coupling_hz(p) for p in cal.pairs} == {g_c}
    measured = cal.entry("coupling", "overrides", "X2-D4", "g_11_20")
    recomputed = anchored_charge_coupling_hz(
        cal.circuit("D4"),
        cal.circuit("X2"),
        measured["value"],
        measured["at_interaction_frequency_hz"],
    )
    assert g_c == pytest.approx(recomputed, rel=1e-4)
    # At each pair's own interaction point (Z2 cannot reach Z2-D7's, so D7
    # is swept at Z2's sweet spot) J_2 is 7.3-9.6 MHz.
    j2 = [
        cal.pair(p).j2_hz(cal.value("cz", "edges", p, "interaction_frequency"))
        for p in ("X2-D4", "Z1-D1", "Z2-D7")
    ]
    assert all(7.2e6 < j < 9.7e6 for j in j2)


def test_the_stored_idle_exchange_is_the_derived_one(cal):
    for pair in cal.pairs:
        stored = cal.value("coupling", "charge_coupling_g_c", "idle_exchange", pair)
        assert stored == pytest.approx(cal.exchange_hz(pair), rel=1e-9), pair


def test_idle_exchange_and_zz_are_frequency_dependent(cal):
    exchange = [cal.exchange_hz(p) for p in cal.pairs]
    zz = [cal.idle_zz_hz(p) for p in cal.pairs]
    assert 5.6e6 < min(exchange) and max(exchange) < 6.1e6
    assert -7.5e3 < min(zz) and max(zz) < -4.4e3
    # Tuning the data qubit up towards the auxiliary raises J_qq.
    d4 = cal.circuit("D4")
    assert cal.exchange_hz("X2-D4", data_hz=d4.f01_hz(0.4)) > cal.exchange_hz("X2-D4")


def test_flux_quanta_and_detuning_convert_both_ways(cal):
    # Hellings et al.: X2 moves by -738 MHz at about a quarter flux quantum.
    assert cal.flux_for_detuning("X2", -738e6) == pytest.approx(0.265, abs=0.005)
    for qudit, flux in (("X2", 0.1), ("D4", 0.3)):
        detuning = cal.detuning_for_flux(qudit, flux)
        assert cal.flux_for_detuning(qudit, detuning) == pytest.approx(flux, abs=1e-8)
    # A data qubit idles at its lower sweet spot, so flux tunes it up.
    assert cal.detuning_for_flux("D4", 0.3) > 0


def test_the_17_qudit_device_carries_the_shipped_calibration(cal):
    dev = pulse.quantum_system("transmon_surface17")
    assert dev.calibration == cal
    assert dev._description is transmon_surface17()._description


def _index_entry(name):
    """The ``_shipped_calibrations`` entry of the shipped file *name*."""
    raw = Path(_SHIPPED, f"{name}.json").read_bytes()
    data = json.loads(raw)
    exchange = data["coupling"]["charge_coupling_g_c"]["idle_exchange"]
    gates, errors = data["single_qubit_gates"], data["readout"]["assignment"]
    x12 = gates["x12"]
    return (
        hashlib.sha256(raw).hexdigest()[:16],
        data["_schema"]["default_measurement_run"],
        tuple(data["measurement_runs"]),
        {p: (e["value"], tuple(e["idle_frequencies_hz"])) for p, e in exchange.items()},
        tuple(
            (
                q,
                (
                    gates["x"][q]["beta"]["value"],
                    (
                        x12["duration"]["value"],
                        x12["sigma"]["value"],
                        x12[q]["amplitude"]["value"],
                        x12[q]["beta"]["value"],
                    ),
                    (errors[q]["error_0to1"]["value"], errors[q]["error_1to0"]["value"]),
                ),
            )
            for q in data["qudits"]
        ),
    )


def _tuning_entry(name):
    """What ``SHIPPED_TUNINGS`` holds of the shipped calibration *name*."""
    data = json.loads(Path(_SHIPPED, f"{name}.json").read_text())
    return (
        data["coupling"]["charge_coupling_g_c"]["used"],
        data["coupling"]["charge_coupling_g_c"][data["coupling"]["charge_coupling_g_c"]["used"]][
            "value"
        ],
        tuple(
            (
                q,
                (
                    v["ec"]["value"],
                    v["ej_max"]["value"],
                    v["asymmetry_d"]["value"],
                    {"upper": 0.0, "lower": 0.5}[v["sweet_spot_idle"]],
                ),
            )
            for q, v in data["qudits"].items()
            if v.get("flux_quanta_capable")
        ),
    )


def test_the_shipped_tunings_are_of_the_shipped_files():
    from cudaq_pulse.quantum_systems._shipped_calibrations import SHIPPED
    from cudaq_pulse.quantum_systems._shipped_tunings import SHIPPED_TUNINGS

    assert set(SHIPPED_TUNINGS) <= set(SHIPPED)
    for name, entry in SHIPPED_TUNINGS.items():
        expected = _tuning_entry(name)
        assert entry == expected, (
            f"quantum_systems/_shipped_tunings.py is stale for {name}; its tunings are\n"
            f"{expected!r}"
        )


def test_the_shipped_calibration_index_is_of_the_shipped_files():
    from cudaq_pulse._source_stamp import CALIBRATIONS
    from cudaq_pulse.quantum_systems._shipped_calibrations import SHIPPED

    assert set(SHIPPED) == set(CALIBRATIONS)
    for name, entry in SHIPPED.items():
        expected = _index_entry(name)
        assert entry == expected, (
            f"targets/_shipped_calibrations.py is stale for {name}; its entry is\n{expected!r}"
        )
        assert entry[0] == CALIBRATIONS[name]


def test_a_device_of_the_shipped_calibration_does_not_read_its_file(monkeypatch):
    from cudaq_pulse._source_stamp import CALIBRATIONS
    from cudaq_pulse.quantum_systems import calibration, transmon

    read = transmon_surface17()
    read_device = pulse.quantum_system("transmon_surface17")
    monkeypatch.setattr(transmon, "_BUILT", {})
    monkeypatch.setattr(calibration, "_LOADED", {})
    monkeypatch.setattr(calibration, "_read", lambda path: pytest.fail(f"read {path}"))
    assert transmon._described_surface17() == read._description
    assert pulse.quantum_system("transmon_surface17").fingerprint == read_device.fingerprint
    shipped = calibration.shipped_calibration("transmon_surface17")
    digest = CALIBRATIONS["transmon_surface17"]
    assert shipped._identity == ("shipped", "transmon_surface17", "2022-main", digest)
    assert "_data" not in vars(shipped)


def test_an_indexed_shipped_calibration_reads_its_file_for_other_values(monkeypatch, cal):
    from cudaq_pulse.quantum_systems import calibration

    monkeypatch.setattr(calibration, "_LOADED", {})
    shipped = calibration.shipped_calibration("transmon_surface17", "2022-a")
    assert "_data" not in vars(shipped)
    assert (
        shipped.measurement_run == "2022-a" and shipped.measurement_runs() == cal.measurement_runs()
    )
    assert shipped.value("qudits", "D4", "ec") == cal.value("qudits", "D4", "ec")
    assert (
        shipped.fingerprint
        == Calibration.load("transmon_surface17", measurement_run="2022-a").fingerprint
    )
    with pytest.raises(pulse.InvalidValueError, match="no measurement run 'nope'"):
        calibration.shipped_calibration("transmon_surface17", "nope")


def test_a_device_loads_a_calibration_from_its_config(tmp_path, cal):
    # A calibration whose charge coupling is 10 % higher strengthens every
    # pair; without the stored idle exchange, the quantum system computes it.
    data = json.loads(_SHIPPED_SURFACE17.read_text())
    couplings = data["coupling"]["charge_coupling_g_c"]
    couplings["calibration_A"]["value"] *= 1.1
    for entry in couplings["idle_exchange"].values():
        del entry["idle_frequencies_hz"]
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(data))
    base = pulse.quantum_system("transmon_surface17")
    custom = pulse.quantum_system("transmon_surface17", calibration=str(path))
    assert custom.calibration.name == "custom" and custom != base

    def x2_d4(dev):
        labels = {q.index: q.label for q in dev.physical_qudits.values()}
        (c,) = (c for c in dev.couplings if tuple(labels[q] for q in c.qudits) == ("X2", "D4"))
        return c.strength_hz

    assert x2_d4(custom) == pytest.approx(1.1 * x2_d4(base), rel=1e-9)
    with pytest.raises(ValueError, match="measurement_run"):
        pulse.quantum_system("transmon_surface17", measurement_run="2025-flux-tuning")
    other = pulse.quantum_system(
        "transmon_surface17", calibration="transmon_surface17", measurement_run="2025-flux-tuning"
    )
    assert other.calibration.measurement_run == "2025-flux-tuning"


def test_the_published_readout_errors_are_those_of_tab_i(cal):
    for q in _k.QUDITS:
        entry = cal.entry("readout", "assignment", q.label, "two_state_error")
        assert entry["kind"] == "published" and "Tab. I" in entry["source"]
        assert entry["value"] == pytest.approx(q.readout_error_2state, abs=1e-12), q.label


def test_the_calibrated_pulses_and_readout_errors_reach_the_quantum_system(cal):
    qs = pulse.quantum_system("transmon_surface17")
    published = transmon_surface17(calibration=None)
    for q in _k.QUDITS:
        qudit = qs.qudit(q.label)
        x = qudit.get_calibration(pulse.gates.X)
        assert x.beta_ns == pytest.approx(
            cal.value("single_qubit_gates", "x", q.label, "beta") * 1e9
        )
        # Between the phase-error (half) and leakage (full) DRAG conditions -1 / alpha.
        motzoi_ns = -1.0 / (2.0 * math.pi * q.anharmonicity_hz * 1e-9)
        assert 0.5 * motzoi_ns < x.beta_ns < motzoi_ns, q.label
        assert qudit.get_calibration(pulse.gates.Y90).beta_ns == x.beta_ns
        x12 = qudit.get_calibration(pulse.gates.X, transition=(1, 2))
        assert (x12.duration_ns, x12.sigma_ns) == pytest.approx((20.0, 5.0))
        # A pi rotation of the 1-2 transition, whose matrix element is sqrt(2) the 0-1 one.
        area = 5.0 * math.sqrt(2.0 * math.pi) * math.erf(2.0 / math.sqrt(2.0))
        assert x12.amplitude == pytest.approx(math.pi / (math.sqrt(2.0) * area), rel=0.01)
        readout = qs.physical_qudits[qudit.index].readout
        error = cal.value("readout", "assignment", q.label, "error_0to1")
        assert (readout.error_0to1, readout.error_1to0) == (error, error)
        bare = published.qudit(q.label)
        assert bare.get_calibration(pulse.gates.X).beta_ns == 0.0
        with pytest.raises(pulse.NotFoundError, match=r"X \(1, 2\)"):
            bare.get_calibration(pulse.gates.X, transition=(1, 2))
        assert published.physical_qudits[bare.index].readout.error_1to0 == 0.0


def test_the_shipped_quantum_system_is_the_published_one_calibrated(cal):
    """transmon_surface17 builds its qudits and couplings once, with the
    shipped values in place: as the calibration applied to the published
    quantum system gives them, equal pulses shared alike."""
    import pickle

    from cudaq_pulse.quantum_systems import transmon

    applied = cal._apply(transmon._build_surface17(None))
    built = transmon._build_surface17(cal)
    assert built == applied
    assert pickle.dumps(built) == pickle.dumps(applied)


@pulse.kernel
def _read_ground(q):
    r = get_readout_line(q)
    readout(r, r.get_calibration(pulse.gates.READOUT))


@pulse.kernel
def _read_excited(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, d.get_calibration(pulse.gates.X))
    sync(d, r)
    readout(r, r.get_calibration(pulse.gates.READOUT))


def _reported(result, level):
    return sum(b.probability for b in result.branches if b.levels == (level,))


def test_model_readout_reproduces_the_published_two_state_errors(cal):
    """Tab. I's eps(2) = 1 - [P(0|0) + P(1|1)] / 2 after preselecting |0>
    (App. F); where relaxation during the readout alone exceeds it, the
    assignment error is 0 and the simulation errs a little more."""
    qs = pulse.quantum_system("transmon_surface17")
    ground = np.array([1.0, 0.0, 0.0])
    for q in _k.QUDITS:
        qudit = qs.qudit(q.label)
        runs = [
            pulse.simulate(k, qudit, levels=3, initial_state=ground, readout="model")
            for k in (_read_ground, _read_excited)
        ]
        eps = 1 - (_reported(runs[0], 0) + _reported(runs[1], 1)) / 2
        if cal.value("readout", "assignment", q.label, "error_1to0") > 0:
            assert eps == pytest.approx(q.readout_error_2state, abs=1e-6), q.label
        else:
            assert q.readout_error_2state < eps < q.readout_error_2state + 1.5e-3, q.label


def _closed(qs, label):
    """*qs* with qudit *label* free of decoherence and thermal excitation."""
    i = qs.qudit(label).index
    q = qs.physical_qudits[i]
    free = q.replace(t1_us=0.0, t2_star_us=0.0, t2_echo_us=None, thermal_population=0.0)
    return qs.replace(qudits={**qs.physical_qudits, i: free}).qudits[i]


@pulse.kernel
def _x(q):
    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.X))


def test_the_drag_coefficient_removes_most_of_the_x_error():
    # The published values alone play X with beta 0.
    systems = (transmon_surface17(calibration=None), pulse.quantum_system("transmon_surface17"))
    for label in ("D5", "Z1"):
        published, calibrated = (
            1 - pulse.simulate(_x, _closed(qs, label), levels=3).density_matrix[1, 1].real
            for qs in systems
        )
        assert published > 1.5e-3 and calibrated < 2e-4, (label, published, calibrated)


@pulse.kernel
def _repair(q):
    d = get_drive_line(q)
    drive(d, d.get_calibration(pulse.gates.X, transition=(1, 2)), tone=d.tones[(1, 2)])


def test_the_1_2_pulse_returns_a_leaked_qubit():
    qs = pulse.quantum_system("transmon_surface17")
    leaked = np.array([0.0, 0.0, 1.0, 0.0])
    for label in ("D5", "X1"):
        rho = pulse.simulate(
            _repair, qs.qudit(label), levels=4, initial_state=leaked
        ).density_matrix
        assert rho[1, 1].real > 0.996, label
        closed = pulse.simulate(_repair, _closed(qs, label), levels=4, initial_state=leaked)
        assert closed.density_matrix[1, 1].real > 0.999, label


def test_a_published_only_target_uses_the_published_value_over_sqrt2():
    target = transmon_surface17(calibration=None)
    assert {round(c.strength_hz) for c in target.couplings} == {
        round(7.0e6 / math.sqrt(2.0)),
        round(7.4e6 / math.sqrt(2.0)),
    }
