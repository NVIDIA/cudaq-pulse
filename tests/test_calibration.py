# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped transmon_surface17 calibration and loading calibrations onto quantum systems."""

import hashlib
import json
import math
import re
from pathlib import Path

import cudaq_pulse as pulse
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
    return (
        hashlib.sha256(raw).hexdigest()[:16],
        data["_schema"]["default_measurement_run"],
        tuple(data["measurement_runs"]),
        {p: (e["value"], tuple(e["idle_frequencies_hz"])) for p, e in exchange.items()},
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


def test_a_published_only_target_uses_the_published_value_over_sqrt2():
    target = transmon_surface17(calibration=None)
    assert {round(c.strength_hz) for c in target.couplings} == {
        round(7.0e6 / math.sqrt(2.0)),
        round(7.4e6 / math.sqrt(2.0)),
    }
