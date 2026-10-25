# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The shipped Krinner calibration and loading calibrations onto devices."""

import json
import math
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.targets import _krinner17 as _k
from cudaq_pulse.targets import transmon_krinner_17q
from cudaq_pulse.targets.calibration import _SHIPPED, Calibration
from cudaq_pulse.targets.transmon_model import anchored_charge_coupling_hz

KINDS = {"published", "digitized", "fitted"}


@pytest.fixture(scope="module")
def cal():
    return Calibration.load("krinner17")


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


def test_every_value_carries_its_kind_source_and_cooldown():
    data = json.loads((_SHIPPED / "krinner17.json").read_text())
    leaves = list(_leaves(data))
    assert len(leaves) > 200
    for path, leaf in leaves:
        name = "/".join(path)
        assert leaf["kind"] in KINDS, name
        assert leaf["source"], name
        assert leaf["cooldown"] in data["cooldowns"], name
        assert "units" in leaf, name


def test_the_shipped_file_holds_no_local_paths_or_working_notes():
    text = (_SHIPPED / "krinner17.json").read_text()
    assert not re.search(r"\.py\b|/Users/|_scratch|\bTODO\b|\bdraft\b", text, re.IGNORECASE)


def test_cooldowns_and_provenance(cal):
    assert cal.cooldown == "K22-main" and "HEL25" in cal.cooldowns
    ec = cal.entry("qubits", "D4", "ec")
    assert ec["kind"] == "fitted" and ec["units"] == "Hz" and "Tab. I" in ec["source"]
    assert cal.value("coupling", "device_default", "g_11_20") == 7.0e6
    with pytest.raises(ValueError, match="no cooldown"):
        Calibration.load("krinner17", cooldown="nope")
    with pytest.raises(FileNotFoundError, match="krinner17"):
        Calibration.load("missing")


def test_every_qubit_has_a_circuit_that_reproduces_its_idle_frequency(cal):
    assert len(cal.flux_qubits) == 17
    for q in _k.QUBITS:
        circuit = cal.circuit(q.label)
        assert circuit.f01_hz(cal.idle_flux(q.label)) == pytest.approx(q.idle_frequency_hz, abs=1e3)
        assert circuit.anharmonicity_hz(cal.idle_flux(q.label)) == pytest.approx(
            q.anharmonicity_hz, abs=1e3
        )


@pytest.mark.parametrize("pair", ["X2-D4", "Z1-D1", "Z2-D7"])
def test_each_charge_coupling_reproduces_its_anchor(cal, pair):
    """g_c gives the published |11> <-> |20> coupling at the interaction
    point (Z2 cannot reach Z2-D7's, so D7 is swept at Z2's sweet spot)."""
    entry = cal.entry("coupling", "charge_coupling_g_c", "per_pair", pair)
    auxiliary, data = pair.split("-")
    recomputed = anchored_charge_coupling_hz(
        cal.circuit(data),
        cal.circuit(auxiliary),
        entry["g_11_20_hz"],
        entry["at_interaction_frequency_hz"],
    )
    assert entry["value"] == pytest.approx(recomputed, rel=1e-9)
    j2 = cal.pair(pair).j2_hz(entry["at_interaction_frequency_hz"])
    assert j2 == pytest.approx(entry["g_11_20_hz"], rel=1e-9)
    # The idle exchange stored with the pair, which devices use without numpy.
    assert entry["idle_exchange_hz"] == pytest.approx(cal.exchange_hz(pair), rel=1e-9)


def test_idle_exchange_and_zz_are_frequency_dependent(cal):
    exchange = [cal.exchange_hz(p) for p in cal.pairs]
    zz = [cal.idle_zz_hz(p) for p in cal.pairs]
    assert 4.3e6 < min(exchange) and max(exchange) < 5.8e6
    assert -6e3 < min(zz) and max(zz) < -3e3
    # Tuning the data qubit up towards the auxiliary raises J_qq.
    d4 = cal.circuit("D4")
    assert cal.exchange_hz("X2-D4", data_hz=d4.f01_hz(0.4)) > cal.exchange_hz("X2-D4")


def test_flux_quanta_and_detuning_convert_both_ways(cal):
    # Hellings et al.: X2 moves by -738 MHz at about a quarter flux quantum.
    assert cal.flux_for_detuning("X2", -738e6) == pytest.approx(0.265, abs=0.005)
    for qubit, flux in (("X2", 0.1), ("D4", 0.3)):
        detuning = cal.detuning_for_flux(qubit, flux)
        assert cal.flux_for_detuning(qubit, detuning) == pytest.approx(flux, abs=1e-8)
    # A data qubit idles at its lower sweet spot, so flux tunes it up.
    assert cal.detuning_for_flux("D4", 0.3) > 0


def test_the_krinner_device_carries_the_shipped_calibration(cal):
    dev = pulse.device({"target": "transmon_krinner_17q"})
    assert dev.calibration == cal
    assert dev.target is transmon_krinner_17q()


def test_a_device_loads_a_calibration_from_its_config(tmp_path, cal):
    # A calibration whose X2-D4 anchor is 10 % higher strengthens that pair.
    data = json.loads((_SHIPPED / "krinner17.json").read_text())
    entry = data["coupling"]["charge_coupling_g_c"]["per_pair"]["X2-D4"]
    entry["value"] *= 1.1
    # Without the stored idle exchange, the device computes it from g_c.
    del entry["idle_exchange_hz"], entry["idle_frequencies_hz"]
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(data))
    base = pulse.device({"target": "transmon_krinner_17q"})
    custom = pulse.device({"target": "transmon_krinner_17q", "calibration": str(path)})
    assert custom.calibration.name == "custom" and custom != base

    def x2_d4(dev):
        labels = {q.index: q.label for q in dev.target.qubits.values()}
        (c,) = (
            c
            for c in dev.target.couplings
            if (labels[c.qubit_a], labels[c.qubit_b]) == ("X2", "D4")
        )
        return c.coupling_strength_hz

    assert x2_d4(custom) == pytest.approx(1.1 * x2_d4(base), rel=1e-9)
    with pytest.raises(ValueError, match="cooldown"):
        pulse.device({"target": "transmon_krinner_17q", "cooldown": "HEL25"})
    other = pulse.device(
        {"target": "transmon_krinner_17q", "calibration": "krinner17", "cooldown": "HEL25"}
    )
    assert other.calibration.cooldown == "HEL25"


def test_a_published_only_target_uses_the_published_value_over_sqrt2():
    target = transmon_krinner_17q(calibration=None)
    assert {round(c.coupling_strength_hz) for c in target.couplings} == {
        round(7.0e6 / math.sqrt(2.0)),
        round(7.4e6 / math.sqrt(2.0)),
    }
