# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A quantum system's couplings, gates and residual ZZ entries.

A field left out in Python is Python's own TypeError at the constructor; a
system file or config reports each missing key, at its path and (for a file)
its line, under the file's key names.
"""

from __future__ import annotations

import dataclasses
import json
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import DeviceDefinitionError
from cudaq_pulse.targets import (
    Coupling,
    Gate,
    Qubit,
    ResidualZZ,
    SimulationModel,
    TargetDescription,
    TransmonModel,
)
from cudaq_pulse.targets.base import _COUPLING_OPERATORS, _coupling_operator


def _qudits(n=3):
    return {i: Qubit(i, 5.0e9 + 1.0e8 * i, -2.5e8, 50.0, 40.0) for i in range(n)}


def _system(**fields):
    return TargetDescription(name="checked", qubits=_qudits(), **fields)


def _file_problems(tmp_path, data):
    path = tmp_path / "qpu.json"
    text = json.dumps(data, indent=2)
    path.write_text(text)
    with pytest.raises(DeviceDefinitionError) as raised:
        pulse.device(str(path))
    assert str(raised.value).startswith(f"{len(raised.value.problems)} problem")
    return raised.value.problems, text.splitlines()


def _system_data(**changes):
    data = pulse.device({"quantum_system": _system()}).as_dict()
    data["quantum_system"].update(changes)
    return data


def _line_of(lines, needle, after=0):
    return next(i for i, line in enumerate(lines, start=1) if i > after and needle in line)


# ── The coupling operator comes from the model ─────────────────────────────


def test_a_system_names_no_architecture():
    assert not hasattr(pulse.targets, "Architecture")
    assert "architecture" not in _system_data()["quantum_system"]


def test_the_model_decides_the_coupling_operator():
    assert dict(_COUPLING_OPERATORS) == {TransmonModel: "exchange"}
    assert _coupling_operator(TransmonModel()) == "exchange"
    assert _coupling_operator(TransmonModel(levels={0: 4})) == "exchange"

    class Tuned(TransmonModel):
        __slots__ = ()

    assert _coupling_operator(Tuned()) == "exchange"
    assert _coupling_operator(SimulationModel()) is None
    assert _coupling_operator(None) is None
    assert _system().model == TransmonModel()


def test_the_hamiltonian_terms_of_a_model_with_no_coupling_operator_are_unsupported():
    coupled = [Coupling((0, 1), 1.0e6)]
    for model in (SimulationModel, None):
        with pytest.raises(pulse.UnsupportedError) as raised:
            _system(model=model, couplings=coupled).hamiltonian_terms()
        assert str(raised.value) == (
            f"the model {_system(model=model).model!r} of 'checked' gives its couplings no "
            "operator, so they have no Hamiltonian terms; the models whose couplings have one "
            "are TransmonModel (exchange)"
        )
        assert len(_system(model=model).hamiltonian_terms()) == 3


# ── Couplings, gates and residual ZZ ───────────────────────────────────────


@pulse.kernel
def _pair_program(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    drive(d0, square(8, 0.1))
    sync(d0, d1)
    drive(d1, square(8, 0.1))


def _unmeasured_system():
    return _system(
        couplings=[Coupling((0, 1), None), Coupling((1, 2), 3.0e6), Coupling((2, 0), None)]
    )


def test_a_coupling_not_measured_yet_compiles_schedules_and_exports():
    dev = pulse.device({"quantum_system": _unmeasured_system()})
    lowered = pulse.compile(_pair_program).lower(*dev.qudit_handles[:2])
    assert [e.start_cycles for e in lowered.schedule] == [0, 8]
    assert lowered.export_controller() and lowered.export_emulation()
    assert "pulse.coupling" not in lowered.mlir()
    data = dev.as_dict()
    assert [c["strength_hz"] for c in data["quantum_system"]["couplings"]] == [None, 3.0e6, None]
    assert pulse.device(json.loads(json.dumps(data))) == dev


def test_simulate_refuses_a_coupling_not_measured_yet_by_name():
    dev = pulse.device({"quantum_system": _unmeasured_system()})
    lowered = pulse.compile(_pair_program).lower(*dev.qudit_handles[[2, 0]])
    with pytest.raises(pulse.InvalidValueError) as raised:
        pulse.simulate(lowered, levels=2)
    assert str(raised.value) == (
        "simulating needs the strength of every coupling of the program's qudits; "
        "couplings[2] (qudits 2 and 0) of 'checked' has strength_hz None, not measured yet: "
        "give the strength in Hz, or leave those qudits out of the program"
    )
    everything = pulse.compile(_three_qudits).lower(*dev.qudit_handles)
    with pytest.raises(pulse.InvalidValueError, match=r"couplings\[0\] \(qudits 0 and 1\) and "):
        pulse.simulate(everything, levels=2)
    # A program on measured couplings only simulates.
    pulse.simulate(pulse.compile(_pair_program).lower(*dev.qudit_handles[1:]), levels=2)


@pulse.kernel
def _three_qudits(q0, q1, q2):
    d0 = get_drive_line(q0)
    get_drive_line(q1)
    get_drive_line(q2)
    drive(d0, square(8, 0.1))


def test_the_hamiltonian_terms_need_every_strength():
    with pytest.raises(pulse.InvalidValueError) as raised:
        _unmeasured_system().hamiltonian_terms()
    assert str(raised.value) == (
        "the Hamiltonian terms need every coupling's strength; couplings[0] (qudits 0 and 1) "
        "and couplings[2] (qudits 2 and 0) of 'checked' have strength_hz None, not measured "
        "yet: give the strength in Hz"
    )


def test_the_hamiltonian_terms_name_the_rotating_wave_exchange():
    terms = _system(couplings=[Coupling((2, 1), 1.0e6)]).hamiltonian_terms()
    (coupling,) = (t for t in terms if t["kind"].startswith("coupling"))
    assert coupling["kind"] == "coupling_exchange"
    assert coupling["qubit_indices"] == (2, 1)
    assert coupling["coefficient"] == pytest.approx(2e-3 * math.pi)


def test_a_gate_belongs_to_its_coupling_and_is_read_only():
    gate = Gate("cz", 98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
    coupling = Coupling([3, 1], 5.0e6, gate=gate)
    assert coupling.qudits == (3, 1)
    assert coupling == Coupling((3, 1), 5.0e6, gate=Gate("cz", 98.0, 15.0, 4.4e9))
    assert hash(coupling) == hash(Coupling((3, 1), 5.0e6, gate=gate))
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.interaction_frequency_hz = 4.5e9  # type: ignore[misc]
    assert not hasattr(gate, "params") and not hasattr(gate, "fidelity")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'duration_ns'"):
        Gate("cz")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'strength_hz'"):
        Coupling((0, 1))  # type: ignore[call-arg]
    assert {f for f in Coupling.__dataclass_fields__} == {"qudits", "strength_hz", "gate"}


def test_residual_zz_is_its_qudits_and_rate():
    entry = ResidualZZ([0, 1], -2.0e4)
    assert (entry.qudits, entry.rate_hz) == ((0, 1), -2.0e4)
    assert [f.name for f in dataclasses.fields(ResidualZZ)] == ["qudits", "rate_hz"]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'rate_hz'"):
        ResidualZZ((0, 1))  # type: ignore[call-arg]
    assert _system(residual_zz=[entry]).residual_zz == (entry,)


def test_a_system_file_holds_couplings_gates_and_residual_zz():
    target = _system(
        couplings=[
            Coupling((1, 0), 5.7e6, gate=Gate("cz", 98.0, 15.0, 4.4e9)),
            Coupling((1, 2), None),
        ],
        residual_zz=[ResidualZZ((0, 2), 1.5e4)],
    )
    dev = pulse.device({"quantum_system": target})
    system = dev.as_dict()["quantum_system"]
    assert system["couplings"] == [
        {
            "qudits": [1, 0],
            "strength_hz": 5.7e6,
            "gate": {
                "name": "cz",
                "duration_ns": 98.0,
                "buffer_ns": 15.0,
                "interaction_frequency_hz": 4.4e9,
            },
        },
        {"qudits": [1, 2], "strength_hz": None, "gate": None},
    ]
    assert system["residual_zz"] == [{"qudits": [0, 2], "rate_hz": 1.5e4}]
    assert "crosstalk" not in system
    back = pulse.device(json.loads(json.dumps(dev.as_dict())))
    assert back == dev and back.target == target
    assert back.target.couplings[0].qudits == (1, 0)


def test_a_key_that_is_no_field_is_an_unknown_key(tmp_path):
    gate = {"name": "cz", "duration_ns": 98.0, "fidelity": 0.99, "params": {"cz_step": 1.0}}
    data = _system_data(
        couplings=[{"qudits": [0, 1], "strength_hz": 1.0e6, "gate": gate}],
        crosstalk=[{"qudits": [0, 1], "static_zz_hz": 1.0e4}],
        residual_zz=[{"qudits": [0, 2], "static_zz_hz": 1.0e4}],
    )
    problems, _ = _file_problems(tmp_path, data)
    assert sorted((p.path, p.message) for p in problems) == [
        ("quantum_system.couplings[0].gate.fidelity", "is not a known key"),
        ("quantum_system.couplings[0].gate.params", "is not a known key"),
        ("quantum_system.crosstalk", "is not a known key"),
        ("quantum_system.residual_zz[0].rate_hz", "is missing"),
        ("quantum_system.residual_zz[0].static_zz_hz", "is not a known key"),
    ]


def test_the_fingerprint_holds_the_whole_gate():
    def fingerprint(gate):
        return pulse.device(
            {"target": _system(couplings=[Coupling((0, 1), 5e6, gate)])}
        ).fingerprint

    prints = {
        fingerprint(None),
        fingerprint(Gate("cz", 98.0)),
        fingerprint(Gate("cz", 99.0)),
        fingerprint(Gate("cz", 98.0, buffer_ns=15.0)),
        fingerprint(Gate("cz", 98.0, interaction_frequency_hz=4.4e9)),
    }
    assert len(prints) == 5


def test_a_calibration_matches_a_pair_named_in_either_order():
    from cudaq_pulse.targets import transmon_17q
    from cudaq_pulse.targets.calibration import shipped_calibration

    published = transmon_17q(calibration=None)
    calibration = shipped_calibration("transmon_17q")
    reversed_ = dataclasses.replace(
        published,
        couplings=[Coupling(c.qudits[::-1], c.strength_hz, c.gate) for c in published.couplings],
    )
    forward = calibration.apply(published).couplings
    backward = calibration.apply(reversed_).couplings
    assert [c.strength_hz for c in backward] == [c.strength_hz for c in forward]
    assert [c.qudits for c in backward] == [c.qudits[::-1] for c in forward]
    assert {c.strength_hz for c in forward}.isdisjoint(c.strength_hz for c in published.couplings)
