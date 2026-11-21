# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target's simulation model, not a label, sets its qudits' levels and physics."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.quantum_systems import (
    PhysicalQudit,
    SimulationModel,
    TransmonModel,
    transmon_lattice,
)
from cudaq_pulse.quantum_systems.base import _Description


def _target(model=TransmonModel, n=3):
    return pulse.quantum_system(
        name="models",
        qudits={
            q: PhysicalQudit(
                q, 5.0e9 + 0.1e9 * q, anharmonicity_hz=-300.0e6, t1_us=0.0, t2_star_us=0.0
            )
            for q in range(n)
        },
        model=model,
    )


def test_each_model_class_declares_its_levels():
    assert SimulationModel.levels == 2
    assert TransmonModel.levels == 3
    assert TransmonModel().levels_of(5) == 3
    assert "architecture" not in _Description.__slots__


def test_a_target_takes_a_model_class_or_instance_and_defaults_to_transmons():
    assert _target(TransmonModel).model == TransmonModel() == _target(TransmonModel()).model
    assert pulse.quantum_system(name="plain", qudits={}).model == TransmonModel()
    assert transmon_lattice(2).model == TransmonModel()
    assert _target(None).model is None
    with pytest.raises(pulse.InvalidTypeError, match="model must be a SimulationModel class"):
        _target("transmon")


def test_levels_are_overridden_by_qudit():
    model = TransmonModel(levels={0: 4, 2: 2})
    assert [model.levels_of(q) for q in range(4)] == [4, 3, 2, 3]
    assert repr(model) == "TransmonModel(levels={0: 4, 2: 2})"
    assert model == TransmonModel(levels={2: 2, 0: 4}) != TransmonModel()
    assert hash(model) == hash(TransmonModel(levels={2: 2, 0: 4}))
    for wrong, message in [
        ({0: 1}, "qudit 0 needs 2 or more levels"),
        ({0: 3.0}, "qudit 0 needs 2 or more levels"),
        ({-1: 3}, "levels keys are qudit indices"),
        ({True: 3}, "levels keys are qudit indices"),
    ]:
        with pytest.raises(pulse.InvalidValueError, match=message):
            TransmonModel(levels=wrong)
    with pytest.raises(pulse.InvalidTypeError, match="levels must be a dict"):
        TransmonModel(levels=[3])  # type: ignore[arg-type]


def test_the_model_sets_the_transition_table():
    transmon = _target()
    assert transmon.transition_freq.shape == (3, 3, 3)
    assert transmon.transition_freq[1, 1, 2] == pytest.approx(5.1e9 - 300e6)
    plain = _target(SimulationModel)
    assert plain.transition_freq.shape == (3, 2, 2)
    assert plain.transition_freq[1, 0, 1] == 5.1e9
    mixed = _target(TransmonModel(levels={0: 4, 2: 2}))
    table = mixed.transition_freq
    assert table.shape == (3, 4, 4)
    assert table[0, 2, 3] == pytest.approx(5.0e9 - 2 * 300e6)
    assert math.isnan(table[1, 2, 3]) and math.isnan(table[2, 1, 2])
    assert table[2, 0, 1] == 5.2e9


def test_devices_of_different_models_are_different_devices():
    quantum_systems = [_target(m) for m in (TransmonModel, SimulationModel)]
    assert quantum_systems[0] != quantum_systems[1]
    assert quantum_systems[0] == _target(TransmonModel())
    assert quantum_systems[0] != _target(TransmonModel(levels={0: 4}))


@pulse.kernel
def _drive_last(q):
    drive(get_drive_line(q), square(8, 0.1))


@pytest.mark.parametrize(
    ("model", "levels", "expected"),
    [
        (TransmonModel(levels={0: 4}), None, "4, 3, 3"),
        (TransmonModel, {2: 2}, "3, 3, 2"),
        (TransmonModel, 2, "2, 2, 2"),
    ],
)
def test_qudits_a_program_leaves_out_keep_the_levels_they_would_have(model, levels, expected):
    quantum_system = _target(model)
    lowered = pulse.compile(_drive_last, quantum_system=quantum_system).lower(
        quantum_system.qudits[2]
    )
    result = pulse.simulate(lowered, levels=levels)
    assert result.qudits == (2,)
    assert f"level_counts = [{expected}]" in lowered.mlir()


def test_qudit_property_tables_keyed_by_qudit_index():
    expected_t1 = {0: 45.0, 1: 60.0, 2: 80.0}
    expected_t2 = {0: 30.0, 1: 20.0, 2: 50.0}
    expected_alpha = {0: -280.0e6, 1: -310.0e6, 2: -330.0e6}
    qs = pulse.quantum_system(
        name="tables",
        qudits={
            i: PhysicalQudit(
                i,
                5.0e9 + 0.1e9 * i,
                expected_t1[i],
                expected_t2[i],
                anharmonicity_hz=expected_alpha[i],
            )
            for i in expected_t1
        },
    )
    for table, expected in (
        (qs.t1_times, expected_t1),
        (qs.t2_times, expected_t2),
        (qs.anharmonicities, expected_alpha),
    ):
        assert type(table) is dict
        assert table == pytest.approx(expected)


def test_model_energies_and_transitions_hz():
    f01 = 5.1e9
    alpha = -290.0e6
    qs = pulse.quantum_system(
        name="energies",
        qudits={0: PhysicalQudit(0, f01, 50.0, 30.0, anharmonicity_hz=alpha)},
    )
    q = qs.physical_qudits[0]

    transmon = TransmonModel()
    assert transmon.energies_hz(q, 3) == pytest.approx([0.0, f01, 2 * f01 + alpha])
    assert transmon.transitions_hz(q, 3) == pytest.approx(
        {(0, 1): f01, (0, 2): 2 * f01 + alpha, (1, 2): f01 + alpha}
    )

    plain = SimulationModel()
    assert plain.energies_hz(q, 3) == pytest.approx([0.0, f01, 2 * f01])
    assert plain.transitions_hz(q, 3) == pytest.approx({(0, 1): f01, (0, 2): 2 * f01, (1, 2): f01})


def test_transmon_energies_without_anharmonicity():
    f01 = 5.0e9
    qs = pulse.quantum_system(
        name="no_alpha",
        qudits={0: PhysicalQudit(0, f01, 50.0, 30.0, anharmonicity_hz=None)},
    )
    q = qs.physical_qudits[0]

    model = TransmonModel()
    assert model.energies_hz(q, 2) == pytest.approx([0.0, f01])
    with pytest.raises(pulse.InvalidValueError, match="gives no anharmonicity_hz"):
        model.energies_hz(q, 3)
