# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A target's simulation model, not a label, sets its qudits' levels and physics."""

from __future__ import annotations

import dataclasses
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.targets import (
    Qubit,
    SimulationModel,
    TargetDescription,
    TransmonModel,
    transmon_generic,
)


def _target(model=TransmonModel, n=3):
    return TargetDescription(
        name="models",
        qubits={
            q: Qubit(q, 5.0e9 + 0.1e9 * q, -300.0e6, t1_us=0.0, t2_star_us=0.0) for q in range(n)
        },
        model=model,
    )


def test_each_model_class_declares_its_levels():
    assert SimulationModel.levels == 2
    assert TransmonModel.levels == 3
    assert TransmonModel().levels_of(5) == 3
    assert "architecture" not in [f.name for f in dataclasses.fields(TargetDescription)]


def test_a_target_takes_a_model_class_or_instance_and_defaults_to_transmons():
    assert _target(TransmonModel).model == TransmonModel() == _target(TransmonModel()).model
    assert TargetDescription(name="plain").model == TransmonModel()
    assert transmon_generic(2).model == TransmonModel()
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
    transmon = pulse.device({"target": _target()})
    assert transmon.transition_freq.shape == (3, 3, 3)
    assert transmon.transition_freq[1, 1, 2] == pytest.approx(5.1e9 - 300e6)
    plain = pulse.device({"target": _target(SimulationModel)})
    assert plain.transition_freq.shape == (3, 2, 2)
    assert plain.transition_freq[1, 0, 1] == 5.1e9
    mixed = pulse.device({"target": _target(TransmonModel(levels={0: 4, 2: 2}))})
    table = mixed.transition_freq
    assert table.shape == (3, 4, 4)
    assert table[0, 2, 3] == pytest.approx(5.0e9 - 2 * 300e6)
    assert math.isnan(table[1, 2, 3]) and math.isnan(table[2, 1, 2])
    assert table[2, 0, 1] == 5.2e9


def test_devices_of_different_models_are_different_devices():
    devices = [pulse.device({"target": _target(m)}) for m in (TransmonModel, SimulationModel)]
    assert devices[0] != devices[1]
    assert devices[0] == pulse.device({"target": _target(TransmonModel())})
    assert devices[0] != pulse.device({"target": _target(TransmonModel(levels={0: 4}))})


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
    device = pulse.device({"target": _target(model)})
    lowered = pulse.compile(_drive_last, device=device).lower(device.qudit_handles[2])
    result = pulse.simulate(lowered, levels=levels)
    assert result.qudits == (2,)
    assert f"pulse.levels = array<i64: {expected}>" in lowered.mlir()
