# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step ``simulate`` keeps for each target: it follows every change of
what the step depends on, holds no target alive and stays small."""

from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import threading
import warnings
import weakref

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.runtime import simulate as simulate_module
from cudaq_pulse.targets import Coupling, DeviceDescription, PhysicalQudit
from cudaq_pulse.targets._registry import _registry, _resolved, register_quantum_system
from cudaq_pulse.targets.models import SimulationModel, TransmonModel

from tests._handles import Q
from tests.runtime._lowering import on


def _target(alpha_hz=-200.0e6, couplings=()):
    return DeviceDescription(
        name="step-cache",
        qudits={q: PhysicalQudit(q, 5.0e9 + 2.0e9 * q, alpha_hz, 0.0, 0.0) for q in range(2)},
        couplings=list(couplings),
    )


@pulse.kernel
def _flip(q0):
    d0 = get_drive_line(q0)
    drive(d0, square(4, 0.1))


@pulse.kernel
def _flip_both(q0, q1):
    drive(get_drive_line(q0), square(4, 0.1))
    drive(get_drive_line(q1), square(4, 0.1))


@pytest.fixture
def steps(monkeypatch):
    """The step of each simulation run from here on."""
    taken = []
    real = simulate_module._simulate_compiled

    def spy(*args, **options):
        taken.append(options["dt_ns"])
        return real(*args, **options)

    monkeypatch.setattr(simulate_module, "_simulate_compiled", spy)
    return taken


def _coarse(call):
    """The step warnings *call* gives, by the spread they name."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pulse.PulseWarning)
        call()
    return [str(w.message).split(" turns at ")[1].split(",")[0] for w in caught]


# Transmon level k sits k (k - 1) / 2 alpha off the frame: 5 levels spread
# over 6 alpha, so 20 steps per period (in whole fractions of the 0.5 ns
# cycle) are 0.5 / 12 ns at -200 MHz and 0.5 / 24 ns at -400 MHz; 3 levels
# over alpha, so 0.25 ns resolves -200 MHz and not -400 MHz.
_STEP_5 = {-200.0e6: 0.5 / 12, -400.0e6: 0.5 / 24}


def test_a_recalibrated_plant_changes_the_step_and_the_warning(steps):
    lowered = on(_target(), _flip, Q[0])
    calibrated, recalibrated = _target(), _target(alpha_hz=-400.0e6)
    for plant in (calibrated, recalibrated, calibrated, recalibrated):
        pulse.simulate(lowered, levels=5, plant=plant)
        assert steps[-1] == _STEP_5[plant.qudits[0].anharmonicity_hz]
    spreads = [
        _coarse(lambda p=plant: pulse.simulate(lowered, levels=3, dt_ns=0.25, plant=p))
        for plant in (calibrated, recalibrated, calibrated, recalibrated)
    ]
    assert spreads == [[], ["400 MHz"], [], ["400 MHz"]]


def test_a_recalibrated_device_changes_the_step_and_the_warning(steps):
    for alpha in (-200.0e6, -400.0e6, -200.0e6):
        pulse.simulate(on(_target(alpha), _flip, Q[0]), levels=5)
        assert steps[-1] == _STEP_5[alpha]
    lowered = on(_target(), _flip, Q[0])
    assert _coarse(lambda: pulse.simulate(lowered, levels=3, dt_ns=0.25)) == []
    # A device is read-only by contract, not enforced: a target set on it in
    # place is the one simulated, and the one its step is kept for.
    lowered.device._description = _target(alpha_hz=-400.0e6)
    assert _coarse(lambda: pulse.simulate(lowered, levels=3, dt_ns=0.25)) == ["400 MHz"]
    pulse.simulate(lowered, levels=5)
    assert steps[-1] == _STEP_5[-400.0e6]


def test_a_device_s_clock_set_in_place_changes_the_step(steps):
    lowered = on(_target(), _flip, Q[0])
    pulse.simulate(lowered, levels=5)
    assert steps[-1] == 0.5 / 12
    lowered.device.clock_ghz = 2.5  # a 0.4 ns cycle: 10 steps
    pulse.simulate(lowered, levels=5)
    assert steps[-1] == 0.4 / 10


def test_couplings_of_another_device_change_the_step(steps):
    # Coupled qudits 2 GHz apart turn at their detuning: 20 steps per period
    # are 0.025 ns, finer than the 0.05 ns default.
    coupled = _target(couplings=[Coupling((0, 1), 5.0e6)])
    for target, step in ((coupled, 0.025), (_target(), 0.05), (coupled, 0.025)):
        pulse.simulate(on(target, _flip_both, Q[0], Q[1]), levels=2)
        assert steps[-1] == step


def test_a_target_registered_again_under_its_name_changes_the_step(steps):
    name = "step-cache-test"
    try:
        for alpha in (-200.0e6, -400.0e6, -200.0e6):
            register_quantum_system(name, lambda arg, a=alpha: _target(a))
            dev = pulse.device(name)
            pulse.simulate(pulse.compile(_flip).lower(dev.qudits[0]), levels=5)
            assert steps[-1] == _STEP_5[alpha]
    finally:
        _registry().pop(name, None)
        _resolved.pop(name, None)


def test_what_a_kept_step_depends_on_cannot_change_in_place():
    target = _target()
    qudit = target.qudits[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        qudit.anharmonicity_hz = -400.0e6
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.couplings = (Coupling((0, 1), 5.0e6),)
    with pytest.raises(TypeError):
        target.qudits[0] = qudit.replace(anharmonicity_hz=-400.0e6)
    model = TransmonModel(levels={0: 4})
    with pytest.raises(TypeError):
        model._levels[0] = 6
    for change in (
        lambda: setattr(model, "_levels", {0: 6}),
        lambda: object.__setattr__(model, "levels", 6),
        lambda: delattr(model, "_levels"),
    ):
        with pytest.raises(AttributeError):
            change()
    for change in (
        lambda: setattr(TransmonModel, "levels", 5),
        lambda: setattr(TransmonModel, "energies_hz", lambda self, qudit, levels: []),
        lambda: type.__setattr__(TransmonModel, "levels", 5),
        lambda: delattr(SimulationModel, "energies_hz"),
        lambda: setattr(SimulationModel, "__init__", lambda self: None),
    ):
        with pytest.raises(TypeError, match="immutable type"):
            change()
    assert TransmonModel.levels == 3
    assert model.levels_of(0) == 4


def test_a_subclass_of_a_model_may_change(steps):
    class Fine(TransmonModel):
        pass

    Fine.levels = 5
    Fine.levels_of = lambda self, qudit: 2
    del Fine.levels_of
    lowered = on(dataclasses.replace(_target(), model=Fine()), _flip, Q[0])
    pulse.simulate(lowered)
    assert steps[-1] == 0.5 / 12
    Fine.levels = 2
    pulse.simulate(lowered)
    assert steps[-1] == 0.05


@pytest.mark.parametrize("model", [SimulationModel(), TransmonModel(levels={0: 4, 1: 2})])
@pytest.mark.parametrize("move", [pickle.loads, copy.copy, copy.deepcopy])
def test_a_model_pickles_and_copies_whole(model, move):
    moved = move(pickle.dumps(model)) if move is pickle.loads else move(model)
    assert type(moved) is type(model) and repr(moved) == repr(model)
    assert [moved.levels_of(q) for q in range(3)] == [model.levels_of(q) for q in range(3)]
    with pytest.raises(AttributeError):
        moved._levels = {0: 6}
    target = dataclasses.replace(_target(), model=model)
    for again in (pickle.loads(pickle.dumps(target)), copy.deepcopy(target)):
        assert repr(again.model) == repr(model)


def test_a_model_of_the_user_s_own_is_not_kept(steps):
    class Spread(TransmonModel):
        def __init__(self, alpha_hz):
            super().__init__()
            self.alpha_hz = alpha_hz

        def energies_hz(self, qudit, levels):
            return [k * qudit.frequency_hz + k * (k - 1) / 2 * self.alpha_hz for k in range(levels)]

    model = Spread(-200.0e6)
    target = dataclasses.replace(_target(), model=model)
    lowered = on(target, _flip, Q[0])
    pulse.simulate(lowered, levels=5)
    model.alpha_hz = -400.0e6
    pulse.simulate(lowered, levels=5)
    assert steps[-2:] == [0.5 / 12, 0.5 / 24]
    assert not [k for k in simulate_module._STEPS if id(target) in k[:2]]


def test_a_step_by_qudit_levels_is_resolved_like_one_for_all(steps):
    lowered = on(_target(), _flip, Q[0])
    before = dict(simulate_module._STEPS)
    pulse.simulate(lowered, levels={0: 5})
    pulse.simulate(lowered, levels={0: 5}, plant=_target(alpha_hz=-400.0e6))
    assert steps == [0.5 / 12, 0.5 / 24]
    assert simulate_module._STEPS.keys() <= before.keys()


def test_kept_steps_hold_no_target_and_go_with_it(steps):
    target, plant = _target(), _target(alpha_hz=-400.0e6)
    lowered = on(target, _flip, Q[0])
    pulse.simulate(lowered, levels=5)
    pulse.simulate(lowered, levels=5, plant=plant)
    ids = {id(target), id(plant)}
    assert len([k for k in simulate_module._STEPS if ids & set(k[:2])]) == 2
    assert ids <= simulate_module._WATCHED.keys()
    refs = [weakref.ref(target), weakref.ref(plant), weakref.ref(lowered.device)]
    del target, plant, lowered
    gc.collect()
    assert [r() for r in refs] == [None, None, None]
    assert not [k for k in simulate_module._STEPS if ids & set(k[:2])]
    assert not ids & simulate_module._WATCHED.keys()


@pytest.mark.slow
def test_kept_steps_stay_few_over_many_steps_and_targets():
    lowered = on(_target(), _flip, Q[0])
    for i in range(2 * simulate_module._MAX_STEPS):
        pulse.simulate(lowered, levels=2, dt_ns=0.05 / (1 + i))
        assert len(simulate_module._STEPS) <= simulate_module._MAX_STEPS
    watched = len(simulate_module._WATCHED)
    for _ in range(50):
        pulse.simulate(on(_target(), _flip, Q[0]), levels=2)
    gc.collect()
    assert len(simulate_module._WATCHED) <= watched + 1


def test_threads_sharing_kept_steps_each_get_their_target_s(monkeypatch):
    taken = []
    real = simulate_module._simulate_compiled

    def spy(lowered, target, *args, **options):
        taken.append((target.qudits[0].anharmonicity_hz, options["dt_ns"]))
        return real(lowered, target, *args, **options)

    monkeypatch.setattr(simulate_module, "_simulate_compiled", spy)
    lowered = on(_target(), _flip, Q[0])
    plants = [_target(), _target(alpha_hz=-400.0e6)]
    start = threading.Barrier(4)

    def run(first):
        start.wait()
        for i in range(40):
            pulse.simulate(lowered, levels=5, plant=plants[(first + i) % 2])

    threads = [threading.Thread(target=run, args=(n % 2,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 160
    assert all(step == _STEP_5[alpha] for alpha, step in taken)
