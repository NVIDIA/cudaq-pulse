# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pickling targets, devices, handles, kernels, programs and errors, as
process pools and caches on disk do."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import pathlib
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Controller, ControlSystem, _pickling, _source_stamp
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import ControllerLimitError, DriveLine
from cudaq_pulse.kernel import _kernel
from cudaq_pulse.kernel.ir_builder import SymbolicValueError
from cudaq_pulse.targets import (
    Coupling,
    DeviceDescription,
    DragPulse,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    ResidualZZ,
    calibration,
)
from cudaq_pulse.targets._registry import make_quantum_system
from cudaq_pulse.targets.calibration import Calibration

from tests._handles import LADDER, Q, ladder_target

# The shipped target and the calibration it carries by default.
_SHIPPED_TARGET = "transmon_17q"
_SHIPPED_CALIBRATION = "transmon_17q"


def _round_trip(value):
    return pickle.loads(pickle.dumps(value))


def _refuse_compile(*args, **kwargs):
    raise AssertionError("the program was compiled again")


# ── Kernels ──────────────────────────────────────────────────────────────────


@pulse.kernel
def rabi(q, amp: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(40, amp, 10.0))
    sync(d, r)
    return readout(r, square(100, 0.05))


@pulse.kernel
def ramsey(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(20, 0.1))
    wait(d, tau)
    drive(d, square(20, 0.1))


@pulse.kernel
def branch(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    d1 = get_drive_line(r)
    drive(d, drag(16, 0.25, 4.0, 0.3))
    shift_phase(d.tone, 0.5)
    sync(d, ro)
    m = readout(ro, square(100, 0.05))
    sync(ro, d1)
    if m:
        drive(d1, square(20, 0.3))
    else:
        drive(d1, gaussian(20, 0.1, 5.0))


@pulse.classical(max_latency_ns=400)
def flip(bits: pulse.Bits) -> pulse.Bits:
    return [1 - bits[0]]


@pulse.kernel
def reset(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    fix = flip.submit([m]).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3))


@pulse.kernel
def idle(tau: float):
    pass


_SHIPPED_DEVICE = pulse.quantum_system(_SHIPPED_TARGET)
_SLOW_CONTROLLER = ControlSystem(controllers=[Controller(clock_ghz=1.0)])
_DELAYED = ControlSystem(
    controllers=[Controller(drive_lines=[DriveLine(0, delay_ns=1.5)], readout_lines=[0])]
)


def _programs():
    return {
        "bound": rabi.lower(Q[0], 0.3),
        "compiled": pulse.compile(rabi, quantum_system=LADDER).lower(Q[1], 0.2),
        "generic": rabi.lower(Q[0], 0.3).generic,
        "variable_wait": ramsey.lower(Q[0], 40.0),
        "branch": pulse.compile(branch, schedule="asap").lower(Q[0], Q[1]),
        "classical_call": pulse.compile(reset, schedule="asap").lower(Q[0]),
        "other_clock": pulse.compile(rabi, control_system=_SLOW_CONTROLLER).lower(Q[0], 0.3),
        "line_delays": pulse.compile(rabi, control_system=_DELAYED).lower(Q[0], 0.3),
        "shipped": rabi.lower(_SHIPPED_DEVICE.qudits[2], 0.1),
    }


def _views(program):
    """What a program shows, without compiling it again."""
    shown = {
        "mlir": program.mlir(),
        "schedule": program.schedule,
        "segments": program.segments(),
        "metrics": program.metrics,
        "parameters": program.parameters,
        "arguments": program.arguments,
        "qudits": program.qudits,
        "options": program.options,
        "target": program.quantum_system,
        "classical": program.classical_report(),
    }
    if not program.parameters:
        shown["controller"] = program.export_controller()
        shown["emulation"] = program.export_emulation()
    return shown


# ── Targets and devices ──────────────────────────────────────────────────────


def test_targets_and_their_parts_unpickle_equal_with_equal_hashes():
    qudit = PhysicalQudit(
        3,
        5.1e9,
        -300e6,
        50.0,
        30.0,
        label="D3",
        drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.2),
        calibrations={pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)},
        metadata={"fridge": "B", "fit": [1, 2.5]},
    )
    cz = FluxCZ(duration_ns=40.0, interaction_frequency_hz=4.4e9)
    coupling = Coupling((3, 4), 5e6, calibrations={pulse.gates.CZ: cz})
    residual = ResidualZZ((3, 4), -1.5e4)
    shipped = _SHIPPED_DEVICE._description
    for value in (qudit, coupling, residual, LADDER._description, shipped):
        back = _round_trip(value)
        assert back == value and hash(back) == hash(value)
    back = _round_trip(shipped)
    assert back.qudits[0].drive == shipped.qudits[0].drive
    assert back.qudits[0].readout == shipped.qudits[0].readout
    assert back.couplings == shipped.couplings and back.residual_zz == shipped.residual_zz
    assert back.qudits[0].calibrations == shipped.qudits[0].calibrations


def test_targets_compare_by_content():
    a = DeviceDescription(name="t", qudits={0: PhysicalQudit(0, 5e9, -3e8, 50.0, 30.0)})
    b = DeviceDescription(name="t", qudits={0: PhysicalQudit(0, 5e9, -3e8, 50.0, 30.0)})
    c = DeviceDescription(name="t", qudits={0: PhysicalQudit(0, 5.1e9, -3e8, 50.0, 30.0)})
    assert a == b and hash(a) == hash(b)
    assert a != c and a != "t"


@pytest.mark.parametrize(
    "make",
    [
        lambda: LADDER,
        lambda: pulse.quantum_system(_SHIPPED_TARGET),
        lambda: pulse.quantum_system(_SHIPPED_TARGET, layout=[4, 5, 6], clock_ghz=1.0),
    ],
    ids=["ladder", "shipped", "config"],
)
def test_a_device_unpickles_equal_and_is_built_once_per_process(make, monkeypatch):
    quantum_system = make()
    data = pickle.dumps(quantum_system)
    back = pickle.loads(data)
    assert back == quantum_system and hash(back) == hash(quantum_system)
    assert pickle.loads(data) is back
    # Another process has not seen the device: it builds it again.
    monkeypatch.setattr(_pickling, "_devices", {})
    built = pickle.loads(data)
    assert built == quantum_system and built is not quantum_system
    assert built.layout == quantum_system.layout and built.clock_ghz == quantum_system.clock_ghz
    assert built._description == quantum_system._description
    assert (built.calibration is None) == (quantum_system.calibration is None)
    assert pickle.loads(data) is built


def _shipped_and_uncalibrated():
    shipped = pulse.quantum_system(_SHIPPED_TARGET)
    uncalibrated = pulse.quantum_system(make_quantum_system(_SHIPPED_TARGET))
    assert shipped.calibration is not None and uncalibrated.calibration is None
    return shipped, uncalibrated


def test_a_device_and_its_target_without_the_calibration_are_two_devices():
    shipped, uncalibrated = _shipped_and_uncalibrated()
    assert shipped.fingerprint != uncalibrated.fingerprint and shipped != uncalibrated
    assert shipped.qudits[0] != uncalibrated.qudits[0]
    named = pulse.quantum_system(_SHIPPED_TARGET, calibration=_SHIPPED_CALIBRATION)
    assert named == shipped and named.calibration == shipped.calibration
    # One kernel, called on each, returns each its own device.
    assert rabi.lower(shipped.qudits[0], 0.1).quantum_system is shipped
    assert rabi.lower(uncalibrated.qudits[0], 0.1).quantum_system is uncalibrated


@pytest.mark.parametrize("kept_first", [1, 0], ids=["uncalibrated kept", "shipped kept"])
def test_a_process_holding_one_unpickles_the_other_as_itself(kept_first, monkeypatch):
    quantum_systems = _shipped_and_uncalibrated()
    kept, sent = quantum_systems[kept_first], quantum_systems[1 - kept_first]
    data = pickle.dumps(sent)
    monkeypatch.setattr(_pickling, "_devices", {})
    assert _round_trip(kept) is kept
    back = pickle.loads(data)
    assert back == sent and back != kept
    assert back.calibration == sent.calibration


def test_a_device_unpickles_with_its_own_name_and_source(monkeypatch):
    target = make_quantum_system(_SHIPPED_TARGET)
    first = pulse.QuantumSystem(target, name="first", source="a")
    second = pulse.QuantumSystem(target, name="second", source="b")
    assert first == second
    data = pickle.dumps(second)
    monkeypatch.setattr(_pickling, "_devices", {})
    assert _round_trip(first) is first
    back = pickle.loads(data)
    assert (back.name, back.source) == ("second", "b")


def test_a_shipped_calibration_pickles_as_its_name(monkeypatch):
    shipped = pulse.quantum_system(_SHIPPED_TARGET)
    program = rabi.lower(shipped.qudits[0], 0.3)
    data = pickle.dumps(program)
    assert len(data) < 20_000 and len(pickle.dumps(shipped.calibration)) < 200
    monkeypatch.setattr(_pickling, "_devices", {})
    back = pickle.loads(data)
    assert back.quantum_system is not shipped and back.quantum_system == shipped
    assert back.quantum_system.calibration is shipped.calibration
    assert _views(back) == _views(program)


def test_an_unpickled_shipped_calibration_reads_its_file_when_used(monkeypatch):
    shipped = pulse.quantum_system(_SHIPPED_TARGET).calibration
    data = pickle.dumps(shipped)
    monkeypatch.setattr(calibration, "_LOADED", {})
    back = pickle.loads(data)
    assert back is not shipped and "_data" not in vars(back)
    assert (back.name, back.cooldown) == (shipped.name, shipped.cooldown)
    assert back.quantum_system == shipped.quantum_system and back.cooldowns == shipped.cooldowns
    assert back == shipped and back.idle_zz_hz(back.pairs[0]) == shipped.idle_zz_hz(back.pairs[0])
    assert pickle.loads(data) is back


def test_a_shipped_calibration_of_another_file_is_refused(monkeypatch):
    data = pickle.dumps(pulse.quantum_system(_SHIPPED_TARGET).calibration)
    monkeypatch.setattr(_source_stamp, "CALIBRATIONS", {_SHIPPED_CALIBRATION: "0123456789abcdef"})
    with pytest.raises(pulse.UnsupportedError, match="ships another .file 0123456789abcdef"):
        pickle.loads(data)
    monkeypatch.setattr(_source_stamp, "CALIBRATIONS", {})
    with pytest.raises(pulse.UnsupportedError, match="does not ship it"):
        pickle.loads(data)


def test_a_calibration_of_its_own_values_pickles_by_value_once_per_pickle():
    shipped = pulse.quantum_system(_SHIPPED_TARGET).calibration
    mine = Calibration(shipped._data, name="mine")
    quantum_system = pulse.quantum_system(make_quantum_system(_SHIPPED_TARGET), calibration=mine)
    back = _round_trip(mine)
    assert back is not mine and back == mine and back.name == "mine"
    one = len(pickle.dumps(rabi.lower(quantum_system.qudits[0], 0.1)))
    ten = len(pickle.dumps([rabi.lower(quantum_system.qudits[i], 0.1) for i in range(10)]))
    assert one > 100_000 and ten < one + 20_000
    assert copy.copy(mine) is mine and copy.deepcopy(mine) is mine


def test_a_calibration_of_other_values_makes_another_device():
    shipped = pulse.quantum_system(_SHIPPED_TARGET).calibration
    data = copy.deepcopy(shipped._data)
    same = Calibration(data, name="mine")
    data["cooldowns"]["2022-main"] = "edited"
    other = Calibration(data, name="mine")
    target = make_quantum_system(_SHIPPED_TARGET)
    quantum_systems = [pulse.quantum_system(target, calibration=c) for c in (same, other)]
    assert quantum_systems[0] != quantum_systems[1]
    again = pulse.quantum_system(target, calibration=Calibration(shipped._data, name="m"))
    assert again == quantum_systems[0]


def test_handles_and_registers_unpickle_as_their_device_and_indices(monkeypatch):
    monkeypatch.setattr(_pickling, "_devices", {})
    dev = pulse.quantum_system(_SHIPPED_TARGET, layout=[4, 5, 6])
    handles = dev.qudits
    q, pair, picked, everything = _round_trip((handles[1], handles[0:2], handles[[2, 0]], handles))
    assert q == handles[1] and (q.index, q.logical) == (5, 1)
    assert pair == handles[0:2] and picked == handles[[2, 0]]
    assert list(everything) == list(handles) and everything[2] == handles[2]
    assert q.quantum_system is pair[0].quantum_system is everything[0].quantum_system
    assert q.quantum_system == dev


def test_devices_handles_programs_and_classical_functions_copy_as_themselves():
    program = rabi.lower(Q[0], 0.3)
    for value in (LADDER, Q[0], program, flip):
        assert copy.copy(value) is value and copy.deepcopy(value) is value
    assert copy.deepcopy([Q[0], flip])[1] is flip


_CHECKS = [flip]


@pulse.kernel
def reset_from_a_list(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    fix = _CHECKS[0].submit([m]).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3))


def test_a_kernel_reading_a_list_of_classical_functions_compiles_once():
    first = reset_from_a_list.lower(Q[0])
    assert reset_from_a_list.lower(Q[0]) is first
    _CHECKS.append(flip)
    try:
        assert reset_from_a_list.lower(Q[0]) is not first
    finally:
        _CHECKS.pop()


# ── Kernels ──────────────────────────────────────────────────────────────────


def test_a_kernel_pickles_by_reference_as_a_function_does():
    assert _round_trip(rabi) is rabi

    @pulse.kernel
    def local(q):
        d = get_drive_line(q)
        wait(d, 4)

    def plain(q):
        pass

    for value in (local, plain):
        with pytest.raises((pickle.PicklingError, AttributeError), match="local"):
            pickle.dumps(value)


@pytest.fixture
def cloudpickle():
    return pytest.importorskip("cloudpickle")


def _local_kernel():
    @pulse.kernel(schedule="asap", cache_size=3, unroll_threshold=8)
    def local(q, amp: float):
        d = get_drive_line(q)
        drive(d, square(20, amp))

    return local


def test_cloudpickle_sends_a_kernel_its_name_does_not_find_by_value(cloudpickle, monkeypatch):
    local = _local_kernel()
    expected = local.lower(Q[0], 0.2).mlir()
    data = cloudpickle.dumps(local)
    assert cloudpickle.loads(data) is local
    monkeypatch.setattr(_pickling, "_kernels", {})
    back = cloudpickle.loads(data)
    assert back is not local and isinstance(back, type(local))
    assert back.__qualname__ == local.__qualname__ and back.__wrapped__.__name__ == "local"
    assert (back.schedule, back.unroll_threshold) == ("asap", 8)
    assert back.cache_info().capacity == 3
    assert back.lower(Q[0], 0.2).mlir() == expected
    assert cloudpickle.loads(data) is back


def test_cloudpickle_sends_a_kernel_its_name_finds_by_reference(cloudpickle):
    assert cloudpickle.loads(cloudpickle.dumps(rabi)) is rabi
    assert cloudpickle.loads(cloudpickle.dumps(pulse.compile(rabi))).kernel is rabi


_MAIN = """
import sys
import {pickler}
import cudaq_pulse as pulse
from tests._handles import Q

@pulse.kernel(schedule="asap", cache_size=3)
def sweep(q, amp: float):
    d = get_drive_line(q)
    drive(d, square(20, amp))

sys.stdout.buffer.write({pickler}.dumps((sweep, Q[0], sweep.lower(Q[0], 0.2).mlir())))
"""


def _sent_from_main(pickler, tmp_path, as_script):
    source = _MAIN.format(pickler=pickler)
    if as_script:
        script = tmp_path / "main.py"
        script.write_text(source)
        command = [sys.executable, str(script)]
    else:
        command = [sys.executable, "-c", source]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    root = pathlib.Path(__file__).resolve().parents[1]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, check=True).stdout


@pytest.mark.parametrize("as_script", [True, False], ids=["script", "prompt"])
def test_cloudpickle_sends_a_kernel_of_main_to_another_process(cloudpickle, tmp_path, as_script):
    data = _sent_from_main("cloudpickle", tmp_path, as_script)
    kernel, qudit, expected = cloudpickle.loads(data)
    assert kernel.__qualname__ == "sweep" and kernel.schedule == "asap"
    assert kernel.cache_info().capacity == 3
    assert kernel.lower(qudit, 0.2).mlir() == expected
    assert cloudpickle.loads(data)[0] is kernel


def test_pickle_sends_a_kernel_of_main_by_reference(tmp_path):
    data = _sent_from_main("pickle", tmp_path, as_script=True)
    with pytest.raises(AttributeError, match="sweep"):
        pickle.loads(data)


def _repeated(unroll_threshold):
    @pulse.kernel(unroll_threshold=unroll_threshold)
    def repeated(q):
        d = get_drive_line(q)
        for _ in range(4):
            drive(d, square(20, 0.1))

    return repeated


@pytest.mark.parametrize("threshold", [None, 0, 3])
def test_kernels_and_compiled_kernels_unpickle_with_their_unroll_threshold(
    threshold, cloudpickle, monkeypatch
):
    kernel = _repeated(threshold)
    data = cloudpickle.dumps(kernel)
    monkeypatch.setattr(_pickling, "_kernels", {})
    back = cloudpickle.loads(data)
    assert back.unroll_threshold == threshold
    assert back.lower(Q[0]).mlir() == kernel.lower(Q[0]).mlir()
    given = {} if threshold is None else {"unroll_threshold": threshold}
    compiled = pulse.compile(rabi, quantum_system=LADDER, **given)
    resolved = 8 if threshold is None else threshold
    assert _round_trip(compiled)._dispatch.options.unroll_threshold == resolved


def test_a_compiled_kernel_unpickles_without_its_specializations():
    compiled = pulse.compile(rabi, quantum_system=LADDER, schedule="asap")
    compiled.lower(Q[0], 0.1)
    back = _round_trip(compiled)
    assert back.kernel is rabi and back.quantum_system == LADDER
    assert back.cache_info().size == 0
    assert back.lower(Q[0], 0.1).mlir() == compiled.lower(Q[0], 0.1).mlir()
    with pytest.raises(pulse.QuantumSystemMismatchError):
        back.lower(pulse.quantum_system(_SHIPPED_TARGET).qudits[0], 0.1)


def test_a_pinned_compiled_kernel_unpickles_with_its_program(monkeypatch):
    # Only float parameters: compile(..., quantum_system=) builds the program at once.
    compiled = pulse.compile(idle, quantum_system=LADDER)
    expected = compiled.mlir(), compiled.parameters
    data = pickle.dumps(compiled)
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    back = pickle.loads(data)
    assert (back.mlir(), back.parameters) == expected


def test_a_classical_function_pickles_by_the_name_it_was_declared_with():
    assert _round_trip(flip) is flip


# ── Programs ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_programs()))
def test_a_program_unpickles_identical_and_runs_without_compiling(name, monkeypatch):
    program = _programs()[name]
    expected = _views(program)
    data = pickle.dumps(program)
    monkeypatch.setattr(_pickling, "_devices", {})
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    back = pickle.loads(data)
    assert type(back) is type(program)
    assert _views(back) == expected


@pulse.kernel
def carried_phase(q):
    d = get_drive_line(q)
    for _ in range(4):
        drive(d, square(20, 0.1))
        shift_phase(d.tone, 0.25)
    shift_frequency(d.tone, 1.0e6)


@pulse.kernel(max_duration_cycles=4000)
def swept(q, amp: float):
    d = get_drive_line(q)
    for _ in range(50):
        drive(d, gaussian(16, amp, 4.0))
        drive(d, gaussian(16, 0.5, 4.0))


_PERSISTENT = ControlSystem(
    controllers=[
        Controller(
            persistent_frames=True,
            waveform_memory_samples=4096,
            waveform_memory="shared",
        )
    ]
)
_AMPLITUDE_LIMIT = ControlSystem(controllers=[Controller(max_amplitude=0.6)])


def _exported():
    return {
        "persistent_frames": pulse.compile(
            carried_phase, control_system=_PERSISTENT, unroll_threshold=0
        ).lower(Q[0]),
        "rolled_call": pulse.compile(swept, unroll_threshold=0).lower(Q[0], 0.2),
        "checked_call": pulse.compile(
            swept, control_system=_AMPLITUDE_LIMIT, unroll_threshold=0
        ).lower(Q[0], 0.2),
    }


@pytest.mark.parametrize("name", ["persistent_frames", "rolled_call", "checked_call"])
def test_a_program_unpickles_with_its_controller_exports_in_both_formats(name, monkeypatch):
    program = _exported()[name]
    expected = (
        program.export_controller(),
        program.export_controller(version=2),
        program._concrete().drive_frames([]),
    )
    data = pickle.dumps(program)
    monkeypatch.setattr(_pickling, "_devices", {})
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    back = pickle.loads(data)
    assert type(back) is type(program) and type(back.generic) is type(program.generic)
    assert (
        back.export_controller(),
        back.export_controller(version=2),
        back._concrete().drive_frames([]),
    ) == expected


def test_frames_that_persist_unpickle_declared_and_exported():
    program = _round_trip(_exported()["persistent_frames"])
    assert program.options.machine == _PERSISTENT._machine
    assert program.options.machine.controller.persistent_frames is True
    flat, structured = program.export_controller(), program.export_controller(version=2)
    [frame] = flat["frame_state"]
    assert frame["frequency_hz"] == LADDER.physical_qudits[0].frequency_hz + 1.0e6
    assert structured["frame_state"] == flat["frame_state"]
    assert structured["controller"]["persistent_frames"] is True
    assert structured["controller"]["waveform_memory"] == "shared"


def test_an_unpickled_module_checks_each_calls_values_against_the_controller(monkeypatch):
    generic = _exported()["checked_call"].generic
    back = _round_trip(generic)
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    assert type(back) is type(generic)
    assert back.bind([0.3]).export_controller() == generic.bind([0.3]).export_controller()
    with pytest.raises(ControllerLimitError, match=r"peaks at amplitude 0\.9; .* at most 0\.6"):
        back.bind([0.9])
    with pytest.raises(ControllerLimitError, match="Controller.max_amplitude"):
        back._with_values([0.9])


def test_an_unpickled_module_binds_new_values_as_the_original_does(monkeypatch):
    generic = ramsey.lower(Q[0], 40.0).generic
    back = _round_trip(generic)
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    for tau in (0.0, 60.0, 123.0):
        assert back.bind([tau]).schedule == generic.bind([tau]).schedule
        assert back.bind([tau]).mlir() == generic.bind([tau]).mlir()
    with pytest.raises(ValueError, match="cannot be negative"):
        back.bind([-10.0])


def test_an_unpickled_module_checks_its_wait_lengths_as_compiled():
    generic = ramsey.lower(Q[0], 40.0).generic
    back = _round_trip(generic)
    assert back.waits.ok.__code__.co_code == generic.waits.ok.__code__.co_code
    for tau in (0.0, 60.0, -10.0, 1e30):
        assert back.waits.ok([tau]) == generic.waits.ok([tau])


def test_calls_pickled_together_share_their_module():
    first, second = _round_trip((rabi.lower(Q[0], 0.1), rabi.lower(Q[0], 0.2)))
    assert first.generic is second.generic
    assert first._module is second._module is first.generic._module
    assert first.arguments == (0.1,) and second.arguments == (0.2,)


@pytest.mark.parametrize(
    "part, other, named",
    [
        (0, "0.0.1", "0.0.1 (build"),
        (1, "0123456789abcdef", "build 0123456789abcdef"),
        (2, "fedcba9876543210", "sources fedcba9876543210"),
        (3, "cpython-0", "cpython-0)"),
    ],
    ids=["version", "native", "python sources", "python"],
)
def test_a_program_of_another_build_is_refused(part, other, named, monkeypatch):
    program = rabi.lower(Q[0], 0.3)
    stale = list(_pickling.build_stamp())
    stale[part] = other
    monkeypatch.setattr(_pickling, "_stamp", tuple(stale))
    data = pickle.dumps(program)
    monkeypatch.setattr(_pickling, "_stamp", None)
    with pytest.raises(pulse.UnsupportedError, match="compile the kernel again") as caught:
        pickle.loads(data)
    assert named in str(caught.value)
    assert pulse.__version__ in str(caught.value)


def test_the_build_stamp_names_this_build():
    version, build, sources, python = _pickling.build_stamp()
    assert version == pulse.__version__
    assert len(build) == len(sources) == 16 and int(build, 16) >= 0 and int(sources, 16) >= 0
    assert python == sys.implementation.cache_tag


def test_the_source_stamp_hashes_the_sources_as_they_are():
    staged = pathlib.Path(_pickling.__file__)
    if not staged.is_symlink():
        pytest.skip("an installed package: its sources were hashed when it was built")
    root = staged.resolve().parent
    found = sorted(
        p.relative_to(root).as_posix()
        for pattern in ("*.py", "*.pyi", "*.json", "*.toml")
        for p in root.rglob(pattern)
    )
    digests = "".join(
        f" {name} {hashlib.sha256((root / name).read_bytes()).hexdigest()}"
        for name in [*found, "py.typed"]
    )
    from cudaq_pulse._source_stamp import SOURCE_STAMP

    if SOURCE_STAMP != hashlib.sha256(digests.encode()).hexdigest()[:16]:
        build = os.path.relpath(staged.parents[2])
        pytest.fail(
            f"the source stamp is stale: the Python sources changed since {build} was "
            f"built, so pickled programs would carry the old stamp. "
            f"Run `ninja -C {build}`, then the tests again.",
            pytrace=False,
        )


# ── Control systems ──────────────────────────────────────────────────────────


@pulse.classical(max_latency_ns=200, link="decoder")
def correction(m: pulse.Bits) -> float:
    if m[0]:
        return 0.1
    return 0.0


@pulse.kernel
def corrected_pair(a, b, amp: float):
    ra = get_readout_line(a)
    da = get_drive_line(a)
    rb = get_readout_line(b)
    db = get_drive_line(b)
    ma = readout(ra, square(40, 0.1))
    mb = readout(rb, square(40, 0.1))
    drive(da, square(8, correction.submit([ma]).result()))
    drive(db, square(8, amp))
    drive(db, square(8, correction.submit([mb]).result()))


@pulse.kernel
def anywhere(q: pulse.Qudit, amp: float):
    d = get_drive_line(q)
    drive(d, square(4, amp))
    drive(d, square(4, 0.2))


def _qpu(slots=1, **processor):
    """Two controllers on their own clocks with a processor each, one
    without lines, and a link with an instance per controller."""
    cores = [pulse.Processor(links=["decoder"], **processor) for _ in range(2)]
    controllers = [
        pulse.Controller(
            drive_lines=[2 * k, 2 * k + 1],
            readout_lines=[2 * k, 2 * k + 1],
            clock_ghz=clock,
            processors=[core],
        )
        for k, (clock, core) in enumerate(zip((1.0, 0.5), cores, strict=True))
    ]
    link = Link(fixed_ns=10, slots=slots, per_controller=True)
    system = pulse.ControlSystem(
        controllers=[*controllers, pulse.Controller()], links={"decoder": link}
    )
    return pulse.quantum_system(ladder_target(4), control_system=system)


_QPU = _qpu()


def _declared_clock():
    controllers = [
        pulse.Controller(drive_lines=[k], readout_lines=[k], clock_ghz=clock)
        for k, clock in enumerate((2.4, 1.0))
    ]
    system = pulse.ControlSystem(controllers=controllers)
    return pulse.quantum_system(ladder_target(2), control_system=system, clock_ghz=12.0)


def _multi_controller_programs():
    q = _QPU.qudits
    compiled = pulse.compile(corrected_pair, quantum_system=_QPU, schedule="asap")
    placed = pulse.compile(anywhere, quantum_system=_QPU, schedule="asap", passes=[])
    return {
        "call": compiled.lower(q[0], q[2], 0.3),
        "placed_on_0": placed.lower(q[0], 0.1),
        "placed_on_2": placed.lower(q[2], 0.1),
    }


def _per_controller(program):
    """Each controller's programs, as the bytes a backend reads."""
    return [
        (
            json.dumps(program.export_controller(controller=k)),
            json.dumps(program.export_emulation(controller=k)),
        )
        for k in range(len(program.quantum_system.control_system.controllers))
    ]


def test_a_control_system_and_its_parts_unpickle_equal_with_equal_hashes():
    system = _QPU.control_system
    shared = Link(controllers=[system.controllers[1]], per_controller=True)
    for part in (system, *system.controllers, system.controllers[0].processors[0], shared):
        back = _round_trip(part)
        assert back is not part and back == part and hash(back) == hash(part)
    back = _round_trip(system)
    assert back.links["decoder"].per_controller
    assert back.as_dict() == system.as_dict() and back._machine == system._machine


@pytest.mark.parametrize("make", [lambda: _QPU, _declared_clock], ids=["derived", "declared"])
def test_a_device_unpickles_with_its_control_system_and_program_clock(make, monkeypatch):
    quantum_system = make()
    data = pickle.dumps(quantum_system)
    monkeypatch.setattr(_pickling, "_devices", {})
    back = pickle.loads(data)
    assert back is not quantum_system and back == quantum_system
    assert (
        back.control_system == quantum_system.control_system
        and back.clock_ghz == quantum_system.clock_ghz
    )
    assert back.as_dict() == quantum_system.as_dict()


def test_devices_differing_only_in_their_control_systems_unpickle_as_each(monkeypatch):
    one, two = _qpu(slots=1), _qpu(slots=2)
    assert one.fingerprint == two.fingerprint and one != two
    data = pickle.dumps(two)
    monkeypatch.setattr(_pickling, "_devices", {})
    pickle.dumps(one)
    back = pickle.loads(data)
    assert back is not one and back == two


def test_usage_reports_unpickle_equal():
    report = _multi_controller_programs()["call"].controller_usage
    controller = report.controllers[0]
    processor = controller.processors[0]
    for part in (report, controller, processor, processor.functions[0]):
        assert _round_trip(part) == part
    assert str(_round_trip(report)) == str(report)


def _usage_error():
    qpu = _qpu(max_ops_per_function=1)
    return _raised(
        lambda: pulse.compile(corrected_pair, quantum_system=qpu).lower(*qpu.qudits[:2], 0.3)
    )


def test_a_limit_error_unpickles_with_its_usage_report():
    error = _usage_error()
    assert error.usage is not None and error.usage.exceeded
    back = _round_trip(error)
    assert back.usage == error.usage and str(back) == str(error)


def test_a_multi_controller_program_exports_byte_identically_per_controller(monkeypatch):
    programs = _multi_controller_programs()
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    for name, program in programs.items():
        data = pickle.dumps(program)
        monkeypatch.setattr(_pickling, "_devices", {})
        back = pickle.loads(data)
        assert type(back) is type(program), name
        assert back.schedule == program.schedule, name
        assert back.classical_report() == program.classical_report(), name
        assert _per_controller(back) == _per_controller(program), name
    # Its generic module binds other values as the original does.
    generic = _round_trip(programs["call"].generic)
    assert _per_controller(generic.bind([0.2])) == _per_controller(
        programs["call"].generic.bind([0.2])
    )


_EXPORT_IN_ANOTHER_PROCESS = """
import json, pickle, sys
from cudaq_pulse.kernel import _kernel
from tests.test_pickling import _per_controller, _refuse_compile

_kernel.build = _refuse_compile
programs = pickle.loads(sys.stdin.buffer.read())
print(json.dumps([_per_controller(program) for program in programs]))
"""


@pytest.mark.slow
def test_another_process_exports_each_controllers_program_compiled_here():
    programs = list(_multi_controller_programs().values())
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    root = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-c", _EXPORT_IN_ANOTHER_PROCESS],
        input=pickle.dumps(programs),
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    expected = [[list(pair) for pair in _per_controller(program)] for program in programs]
    assert json.loads(done.stdout) == expected


def test_a_compiled_kernel_of_runtime_qudits_unpickles_compiling_once_for_any():
    compiled = pulse.compile(anywhere, quantum_system=_QPU, schedule="asap", passes=[])
    back = _round_trip(compiled)
    assert type(back._dispatch) is type(compiled._dispatch)
    for q in (0, 2):
        expected = _per_controller(compiled.lower(_QPU.qudits[q], 0.1))
        assert _per_controller(back.lower(_QPU.qudits[q], 0.1)) == expected
    assert back.cache_info().compiled == 1


# ── Errors ───────────────────────────────────────────────────────────────────


def _raised(make):
    try:
        make()
    except Exception as error:
        return error
    raise AssertionError("nothing raised")


def _limit_error():
    machine = ControlSystem(controllers=[Controller(instruction_cycles=4, min_idle_cycles=40)])
    return _raised(lambda: pulse.compile(ramsey, control_system=machine).lower(Q[0], 0.75))


@pytest.mark.parametrize(
    "make",
    [
        lambda: ControllerLimitError([f"violation {k}" for k in range(13)]),
        _limit_error,
        lambda: _raised(lambda: pulse.quantum_system(_SHIPPED_TARGET, layout=[99])),
        lambda: _raised(lambda: pulse.quantum_system(42)),
        lambda: SymbolicValueError("tau must be known", ("tau",), "a duration"),
        lambda: _raised(lambda: rabi.lower(0, 0.1)),
        lambda: _raised(lambda: Q[999]),
        lambda: pulse.InternalError("a check failed"),
        _usage_error,
    ],
    ids=[
        "limits",
        "raised",
        "definition",
        "type",
        "symbolic",
        "no_device",
        "index",
        "internal",
        "usage",
    ],
)
def test_errors_unpickle_with_their_attributes(make):
    error = make()
    back = _round_trip(error)
    assert type(back) is type(error) and back.args == error.args
    assert str(back) == str(error)
    assert vars(back) == vars(error)


def test_a_tracing_error_unpickles_without_where_tracing_stopped():
    @pulse.kernel
    def broken(q):
        d = get_drive_line(q)
        drive(d, square(8, 0.1), tone=d)

    error = _raised(lambda: broken.lower(Q[0]))
    assert hasattr(error, "pulse_source")
    back = _round_trip(error)
    assert not hasattr(back, "pulse_source") and str(back) == str(error)
    assert getattr(back, "__notes__", None) == getattr(error, "__notes__", None)


# ── Simulation results ───────────────────────────────────────────────────────

_PLUS = [2**-0.5, 2**-0.5]


def test_a_simulation_result_unpickles_with_its_branches_and_level_populations():
    result = pulse.simulate(reset, Q[0], initial_state=_PLUS, levels=2)
    back = _round_trip(result)
    assert back.branches == result.branches and len(back.branches) == 2
    assert [b.answers("flip") for b in back.branches] == [[[1]], [[0]]]
    assert back.measurements == result.measurements and back.qudits == result.qudits
    assert back.level_populations() == result.level_populations()
    assert back.level_population(1, qudit=Q[0]) == result.level_population(1, qudit=Q[0])
    np.testing.assert_array_equal(back.density_matrix, result.density_matrix)


def test_branches_level_populations_and_branch_coverage_pickle_on_their_own():
    result = pulse.simulate(reset, Q[0], initial_state=_PLUS, shots_count=20, seed=1, levels=2)
    for value in (
        result.branches,
        result.branches[0],
        result.branches[0].calls[0],
        result.measurements[0],
        result.level_populations(),
        result.level_population(0),
        pulse.BranchCoverage.SAMPLED,
    ):
        back = _round_trip(value)
        assert back == value and type(back) is type(value)
    np.testing.assert_array_equal(_round_trip(result).records, result.records)


# ── A process pool sweep ─────────────────────────────────────────────────────


def _run_in_worker(program):
    build, _kernel.build = _kernel.build, _refuse_compile
    try:
        return program.schedule, program.export_controller()
    finally:
        _kernel.build = build


def _compile_in_worker(kernel, qudit, amp):
    return kernel.lower(qudit, amp).mlir()


def _simulate_in_worker(program):
    return pulse.simulate(program, initial_state=_PLUS, levels=2)


def _over_the_limit():
    raise _limit_error()


def test_a_process_pool_sweep_runs_programs_compiled_in_the_parent():
    compiled = pulse.compile(rabi, quantum_system=LADDER)
    programs = [compiled.lower(Q[0], amp) for amp in (0.1, 0.2, 0.3, 0.4)]
    expected = [(p.schedule, p.export_controller()) for p in programs]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=context) as pool:
        assert list(pool.map(_run_in_worker, programs)) == expected
        assert pool.submit(_compile_in_worker, rabi, Q[3], 0.2).result() == (
            rabi.lower(Q[3], 0.2).mlir()
        )
        # A result comes back from the worker as the parent computes it.
        program = pulse.compile(reset, quantum_system=LADDER).lower(Q[0])
        simulated = pool.submit(_simulate_in_worker, program).result()
        with pytest.raises(ControllerLimitError, match="starts 21 cycles after") as caught:
            pool.submit(_over_the_limit).result()
    assert caught.value.violations == _limit_error().violations
    local = _simulate_in_worker(program)
    assert simulated.branches == local.branches
    assert simulated.level_populations() == local.level_populations()
