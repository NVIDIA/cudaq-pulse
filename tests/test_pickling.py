# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pickling targets, devices, handles, kernels, programs and errors, as
process pools and caches on disk do."""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import _pickling
from cudaq_pulse.classical import Bits
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits
from cudaq_pulse.kernel import _kernel
from cudaq_pulse.kernel.ir_builder import SymbolicValueError
from cudaq_pulse.targets import Coupling, Qubit, TargetDescription

from tests._handles import LADDER, Q


def _round_trip(value):
    return pickle.loads(pickle.dumps(value))


def _refuse_compile(*args, **kwargs):
    raise AssertionError("the program was compiled again")


# ── Kernels ──────────────────────────────────────────────────────────────────


@pulse.kernel
def rabi(q, amp: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(40, amp, 10.0), t)
    sync(d, r)
    return readout(r, square(100, 0.05), rt)


@pulse.kernel
def ramsey(q, tau: float):
    d, t = get_drive_line(q)
    drive(d, square(20, 0.1), t)
    wait(d, tau)
    drive(d, square(20, 0.1), t)


@pulse.kernel
def branch(q, r):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    d1, t1 = get_drive_line(r)
    drive(d, drag(16, 0.25, 4.0, 0.3), t)
    shift_phase(t, 0.5)
    sync(d, ro)
    m = readout(ro, square(100, 0.05), rt)
    sync(ro, d1)
    if m:
        drive(d1, square(20, 0.3), t1)
    else:
        drive(d1, gaussian(20, 0.1, 5.0), t1)


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=400)
def flip(bits):
    return [1 - bits[0]]


@pulse.kernel
def reset(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    fix = flip.submit([m]).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3), t)


@pulse.kernel
def idle(tau: float):
    pass


_KRINNER = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
_SLOW_CONTROLLER = MachineModel(controller=ControllerLimits(clock_ghz=1.0))
_DELAYED = MachineModel(line_delays_ns={("drive", 0): 1.5})


def _programs():
    return {
        "bound": rabi.lower(Q[0], 0.3),
        "compiled": pulse.compile(rabi, device=LADDER).lower(Q[1], 0.2),
        "generic": rabi.lower(Q[0], 0.3).generic,
        "variable_wait": ramsey.lower(Q[0], 40.0),
        "branch": pulse.compile(branch, schedule="asap").lower(Q[0], Q[1]),
        "classical_call": pulse.compile(reset, schedule="asap").lower(Q[0]),
        "other_clock": pulse.compile(rabi, machine=_SLOW_CONTROLLER).lower(Q[0], 0.3),
        "line_delays": pulse.compile(rabi, machine=_DELAYED).lower(Q[0], 0.3),
        "krinner": rabi.lower(_KRINNER.qudit_handles[2], 0.1),
    }


def _views(program):
    """What a program shows, without compiling it again."""
    shown = {
        "mlir": program.mlir(),
        "schedule": program.schedule,
        "epochs": program.epochs(),
        "metrics": program.metrics,
        "parameters": program.parameters,
        "arguments": program.arguments,
        "qudits": program.qudits,
        "options": program.options,
        "target": program.target,
        "classical": program.classical_report(),
    }
    if not program.parameters:
        shown["controller"] = program.export_controller()
        shown["emulation"] = program.export_emulation()
    return shown


# ── Targets and devices ──────────────────────────────────────────────────────


def test_targets_and_their_parts_unpickle_equal_with_equal_hashes():
    qubit = Qubit(3, 5.1e9, -300e6, 50.0, 30.0, label="D3", drive_params={"x_amp": 0.2})
    coupling = Coupling(3, 4, 5e6, gate_type="cz", gate_params={"theta": 0.1})
    krinner = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q).target
    for value in (qubit, coupling, LADDER.target, krinner, pulse.Target.TRANSMON_KRINNER_17Q):
        back = _round_trip(value)
        assert back == value and hash(back) == hash(value)
    back = _round_trip(krinner)
    assert back.qubits[0].drive_params == krinner.qubits[0].drive_params
    assert back.couplings == krinner.couplings and back.crosstalk == krinner.crosstalk


def test_targets_compare_by_content():
    a = TargetDescription(name="t", qubits={0: Qubit(0, 5e9, -3e8, 50.0, 30.0)})
    b = TargetDescription(name="t", qubits={0: Qubit(0, 5e9, -3e8, 50.0, 30.0)})
    c = TargetDescription(name="t", qubits={0: Qubit(0, 5.1e9, -3e8, 50.0, 30.0)})
    assert a == b and hash(a) == hash(b)
    assert a != c and a != "t"


@pytest.mark.parametrize(
    "make",
    [
        lambda: LADDER,
        lambda: pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q),
        lambda: pulse.device(
            {"target": "transmon_krinner_17q", "layout": [4, 5, 6], "clock_ghz": 1.0}
        ),
    ],
    ids=["ladder", "shipped", "config"],
)
def test_a_device_unpickles_equal_and_is_built_once_per_process(make, monkeypatch):
    device = make()
    data = pickle.dumps(device)
    back = pickle.loads(data)
    assert back == device and hash(back) == hash(device)
    assert pickle.loads(data) is back
    # Another process has not seen the device: it builds it again.
    monkeypatch.setattr(_pickling, "_devices", {})
    built = pickle.loads(data)
    assert built == device and built is not device
    assert built.layout == device.layout and built.clock_ghz == device.clock_ghz
    assert built.target == device.target
    assert (built.calibration is None) == (device.calibration is None)
    assert pickle.loads(data) is built


def test_handles_and_registers_unpickle_as_their_device_and_indices(monkeypatch):
    monkeypatch.setattr(_pickling, "_devices", {})
    dev = pulse.device({"target": "transmon_krinner_17q", "layout": [4, 5, 6]})
    handles = dev.qudit_handles
    q, pair, picked, everything = _round_trip((handles[1], handles[0:2], handles[[2, 0]], handles))
    assert q == handles[1] and (q.index, q.logical) == (5, 1)
    assert pair == handles[0:2] and picked == handles[[2, 0]]
    assert list(everything) == list(handles) and everything[2] == handles[2]
    assert q.device is pair[0].device is everything[0].device
    assert q.device == dev


def test_devices_handles_programs_and_classical_functions_copy_as_themselves():
    program = rabi.lower(Q[0], 0.3)
    for value in (LADDER, Q[0], program, flip):
        assert copy.copy(value) is value and copy.deepcopy(value) is value
    assert copy.deepcopy([Q[0], flip])[1] is flip


_CHECKS = [flip]


@pulse.kernel
def reset_from_a_list(q):
    d, t = get_drive_line(q)
    ro, rt = get_readout_line(q)
    m = readout(ro, square(100, 0.05), rt)
    fix = _CHECKS[0].submit([m]).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3), t)


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
        d, t = get_drive_line(q)
        wait(d, 4)

    def plain(q):
        pass

    for value in (local, plain):
        with pytest.raises((pickle.PicklingError, AttributeError), match="local"):
            pickle.dumps(value)


def test_a_compiled_kernel_unpickles_without_its_specializations():
    compiled = pulse.compile(rabi, device=LADDER, schedule="asap")
    compiled.lower(Q[0], 0.1)
    back = _round_trip(compiled)
    assert back.kernel is rabi and back.device == LADDER
    assert back.cache_info().size == 0
    assert back.lower(Q[0], 0.1).mlir() == compiled.lower(Q[0], 0.1).mlir()
    with pytest.raises(pulse.DeviceMismatchError):
        back.lower(pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q).qudit_handles[0], 0.1)


def test_a_pinned_compiled_kernel_unpickles_with_its_program(monkeypatch):
    # Only float parameters: compile(..., device=) builds the program at once.
    compiled = pulse.compile(idle, device=LADDER)
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


def test_an_unpickled_module_binds_new_values_as_the_original_does(monkeypatch):
    generic = ramsey.lower(Q[0], 40.0).generic
    back = _round_trip(generic)
    monkeypatch.setattr(_kernel, "build", _refuse_compile)
    for tau in (0.0, 60.0, 123.0):
        assert back.bind([tau]).schedule == generic.bind([tau]).schedule
        assert back.bind([tau]).mlir() == generic.bind([tau]).mlir()
    with pytest.raises(ValueError, match="cannot be negative"):
        back.bind([-10.0])


def test_calls_pickled_together_share_their_module():
    first, second = _round_trip((rabi.lower(Q[0], 0.1), rabi.lower(Q[0], 0.2)))
    assert first.generic is second.generic
    assert first._module is second._module is first.generic._module
    assert first.arguments == (0.1,) and second.arguments == (0.2,)


def test_a_program_of_another_build_is_refused(monkeypatch):
    program = rabi.lower(Q[0], 0.3)
    monkeypatch.setattr(_pickling, "_stamp", (_pickling._FORMAT, "0.0.1", "0123456789abcdef"))
    data = pickle.dumps(program)
    monkeypatch.setattr(_pickling, "_stamp", None)
    with pytest.raises(pulse.UnsupportedError, match="compile the kernel again") as caught:
        pickle.loads(data)
    assert "0.0.1 (build 0123456789abcdef" in str(caught.value)
    assert pulse.__version__ in str(caught.value)


def test_the_build_stamp_names_this_build():
    format_, version, build = _pickling.build_stamp()
    assert format_ == _pickling._FORMAT and version == pulse.__version__
    assert len(build) == 16 and int(build, 16) >= 0


# ── Errors ───────────────────────────────────────────────────────────────────


def _raised(make):
    try:
        make()
    except Exception as error:
        return error
    raise AssertionError("nothing raised")


def _limit_error():
    machine = MachineModel(controller=ControllerLimits(issue_cycles=4, min_wait_cycles=40))
    return _raised(lambda: pulse.compile(ramsey, machine=machine).lower(Q[0], 0.75))


@pytest.mark.parametrize(
    "make",
    [
        lambda: ControllerLimitError([f"violation {k}" for k in range(13)]),
        _limit_error,
        lambda: _raised(lambda: pulse.device({"target": "transmon_krinner_17q", "layout": [99]})),
        lambda: _raised(lambda: pulse.get_device("transmon_krinner_17q")),
        lambda: SymbolicValueError("tau must be known", ("tau",), "a duration"),
        lambda: _raised(lambda: rabi.lower(0, 0.1)),
        lambda: _raised(lambda: Q[999]),
        lambda: pulse.InternalError("a check failed"),
    ],
    ids=["limits", "raised", "definition", "type", "symbolic", "no_device", "index", "internal"],
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
        d, t = get_drive_line(q)
        drive(d, square(8, 0.1), d)

    error = _raised(lambda: broken.lower(Q[0]))
    assert hasattr(error, "pulse_source")
    back = _round_trip(error)
    assert not hasattr(back, "pulse_source") and str(back) == str(error)
    assert getattr(back, "__notes__", None) == getattr(error, "__notes__", None)


# ── A process pool sweep ─────────────────────────────────────────────────────


def _run_in_worker(program):
    build, _kernel.build = _kernel.build, _refuse_compile
    try:
        return program.schedule, program.export_controller()
    finally:
        _kernel.build = build


def _compile_in_worker(kernel, qudit, amp):
    return kernel.lower(qudit, amp).mlir()


def _over_the_limit():
    raise _limit_error()


def test_a_process_pool_sweep_runs_programs_compiled_in_the_parent():
    compiled = pulse.compile(rabi, device=LADDER)
    programs = [compiled.lower(Q[0], amp) for amp in (0.1, 0.2, 0.3, 0.4)]
    expected = [(p.schedule, p.export_controller()) for p in programs]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=context) as pool:
        assert list(pool.map(_run_in_worker, programs)) == expected
        assert pool.submit(_compile_in_worker, rabi, Q[3], 0.2).result() == (
            rabi.lower(Q[3], 0.2).mlir()
        )
        with pytest.raises(ControllerLimitError, match="starts 20 cycles after") as caught:
            pool.submit(_over_the_limit).result()
    assert caught.value.violations == _limit_error().violations
