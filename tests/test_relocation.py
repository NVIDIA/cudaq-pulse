"""Runtime qudits: a kernel whose qudit parameters are annotated ``Qudit`` or
``Register`` compiles once and binds its qudits at each call."""

import dataclasses
import json

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Device, Qudit, Register, Target, get_device
from cudaq_pulse.controller import ControllerLimits, check
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.ops import (
    drive,
    frequency,
    gaussian,
    get_drive_line,
    get_readout_line,
    readout,
    set_frequency,
    shift_phase,
    square,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples

DEV = get_device(Target.TRANSMON_KRINNER_17Q)
H = DEV.qudit_handles


def _body(q, r, amps):
    d, t = get_drive_line(q)
    drive(d, gaussian(40, amps[0], 10.0), t)
    t = shift_phase(t, 0.3)
    drive(d, gaussian(40, amps[1], 10.0), t)
    for x in r:
        dx, tx = get_drive_line(x)
        drive(dx, square(20, 0.1), tx)
    t = set_frequency(t, frequency(q, (1, 2)))
    drive(d, square(20, amps[0]), t)
    rl, rt = get_readout_line(q)
    return readout(rl, square(100, 0.05), rt)


@pulse.kernel
def static_k(q, r, amps: list[float]):
    return _body(q, r, amps)


@pulse.kernel
def runtime_k(q: Qudit, r: Register, amps: list[float]):
    return _body(q, r, amps)


def _plain(export):
    """An export without the symbols only a runtime compile records."""
    export = {k: v for k, v in export.items() if k not in ("program_id", "symbols")}
    export["events"] = [{k: v for k, v in e.items() if k != "symbols"} for e in export["events"]]
    return json.dumps(export, sort_keys=True)


def _recalibrated(device, carrier, shift_hz):
    target = device.target
    qubits = dict(target.qubits)
    qubits[carrier] = dataclasses.replace(
        qubits[carrier], frequency_hz=qubits[carrier].frequency_hz + shift_hz
    )
    return Device(dataclasses.replace(target, qubits=qubits), name=device.name)


@pytest.mark.parametrize("schedule", ["alap", "asap"])
def test_runtime_qudits_match_static_compiles(schedule):
    options = {"schedule": schedule}
    static = pulse.kernel(static_k.__wrapped__, **options)
    runtime = pulse.kernel(runtime_k.__wrapped__, **options)
    calls = [(H[3], H[[5, 7]], [0.2, 0.4]), (H[9], H[[0, 16]], [0.1, 0.3])]
    for args in calls:
        a, b = static.lower(*args), runtime.lower(*args)
        assert _plain(a.export_controller()) == _plain(b.export_controller())
        assert a.qudits == b.qudits
        assert a.schedule == b.schedule
    assert static.cache_info().compiled == 2
    assert runtime.cache_info().compiled == 1


def test_recalibration_reuses_the_compile():
    runtime = pulse.kernel(runtime_k.__wrapped__)
    other = _recalibrated(DEV, DEV.layout[3], 2.5e6)
    assert other.fingerprint != DEV.fingerprint
    a = runtime.lower(H[3], H[[5]], [0.2, 0.4])
    b = runtime.lower(other.qudit_handles[3], other.qudit_handles[[5]], [0.2, 0.4])
    assert runtime.cache_info().compiled == 1
    fa = {e["frequency_hz"] for e in a.export_controller()["events"] if e["kind"] == "drive"}
    fb = {e["frequency_hz"] for e in b.export_controller()["events"] if e["kind"] == "drive"}
    assert fa != fb
    static = static_k.lower(other.qudit_handles[3], other.qudit_handles[[5]], [0.2, 0.4])
    assert _plain(static.export_controller()) == _plain(b.export_controller())


def test_program_id_is_stable_and_symbols_bind():
    a = runtime_k.lower(H[3], H[[5, 7]], [0.2, 0.4]).export_controller()
    b = runtime_k.lower(H[9], H[[0, 16]], [0.1, 0.3]).export_controller()
    assert a["program_id"] == b["program_id"]
    symbols = {s["symbol"]: s for s in b["symbols"]}
    assert symbols["s0"]["value"] == DEV.layout[9]
    assert symbols["amps[1]"]["value"] == 0.3
    assert symbols["s0.f12"]["transition"] == [1, 2]
    assert symbols["s0.f01"]["value"] == DEV.frequencies[DEV.layout[9]]
    retuned = [e for e in b["events"] if e["symbols"].get("frequency_hz") == ["s0.f12"]]
    assert retuned and all(e["symbols"]["qudit"] == "s0" for e in retuned)


def test_float_only_exports_have_no_symbols():
    @pulse.kernel
    def k(q, a: float):
        d, t = get_drive_line(q)
        drive(d, square(20, a), t)

    export = k.lower(H[3], 0.1).export_controller()
    assert "symbols" not in export and "program_id" not in export
    assert all("symbols" not in e for e in export["events"])


def test_frequency_of_a_static_qudit_is_a_constant():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(20, 0.1), set_frequency(t, frequency(q, (2, 1))))

    events = k.lower(H[4]).export_controller()["events"]
    qubit = DEV.target.qubits[DEV.layout[4]]
    assert events[0]["frequency_hz"] == pytest.approx(qubit.frequency_hz + qubit.anharmonicity_hz)


@pytest.mark.parametrize("transition", [(0, 0), (0, 3), (-1, 1), (0, 1.0)])
def test_frequency_rejects_bad_transitions(transition):
    @pulse.kernel
    def k(q: Qudit):
        d, t = get_drive_line(q)
        drive(d, square(20, 0.1), set_frequency(t, frequency(q, transition)))

    with pytest.raises((CompilationError, ValueError)):
        k.lower(H[0])


def test_duplicate_qudits_are_rejected():
    with pytest.raises(TypeError, match="more than once"):
        runtime_k.lower(H[3], H[[3, 5]], [0.2, 0.4])


def test_qudits_of_two_devices_are_rejected():
    other = _recalibrated(DEV, DEV.layout[3], 1e6)
    with pytest.raises(pulse.DeviceMismatchError):
        runtime_k.lower(H[3], other.qudit_handles[[5]], [0.2, 0.4])


def test_register_length_is_part_of_the_key():
    runtime = pulse.kernel(runtime_k.__wrapped__)
    runtime.lower(H[3], H[[5]], [0.2, 0.4])
    runtime.lower(H[3], H[[5, 7]], [0.2, 0.4])
    runtime.lower(H[4], H[[6, 8]], [0.2, 0.4])
    assert runtime.cache_info().compiled == 2


def test_unbound_view_cannot_lower_to_llvm():
    @pulse.kernel
    def k(q: Qudit):
        d, t = get_drive_line(q)
        drive(d, square(20, 0.1), t)

    lowered = k.lower(H[3])
    assert lowered.carriers == (DEV.layout[3],)
    assert "llvm" in lowered.mlir("llvm")
    generic = lowered.generic
    assert generic is not None and generic._unbound()
    with pytest.raises(Exception, match="carrier|bound|qudit"):
        generic.mlir("llvm")


def test_simulation_matches_static():
    pytest.importorskip("cudaq_pulse.runtime.simulate")
    a = static_k.lower(H[1], H[[2]], [0.2, 0.4])
    b = runtime_k.lower(H[1], H[[2]], [0.2, 0.4])
    ra, rb = pulse.simulate(a), pulse.simulate(b)
    assert ra.qudits == rb.qudits
    np.testing.assert_allclose(ra.density_matrix, rb.density_matrix, atol=0)


@settings(max_examples=examples(20), deadline=None)
@given(
    st.lists(st.integers(0, len(H) - 1), min_size=2, max_size=4, unique=True),
    st.floats(0.01, 0.5),
)
def test_any_placement_matches_static(indices, amp):
    q, *rest = indices
    args = (H[q], H[rest], [amp, amp / 2])
    a, b = static_k.lower(*args), runtime_k.lower(*args)
    assert _plain(a.export_controller()) == _plain(b.export_controller())


@pulse.kernel
def _two_readouts(a: Qudit, b: Qudit):
    la, ta = get_readout_line(a)
    lb, tb = get_readout_line(b)
    readout(la, square(20, 0.05), ta)
    readout(lb, square(20, 0.05), tb)


def test_acquirer_limits_are_checked_per_call():
    c = DEV.layout
    limits = ControllerLimits(acquirers={c[0]: 0, c[1]: 0, c[2]: 1}, acquirer_hold_cycles=30)
    compiled = pulse.compile(_two_readouts, schedule="asap")
    shared = compiled.lower(H[0], H[1])
    apart = compiled.lower(H[0], H[2])
    assert compiled.cache_info().compiled == 1
    assert any("acquirer 0" in v for v in check(shared, limits))
    assert check(apart, limits) == []


def test_list_promoted_to_static_warns():
    @pulse.kernel
    def k(q: Qudit, lengths: list[float]):
        d, t = get_drive_line(q)
        drive(d, square(lengths[0], 0.1), t)

    with pytest.warns(RuntimeWarning, match="lengths"):
        k.lower(H[0], [20.0])
