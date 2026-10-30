# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Runtime qudits: a kernel's parameters annotated ``Qudit`` or ``Register``
compile once and bind their qudits at each call; unannotated ones are
static."""

import dataclasses
import json
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Device, Qudit, Register
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits, check
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.ops import (
    drive,
    gaussian,
    get_drive_line,
    get_readout_line,
    readout,
    set_frequency,
    shift_frequency,
    shift_phase,
    square,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples

DEV = pulse.device("transmon_17q")
H = DEV.qudit_handles


def _body(q, r, amps, f12):
    d = get_drive_line(q)
    drive(d, gaussian(40, amps[0], 10.0))
    shift_phase(d.tone, 0.3)
    drive(d, gaussian(40, amps[1], 10.0))
    for x in r:
        dx = get_drive_line(x)
        drive(dx, square(20, 0.1))
    set_frequency(d.tone, f12)
    drive(d, square(20, amps[0]))
    rl = get_readout_line(q)
    return readout(rl, square(100, 0.05))


@pulse.kernel
def static_k(q, r, amps: list[float], f12: float):
    return _body(q, r, amps, f12)


def _f12(q):
    """The 1-2 frequency of *q*, read from its device on the host."""
    return float(q.device.transition_freq[q.logical, 1, 2])


@pulse.kernel
def runtime_k(q: Qudit, r: Register, amps: list[float], f12: float):
    return _body(q, r, amps, f12)


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
    calls = [(H[3], H[[5, 7]], [0.2, 0.4], _f12(H[3])), (H[9], H[[0, 16]], [0.1, 0.3], _f12(H[9]))]
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
    a = runtime.lower(H[3], H[[5]], [0.2, 0.4], _f12(H[3]))
    b = runtime.lower(
        other.qudit_handles[3], other.qudit_handles[[5]], [0.2, 0.4], _f12(other.qudit_handles[3])
    )
    assert runtime.cache_info().compiled == 1
    fa = {e["frequency_hz"] for e in a.export_controller()["events"] if e["kind"] == "drive"}
    fb = {e["frequency_hz"] for e in b.export_controller()["events"] if e["kind"] == "drive"}
    assert fa != fb
    static = static_k.lower(
        other.qudit_handles[3], other.qudit_handles[[5]], [0.2, 0.4], _f12(other.qudit_handles[3])
    )
    assert _plain(static.export_controller()) == _plain(b.export_controller())


def test_program_id_is_stable_and_symbols_bind():
    a = runtime_k.lower(H[3], H[[5, 7]], [0.2, 0.4], _f12(H[3])).export_controller()
    b = runtime_k.lower(H[9], H[[0, 16]], [0.1, 0.3], _f12(H[9])).export_controller()
    assert a["program_id"] == b["program_id"]
    symbols = {s["symbol"]: s for s in b["symbols"]}
    assert symbols["s0"]["value"] == DEV.layout[9]
    assert symbols["amps[1]"]["value"] == 0.3
    assert symbols["f12"]["value"] == _f12(H[9])
    assert symbols["s0.f01"]["value"] == DEV.frequencies[DEV.layout[9]]
    retuned = [e for e in b["events"] if e["symbols"].get("frequency_hz") == ["f12"]]
    assert retuned and all(e["symbols"]["qudit"] == "s0" for e in retuned)


def test_static_qudit_exports_have_no_symbols():
    def k(q, a: float):
        d = get_drive_line(q)
        drive(d, square(20, a))

    export = pulse.kernel(k).lower(H[3], 0.1).export_controller()
    assert "symbols" not in export and "program_id" not in export
    assert all("symbols" not in e for e in export["events"])


def test_duplicate_qudits_are_rejected():
    with pytest.raises(TypeError, match="more than once"):
        runtime_k.lower(H[3], H[[3, 5]], [0.2, 0.4], _f12(H[3]))


def test_binding_a_placed_module_checks_its_values():
    generic = runtime_k.lower(H[3], H[[5]], [0.2, 0.4], _f12(H[3])).generic
    with pytest.raises(pulse.InvalidTypeError, match="bind\\(\\) got 1 value for 3"):
        generic.bind([0.1])
    view = generic.bind([0.1, 0.2, float("nan")])
    with pytest.raises(pulse.InvalidValueError, match="is nan"):
        view.mlir()


def test_qudits_of_two_devices_are_rejected():
    other = _recalibrated(DEV, DEV.layout[3], 1e6)
    with pytest.raises(pulse.DeviceMismatchError):
        runtime_k.lower(H[3], other.qudit_handles[[5]], [0.2, 0.4], _f12(H[3]))


def test_register_length_is_part_of_the_key():
    runtime = pulse.kernel(runtime_k.__wrapped__)
    runtime.lower(H[3], H[[5]], [0.2, 0.4], _f12(H[3]))
    runtime.lower(H[3], H[[5, 7]], [0.2, 0.4], _f12(H[3]))
    runtime.lower(H[4], H[[6, 8]], [0.2, 0.4], _f12(H[4]))
    assert runtime.cache_info().compiled == 2


def test_unbound_view_cannot_lower_to_llvm():
    @pulse.kernel
    def k(q: Qudit):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))

    lowered = k.lower(H[3])
    assert lowered.carriers == (DEV.layout[3],)
    assert "llvm" in lowered.mlir("llvm")
    generic = lowered.generic
    assert generic is not None and generic._unbound()
    with pytest.raises(pulse.InvalidValueError, match="runtime qudits; lower it with handles"):
        generic.mlir("llvm")


def test_simulation_matches_static():
    pytest.importorskip("cudaq_pulse.runtime.simulate")
    a = static_k.lower(H[1], H[[2]], [0.2, 0.4], _f12(H[1]))
    b = runtime_k.lower(H[1], H[[2]], [0.2, 0.4], _f12(H[1]))
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
    args = (H[q], H[rest], [amp, amp / 2], _f12(H[q]))
    a, b = static_k.lower(*args), runtime_k.lower(*args)
    assert _plain(a.export_controller()) == _plain(b.export_controller())


@pulse.kernel
def _two_readouts(a: Qudit, b: Qudit):
    la = get_readout_line(a)
    lb = get_readout_line(b)
    readout(la, square(20, 0.05))
    readout(lb, square(20, 0.05))


def test_acquirer_limits_are_checked_per_call():
    c = DEV.layout
    limits = ControllerLimits(acquirers={c[0]: 0, c[1]: 0, c[2]: 1}, acquirer_hold_cycles=30)
    compiled = pulse.compile(_two_readouts, schedule="asap")
    shared = compiled.lower(H[0], H[1])
    apart = compiled.lower(H[0], H[2])
    assert compiled.cache_info().compiled == 1
    assert any("acquirer 0" in v for v in check(shared, limits))
    assert check(apart, limits) == []


def test_bound_views_print_carriers():
    @pulse.kernel
    def k(q: Qudit):
        line = get_drive_line(q)
        drive(line, square(20, 0.1))

    carrier = DEV.layout[4]
    text = pulse.compile(k, passes=()).lower(H[4]).mlir()
    assert f"qubit = {carrier} : i64" in text
    assert "pulse.slot" in text and "!pulse.qref" not in text.split("func.func")[1].split(")")[0]


@pytest.mark.parametrize(
    "annotation, read, name, rewrite",
    [
        (Qudit, "q.logical", "'q'", "value: float, called with TABLE[q.logical]"),
        (None, "q.index", "'q'", "value: float, called with TABLE[q.index]"),
    ],
)
def test_reading_a_qudit_attribute_names_the_rewrite(annotation, read, name, rewrite):
    table = [0.1] * len(H)
    source = f"""
def k(q):
    d = get_drive_line(q)
    drive(d, square(20, table[{read}]))
"""
    scope = {"table": table, "get_drive_line": get_drive_line, "drive": drive, "square": square}
    exec(source, scope)
    k = scope["k"]
    if annotation is not None:
        k.__annotations__["q"] = annotation
    with pytest.raises(CompilationError, match=rf"cannot read \.\w+ of qudit {name}") as e:
        pulse.kernel(k).lower(H[2])
    assert rewrite in str(e.value)


def test_reading_a_register_element_attribute_names_the_list_rewrite():
    table = [0.1] * len(H)

    @pulse.kernel
    def k(qs: Register):
        for q in qs:
            d = get_drive_line(q)
            drive(d, square(20, table[q.index]))

    with pytest.raises(CompilationError, match=r"of qudit 'qs\[0\]'") as e:
        k.lower(H[[1, 2]])
    assert "values: list[float], called with [TABLE[q.index] for q in qs]" in str(e.value)


def test_a_runtime_qudit_frequency_is_one_tone_until_its_line_is_retuned():
    machine = pulse.compiler.MachineModel(controller=ControllerLimits(tones_per_line=1))

    @pulse.kernel
    def native(q: Qudit):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))
        drive(d, square(20, 0.2), tone=shift_phase(d.tone, 0.5))

    @pulse.kernel
    def retuned(q: Qudit):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))
        drive(d, square(20, 0.2), tone=shift_frequency(d.tone, 1e6))

    assert pulse.compile(native, machine=machine).lower(H[3]).export_controller()
    with pytest.raises(ControllerLimitError, match="2 tone frequencies"):
        pulse.compile(retuned, machine=machine).lower(H[3])


def test_list_promoted_to_static_warns():
    @pulse.kernel
    def k(q: Qudit, lengths: list[float]):
        d = get_drive_line(q)
        drive(d, square(lengths[0], 0.1))

    with pytest.warns(RuntimeWarning, match="lengths"):
        k.lower(H[0], [20.0])


def test_only_annotated_qudits_are_placed_at_call_time():
    def body(q, r):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))
        for x in r:
            dx = get_drive_line(x)
            drive(dx, square(20, 0.1))

    def annotated(q: Qudit, r: Register):
        body(q, r)

    def unannotated(q, r):
        body(q, r)

    placed, static = pulse.kernel(annotated), pulse.kernel(unannotated)
    assert placed.runtime_parameters == ("q", "r") and placed.static_parameters == ()
    assert static.runtime_parameters == () and static.static_parameters == ("q", "r")
    for i in (3, 4, 5):
        placed.lower(H[i], H[[i + 1]])
        static.lower(H[i], H[[i + 1]])
    assert placed.cache_info().compiled == 1
    assert static.cache_info().compiled == 3


def _calls(fn):
    """The Python and the C function calls one run of *fn* makes, warm."""
    fn()
    counts = {"call": 0, "c_call": 0}

    def count(frame, event, arg):
        if event in counts:
            counts[event] += 1

    sys.setprofile(count)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return counts["call"], counts["c_call"]


def test_a_runtime_qudit_export_adds_calls_that_do_not_grow_with_the_program():
    gaps = []
    for n in (1, 64):

        def drives(q):
            d = get_drive_line(q)
            for k in range(n):
                drive(d, square(20, 0.1 + 0.001 * k))

        def placed(q: Qudit):
            drives(q)

        runtime = _calls(pulse.kernel(placed).lower(H[0]).export_controller)
        fixed = _calls(pulse.kernel(drives).lower(H[0]).export_controller)
        # Beyond each event's own copy of its symbols, a fixed few calls.
        gaps.append((runtime[0] - fixed[0], runtime[1] - fixed[1] - n))
    assert all(python <= 8 and native <= 16 for python, native in gaps), gaps
