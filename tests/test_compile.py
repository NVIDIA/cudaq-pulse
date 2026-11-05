# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for compiling kernels: ``pulse.compile`` and ``kernel.lower``."""

from __future__ import annotations

import array
import math

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pulse.kernel
def _bell(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf)
    sync(d0, d1)
    drive(d1, wf)


_FREQ_2Q = {0: 5.0e9, 1: 5.1e9}

# ---------------------------------------------------------------------------
# Basic compile() usage
# ---------------------------------------------------------------------------


def test_compile_returns_compiled_kernel():
    ck = pulse.compile(_bell)
    assert isinstance(ck, pulse.CompiledKernel)
    assert isinstance(ck.lower(Q[0], Q[1]), pulse.Lowered)


def test_compile_produces_mlir():
    ck = _bell.lower(Q[0], Q[1])
    mlir = ck.mlir()
    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.gaussian" in mlir
    assert "pulse.drive" in mlir


def test_scheduled_event_start_and_end():
    events = _bell.lower(Q[0], Q[1]).schedule
    assert events
    for e in events:
        assert e.start == e.start_cycles
        assert e.end == e.start_cycles + e.duration_cycles
    unscheduled = pulse.compiler.ScheduledEvent("drive", 0, 0, None, None)
    with pytest.raises(ValueError, match="not scheduled"):
        unscheduled.start  # noqa: B018
    with pytest.raises(ValueError, match="not scheduled"):
        unscheduled.end  # noqa: B018


def test_compile_metrics():
    ck = _bell.lower(Q[0], Q[1])
    m = ck.metrics
    assert isinstance(m, pulse.compiler.CompileMetrics)
    assert m.total_ms > 0
    assert m.trace_ms > 0
    assert m.ffi_ms > 0


def test_compile_no_passes():
    ck = pulse.compile(_bell, passes=()).lower(Q[0], Q[1])
    assert ck.mlir() is not None
    assert "pulse.drive" in ck.mlir()


def test_compile_single_qubit():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf)

    ck = k.lower(Q[0])
    assert "pulse.drive" in ck.mlir()


def _sparse_device():
    from cudaq_pulse.targets import Qubit, TargetDescription

    qubits = {i: Qubit(i, 5.0e9 + 1e8 * i, -300e6, 50.0, 30.0) for i in (0, 2)}
    return pulse.device({"target": TargetDescription(name="sparse", qubits=qubits)})


@pulse.kernel
def _drive_one(q):
    drive(get_drive_line(q), square(40, 0.1))


def test_a_module_records_0_hz_for_a_qudit_its_target_lacks():
    dev = _sparse_device()
    for _ in range(2):
        mlir = _drive_one.lower(dev.qudit_handles[1]).mlir()
        assert "pulse.qubit_freq_hz = array<f64: 5.000000e+09, 0.000000e+00, 5.200000e+09>" in mlir


def test_compile_folds_phases():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        shift_phase(d.tone, math.pi / 4)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf)

    ck = k.lower(Q[0])
    assert ck.mlir() is not None
    assert ck.metrics.total_ms > 0


def test_compile_with_fusion():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        sq1 = square(50, 0.2)
        drive(d, sq1)
        sq2 = square(50, 0.2)
        drive(d, sq2)

    ck = k.lower(Q[0])
    assert ck.mlir() is not None


@pulse.kernel
def _ramsey(q, phase: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, 0.3, 10.0))
    shift_phase(d.tone, phase)
    drive(d, gaussian(40, 0.3, 10.0))


def test_a_phase_from_a_float_folds_into_the_drive_once_bound():
    bound = pulse.compile(_ramsey).lower(Q[0], 0.25)
    assert "pulse.shift_phase" in bound.generic.mlir()
    assert "pulse.shift_phase" not in bound.mlir()
    assert "frame_phase_offset = 2.500000e-01" in bound.mlir()
    # Without fold_phases the frame change stays, bound or not.
    assert "pulse.shift_phase" in pulse.compile(_ramsey, passes=()).lower(Q[0], 0.25).mlir()


@pulse.kernel
def _two_squares(q, a: float, b: float):
    d = get_drive_line(q)
    drive(d, square(50, a))
    drive(d, square(50, b))


def test_squares_of_amplitudes_bound_equal_fuse():
    compiled = pulse.compile(_two_squares)
    assert "fused" in compiled.lower(Q[0], 0.2, 0.2).mlir()
    assert "fused" not in compiled.lower(Q[0], 0.2, 0.3).mlir()
    unfused = pulse.compile(_two_squares, passes=("verify", "fold_constants", "fold_phases"))
    assert "fused" not in unfused.lower(Q[0], 0.2, 0.2).mlir()


def test_passes_are_members_or_their_names():
    assert pulse.Pass.FOLD_PHASES == "fold_phases"
    as_members = pulse.compile(_bell, passes=pulse.compiler.DEFAULT_PASSES).lower(Q[0], Q[1])
    as_names = pulse.compile(_bell, passes=[str(p) for p in pulse.compiler.DEFAULT_PASSES])
    assert as_members.mlir() == as_names.lower(Q[0], Q[1]).mlir()
    assert as_members.mlir() == pulse.compile(_bell).lower(Q[0], Q[1]).mlir()


@pytest.mark.parametrize("passes", [(p,) for p in pulse.Pass] + [tuple(pulse.Pass)[::-1]])
def test_every_pass_runs_alone_and_in_any_order(passes):
    lowered = pulse.compile(_ramsey, passes=passes).lower(Q[0], 0.25)
    assert "pulse.drive" in lowered.mlir()
    assert lowered.schedule


def test_an_unknown_pass_names_the_closest_and_lists_them_all():
    with pytest.raises(
        ValueError, match="Unknown pass 'fuse_pulse'. Did you mean 'fuse_pulses'"
    ) as error:
        pulse.compile(_bell, passes=("verify", "fuse_pulse"))
    assert all(p.value in str(error.value) for p in pulse.Pass)
    with pytest.raises(ValueError, match="not the string 'verify'"):
        pulse.compile(_bell, passes="verify")


def test_compile_bad_schedule_raises():
    with pytest.raises(ValueError, match="Unknown schedule"):
        pulse.compile(_bell, schedule="bogus").lower(Q[0], Q[1])


def test_lower_binds_arguments_like_python():
    with pytest.raises(TypeError, match=r"_bell\(\) missing 2 required positional arguments"):
        _bell.lower()


@pulse.kernel
def _two_amplitudes(q0, amplitude: float, detuning: float):
    d0 = get_drive_line(q0)
    drive(d0, square(40, amplitude))
    shift_frequency(d0.tone, detuning)
    drive(d0, square(40, amplitude))


def test_bind_takes_one_real_number_per_parameter():
    generic = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic
    assert generic.parameters == ("amplitude", "detuning")
    expected = generic.bind([0.2, 2e6])
    values = ([0.2, 2e6], (0.2, 2_000_000), [_Real(0.2), 2e6], iter([0.2, 2e6]))
    bound = [generic.bind(v) for v in values]
    assert all(b.mlir() == expected.mlir() for b in bound)
    density = pulse.simulate(expected).density_matrix
    assert all((pulse.simulate(b).density_matrix == density).all() for b in bound)


class _Real:
    def __init__(self, x):
        self.x = x

    def __float__(self):
        return self.x


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            [0.2],
            r"got 1 value for 2 float parameters "
            r"\(amplitude, detuning\); missing the value of 'detuning'",
        ),
        ([], r"missing the value of 'amplitude', 'detuning'"),
        ([0.2, 1e6, 3.0], r"got 3 values .*; 1 more than it has"),
        (0.2, r"takes a list or tuple of values, .*got float"),
    ],
)
def test_bind_names_the_parameters_when_the_count_does_not_fit(values, message):
    generic = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic
    with pytest.raises(pulse.InvalidTypeError, match=message):
        generic.bind(values)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (
            [True, 1e6],
            pulse.InvalidTypeError,
            r"the value of 'amplitude' must be a real number, got bool True",
        ),
        (
            [0.2, "1e6"],
            pulse.InvalidTypeError,
            r"the value of 'detuning' must be a real number, got str '1e6'",
        ),
        ([0.2, 1j], pulse.InvalidTypeError, r"'detuning' must be a real number, got complex"),
        ([0.2, None], pulse.InvalidTypeError, r"'detuning' must be a real number, got NoneType"),
        ([math.nan, 1e6], pulse.InvalidValueError, r"argument 'amplitude' is nan; a float"),
        ([0.2, -math.inf], pulse.InvalidValueError, r"argument 'detuning' is -inf; a float"),
    ],
)
def test_a_bound_value_that_does_not_fit_fails_where_the_module_is_used(values, error, message):
    bound = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic.bind(values)
    for use in (bound.mlir, lambda: pulse.simulate(bound)):
        with pytest.raises(error, match=message):
            use()


def test_generic_is_the_module_every_call_rebinds_from():
    call = _two_amplitudes.lower(Q[0], 0.1, 1e6)
    generic = call.generic
    assert generic is not call
    assert generic.parameters == ("amplitude", "detuning")
    assert generic.generic is generic
    assert generic.bind([0.2, 2e6]).generic is generic

    @pulse.kernel
    def no_floats(q):
        drive(get_drive_line(q), square(40, 0.1))

    # A module that holds no values is its own: binding it takes none.
    plain = no_floats.lower(Q[0])
    assert plain.generic is plain
    assert plain.generic.bind([]).mlir() == plain.mlir()


def test_bind_of_a_module_with_its_values_takes_none():
    bound = _two_amplitudes.lower(Q[0], 0.1, 1e6)
    assert bound.parameters == ()
    assert bound.bind([]).arguments == ()
    with pytest.raises(pulse.InvalidTypeError, match=r"no float parameters; a module of one call"):
        bound.bind([0.2])


# ---------------------------------------------------------------------------
# PackedIRBuilder tests
# ---------------------------------------------------------------------------


def test_packed_ir_builder_basic():
    """PackedIRBuilder produces valid int64 buffer."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (dl, t) = b.emit("pulse.get_drive_line", (q,), ("drive_line", "tone"))
    (wf,) = b.emit(
        "pulse.gaussian", (), ("waveform",), {"duration": 40, "amplitude": 0.5, "sigma": 10.0}
    )
    b.emit("pulse.drive", (dl, wf, t), ("drive_line", "tone"))

    buf = b.get_buffer()
    assert isinstance(buf, array.array)
    assert buf.typecode == "q" and buf.itemsize == 8
    assert len(buf) > 0
    assert b.n_qubits == 1


def test_packed_ir_builder_readout():
    """PackedIRBuilder encodes readout ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (rl, t) = b.emit("pulse.get_readout_line", (q,), ("readout_line", "tone"))
    (wf,) = b.emit("pulse.square", (), ("waveform",), {"duration": 400, "amplitude": 0.1})
    b.emit("pulse.readout", (rl, wf, t), ("readout_line", "tone", "measurement"))

    buf = b.get_buffer()
    assert (buf[0] & 0xFF) == 1  # ALLOC_READOUT
    assert len(buf) > 8


def test_packed_ir_builder_sync():
    """PackedIRBuilder encodes variable-length sync ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9, 1: 5.1e9})
    (q0,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (q1,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (d0, t0) = b.emit("pulse.get_drive_line", (q0,), ("drive_line", "tone"))
    (d1, t1) = b.emit("pulse.get_drive_line", (q1,), ("drive_line", "tone"))
    b.emit("pulse.sync", (d0, d1), ("drive_line", "drive_line"))

    buf = b.get_buffer()
    assert len(buf) > 0


def test_compile_module_available():
    """compile() produces an in-memory PulseModule."""
    ck = _bell.lower(Q[0], Q[1])
    assert ck._module is not None
    assert ck.mlir() is not None
