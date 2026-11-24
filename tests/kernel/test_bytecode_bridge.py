# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the bytecode kernel capture backend.

Covers all pulse operations, control flow, linear-type rebinding,
constant folding, qudit allocation patterns, and error cases.
"""

import math

import numpy as np
import pytest
from cudaq_pulse.kernel.bytecode_bridge import compile_kernel_bytecode
from cudaq_pulse.kernel.decorator import kernel
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.ops import (
    cosine,
    drag,
    drive,
    gaussian,
    gaussian_square,
    get_drive_line,
    get_readout_line,
    readout,
    set_frequency,
    set_phase,
    shift_frequency,
    shift_phase,
    square,
    sync,
    tanh_ramp,
    wait,
)

from tests._handles import Q
from tests._ir import python_ir


def _helper_waveform():
    return gaussian(32, 0.2, 8.0)


def _helper_drive(line, waveform):
    drive(line, waveform)


class _Sweep:
    def __init__(self):
        self.amp = 0.0

    def step(self):
        self.amp += 0.25
        return self.amp


def _advances_an_iterator(q0):
    d0 = get_drive_line(q0)
    amps = iter([0.25, 0.5, 0.75])
    for _ in range(3):
        drive(d0, square(10, next(amps)))


def _advances_an_object(q0):
    d0 = get_drive_line(q0)
    sweep = _Sweep()
    for _ in range(3):
        drive(d0, square(10, sweep.step()))


def _changes_an_array(q0):
    d0 = get_drive_line(q0)
    amp = np.zeros(1)
    for _ in range(3):
        amp += 0.25
        drive(d0, square(10, float(amp[0])))


# ── Basic compilation ────────────────────────────────────────────────


class TestBasicCompilation:
    def test_empty_kernel(self):

        def empty(q0):
            pass

        ir = compile_kernel_bytecode(empty)(Q[0])
        assert ir.name == "empty"
        assert len(ir.ops) == 1  # just pulse.qudit_arg

    def test_kernel_name_preserved(self):

        def my_special_name(q0):
            pass

        ir = compile_kernel_bytecode(my_special_name)(Q[0])
        assert ir.name == "my_special_name"

    def test_drive_kernel(self):

        def drive_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(drive_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds == [
            "pulse.qudit_arg",
            "pulse.get_drive_line",
            "pulse.gaussian",
            "pulse.drive",
        ]

    def test_multiple_args(self):

        def two_qudit(q0, q1):
            _d0 = get_drive_line(q0)
            _d1 = get_drive_line(q1)

        ir = compile_kernel_bytecode(two_qudit)(Q[0], Q[1])
        arg_ops = [op for op in ir.ops if op.kind == "pulse.qudit_arg"]
        assert len(arg_ops) == 2

    def test_pulse_helpers_are_inlined(self):

        def helper_kernel(q0):
            d0 = get_drive_line(q0)
            wf = _helper_waveform()
            _helper_drive(d0, wf)

        ir = compile_kernel_bytecode(helper_kernel)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds == [
            "pulse.qudit_arg",
            "pulse.get_drive_line",
            "pulse.gaussian",
            "pulse.drive",
        ]

    def test_wrong_arg_count_raises(self):

        def one_arg(q0):
            pass

        emitter = compile_kernel_bytecode(one_arg)
        with pytest.raises(CompilationError, match="expected 1 args, got 0"):
            emitter()

    def test_kwargs_rejected(self):

        def one_arg(q0):
            pass

        emitter = compile_kernel_bytecode(one_arg)
        with pytest.raises(CompilationError, match="keyword arguments"):
            emitter(q0=Q[0])


# ── Waveform creation ────────────────────────────────────────────────


class TestWaveforms:
    def test_gaussian(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        wf_op = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert wf_op.attrs == {"duration": 40, "amplitude": 0.3, "sigma": 10.0}

    def test_square(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = square(20, 0.5)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        wf_op = next(op for op in ir.ops if op.kind == "pulse.square")
        assert wf_op.attrs == {"duration": 20, "amplitude": 0.5}

    def test_drag(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = drag(40, 0.435, 5.0, 0.75)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        wf_op = next(op for op in ir.ops if op.kind == "pulse.drag")
        assert wf_op.attrs == {
            "duration": 40,
            "amplitude": 0.435,
            "sigma": 5.0,
            "beta": 0.75,
        }

    def test_cosine(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = cosine(100, 0.5)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        assert any(op.kind == "pulse.cosine" for op in ir.ops)

    def test_gaussian_square(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian_square(200, 0.3, 10.0, 150)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        wf_op = next(op for op in ir.ops if op.kind == "pulse.gaussian_square")
        assert wf_op.attrs["width"] == 150.0

    def test_tanh_ramp(self):

        def wf_test(q0):
            d0 = get_drive_line(q0)
            wf = tanh_ramp(50, 0.4, 5.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(wf_test)(Q[0])
        assert any(op.kind == "pulse.tanh_ramp" for op in ir.ops)


# ── Readout and measurement ──────────────────────────────────────────


class TestReadout:
    def test_readout_produces_measurement(self):

        def ro_test(q0):
            r0 = get_readout_line(q0)
            wf = square(600, 0.1)
            readout(r0, wf)

        ir = compile_kernel_bytecode(ro_test)(Q[0])
        ro_ops = [op for op in ir.ops if op.kind == "pulse.readout"]
        assert len(ro_ops) == 1
        assert len(ro_ops[0].results) == 3
        assert ro_ops[0].results[2].vtype == "measurement"


# ── Tone manipulation ops ────────────────────────────────────────────


class TestToneOps:
    def test_shift_phase(self):

        def phase_test(q0):
            d0 = get_drive_line(q0)
            shift_phase(d0.tone, 1.5707)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(phase_test)(Q[0])
        sp = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
        assert abs(sp.attrs["phase_rad"] - 1.5707) < 1e-10

    def test_set_phase(self):

        def phase_test(q0):
            d0 = get_drive_line(q0)
            set_phase(d0.tone, 0.0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(phase_test)(Q[0])
        assert any(op.kind == "pulse.set_phase" for op in ir.ops)

    def test_shift_frequency(self):

        def freq_test(q0):
            d0 = get_drive_line(q0)
            shift_frequency(d0.tone, 1e6)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(freq_test)(Q[0])
        sf = next(op for op in ir.ops if op.kind == "pulse.shift_frequency")
        assert sf.attrs["freq_hz"] == 1e6

    def test_set_frequency(self):

        def freq_test(q0):
            d0 = get_drive_line(q0)
            set_frequency(d0.tone, 5.1e9)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(freq_test)(Q[0])
        assert any(op.kind == "pulse.set_frequency" for op in ir.ops)


# ── Wait and sync ────────────────────────────────────────────────────


class TestWaitSync:
    def test_wait(self):

        def wait_test(q0):
            d0 = get_drive_line(q0)
            wait(d0, 100)

        ir = compile_kernel_bytecode(wait_test)(Q[0])
        w = next(op for op in ir.ops if op.kind == "pulse.wait")
        assert w.attrs["duration"] == 100

    def test_sync_variadic(self):

        def sync_test(q0, q1):
            d0 = get_drive_line(q0)
            d1 = get_drive_line(q1)
            sync(d0, d1)

        ir = compile_kernel_bytecode(sync_test)(Q[0], Q[1])
        s = next(op for op in ir.ops if op.kind == "pulse.sync")
        assert len(s.operands) == 2
        assert all(o.vtype == "drive_line" for o in s.operands)


# ── Linear-type rebinding ────────────────────────────────────────────


class TestLinearRebinding:
    def test_drive_rebinds_line_and_tone(self):

        def rebind_test(q0):
            d0 = get_drive_line(q0)
            wf1 = gaussian(40, 0.3, 10.0)
            wf2 = gaussian(40, 0.5, 10.0)
            drive(d0, wf1)
            drive(d0, wf2)

        ir = compile_kernel_bytecode(rebind_test)(Q[0])
        drives = [op for op in ir.ops if op.kind == "pulse.drive"]
        assert len(drives) == 2
        assert drives[0].operands[0] is not drives[1].operands[0]
        assert drives[0].operands[2] is not drives[1].operands[2]

    def test_wait_rebinds_line(self):

        def rebind_test(q0):
            d0 = get_drive_line(q0)
            wait(d0, 50)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(rebind_test)(Q[0])
        w = next(op for op in ir.ops if op.kind == "pulse.wait")
        d = next(op for op in ir.ops if op.kind == "pulse.drive")
        assert w.operands[0] is not d.operands[0]

    def test_shift_phase_rebinds_tone(self):

        def phase_test(q0):
            d0 = get_drive_line(q0)
            shift_phase(d0.tone, 0.5)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(phase_test)(Q[0])
        sp = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
        d = next(op for op in ir.ops if op.kind == "pulse.drive")
        assert sp.results[0] is d.operands[2]


# ── For loops ────────────────────────────────────────────────────────


def _rolled(fn):
    """*fn* traced with every loop that can stay rolled kept rolled."""
    return compile_kernel_bytecode(fn, unroll_threshold=0)


class TestForLoop:
    def test_for_range_emits_scf(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            for i in range(5):  # noqa: B007
                drive(d0, wf)

        ir = _rolled(loop_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds.count("pulse.drive") == 1
        assert "scf.for_end" in kinds
        scf_for = next(op for op in ir.ops if op.kind == "scf.for")
        assert scf_for.attrs == {"lb": 0, "ub": 5, "step": 1, "var": "i"}

    def test_for_loop_with_yield(self):

        def loop_yield(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            for i in range(3):  # noqa: B007
                drive(d0, wf)

        ir = _rolled(loop_yield)(Q[0])
        (yield_op,) = [op for op in ir.ops if op.kind == "scf.yield"]
        end = ir.ops[-1]
        assert end.kind == "scf.for_end"
        assert end.attrs == {"iter_args": ["d0", "d0.tone"]}
        assert [v.vtype for v in yield_op.operands] == ["drive_line", "tone"]

    def test_loop_var_read_unrolls(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            for i in range(2, 9, 3):
                shift_phase(d0.tone, float(i))

        ir = _rolled(loop_test)(Q[0])
        phases = [op.attrs["phase_rad"] for op in ir.ops if op.kind == "pulse.shift_phase"]
        assert phases == [2.0, 5.0, 8.0]
        assert not any(op.kind.startswith("scf.") for op in ir.ops)

    def test_python_state_carried_across_iterations_unrolls(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            amp = 0.1
            for _ in range(3):
                drive(d0, square(10, amp))
                amp = amp * 2

        ir = _rolled(loop_test)(Q[0])
        amps = [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"]
        assert amps == [0.1, 0.2, 0.4]
        assert not any(op.kind.startswith("scf.") for op in ir.ops)

    @pytest.mark.parametrize(
        "loop_test", [_advances_an_iterator, _advances_an_object, _changes_an_array]
    )
    def test_python_state_changed_by_a_call_unrolls(self, loop_test):
        ir = _rolled(loop_test)(Q[0])
        amps = [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"]
        assert amps == [0.25, 0.5, 0.75]
        assert not any(op.kind.startswith("scf.") for op in ir.ops)

    def test_a_body_that_does_not_roll_starts_from_the_state_before_the_loop(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            amps = iter([0.25, 0.5, 0.75])
            played = []
            for _ in range(3):
                played.append(next(amps))
            drive(d0, square(10, played[0]))

        ir = _rolled(loop_test)(Q[0])
        amps = [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"]
        assert amps == [0.25]

    def test_calls_that_change_nothing_stay_rolled(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            amps = [0.25, 0.5]
            for _ in range(3):
                pair = dict(zip(("a", "b"), iter(amps), strict=True))
                drive(d0, square(10, max(pair.values()) * float(np.sqrt(math.cos(0.0)))))

        ir = _rolled(loop_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds.count("scf.for") == 1
        assert [op.attrs["amplitude"] for op in ir.ops if op.kind == "pulse.square"] == [0.5]

    def test_list_mutation_in_loop_unrolls(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            waves = []
            for _ in range(3):
                waves.append(square(10, 0.1))
            for w in range(len(waves)):
                drive(d0, waves[w])

        ir = _rolled(loop_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds.count("pulse.square") == 3
        assert kinds.count("pulse.drive") == 3
        assert "scf.for" not in kinds

    def test_break_in_loop_unrolls(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            for _ in range(4):
                drive(d0, square(10, 0.1))
                break

        ir = _rolled(loop_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds.count("pulse.drive") == 1
        assert "scf.for" not in kinds

    def test_nested_rolled_loops(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            x = square(10, 0.1)
            for _ in range(3):
                for _ in range(4):
                    drive(d0, x)

        ir = _rolled(loop_test)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert kinds.count("scf.for") == 2
        assert kinds.count("pulse.drive") == 1
        assert [op.attrs["ub"] for op in ir.ops if op.kind == "scf.for"] == [3, 4]

    def test_conditional_chain_inside_unrolled_loop(self):
        """`and` chains at the end of a loop body compile to a backward jump
        on 3.12+; every taken branch must still be traced."""

        def loop_test(q0):
            d0 = get_drive_line(q0)
            x = square(10, 0.1)
            for r in range(4):
                if 0 < r and r < 3:
                    drive(d0, x)

        ir = _rolled(loop_test)(Q[0])
        assert [op.kind for op in ir.ops].count("pulse.drive") == 2

    def test_while_loop_rejected(self):

        def loop_test(q0):
            d0 = get_drive_line(q0)
            n = 2
            while n:
                wait(d0, 10)
                n -= 1

        with pytest.raises(CompilationError, match="while loops"):
            _rolled(loop_test)(Q[0])


# ── Qudit allocation patterns ────────────────────────────────────────


class TestQuditAlloc:
    def test_attribute_style_op(self):
        import cudaq_pulse

        def attr_alloc(q):
            _d0 = cudaq_pulse.ops.get_drive_line(q)

        ir = compile_kernel_bytecode(attr_alloc)(Q[0])
        kinds = [op.kind for op in ir.ops]
        assert "pulse.qudit_arg" in kinds
        assert "pulse.get_drive_line" in kinds


# ── Constant folding / arithmetic ────────────────────────────────────


class TestConstantFolding:
    def test_binary_add(self):

        def arith_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(20 + 20, 0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(arith_test)(Q[0])
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert wf.attrs["duration"] == 40

    def test_binary_mul(self):

        def arith_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3 * 2, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(arith_test)(Q[0])
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["amplitude"] - 0.6) < 1e-10

    def test_binary_div(self):

        def arith_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 20.0 / 2)
            drive(d0, wf)

        ir = compile_kernel_bytecode(arith_test)(Q[0])
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["sigma"] - 10.0) < 1e-10

    def test_unary_negation(self):

        def neg_test(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, -0.3, 10.0)
            drive(d0, wf)

        ir = compile_kernel_bytecode(neg_test)(Q[0])
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["amplitude"] - (-0.3)) < 1e-10


# ── Source-less compilation ──────────────────────────────────────────


class TestSourceless:
    def test_exec_compiled_function(self):
        code = compile(
            "def f(q0):\n d0 = get_drive_line(q0)\n",
            "<test>",
            "exec",
        )
        ns = {"get_drive_line": get_drive_line}
        exec(code, ns)
        fn = ns["f"]

        ir = compile_kernel_bytecode(fn)(Q[0])
        assert any(op.kind == "pulse.get_drive_line" for op in ir.ops)


# ── Error cases ──────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_op_raises(self):
        """Calling a function that isn't a pulse op or known builtin raises."""

        def bad_kernel(q0):
            d0 = get_drive_line(q0)
            unknown_function_xyz(d0)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

        emitter = compile_kernel_bytecode(bad_kernel)
        with pytest.raises(CompilationError):
            emitter(Q[0])

    def test_for_over_a_non_iterable_names_its_type(self):
        def bad_loop(q0):
            d0 = get_drive_line(q0)
            for i in 3.5:  # noqa: B007
                wait(d0, 10)

        emitter = compile_kernel_bytecode(bad_loop)
        with pytest.raises(CompilationError, match="cannot iterate over float"):
            emitter(Q[0])


# ── Decorator integration ────────────────────────────────────────────


class TestDecoratorIntegration:
    def test_kernel_decorator_uses_bytecode(self):

        @kernel
        def my_kernel(q0):
            d0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)

        ir = python_ir(my_kernel, Q[0])
        assert ir is not None
        kinds = [op.kind for op in ir.ops]
        assert "pulse.drive" in kinds

    def test_kernel_caching(self):

        @kernel
        def cached(q0):
            pass

        cached.lower(Q[0])
        cached.lower(Q[0])
        assert cached.cache_info().compiled == 1
