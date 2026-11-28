# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every native binding returns its result to Python."""

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native

from tests._handles import Q


@pulse.kernel
def _x(q, amplitude: float):
    d = get_drive_line(q)
    drive(d, square(40, amplitude))


def _public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def test_module_builder_exposes_only_the_packed_decoder_and_the_loaders():
    assert _public(native.PulseModuleBuilder) == {
        "build_from_packed",
        "load",
        "load_callables",
        "load_function",
        "load_link_names",
        "load_submissions",
    }


def test_pulse_module_methods():
    # The cached specialization keeps ``amplitude`` a function argument.
    compiled = _x._dispatch.lookup((Q[0], 0.0)).lowered
    module = compiled._module
    assert module.param_names() == ["amplitude"]
    assert module.run_passes(["pulse-verify"]) is None
    assert "pulse.drive" in module.print()
    concrete = module.specialize([0.1], [], "alap")
    assert "%arg0" not in concrete.print()
    assert concrete.schedule("asap") is None
    assert "start_cycles = 0" in concrete.print()
    assert "llvm.func" in concrete.run_full_lowering()


def test_pipeline_methods_return_results():
    text = _x.lower(Q[0], 0.1).mlir()
    pipeline = native.MLIRPipeline()
    assert "pulse.drive" in pipeline.parse_to_module(text).print()
    qop = pipeline.run_pulse_to_qop(text)
    assert "qop." in qop
    assert "cudm." in pipeline.run_qop_to_cudm(qop)


def test_pipeline_exposes_only_the_live_entry_points():
    assert _public(native.MLIRPipeline) == {
        "jit_llvm_dialect",
        "jit_pulse",
        "parse_to_module",
        "run_pulse_to_qop",
        "run_qop_to_cudm",
    }


def test_pass_failures_carry_the_diagnostics():
    text = (
        "func.func @main() {\n"
        "  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit\n"
        "  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)\n"
        "  %c = arith.constant 40 : i64\n"
        "  %a = arith.constant 0.1 : f64\n"
        "  %w = pulse.square %c, %a, %a : i64, f64, f64 -> !pulse.waveform\n"
        "  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone"
        " -> !pulse.drive_line, !pulse.tone\n"
        "  %d2, %t2 = pulse.drive %d, %w, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone"
        " -> !pulse.drive_line, !pulse.tone\n"
        "  return\n"
        "}\n"
    )
    module = native.MLIRPipeline().parse_to_module(text)
    with pytest.raises(RuntimeError, match="linear pulse value is consumed more than once"):
        module.run_passes(["pulse-verify"])
    with pytest.raises(RuntimeError, match="expected"):
        native.MLIRPipeline().parse_to_module("func.func @main() {")


def test_a_pass_by_name_takes_its_options():
    from tests.workloads import w4_syndrome

    text = pulse.compile(w4_syndrome.build).lower(*Q[: w4_syndrome.NUM_QUDITS]).mlir()
    pipeline = native.MLIRPipeline()
    scheduled = pipeline.parse_to_module(text)
    scheduled.schedule("alap_rcp", 1, 1, 3, 2)
    by_name = pipeline.parse_to_module(text)
    options = "max-drives=1 max-readouts=1 readout-latency=3 switch-penalty=2 alap=true"
    by_name.run_passes([f"pulse-schedule-rcp{{{options}}}", "pulse-frame-changes"])
    assert by_name.print() == scheduled.print() != pipeline.parse_to_module(text).print()


def test_compiled_module_is_named_after_the_kernel():
    compiled = _x.lower(Q[0], 0.1)
    assert compiled.mlir().startswith("module @_x")
    assert (
        "pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.000000e+09]>"
        in compiled.mlir()
    )


@pulse.kernel
def _measured(q):
    r = get_readout_line(q)
    return readout(r, square(400, 0.05))


def test_the_exports_refuse_a_readout_mode_they_do_not_support_yet():
    text = pulse.compile(_measured).lower(Q[0]).mlir()
    assert ", iq <{" in text
    module = native.MLIRPipeline().parse_to_module(text.replace(", iq <{", ", trace <{"))
    message = "readout mode trace is not supported by the exports yet; only iq is"
    with pytest.raises(NotImplementedError, match=message):
        module.events()
    with pytest.raises(NotImplementedError, match=message):
        module.controller_events([], [])


def _listed(module, limit):
    """How many events ``event_tree(limit)`` lists or stands for."""
    from cudaq_pulse._loops import Schedule

    events = module.event_tree(limit)
    return len(events) if type(events) is list else len(Schedule(events[0], tuple))


def test_the_event_count_is_what_the_event_tree_lists_or_stands_for():
    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(16, 0.1))
        for _ in range(7):
            for _ in range(5):
                drive(d, square(16, 0.1))
            if readout(r, square(400, 0.05)):
                drive(d, square(16, 0.2))
            else:
                wait(d, 16)
        for _ in range(0):
            drive(d, square(16, 0.3))

    for options in ({"unroll_threshold": 0}, {}):
        module = pulse.compile(nested, **options).lower(Q[0])._module
        assert len(module.events()) == 1 + 7 * (5 + 3)
        for limit in (-1, 0, 56, 57, 10_000):
            assert module.event_count(limit) == _listed(module, limit) == 57


def test_the_event_count_refuses_the_readout_the_listed_events_refuse():
    @pulse.kernel
    def rounds(q):
        r = get_readout_line(q)
        for _ in range(3):
            readout(r, square(400, 0.05))

    text = pulse.compile(rounds, unroll_threshold=0).lower(Q[0]).mlir()
    assert text.count(", iq <{") == 1 and "scf.for" in text
    module = native.MLIRPipeline().parse_to_module(text.replace(", iq <{", ", trace <{"))
    message = "readout mode trace is not supported by the exports yet; only iq is"
    # event_tree(limit) is events() up to limit events, and refuses with it.
    for limit in (3, 10_000):
        with pytest.raises(NotImplementedError, match=message):
            module.event_tree(limit)
        with pytest.raises(NotImplementedError, match=message):
            module.event_count(limit)
    for limit in (-1, 0, 2):
        assert module.event_count(limit) == _listed(module, limit) == 3
    # A loop of no iteration lists nothing, so nothing of it is refused.
    none = native.MLIRPipeline().parse_to_module(
        text.replace(", iq <{", ", trace <{").replace("constant 3 : index", "constant 0 : index")
    )
    assert none.events() == [] and none.event_count(10_000) == 0


def test_an_event_count_past_the_length_of_a_sequence_is_none():
    line = "!pulse.drive_line, !pulse.tone"

    def squared(trips):
        """A drive in two nested loops of *trips* iterations each."""
        return f"""func.func @main() {{
  %q = pulse.qudit_alloc {{qudit = 0 : i64}} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({line})
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %trips = arith.constant {trips} : index
  %n = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %w = pulse.square %n, %amp, %amp : i64, f64, f64 -> !pulse.waveform
  %out:2 = scf.for %i = %c0 to %trips step %c1 iter_args(%l = %d, %lt = %t) -> ({line}) {{
    %in:2 = scf.for %j = %c0 to %trips step %c1 iter_args(%m = %l, %mt = %lt) -> ({line}) {{
      %m1, %mt1 = pulse.drive %m, %w, %mt : !pulse.drive_line, !pulse.waveform, !pulse.tone
          -> {line}
      scf.yield %m1, %mt1 : {line}
    }}
    scf.yield %in#0, %in#1 : {line}
  }}
  return
}}
"""

    parse = native.MLIRPipeline().parse_to_module
    module = parse(squared(2**31))
    assert module.event_count(10_000) == _listed(module, 10_000) == 2**62
    # len() of the sequence that stands for more events than that fails.
    for trips in (2**32, 2**33):
        module = parse(squared(trips))
        assert module.event_count(10_000) is None
        with pytest.raises(OverflowError):
            _listed(module, 10_000)


# ── The decoder creates constants in the order of the operands ───────


def _decoded(kernel) -> list[str]:
    """The lines of *kernel*'s IR as the decoder wrote it: with no passes,
    nothing has moved or merged a constant."""
    text = pulse.compile(kernel, passes=()).lower(Q[0]).mlir()
    return [line.strip() for line in text.splitlines()]


def test_a_waveforms_literals_become_constants_in_the_order_of_its_operands():
    """The constants are the same, in the same order, whichever C++ compiler
    built the package. The decoder once created them in the order that
    compiler evaluated a call's arguments: right to left in the Linux x86_64
    build, left to right in the others."""

    @pulse.kernel
    def six(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.11, 10.0))
        drive(d, square(8, [0.21, 0.22]))
        drive(d, drag(48, 0.31, 12.0, 0.33))
        drive(d, cosine(56, 0.41))
        drive(d, tanh_ramp(64, 0.51, 4.0))
        drive(d, gaussian_square(72, 0.61, 6.0, 32))

    lines = _decoded(six)
    made = " = arith.constant "
    constants = [line.split(made) for line in lines if made in line]
    expected = [
        ("40 : i64", "1.100000e-01 : f64", "1.000000e+01 : f64"),
        ("8 : i64", "2.100000e-01 : f64", "2.200000e-01 : f64"),
        ("48 : i64", "3.100000e-01 : f64", "1.200000e+01 : f64", "3.300000e-01 : f64"),
        ("56 : i64", "4.100000e-01 : f64"),
        ("64 : i64", "5.100000e-01 : f64", "4.000000e+00 : f64"),
        # The frontend passes gaussian_square's edge length, (72 - 32) / 2.
        ("72 : i64", "6.100000e-01 : f64", "6.000000e+00 : f64", "20 : i64"),
    ]
    assert [value for _, value in constants] == [value for row in expected for value in row]
    # Each waveform takes its own, in that order. Its operands are read
    # through the constants they name, not by the names: the number in a
    # %cst name counts every name before it that was made unique, the
    # drives' results among them.
    value = dict(constants)
    ends = " -> !pulse.waveform"
    waveforms = [line.split(" : ")[0].split(" ", 3)[2:] for line in lines if line.endswith(ends)]
    assert [op for op, _ in waveforms] == [
        "pulse.gaussian",
        "pulse.square",
        "pulse.drag",
        "pulse.cosine",
        "pulse.tanh_ramp",
        "pulse.gaussian_square",
    ]
    assert [tuple(value[name] for name in names.split(", ")) for _, names in waveforms] == expected


def test_a_loops_bounds_become_constants_in_the_order_of_its_operands():
    @pulse.kernel(unroll_threshold=0)
    def train(q):
        d = get_drive_line(q)
        for _ in range(3):
            drive(d, square(8, 0.1))

    lines = _decoded(train)
    assert [line for line in lines if line.endswith(" : index")] == [
        "%c0 = arith.constant 0 : index",
        "%c3 = arith.constant 3 : index",
        "%c1 = arith.constant 1 : index",
    ]
    assert sum("scf.for %arg0 = %c0 to %c3 step %c1 " in line for line in lines) == 1
