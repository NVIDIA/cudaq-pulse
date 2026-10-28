# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Regression tests for semantic correctness at public API boundaries."""

from __future__ import annotations

import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse._native._cudaq_pulse_native import PulseModuleBuilder
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q


def _drive_timings(mlir: str) -> list[tuple[int, int]]:
    timings = []
    for match in re.finditer(r"pulse\.drive.*?\{([^}]*)\}", mlir):
        attrs = match.group(1)
        start = re.search(r"start_cycles\s*=\s*(-?\d+)", attrs)
        duration = re.search(r"duration_cycles\s*=\s*(-?\d+)", attrs)
        assert start and duration
        timings.append((int(start.group(1)), int(duration.group(1))))
    return timings


def test_constant_loop_is_retained_and_scheduled_per_iteration():

    @pulse.kernel
    def kernel(q):
        line, tone = get_drive_line(q)
        waveform = square(10, 0.2)
        for _ in range(5):
            drive(line, waveform, tone)

    compiled = pulse.compile(kernel, schedule="asap", passes=()).lower(Q[0])
    assert compiled.mlir().count("= pulse.drive ") == 1
    assert "scf.for" in compiled.mlir()
    assert [(e.start_cycles, e.duration_cycles) for e in compiled.schedule] == [
        (0, 10),
        (10, 10),
        (20, 10),
        (30, 10),
        (40, 10),
    ]


def test_range_start_stop_step_values_are_preserved():

    @pulse.kernel
    def kernel(q):
        _, tone = get_drive_line(q)
        for index in range(2, 9, 3):
            shift_phase(tone, float(index))

    compiled = pulse.compile(kernel, passes=()).lower(Q[0])
    assert compiled.mlir().count("pulse.shift_phase") == 3
    for value in (2.0, 5.0, 8.0):
        assert f"{value:.6e}" in compiled.mlir()


@pytest.mark.parametrize("symbolic", [False, True])
def test_tones_kept_in_a_list_and_retuned_stay_in_a_rolled_loop(symbolic):

    @pulse.kernel
    def constant(qs):
        lines = [get_drive_line(q) for q in qs]
        for (d, t), f in zip(lines, (5.0e9, 5.1e9)):
            set_frequency(t, f)
        for _ in range(20):
            for d, t in lines:
                drive(d, square(20, 0.3), t)

    @pulse.kernel
    def runtime(qs, f0: float, f1: float):
        lines = [get_drive_line(q) for q in qs]
        for (d, t), f in zip(lines, (f0, f1)):
            set_frequency(t, f)
        for _ in range(20):
            for d, t in lines:
                drive(d, square(20, 0.3), t)

    args = (5.0e9, 5.1e9) if symbolic else ()
    compiled = pulse.compile(runtime if symbolic else constant).lower(Q[0:2], *args)
    text = compiled.mlir()
    assert text.count("scf.for") == 1
    assert text.count("= pulse.drive ") == 2
    assert text.count("pulse.set_frequency") == 2
    drives = [e for e in compiled.schedule if e.kind == "drive"]
    assert len(drives) == 40
    assert max(e.start_cycles + e.duration_cycles for e in drives) == 20 * 20


def test_compile_keeps_a_measurement_dependent_branch():

    @pulse.kernel
    def kernel(q):
        line, tone = get_readout_line(q)
        result = readout(line, square(10, 0.2), tone)
        if result:
            wait(line, 10)

    compiled = kernel.lower(Q[0])
    assert "pulse.is_set" in compiled.mlir() and "scf.if" in compiled.mlir()
    compiled._module.run_passes(["pulse-verify"])


def test_documented_waveforms_and_algebra_lower_to_typed_ops():

    @pulse.kernel
    def kernel(q):
        line, tone = get_drive_line(q)
        left = cosine(4, 0.2)
        right = custom_samples([0.1, 0.2, 0.3, 0.4])
        combined = wf_add(left, right)
        scaled = wf_scale(combined, 2.0)
        negated = wf_neg(scaled)
        drive(line, negated, tone)

    compiled = pulse.compile(kernel, passes=(), schedule="asap").lower(Q[0])
    assert "pulse.cosine" in compiled.mlir()
    assert "pulse.custom_samples" in compiled.mlir()
    assert "pulse.add" in compiled.mlir()
    assert "pulse.scale" in compiled.mlir()
    assert "pulse.neg" in compiled.mlir()
    assert _drive_timings(compiled.mlir()) == [(0, 4)]


def test_custom_waveform_is_sampled_not_named():

    @pulse.kernel
    def kernel(q):
        line, tone = get_drive_line(q)
        drive(line, custom(8, abs), tone)

    mlir = pulse.compile(kernel, passes=()).lower(Q[0]).mlir()
    assert "pulse.custom_samples" in mlir and "@abs" not in mlir


def test_concrete_numeric_arguments_are_not_made_symbolic():

    @pulse.kernel
    def kernel(q, duration, amplitude):
        line, tone = get_drive_line(q)
        drive(line, gaussian(duration, amplitude, 5.0), tone)

    compiled = kernel.lower(Q[0], 20, 0.4)
    assert compiled.parameters == ()
    assert _drive_timings(compiled.mlir()) == [(0, 20)]


def test_an_unannotated_parameter_specializes_the_kernel():

    @pulse.kernel
    def kernel(q, tau):
        line, _ = get_drive_line(q)
        wait(line, tau)

    compiled = pulse.compile(kernel)
    lowered = compiled.lower(Q[0], tau=17)
    assert "%arg0" not in lowered.mlir()
    assert lowered.mlir().count("duration_cycles = 17") == 1
    compiled.lower(Q[0], tau=18)
    assert compiled.cache_info().compiled == 2


def test_a_float_that_sets_a_duration_is_specialized_on():

    @pulse.kernel
    def kernel(q, sigma: float, delay: int):
        line, tone = get_drive_line(q)
        waveform = gaussian(int(4 * sigma), 0.2, sigma)
        drive(line, waveform, tone)
        wait(line, delay * 2 + 1)

    # sigma sets a duration, so the kernel is specialized on its value.
    specialized = pulse.compile(kernel, schedule="asap").lower(Q[0], sigma=5.0, delay=7)
    assert kernel.static_parameters == ("q", "sigma", "delay")
    assert _drive_timings(specialized.mlir()) == [(0, 20)]
    assert "duration_cycles = 15" in specialized.mlir()


def test_unknown_pass_is_rejected():

    @pulse.kernel
    def kernel():
        pass

    with pytest.raises(ValueError, match="Unknown pass 'typo'"):
        pulse.compile(kernel, passes=("typo",)).lower()


def test_asap_and_alap_have_distinct_correct_placement():

    @pulse.kernel
    def kernel(q0, q1):
        line0, tone0 = get_drive_line(q0)
        line1, tone1 = get_drive_line(q1)
        drive(line0, square(40, 0.2), tone0)
        drive(line1, square(100, 0.2), tone1)

    args = [Q[0], Q[1]]
    asap = pulse.compile(kernel, schedule="asap", passes=()).lower(*args)
    alap = pulse.compile(kernel, schedule="alap", passes=()).lower(*args)
    assert _drive_timings(asap.mlir()) == [(0, 40), (0, 100)]
    assert _drive_timings(alap.mlir()) == [(60, 40), (0, 100)]


def test_compile_uses_resource_machine_for_overlapping_intervals():

    @pulse.kernel
    def kernel(q0, q1):
        line0, tone0 = get_drive_line(q0)
        line1, tone1 = get_drive_line(q1)
        drive(line0, square(100, 0.2), tone0)
        drive(line1, square(40, 0.2), tone1)

    compiled = pulse.compile(
        kernel, schedule="rcp", passes=(), machine=MachineModel(max_concurrent_drives=1)
    ).lower(Q[0], Q[1])
    assert _drive_timings(compiled.mlir()) == [(0, 100), (100, 40)]


def test_scheduler_tracks_line_lineage_and_resource_intervals():
    @pulse.kernel
    def lineage(q0):
        line, tone = get_drive_line(q0)
        for duration in range(10, 40, 10):
            drive(line, square(duration, 0.2), tone)

    compiled = pulse.compile(lineage, passes=(), schedule="asap").lower(Q[0])
    assert [e.start_cycles for e in compiled.schedule] == [0, 10, 30]

    @pulse.kernel
    def resources(qs):
        for i in range(3):
            line, tone = get_drive_line(qs[i])
            drive(line, square(100, 0.2), tone)

    compiled = pulse.compile(
        resources, passes=(), schedule="rcp", machine=MachineModel(max_concurrent_drives=2)
    ).lower(Q[0:3])
    assert [e.start_cycles for e in compiled.schedule] == [0, 0, 100]


def test_alap_delays_independent_shorter_operation():
    @pulse.kernel
    def k(q0, q1):
        line0, tone0 = get_drive_line(q0)
        line1, tone1 = get_drive_line(q1)
        drive(line0, square(40, 0.2), tone0)
        drive(line1, square(100, 0.2), tone1)

    compiled = pulse.compile(k, passes=()).lower(Q[0], Q[1])
    assert [(e.line, e.start_cycles) for e in compiled.schedule] == [(0, 60), (1, 0)]


def test_packed_decoder_rejects_truncated_and_unknown_records():
    builder = PulseModuleBuilder()
    frequencies = np.array([], dtype=np.float64)
    with pytest.raises(Exception, match="truncated record"):
        builder.build_from_packed(np.array([3 | (4 << 8)], dtype=np.int64), 2.0, 0, frequencies)
    with pytest.raises(Exception, match="unknown opcode"):
        builder.build_from_packed(np.array([255], dtype=np.int64), 2.0, 0, frequencies)


def test_a_handle_binds_its_physical_target_index():

    @pulse.kernel
    def kernel(qubit):
        line, _tone = get_drive_line(qubit)
        wait(line, 10)

    compiled = pulse.compile(kernel, passes=()).lower(Q[4])
    assert "qubit = 4 : i64" in compiled.mlir()
    # The test target puts qudit i at 5 GHz + i * 100 MHz.
    assert "frequency_hz = 5.400000e+09 : f64" in compiled.mlir()


def test_virtual_time_arguments_are_not_silently_truncated():

    @pulse.kernel
    def kernel(qubit):
        line, tone = get_drive_line(qubit)
        drive(line, square(10.5, 0.2), tone)

    with pytest.raises(CompilationError, match="must be an integer"):
        kernel.lower(Q[0])


def test_alap_rcp_never_schedules_before_time_zero():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "compiler" / "compile_time.py"
    spec = importlib.util.spec_from_file_location("compile_time", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    workloads = {name: (k, n, f) for name, k, n, f in module.WORKLOADS}
    kernel, n, freq = workloads["qaoa4"]
    compiled = pulse.compile(
        kernel,
        schedule="alap_rcp",
        machine=MachineModel(max_concurrent_drives=2, max_concurrent_readouts=1),
    ).lower(*[Q[i] for i in range(n)])
    assert min(e.start_cycles for e in compiled.schedule) >= 0
    compiled._module.run_passes(["pulse-verify"])
