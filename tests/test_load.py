# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.load``: a scheduled function another compiler built."""

import pytest

import cudaq_pulse as pulse

MODULE = """
module {
  func.func private @other(%q: !pulse.qudit) attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
    return
  }
  func.func private @round(%a: !pulse.qudit, %b: !pulse.qudit) -> i1 attributes {made.by = "elsewhere", pulse.schedule_state = #pulse.schedule_state<timed>} {
    %da, %ta = pulse.get_drive_line %a {made.channel = 3 : i64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %rb, %rt = pulse.get_readout_line %b : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %da1, %ta1 = pulse.drive %da, %w, %ta <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %db1, %tb1 = pulse.drive %db, %w, %tb <{duration_cycles = 16 : i64, start_cycles = 16 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %m = arith.constant 64 : i64
    %r = pulse.square %m, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rb1, %rt1, %out = pulse.readout %rb, %r, %rt, iq <{duration_cycles = 64 : i64, start_cycles = 32 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %out : !pulse.measurement -> i1
    return %bit : i1
  }
}
"""


@pytest.fixture
def system():
    return pulse.quantum_system("transmon_surface17")


def events(program):
    return [
        (e["kind"], e["qudit"], e["start_cycles"], e["duration_cycles"])
        for e in program.export_to_controller()["events"]
    ]


def test_schedule_is_kept_and_qudits_are_bound_in_order(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[2], system.qudits[0]])
    assert isinstance(program, pulse.ScheduledKernel)
    assert program.qudits == (0, 2)
    assert events(program) == [
        ("drive", 2, 0, 16),
        ("drive", 0, 16, 16),
        ("readout", 0, 32, 64),
    ]


def test_foreign_attributes_and_results_are_dropped(system):
    program = pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    text = program.mlir()
    assert "made." not in text
    assert "func.func @main()" in text
    assert program.metrics.op_count == 3
    assert "3 ops" in repr(program)


def test_lines_get_the_frequencies_of_their_qudits(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[1], system.qudits[2]])
    exported = program.export_to_controller()
    drives = [e for e in exported["events"] if e["kind"] == "drive"]
    assert [e["frequency_hz"] for e in drives] == [system.frequencies[1], system.frequencies[2]]


def test_bytes_are_accepted(system):
    program = pulse.load(MODULE.encode(), entry="round", qudits=system.qudits[:2])
    assert len(events(program)) == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda kw: kw.update(entry="missing"), "no function named missing"),
        (lambda kw: kw.update(qudits=kw["qudits"][:1]), "takes 2 qudit"),
    ],
)
def test_wrong_entry_or_count(system, change, message):
    kw = dict(entry="round", qudits=list(system.qudits[:2]))
    change(kw)
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(MODULE, **kw)


def test_unscheduled_function_is_refused(system):
    text = MODULE.replace(
        'made.by = "elsewhere", pulse.schedule_state = #pulse.schedule_state<timed>',
        'made.by = "elsewhere"',
    )
    with pytest.raises(pulse.CompilationError, match="not scheduled"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_overlap_on_a_line_fails_verification(system):
    text = MODULE.replace("start_cycles = 16 : i64}> : !pulse.drive_line", "start_cycles = 0 : i64}> : !pulse.drive_line")
    text = text.replace("%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1")
    with pytest.raises(pulse.CompilationError):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_qudits_must_be_distinct_and_of_one_system(system):
    with pytest.raises(pulse.InvalidValueError, match="twice"):
        pulse.load(MODULE, entry="round", qudits=[system.qudits[0], system.qudits[0]])
    other = pulse.quantum_system("transmon_surface17")
    if other is not system:
        with pytest.raises(pulse.InvalidValueError, match="more than one"):
            pulse.load(MODULE, entry="round", qudits=[system.qudits[0], other.qudits[1]])


def test_a_stated_clock_must_be_the_systems(system):
    period_fs = round(1e6 / system.clock_ghz)
    stated = "module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = %d>} {"
    same = MODULE.replace("module {", stated % period_fs, 1)
    assert len(events(pulse.load(same, entry="round", qudits=system.qudits[:2]))) == 3
    other = MODULE.replace("module {", stated % (2 * period_fs), 1)
    with pytest.raises(pulse.CompilationError, match="clock"):
        pulse.load(other, entry="round", qudits=system.qudits[:2])
