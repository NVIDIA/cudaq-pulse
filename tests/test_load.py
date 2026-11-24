# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.load``: a scheduled function another compiler built."""

import re

import cudaq_pulse as pulse
import pytest

MODULE = """
module {
  func.func private @other(%q: !pulse.qudit) attributes {pulse.schedule_state =
        #pulse.schedule_state<timed>} {
    return
  }
  func.func private @round(%a: !pulse.qudit, %b: !pulse.qudit) -> i1 attributes {made.by =
        "elsewhere", pulse.schedule_state = #pulse.schedule_state<timed>} {
    %da, %ta = pulse.get_drive_line %a {made.channel = 3 : i64} : (!pulse.qudit) ->
          (!pulse.drive_line, !pulse.tone)
    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %rb, %rt = pulse.get_readout_line %b : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %da1, %ta1 = pulse.drive %da, %w, %ta <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %db1, %tb1 = pulse.drive %db, %w, %tb <{duration_cycles = 16 : i64, start_cycles = 16 : i64}>
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %m = arith.constant 64 : i64
    %r = pulse.square %m, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rb1, %rt1, %out = pulse.readout %rb, %r, %rt, iq <{duration_cycles = 64 : i64, start_cycles =
          32 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line,
          !pulse.tone, !pulse.measurement
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
        ", pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %da,", "} {\n    %da,"
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="not scheduled"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_overlap_on_a_line_fails_verification(system):
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64")
    assert text != MODULE
    text = text.replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
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
    stated = (
        "module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = %d>} {"
    )
    same = MODULE.replace("module {", stated % period_fs, 1)
    assert len(events(pulse.load(same, entry="round", qudits=system.qudits[:2]))) == 3
    other = MODULE.replace("module {", stated % (2 * period_fs), 1)
    with pytest.raises(pulse.CompilationError, match="clock"):
        pulse.load(other, entry="round", qudits=system.qudits[:2])


T = "!pulse.drive_line, !pulse.tone"
TIMED = "attributes {pulse.schedule_state = #pulse.schedule_state<timed>}"
CALLS = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {TIMED} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func private @pair(%a: !pulse.drive_line, %at: !pulse.tone, %b: !pulse.drive_line, %bt:
        !pulse.tone) -> ({T}, {T}) {TIMED} {{
    %a1, %at1 = func.call @x(%a, %at) {{pulse.start_cycles = 0 : i64}} : ({T}) -> ({T})
    %b1, %bt1 = func.call @x(%b, %bt) {{pulse.start_cycles = 8 : i64, pulse.length_cycles = 8 :
          i64}} : ({T}) -> ({T})
    return %a1, %at1, %b1, %bt1 : {T}, {T}
  }}
  func.func @shot(%qa: !pulse.qudit, %qb: !pulse.qudit) {TIMED} {{
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %b, %bt = pulse.get_drive_line %qb : (!pulse.qudit) -> ({T})
    %a1, %at1, %b1, %bt1 = func.call @pair(%a, %at, %b, %bt) {{pulse.start_cycles = 0 : i64,
          pulse.length_cycles = 16 : i64}} : ({T}, {T}) -> ({T}, {T})
    %b2, %bt2 = func.call @x(%b1, %bt1) {{pulse.start_cycles = 40 : i64}} : ({T}) -> ({T})
    %b3, %bt3, %a3, %at3 = func.call @pair(%b2, %bt2, %a1, %at1) {{pulse.start_cycles = 100 :
          i64}} : ({T}, {T}) -> ({T}, {T})
    return
  }}
}}
"""


def test_calls_with_a_stated_start_are_written_in_place(system):
    program = pulse.load(CALLS, entry="shot", qudits=system.qudits[:2])
    assert events(program) == [
        ("drive", 0, 0, 8),
        ("drive", 1, 8, 8),
        ("drive", 1, 40, 8),
        ("drive", 1, 100, 8),
        ("drive", 0, 108, 8),
    ]
    assert "call" not in program.mlir()


@pytest.mark.parametrize(
    "old, new, message",
    [
        (" {pulse.start_cycles = 40 : i64}", "", "without stating when it starts"),
        (
            "pulse.length_cycles = 16",
            "pulse.length_cycles = 12",
            "and it takes 16",
        ),
        ("pulse.start_cycles = 40 : i64", "pulse.start_cycles = 12 : i64", None),
        (
            "pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %n",
            "made.by = 1 : i64} {\n    %n",
            "which is not scheduled",
        ),
    ],
)
def test_calls_that_cannot_be_written_in_place(system, old, new, message):
    text = CALLS.replace(old, new)
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_function_that_calls_itself_is_refused(system):
    text = f"""
module {{
  func.func @again(%q: !pulse.qudit) {TIMED} {{
    func.call @again(%q) {{pulse.start_cycles = 0 : i64}} : (!pulse.qudit) -> ()
    return
  }}
}}
"""
    with pytest.raises(pulse.CompilationError, match="calls itself"):
        pulse.load(text, entry="again", qudits=system.qudits[:1])


# ── Classical functions ──────────────────────────────────────────────────────


def _links():
    from cudaq_pulse.classical import Link

    return pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=100)})


@pulse.classical(max_latency_ns=400, link="host")
def decide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits([bits[0]])


@pulse.kernel
def rounds(a, c):
    ro = get_readout_line(a)  # noqa: F821
    da = get_drive_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    for _ in range(2):
        drive(da, square(16, 0.5))  # noqa: F821
        sync(da, ro)  # noqa: F821
        m = readout(ro, square(100, 0.05))  # noqa: F821
        fix = decide.submit([m]).result()
        sync(dc, ro)  # noqa: F821
        if fix[0]:
            drive(dc, square(16, 0.5))  # noqa: F821


def _as_written(system, machine):
    """The compiled kernel and its module as another compiler would write it:
    a function of its qudits, with nothing of the target in it."""
    import re

    native = pulse.compile(rounds, quantum_system=system, control_system=machine).lower(
        system.qudits[0], system.qudits[2]
    )
    text = native.mlir()
    names = []
    for qudit in (0, 2):
        found = re.search(
            r"\n\s*(%\w+) = pulse\.qudit_alloc \{qudit = "
            + str(qudit)
            + r" : i64\} : !pulse\.qudit",
            text,
        )
        assert found is not None
        names.append(found.group(1))
        text = text.replace(found.group(0), "")
    text = text.replace(
        "func.func @main()",
        "func.func @main(" + ", ".join(f"{n}: !pulse.qudit" for n in names) + ")",
    )
    text = re.sub(r" \{frequency_hz = [^}]*\}", "", text)
    text = re.sub(r"classical\.links = \{.*?\}\}, ", "", text, count=1)
    assert "classical.links" not in text and "classical.callable" in text
    return native, text


def _calls(program):
    exported = program.export_to_controller()
    return exported["calls"], exported["links"], exported["manifest"]


def test_classical_functions_are_read_from_the_module(system):
    machine = _links()
    native, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert events(loaded) == events(native)
    assert _calls(loaded) == _calls(native)
    assert len(_calls(loaded)[0]) == 2
    assert [fn.implementation for fn in loaded.classical] == [None]


def test_a_given_classical_function_is_checked_and_its_latency_used(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=machine, classical=[decide]
    )
    assert loaded.classical[0].implementation is decide.implementation

    @pulse.classical(name="decide", max_latency_ns=900, link="host")
    def slow(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[slow])

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def wide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
        return pulse.Bits([bits[0], bits[0]])

    with pytest.raises(pulse.CompilationError, match=r"gives \(bits:1\).*gives \(bits:2\)"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[wide])

    @pulse.classical(max_latency_ns=400, link="host")
    def other(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named other"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[other])
    with pytest.raises(pulse.InvalidTypeError, match="takes classical functions"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=["decide"])


def test_the_links_are_the_control_systems(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=900)})
    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


LOOP = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {TIMED} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func @shot(%qa: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %l1, %lt1 = func.call @x(%l, %lt) {{pulse.start_cycles = 4 : i64, pulse.period_cycles = 20 :
            i64}} : ({T}) -> ({T})
      %l2, %lt2 = func.call @x(%l1, %lt1) {{pulse.start_cycles = 12 : i64, pulse.period_cycles =
            20 : i64}} : ({T}) -> ({T})
      scf.yield %l2, %lt2 : {T}
    }}
    return
  }}
}}
"""


def test_calls_in_a_loop_keep_the_loop_and_state_its_period(system):
    program = pulse.load(LOOP, entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [
        (4, 8),
        (12, 8),
        (24, 8),
        (32, 8),
        (44, 8),
        (52, 8),
    ]
    text = program.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.drive ") == 2
    assert "call" not in text


# ── What a review found ──────────────────────────────────────────────────────


def test_a_failed_load_carries_a_documented_code(system):
    with pytest.raises(pulse.CompilationError) as wrong:
        pulse.load(MODULE, entry="missing", qudits=system.qudits[:2])
    assert wrong.value.code == "wrong_argument"
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64").replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
    with pytest.raises(pulse.CompilationError) as overlap:
        pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert overlap.value.code == "other"


def test_a_nested_call_without_a_length_still_counts(system):
    text = re.sub(r", pulse\.length_cycles = 8 :\s+i64", "", CALLS).replace(
        "pulse.length_cycles = 16", "pulse.length_cycles = 12"
    )
    assert text.count("pulse.length_cycles") == 1
    with pytest.raises(pulse.CompilationError, match="and it takes 16"):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_function_with_a_qudit_of_its_own_is_refused(system):
    text = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    %own = pulse.qudit_alloc {qudit = 5 : i64} : !pulse.qudit\n"
        "    %n = arith.constant 16 : i64",
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="allocates a qudit"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


@pytest.mark.parametrize("limit", [{"max_instructions": 1000}, {"max_program_duration": 1e6}])
def test_program_limits_are_not_checked_yet_and_say_so(system, limit):
    machine = pulse.Controller(**limit)
    with pytest.raises(pulse.UnsupportedError, match="max_instructions or max_program_duration"):
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2], control_system=machine)


def test_a_slower_link_is_refused(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=5000, feedback_ns=100)})
    with pytest.raises(pulse.CompilationError, match="due|it takes"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


def test_a_given_bool_function_keeps_its_own_types(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def yes(bits: pulse.Bits) -> bool:
        return bool(bits[0])

    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[yes])
    assert [type(r).__name__ for r in loaded.classical[0].results] == [
        type(r).__name__ for r in yes.specialize(bits=1).results
    ]
    assert loaded.classical[0].base_name == "decide"


def test_a_function_is_given_for_its_versions_only():
    from cudaq_pulse._load import _given_functions

    row = ("decide__fast", 1, 0, ["bits:1"], ["bits:1"], (0.0, 400.0, 400.0), "bounded", "eager")
    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named decide"):
        _given_functions([decide], [row + (None, None, None)])
    version = ("decide__1",) + row[1:] + (None, None, None)
    assert set(_given_functions([decide], [version])) == {"decide__1"}
    lazy = version[:7] + ("lazy",) + version[8:]
    with pytest.raises(pulse.CompilationError, match="submit policy"):
        _given_functions([decide], [lazy])
