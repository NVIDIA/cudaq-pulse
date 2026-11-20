# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle replacement: the pass member, its rule, the option that carries it,
and what a compilation without it pays (nothing)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.compiler import DEFAULT_PASSES, CompileOptions

from tests._handles import Q

_ROOT = str(Path(pulse.__file__).parent.parent)
_WITH = (*DEFAULT_PASSES, pulse.Pass.IDLE_REPLACEMENT)


@pulse.kernel
def echo(qudit, window):
    line = get_drive_line(qudit)
    lead = (window - 40) // 2
    wait(line, lead)
    drive(line, square(40, 0.1))
    wait(line, window - 40 - lead)


@pulse.kernel
def two_lines(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def ramsey(q, amp: float, tau: float):
    d = get_drive_line(q)
    drive(d, square(40, amp))
    wait(d, tau)
    drive(d, square(40, amp))


def _rule(**settings):
    return pulse.IdleReplacement(echo, **({"threshold": 100 * pulse.ns} | settings))


# -- The member and the option: both or neither ------------------------------


def test_the_member_is_a_pass_by_name():
    assert pulse.Pass.IDLE_REPLACEMENT == "idle_replacement"
    assert pulse.Pass("idle_replacement") is pulse.Pass.IDLE_REPLACEMENT
    assert pulse.Pass.IDLE_REPLACEMENT not in DEFAULT_PASSES


@pytest.mark.parametrize("passes", [_WITH, ("idle_replacement",), _WITH[::-1]])
def test_the_member_without_a_rule_names_the_option(passes):
    with pytest.raises(pulse.InvalidValueError) as error:
        pulse.compile(two_lines, passes=passes)
    assert str(error.value) == (
        "passes= lists Pass.IDLE_REPLACEMENT, which needs "
        "idle_replacement=pulse.IdleReplacement(fill, threshold=...)"
    )


@pytest.mark.parametrize("passes", [None, DEFAULT_PASSES, ()])
def test_a_rule_without_the_member_names_the_member(passes):
    with pytest.raises(pulse.InvalidValueError) as error:
        pulse.compile(two_lines, passes=passes, idle_replacement=_rule())
    assert str(error.value) == (
        "idle_replacement= is given but passes= does not list Pass.IDLE_REPLACEMENT; "
        "add it, as passes=(*pulse.compiler.DEFAULT_PASSES, pulse.Pass.IDLE_REPLACEMENT)"
    )


def test_the_option_takes_only_a_rule():
    with pytest.raises(pulse.InvalidTypeError, match=r"takes a pulse\.IdleReplacement, got "):
        pulse.compile(two_lines, passes=_WITH, idle_replacement=echo)


def test_the_member_is_no_rewrite_and_its_place_changes_nothing():
    rule = _rule()
    last = CompileOptions.make(passes=_WITH, idle_replacement=rule)
    first = CompileOptions.make(passes=_WITH[::-1][:1] + DEFAULT_PASSES, idle_replacement=rule)
    assert last.passes == first.passes == DEFAULT_PASSES
    assert last == first and hash(last) == hash(first)
    assert last.idle_replacement is rule


# -- The rule -----------------------------------------------------------------


def test_a_rule_holds_what_it_was_given():
    data = Q[0:2]
    rule = pulse.IdleReplacement(
        echo, threshold=200, qudits=data, leading=True, trailing=1, align="end"
    )
    assert rule.fill is echo
    assert rule.threshold == 200 and type(rule.threshold) is int
    assert rule.qudits == (Q[0], Q[1])
    assert (rule.leading, rule.trailing, rule.align) == (True, True, "end")
    plain = _rule()
    assert plain.threshold == 100 * pulse.ns and isinstance(plain.threshold, pulse.Time)
    assert (plain.qudits, plain.leading, plain.trailing, plain.align) == (
        None,
        False,
        False,
        "center",
    )
    assert pulse.IdleReplacement(None, threshold=0).fill is None


def test_a_rule_is_immutable():
    rule = _rule()
    with pytest.raises(AttributeError, match="immutable"):
        rule.threshold = 5
    with pytest.raises(AttributeError, match="immutable"):
        del rule.fill


def test_rules_compare_by_kernel_and_settings():
    assert _rule() == _rule() and hash(_rule()) == hash(_rule())
    assert _rule() != _rule(threshold=200)
    assert _rule() != _rule(leading=True)
    assert _rule() != _rule(align="start")
    assert _rule() != _rule(qudits=[Q[0]])
    assert _rule() != pulse.IdleReplacement(None, threshold=100 * pulse.ns)
    # A time and a count of cycles are different thresholds on every clock.
    assert _rule(threshold=200) != _rule(threshold=200 * pulse.ns)
    assert _rule() != "rule"


def test_a_rule_prints_its_kernel_by_name_and_only_what_was_set():
    assert repr(_rule(threshold=200)) == f"IdleReplacement({__name__}.echo, threshold=200)"
    shown = repr(_rule(qudits=Q[0:2], trailing=True, align="start"))
    assert shown == (
        f"IdleReplacement({__name__}.echo, threshold={100 * pulse.ns!r}, qudits=(0, 1), "
        "trailing=True, align='start')"
    )


@pytest.mark.parametrize(
    ("threshold", "error", "text"),
    [
        (100.0, pulse.InvalidTypeError, "threshold is a float (100.0)"),
        (True, pulse.InvalidTypeError, "not True"),
        ("100", pulse.InvalidTypeError, "not '100'"),
        (-1, pulse.InvalidValueError, "threshold is -1; it cannot be negative"),
        (-1 * pulse.ns, pulse.InvalidValueError, "it cannot be negative"),
    ],
)
def test_a_threshold_is_cycles_or_a_time(threshold, error, text):
    with pytest.raises(error) as raised:
        pulse.IdleReplacement(echo, threshold=threshold)
    assert text in str(raised.value)


def test_a_float_threshold_shows_both_spellings():
    with pytest.raises(pulse.InvalidTypeError) as raised:
        pulse.IdleReplacement(echo, threshold=100.0)
    assert "    threshold=100\n    threshold=100 * pulse.ns" in str(raised.value)


def test_a_threshold_is_required_by_keyword():
    with pytest.raises(TypeError):
        pulse.IdleReplacement(echo)
    with pytest.raises(TypeError):
        pulse.IdleReplacement(echo, 200)


def test_a_fill_is_a_kernel_of_a_qudit_and_a_window():
    def plain(qudit, window):
        return None

    with pytest.raises(pulse.InvalidTypeError, match=r"fill is a @pulse\.kernel"):
        pulse.IdleReplacement(plain, threshold=200)

    @pulse.kernel
    def one(qudit):
        wait(get_drive_line(qudit), 10)

    @pulse.kernel
    def three(qudit, window, start):
        wait(get_drive_line(qudit), window)

    @pulse.kernel
    def with_float(qudit, window: float):
        wait(get_drive_line(qudit), window)

    @pulse.kernel
    def with_runtime_qudit(qudit: pulse.Qudit, window):
        wait(get_drive_line(qudit), window)

    for fill in (one, three, with_float, with_runtime_qudit):
        with pytest.raises(pulse.InvalidTypeError) as raised:
            pulse.IdleReplacement(fill, threshold=200)
        assert f"fill {fill.__name__}(" in str(raised.value)
        assert "does not take (qudit, window)" in str(raised.value)


def test_align_and_qudits_are_checked():
    with pytest.raises(
        pulse.InvalidValueError, match="unknown align 'centre'. Did you mean 'center'"
    ):
        _rule(align="centre")
    with pytest.raises(pulse.InvalidTypeError, match="qudits are qudit handles"):
        _rule(qudits=[0, 1])
    with pytest.raises(pulse.InvalidValueError, match="qudits is empty"):
        _rule(qudits=[])
    assert _rule(qudits=Q[3]).qudits == (Q[3],)


# -- The options of a compilation with a rule ---------------------------------


def test_options_with_a_rule_are_apart_from_those_without():
    plain = CompileOptions.make()
    with_rule = CompileOptions.make(passes=_WITH, idle_replacement=_rule())
    assert plain.idle_replacement is None and type(plain) is CompileOptions
    assert with_rule != plain and with_rule.passes == plain.passes
    assert with_rule == CompileOptions.make(passes=_WITH, idle_replacement=_rule())
    assert with_rule != CompileOptions.make(passes=_WITH, idle_replacement=_rule(threshold=7))
    assert with_rule.without_rule() == plain and type(with_rule.without_rule()) is CompileOptions
    assert repr(with_rule) == f"{plain!r} + {_rule()!r}"


def test_a_copy_of_the_options_keeps_the_rule_and_the_unit_warnings():
    rule = _rule()
    options = CompileOptions.make(passes=_WITH, idle_replacement=rule, unit_warnings=True)
    assert options.unit_warnings is True and options.idle_replacement is rule
    assert options.without_rule().unit_warnings is True
    copy = dataclasses.replace(options, schedule="asap")
    assert copy.idle_replacement is rule and copy.unit_warnings is True
    assert copy.schedule == "asap" and copy != options


# -- A compilation with a rule ------------------------------------------------


def _compiled(kernel, **settings):
    return pulse.compile(kernel, passes=_WITH, idle_replacement=_rule(**settings))


def test_a_wait_as_long_as_a_float_is_refused_with_the_way_out():
    with pytest.raises(pulse.CompilationError) as raised:
        _compiled(ramsey).lower(Q[0], 0.1, 300.0)
    assert str(raised.value).startswith(
        "idle replacement: the length of a wait in ramsey() is computed from the float "
        "parameter 'tau', so its idle windows differ at every call; annotate tau as int to "
        "compile for each value, or compile without idle replacement"
    )
    assert raised.value.code == pulse.ErrorCode.SYMBOLIC_PARAMETER


def test_refusing_a_float_wait_leaves_the_kernel_as_it_was():
    before = ramsey.lower(Q[0], 0.1, 300.0)
    with pytest.raises(pulse.CompilationError):
        _compiled(ramsey).lower(Q[0], 0.1, 300.0)
    assert ramsey.runtime_parameters == ("amp", "tau")
    after = ramsey.lower(Q[0], 0.1, 500.0)
    assert after.generic.mlir() == before.generic.mlir()
    assert [e.duration_cycles for e in after.schedule] == [40, 500, 40]


def test_qudits_that_each_call_binds_are_not_supported_yet():
    @pulse.kernel
    def anywhere(q: pulse.Qudit):
        drive(get_drive_line(q), square(40, 0.1))

    with pytest.raises(pulse.UnsupportedError, match="does not support yet"):
        _compiled(anywhere).lower(Q[0])


# -- The report of idle windows (a rule without a fill) -----------------------


@pulse.kernel
def host(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))
    sync(da, db, rb)
    readout(rb, square(600, 0.05))
    sync(da, db, rb)  # ends the window
    drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))


@pulse.kernel
def rounds(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))
        sync(da, db, rb)
        drive(da, square(40, 0.1))


@pulse.kernel
def crossed(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1), tone=da.tone)  # a's tone plays: a is not idle
    wait(db, 300)
    drive(db, square(40, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def own_readout(a):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    drive(da, square(40, 0.1))
    readout(ra, square(300, 0.05))
    wait(ra, 500)
    sync(da, ra)
    drive(da, square(40, 0.1))


@pulse.kernel
def written_wait(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    wait(da, 400)
    drive(db, square(440, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def before_a_branch(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    m = readout(rb, square(600, 0.05))
    if m:
        drive(db, square(40, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def into_a_branch(a, b):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    m = readout(rb, square(600, 0.05))
    if m:
        drive(da, square(40, 0.1))
    wait(da, 400)
    drive(da, square(40, 0.1))


@pulse.kernel
def detuned(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    shift_frequency(da.tone, 1.0e6)
    drive(db, square(600, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def uneven(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(400, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
        drive(db, square(440, 0.2))


def _windows(kernel, *args, schedule="asap", **settings):
    rule = pulse.IdleReplacement(None, **({"threshold": 100} | settings))
    program = pulse.compile(kernel, schedule=schedule, passes=_WITH, idle_replacement=rule)
    return program.lower(*args).idle_windows


def _spans(windows):
    return [(w.qudit, w.start, w.end, w.rounds) for w in windows]


_OPEN = "not filled: the rule has no fill"


def test_a_gap_the_schedule_made_is_a_window():
    (window,) = _windows(host, Q[0], Q[1])
    line = Path(__file__).read_text().splitlines().index("    sync(da, db, rb)  # ends the window")
    assert window == pulse.compiler.IdleWindow(
        0, 40, 640, 1, f"sync() at line {line + 1} of host()", False, _OPEN
    )
    assert window.cycles == 600


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_the_report_is_the_same_under_every_policy_that_keeps_the_times(schedule):
    assert _spans(_windows(host, Q[0], Q[1], schedule=schedule)) == [(0, 40, 640, 1)]


def test_the_windows_print_as_a_table():
    text = str(_windows(host, Q[0], Q[1]))
    head, row = text.splitlines()
    assert head.split() == ["qudit", "start", "end", "cycles", "rounds", "before", "result"]
    assert row.split()[:5] == ["0", "40", "640", "600", "1"]
    assert row.endswith(_OPEN)
    assert str(_windows(host, Q[0], Q[1], threshold=601)) == "no idle windows"


def test_a_rule_without_a_fill_leaves_the_program_as_it_is():
    rule = pulse.IdleReplacement(None, threshold=100)
    program = pulse.compile(host, passes=_WITH, idle_replacement=rule).lower(Q[0], Q[1])
    plain = pulse.compile(host).lower(Q[0], Q[1])
    assert program.mlir() == plain.mlir()
    assert program._module.serialize() == plain._module.serialize()
    assert program.options.idle_replacement is rule
    assert plain.idle_windows is None
    assert program.bind(()).idle_windows is program.idle_windows


@pytest.mark.parametrize(
    ("threshold", "count"),
    [(600, 1), (601, 0), (300 * pulse.ns, 1), (300.25 * pulse.ns, 0)],
)
def test_a_window_is_at_least_the_threshold(threshold, count):
    # The clock runs two cycles a nanosecond: the window lasts 600 cycles.
    assert len(_windows(host, Q[0], Q[1], threshold=threshold)) == count


def test_the_rule_can_name_its_qudits():
    assert _windows(host, Q[0], Q[1], qudits=[Q[1]]) == ()
    assert _spans(_windows(host, Q[0], Q[1], qudits=[Q[0]])) == [(0, 40, 640, 1)]


def test_a_qudit_whose_tone_plays_from_another_line_is_not_idle():
    # a's line is free from 40 to 780; its tone plays on b's line until 440.
    assert _spans(_windows(crossed, Q[0], Q[1])) == [(0, 440, 780, 1)]


@pytest.mark.parametrize(("schedule", "start", "end"), [("asap", 300, 800), ("alap", 300, 760)])
def test_a_qudit_is_not_idle_while_its_own_readout_runs(schedule, start, end):
    # Under alap the readout is written after the gap it runs in.
    settings = {"leading": True} if schedule == "alap" else {}
    windows = _windows(own_readout, Q[0], schedule=schedule, **settings)
    assert _spans(windows) == [(0, start, end, 1)]
    assert windows[0].result == _OPEN


def test_a_written_wait_is_an_operation_and_no_window():
    assert _windows(written_wait, Q[0], Q[1]) == ()


def test_idle_before_the_first_operation_and_after_the_last_is_left_alone_unless_asked():
    (first,) = _windows(host, Q[0], Q[1], schedule="alap", threshold=30)[:1]
    assert _spans([first]) == [(0, 40, 640, 1)]  # nothing leads under alap here
    (lead,) = _windows(before_a_branch, Q[0], Q[1], schedule="alap")
    assert (lead.start, lead.end) == (0, 600)
    assert lead.result == ("left alone: before the qudit's first operation (leading=True fills it)")
    (asked,) = _windows(before_a_branch, Q[0], Q[1], schedule="alap", leading=True)
    assert asked.result == _OPEN
    tail = _windows(uneven, Q[0], Q[1], 1)[-1]
    assert (tail.start, tail.end, tail.before) == (440, 840, None)
    assert tail.result == "left alone: after the qudit's last operation (trailing=True fills it)"
    assert _windows(uneven, Q[0], Q[1], 1, trailing=True)[-1].result == _OPEN


def test_the_gap_before_a_branch_is_a_window_and_one_that_ends_in_an_arm_is_not():
    (window,) = _windows(before_a_branch, Q[0], Q[1])
    assert (window.start, window.end, window.result) == (40, 640, _OPEN)
    (window,) = _windows(into_a_branch, Q[0], Q[1])
    assert (window.start, window.end) == (40, 600)
    assert window.result == "left alone: in or next to an arm of a branch"


def test_a_window_after_the_kernel_changes_the_tones_frequency_is_left_alone():
    (window,) = _windows(detuned, Q[0], Q[1])
    assert (window.start, window.end) == (40, 600)
    assert window.result == "left alone: the kernel changed the frequency of the qudit's tone"


@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_in_a_loop_counts_its_rounds(n):
    # Unrolled (3) and rolled (9, 1000): one window, as many rounds.
    (window,) = _windows(rounds, Q[0], Q[1], n)
    assert (window.start, window.end, window.rounds, window.result) == (40, 640, n, _OPEN)


def test_a_window_whose_rounds_differ_is_left_alone_at_its_shortest():
    window = _windows(uneven, Q[0], Q[1], 50)[0]
    assert (window.start, window.end, window.rounds) == (40, 400, 50)
    assert window.result == "left alone: its rounds differ"


def test_an_error_of_the_program_is_the_one_it_raises_without_the_rule():
    # The schedule's errors print where the operation is; the operations of
    # a compilation with a rule carry numbers there.
    @pulse.kernel(max_duration_cycles=500)
    def budgeted(a):
        da = get_drive_line(a)
        drive(da, square(400, 0.1))
        wait(da, 200)

    with pytest.raises(RuntimeError) as plain:
        pulse.compile(budgeted).lower(Q[0])
    with pytest.raises(RuntimeError) as ruled:
        pulse.compile(
            budgeted, passes=_WITH, idle_replacement=pulse.IdleReplacement(None, threshold=100)
        ).lower(Q[0])
    assert "max_duration_cycles=500" in str(plain.value)
    assert str(ruled.value) == str(plain.value)
    assert type(ruled.value) is type(plain.value)
    assert ruled.value.__context__ is None


# -- Fills --------------------------------------------------------------------


@pulse.kernel
def pi(qudit, window):
    line = get_drive_line(qudit)
    drive(line, gaussian(40, 0.2, 10))


@pulse.kernel
def whole(qudit, window):
    line = get_drive_line(qudit)
    drive(line, gaussian(window, 0.01, window / 4))


@pulse.kernel
def pair(qudit, window):
    line = get_drive_line(qudit)
    quarter = (window - 80) // 4
    wait(line, quarter)
    drive(line, gaussian(40, 0.2, 10))
    wait(line, 2 * quarter)
    drive(line, gaussian(40, 0.2, 10))
    wait(line, window - 80 - 3 * quarter)


@pulse.kernel
def train(qudit, window):
    line = get_drive_line(qudit)
    for _ in range(12):
        wait(line, 5)
        drive(line, gaussian(40, 0.2, 10))
        wait(line, 5)


@pulse.kernel
def many(a, b, c, d, e):
    lines = [get_drive_line(q) for q in (a, b, c, d)]
    de = get_drive_line(e)
    re = get_readout_line(e)
    for n, line in enumerate(lines):
        drive(line, square(40 + 20 * n, 0.1))
    drive(de, square(40, 0.1))
    sync(*lines, de, re)
    readout(re, square(600, 0.05))
    sync(*lines, de, re)
    for line in lines:
        drive(line, square(40, 0.1))


@pulse.kernel
def middle_tone(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    wait(db, 200)
    drive(db, square(100, 0.1), tone=da.tone)  # a's tone plays 200..300 of a's gap
    wait(db, 300)
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def into_a_loop(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(600, 0.1))
    for _ in range(n):
        sync(da, db)
        drive(da, square(40, 0.1))
        drive(db, square(100, 0.1))


@pulse.kernel
def waits_in_a_row(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    wait(da, 50)
    wait(da, 50)
    drive(da, square(40, 0.1))
    drive(db, square(600, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


_POLICIES = ["asap", "alap", "rcp", "alap_rcp"]


def _filled(kernel, *args, fill=pi, schedule="asap", passes=_WITH, **settings):
    rule = pulse.IdleReplacement(fill, **({"threshold": 100} | settings))
    return pulse.compile(kernel, schedule=schedule, passes=passes, idle_replacement=rule).lower(
        *args
    )


def _events(program):
    return [(e.kind, e.line, e.start, e.end) for e in program.schedule]


def _kept(kernel, *args, schedule="asap", **settings):
    """The program with its windows filled, after checking that it is the
    program without the rule with something written into it."""
    filled = _filled(kernel, *args, schedule=schedule, **settings)
    plain = pulse.compile(kernel, schedule=schedule).lower(*args)
    host = _events(plain)
    assert [event for event in _events(filled) if event in host] == host
    assert filled.duration == plain.duration
    return filled


@pytest.mark.parametrize("schedule", _POLICIES)
def test_a_fill_is_written_into_its_window_and_nothing_else_moves(schedule):
    program = _kept(host, Q[0], Q[1], schedule=schedule)
    assert _events(program) == [
        ("drive", 0, 0, 40),
        ("drive", 1, 0, 40),
        ("readout", 2, 40, 640),
        ("wait", 0, 40, 320),
        ("drive", 0, 320, 360),
        ("wait", 0, 360, 640),
        ("drive", 0, 640, 680),
        ("drive", 1, 640, 680),
    ]
    (window,) = program.idle_windows
    assert (window.qudit, window.start, window.end, window.filled) == (0, 40, 640, True)
    assert window.result == "filled by pi()"


@pytest.mark.parametrize(
    ("align", "written"),
    [
        ("start", [("drive", 0, 40, 80), ("wait", 0, 80, 640)]),
        ("center", [("wait", 0, 40, 320), ("drive", 0, 320, 360), ("wait", 0, 360, 640)]),
        ("end", [("wait", 0, 40, 600), ("drive", 0, 600, 640)]),
    ],
)
@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_fill_shorter_than_its_window_is_placed_as_align_says(schedule, align, written):
    program = _kept(host, Q[0], Q[1], schedule=schedule, align=align)
    assert _events(program)[3:-2] == written


def test_a_fill_as_long_as_its_window_needs_no_wait():
    program = _kept(host, Q[0], Q[1], fill=whole)
    assert _events(program)[3:-2] == [("drive", 0, 40, 640)]


def test_a_fills_pulse_is_not_fused_with_the_kernels_pulse_next_to_it():
    # Square pulses of one amplitude, back to back: the passes would make
    # one pulse of the kernel's two and the fill's between them.
    @pulse.kernel
    def flat(qudit, window):
        drive(get_drive_line(qudit), square(window, 0.1))

    program = _kept(host, Q[0], Q[1], fill=flat)
    assert _events(program)[3:-2] == [("drive", 0, 40, 640)]


def test_a_fill_longer_than_its_window_is_an_error_that_names_both():
    @pulse.kernel
    def long(qudit, window):
        drive(get_drive_line(qudit), gaussian(700, 0.01, 100))

    with pytest.raises(pulse.CompilationError) as raised:
        _filled(host, Q[0], Q[1], fill=long)
    assert str(raised.value).startswith(
        "idle replacement: fill long() takes 700 cycles, 100 more than its window of 600 "
        "cycles on qudit 0; shorten the fill, or raise threshold= so that this window is "
        "left alone"
    )
    assert raised.value.code == pulse.ErrorCode.INVALID_DURATION


def _fill_that(body):
    """A fill whose statements after its line and tone are *body*."""
    scope = {"pulse": pulse, "Q": Q}
    source = (
        "@pulse.kernel\n"
        "def odd(qudit, window):\n"
        "    line = get_drive_line(qudit)\n"
        "    tone = line.tone\n" + "".join(f"    {line}\n" for line in body)
    )
    exec(source, scope)  # noqa: S102
    return scope["odd"]


@pytest.mark.parametrize(
    ("body", "what"),
    [
        (["readout(get_readout_line(qudit), square(100, 0.05))"], "uses get_readout_line"),
        (["sync(line, line)"], "uses sync"),
        (["shift_frequency(tone, 1.0e6)"], "uses shift_frequency"),
        (["set_phase(tone, 0.5)"], "uses set_phase"),
        (
            ["shift_phase(tone, 0.5)", "drive(line, square(40, 0.1))"],
            "leaves its tone's phase shifted by 0.5 rad (its shifts must add up to zero)",
        ),
    ],
)
def test_a_fill_drives_and_waits_on_its_own_line_and_nothing_else(body, what):
    with pytest.raises(pulse.CompilationError) as raised:
        _filled(host, Q[0], Q[1], fill=_fill_that(body))
    assert str(raised.value).startswith(f"idle replacement: fill odd() {what}; in this version")
    assert raised.value.code == pulse.ErrorCode.UNSUPPORTED_CONSTRUCT


def test_a_fill_may_shift_its_tones_phase_by_constants_that_add_up_to_zero():
    fill = _fill_that(
        ["shift_phase(tone, 0.5)", "drive(line, gaussian(40, 0.2, 10))", "shift_phase(tone, -0.5)"]
    )
    program = _kept(host, Q[0], Q[1], fill=fill)
    assert ("drive", 0, 320, 360) in _events(program)


def test_a_fill_with_a_schedule_of_its_own_is_refused():
    @pulse.kernel(schedule="asap")
    def blocked(qudit, window):
        drive(get_drive_line(qudit), gaussian(40, 0.2, 10))

    with pytest.raises(pulse.CompilationError, match="has a schedule= of its own"):
        _filled(host, Q[0], Q[1], fill=blocked)


@pytest.mark.parametrize("schedule", _POLICIES)
def test_the_whole_gap_is_written_around_a_pulse_at_the_qudits_tone(schedule):
    # a idles 40..200 and 300..600; b's line plays a's tone in between.
    program = _kept(middle_tone, Q[0], Q[1], schedule=schedule, threshold=60, leading=True)
    spans = [(w.start, w.end, w.result) for w in program.idle_windows]
    written = [e for e in _events(program) if e[1] == 0 and e[0] == "wait"]
    if schedule in ("asap", "rcp"):
        assert spans == [(40, 200, "filled by pi()"), (300, 600, "filled by pi()")]
        assert written == [
            ("wait", 0, 40, 100),
            ("wait", 0, 140, 200),
            ("wait", 0, 200, 300),  # while the tone plays
            ("wait", 0, 300, 430),
            ("wait", 0, 470, 600),
        ]
    else:
        assert spans == [(0, 160, "filled by pi()"), (300, 600, "filled by pi()")]


def test_a_wait_is_written_while_the_qudits_own_readout_runs():
    program = _kept(own_readout, Q[0], schedule="alap", leading=True)
    assert [e for e in _events(program) if e[1] == 0][:4] == [
        ("wait", 0, 0, 300),
        ("wait", 0, 300, 510),
        ("drive", 0, 510, 550),
        ("wait", 0, 550, 760),
    ]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_the_windows_of_several_qudits_that_end_at_one_sync_are_all_filled(schedule):
    program = _kept(many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule)
    assert [(w.qudit, w.start, w.end, w.filled) for w in program.idle_windows] == [
        (qudit, 100, 700, True) for qudit in range(4)
    ]
    drives = [e for e in _events(program) if e[0] == "drive" and 100 <= e[2] < 700]
    assert drives == [("drive", line, at, at + 40) for line in range(4) for at in (230, 530)]


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_fills_that_need_more_drive_lanes_than_there_are_fail_and_name_them(schedule):
    with pytest.raises(pulse.CompilationError) as raised:
        _filled(many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule)
    text = str(raised.value)
    assert text.startswith("idle replacement: with the idle windows of many() filled, ")
    assert (
        f"under schedule={schedule!r} a fill's pulses take drive lanes like any pulse, and "
        "the control system has 4; fill fewer qudits (qudits=), use a fill with fewer pulses, "
        "or allow more drives at once (max_concurrent_drives)"
    ) in text
    assert raised.value.code == pulse.ErrorCode.TIMING
    # Two qudits fit.
    program = _filled(
        many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule, qudits=[Q[0], Q[1]]
    )
    assert [w.qudit for w in program.idle_windows] == [0, 1]


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_in_a_loop_is_filled_in_every_round_and_the_loop_stays_rolled(schedule, n):
    program = _filled(rounds, Q[0], Q[1], n, schedule=schedule)
    (window,) = program.idle_windows
    assert (window.start, window.end, window.rounds, window.filled) == (40, 640, n, True)
    plain = pulse.compile(rounds, schedule=schedule).lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    if n > 8:
        # Rolled: as many ops for a thousand rounds as for nine.
        assert program.metrics.op_count == _filled(rounds, Q[0], Q[1], 9).metrics.op_count
    else:
        fills = [e for e in _events(program) if e[0] == "drive" and e[3] - e[2] == 40 and e[1] == 0]
        assert len(fills) == 3 * n


@pulse.kernel
def memory(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))  # a idles until its drive of the next round


@pulse.kernel
def memory_then_gate(a, b, n: int, last: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))
    drive(da, square(last, 0.1))


@pulse.kernel
def tone_first(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(db, square(100, 0.1), tone=da.tone)
        drive(da, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))


@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_across_the_end_of_a_loop_iteration_is_filled_in_every_round(n):
    # Under alap a's drive of each round waits for the round before: a
    # idles from its drive to its drive of the next round, and after the
    # last round as long again, to the program's end.
    program = _filled(memory, Q[0], Q[1], n, schedule="alap")
    (window,) = program.idle_windows
    assert (window.start, window.end, window.rounds, window.filled) == (40, 600, n, True)
    assert window.before.startswith("drive() at line ")
    plain = pulse.compile(memory, schedule="alap").lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    if n == 3:
        host = _events(plain)
        written = [event for event in _events(program) if event not in host]
        assert written == [
            (kind, 0, 600 * k + a, 600 * k + b)
            for k in range(3)
            for kind, a, b in (("wait", 40, 300), ("drive", 300, 340), ("wait", 340, 600))
        ] + [("wait", 0, 1800, 1840)]
    else:
        assert (
            program.metrics.op_count
            == _filled(memory, Q[0], Q[1], 9, schedule="alap").metrics.op_count
        )


def test_a_window_to_the_programs_end_is_written_after_the_kernels_last_statement():
    # Under asap a's drive of the last round is its last operation, inside
    # the loop; a then idles through the last readout, to the end.
    program = _kept(memory, Q[0], Q[1], 20, trailing=True)
    (window,) = program.idle_windows
    assert (window.start, window.end, window.before, window.filled) == (11440, 12040, None, True)


def test_a_loops_window_needs_as_much_room_after_the_last_round():
    program = _kept(memory_then_gate, Q[0], Q[1], 20, 40, schedule="alap")
    assert [(w.start, w.end, w.rounds, w.filled) for w in program.idle_windows] == [
        (40, 600, 20, True)
    ]
    # A longer gate after the loop starts sooner: the last round has less room.
    program = _kept(memory_then_gate, Q[0], Q[1], 20, 200, schedule="alap")
    window = program.idle_windows[0]
    assert (window.rounds, window.filled) == (19, False)
    assert window.result == "left alone: there is less room for it after the loop's last round"


def test_a_loops_window_with_the_tone_playing_at_the_start_of_the_next_round_is_left_alone():
    program = _kept(tone_first, Q[0], Q[1], 20, schedule="alap", threshold=60)
    assert {w.result for w in program.idle_windows if not w.filled} >= {
        "left alone: its tone plays or changes at the start of the next round"
    }


def test_a_fill_may_hold_a_loop():
    program = _kept(host, Q[0], Q[1], fill=train)
    pulses = [e for e in _events(program) if e[1] == 0 and e[0] == "drive" and 40 <= e[2] < 640]
    assert pulses == [("drive", 0, 45 + 50 * k, 85 + 50 * k) for k in range(12)]


def test_a_gap_that_ends_inside_a_loop_is_written_after_the_operation_before_it():
    program = _kept(into_a_loop, Q[0], Q[1], 20)
    first = program.idle_windows[0]
    assert (first.start, first.end, first.rounds, first.filled) == (40, 600, 1, True)
    assert _events(program)[:5] == [
        ("drive", 0, 0, 40),
        ("wait", 0, 40, 300),
        ("drive", 0, 300, 340),
        ("wait", 0, 340, 600),
        ("drive", 1, 0, 600),
    ]


def test_a_window_before_waits_the_passes_merge_is_written_before_the_first():
    passes = (*DEFAULT_PASSES, pulse.Pass.MERGE_WAITS, pulse.Pass.IDLE_REPLACEMENT)
    program = _filled(waits_in_a_row, Q[0], Q[1], schedule="alap", passes=passes, leading=True)
    assert _events(program)[:4] == [
        ("wait", 0, 0, 210),
        ("drive", 0, 210, 250),
        ("wait", 0, 250, 460),
        ("wait", 0, 460, 560),
    ]


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize(
    ("kernel", "args", "settings"),
    [
        (host, (Q[0], Q[1]), {}),
        (crossed, (Q[0], Q[1]), {"leading": True, "trailing": True}),
        (own_readout, (Q[0],), {"leading": True}),
        (before_a_branch, (Q[0], Q[1]), {"leading": True}),
        (middle_tone, (Q[0], Q[1]), {"threshold": 60, "leading": True}),
        (many, (Q[0], Q[1], Q[2], Q[3], Q[4]), {}),
        (rounds, (Q[0], Q[1], 200), {}),
        (into_a_loop, (Q[0], Q[1], 20), {"trailing": True}),
        (memory, (Q[0], Q[1], 20), {"trailing": True}),
        (memory_then_gate, (Q[0], Q[1], 20, 40), {"trailing": True}),
        (memory_then_gate, (Q[0], Q[1], 6, 200), {}),
        (tone_first, (Q[0], Q[1], 20), {"threshold": 60, "leading": True, "trailing": True}),
        (uneven, (Q[0], Q[1], 6), {"leading": True, "trailing": True}),
    ],
)
def test_waits_as_long_as_the_windows_change_no_schedule(
    monkeypatch, schedule, kernel, args, settings
):
    # Each window written as one wait: what finding the windows must get right.
    import cudaq_pulse._idle as idle

    monkeypatch.setattr(idle, "_PADS", True)
    program = _kept(kernel, *args, schedule=schedule, **settings)
    assert all(not w.filled or w.result == "filled by a wait" for w in program.idle_windows)


def test_a_program_with_filled_windows_pickles_with_its_rule_and_its_windows():
    import pickle

    program = _filled(host, Q[0], Q[1])
    again = pickle.loads(pickle.dumps(program))
    assert _events(again) == _events(program)
    assert again.idle_windows == program.idle_windows
    assert again.options == program.options
    assert again.options.idle_replacement.fill is pi


_PULSE = {"cycles": 40}


@pulse.kernel
def from_a_global(qudit, window):
    drive(get_drive_line(qudit), gaussian(_PULSE["cycles"], 0.2, 10))


def test_a_change_of_what_the_fill_reads_compiles_the_program_again(monkeypatch):
    compiled = pulse.compile(
        host,
        schedule="asap",
        passes=_WITH,
        idle_replacement=pulse.IdleReplacement(from_a_global, threshold=100),
    )
    assert ("drive", 0, 320, 360) in _events(compiled.lower(Q[0], Q[1]))
    monkeypatch.setitem(_PULSE, "cycles", 80)
    assert ("drive", 0, 300, 380) in _events(compiled.lower(Q[0], Q[1]))


def test_a_float_argument_of_the_kernel_stays_an_argument_of_the_filled_program():
    @pulse.kernel
    def scaled(a, b, amp: float):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, amp))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, amp))

    compiled = pulse.compile(
        scaled,
        schedule="asap",
        passes=_WITH,
        idle_replacement=pulse.IdleReplacement(pi, threshold=100),
    )
    one, other = compiled.lower(Q[0], Q[1], 0.1), compiled.lower(Q[0], Q[1], 0.2)
    assert one.generic is other.generic
    assert one.generic.parameters == ("amp",)
    assert ("drive", 0, 300, 340) in _events(other)
    assert other.idle_windows[0].filled


def test_a_kernel_that_traces_differently_each_time_is_refused():
    import random

    @pulse.kernel
    def dice(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, 0.1 * random.random()))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))

    with pytest.raises(pulse.UnsupportedError, match="replacing its idle windows traces it twice"):
        _filled(dice, Q[0], Q[1])


# -- A compilation without a rule pays nothing --------------------------------


def test_a_compilation_without_a_rule_is_the_program_it_always_was():
    plain = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert type(plain.options) is CompileOptions
    filled = _compiled(two_lines, leading=True).lower(Q[0], Q[1])
    assert len(filled.schedule) == 6
    again = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert again.idle_windows is None
    assert again.mlir() == plain.mlir()
    assert [(e.kind, e.line, e.start, e.end) for e in again.schedule] == [
        ("drive", 0, 360, 400),
        ("drive", 1, 0, 400),
        ("drive", 0, 400, 440),
    ]


_COLD = """
import sys
import cudaq_pulse as p
from cudaq_pulse.quantum_systems import PhysicalQudit

@p.kernel
def k(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))

qudits = {i: PhysicalQudit(i, 5.0e9 + 1e8 * i, 50.0, 30.0) for i in range(2)}
qs = p.quantum_system(name="cold", qudits=qudits)
p.compile(k, passes=p.compiler.DEFAULT_PASSES).lower(qs.qudits[0], qs.qudits[1]).schedule
k.lower(qs.qudits[0], qs.qudits[1]).mlir()
print("cudaq_pulse._idle" in sys.modules)
"""


def test_a_compilation_without_a_rule_does_not_load_the_module(tmp_path):
    script = tmp_path / "cold.py"
    script.write_text(_COLD)
    env = {**os.environ, "PYTHONPATH": _ROOT, "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
