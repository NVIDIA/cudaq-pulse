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


def test_replacing_windows_is_not_built_yet():
    # Replaced by the tests of the windows and the fills when they are built.
    with pytest.raises(pulse.UnsupportedError, match="is not built yet"):
        _compiled(two_lines).lower(Q[0], Q[1])


# -- A compilation without a rule pays nothing --------------------------------


def test_a_compilation_without_a_rule_is_the_program_it_always_was():
    plain = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert type(plain.options) is CompileOptions
    with pytest.raises(pulse.UnsupportedError):
        _compiled(two_lines).lower(Q[0], Q[1])
    again = pulse.compile(two_lines).lower(Q[0], Q[1])
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
