# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A name, or a place of a list, that the arms of an if on a measurement
bind to different lines or tones: an operation on it after the if is an
error at the if, and a kernel that does none compiles."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

_HEAD = (
    "@pulse.kernel\n"
    "def k(q0, q1):\n"
    "    lines = [get_drive_line(q0), get_drive_line(q1)]\n"
    "    a = lines[0]\n"
    "    b = lines[1]\n"
    "    ta = a.tone\n"
    "    tb = b.tone\n"
    "    r = get_readout_line(q0)\n"
    "    m = readout(r, square(40, 0.05))\n"
)


def _kernel(*body):
    scope = {"pulse": pulse}
    exec(_HEAD + "".join(f"    {line}\n" for line in body), scope)  # noqa: S102
    return scope["k"]


def _events(kernel, **options):
    program = pulse.compile(kernel, **options).lower(Q[0], Q[1])
    return [(e.kind, e.qudit, e.start, e.end) for e in program.schedule]


_SWAP = ("if m:", "    drive(a, square(16, 0.1))", "    a, b = b, a")


@pytest.mark.parametrize(
    ("after", "held"),
    [
        (["drive(a, square(8, 0.5))"], "a holds"),
        (["drive(a, square(8, 0.5), tone=ta)"], "a holds"),
        (["wait(a, 100)"], "a holds"),
        (["wait(b, 100)"], "b holds"),
        (["sync(a, r)"], "a holds"),
    ],
)
@pytest.mark.parametrize("passes", [None, ()])
def test_an_operation_on_a_name_the_arms_bind_to_different_lines_is_an_error(after, held, passes):
    # Before, drive(a, wf, tone=) and, without the verify pass, wait(a, n)
    # compiled into a program that used qudit 1's line on both paths.
    options = {} if passes is None else {"passes": passes}
    with pytest.raises(pulse.CompilationError) as raised:
        _events(_kernel(*_SWAP, *after), **options)
    text = str(raised.value)
    assert text.startswith(
        f"after this if, {held} one line when the measurement is set and another when it is "
        "not (its arms bind it differently, as a, b = b, a in one arm does), so no operation "
        "after the if can be written on it; do in each arm what follows on that arm's line, "
        "or bind it to the same line in both arms"
    )
    assert raised.value.code == pulse.ErrorCode.RUNTIME_VALUE
    # The error is at the if, the tenth line of the kernel, not at the operation.
    assert raised.value.location.line == 10


@pytest.mark.parametrize(
    ("after", "held"),
    [
        (["drive(lines[0], square(8, 0.5))"], "lines[0] holds"),
        (["sync(lines[0], lines[1])"], "lines[0] holds"),
        (["wait(lines[1], 100)"], "lines[1] holds"),
        # The place the failing operation reads, not the first of the list.
        (["drive(a, square(8, 0.5), tone=lines[1].tone)"], "lines[1] holds"),
        (["for line in lines[1:]:", "    drive(line, square(8, 0.5))"], "lines[1] holds"),
    ],
)
def test_a_place_of_a_list_the_arms_bind_to_different_lines_is_named(after, held):
    kernel = _kernel("if m:", "    lines[0], lines[1] = lines[1], lines[0]", *after)
    with pytest.raises(pulse.CompilationError) as raised:
        _events(kernel)
    assert str(raised.value).startswith(f"after this if, {held} one line when")


def test_tones_the_arms_bind_differently_are_named():
    kernel = _kernel("if m:", "    ta, tb = tb, ta", "drive(a, square(8, 0.5), tone=ta)")
    with pytest.raises(pulse.CompilationError) as raised:
        _events(kernel)
    assert str(raised.value).startswith(
        "after this if, ta holds one tone when the measurement is set and another when it is "
        "not (its arms bind it differently, as a, b = b, a in one arm does), so no operation "
        "after the if can be given it; do in each arm what follows on that arm's tone, or "
        "bind it to the same tone in both arms"
    )


_UPPER = ("t12 = a.tones[(1, 2)]", "u12 = b.tones[(1, 2)]")


@pytest.mark.parametrize(
    ("body", "held", "line"),
    [
        # The tones of another transition of two lines.
        (
            (*_UPPER, "if m:", "    t12, u12 = u12, t12", "drive(a, square(8, 0.5), tone=t12)"),
            "t12",
            12,
        ),
        ((*_UPPER, "if m:", "    t12 = u12", "drive(a, square(8, 0.5), tone=t12)"), "t12", 12),
        # Two tones of one line.
        (
            (*_UPPER, "if m:", "    ta, t12 = t12, ta", "drive(a, square(8, 0.5), tone=ta)"),
            "ta",
            12,
        ),
        ((*_UPPER, "if m:", "    ta, t12 = t12, ta", "shift_phase(t12, 0.5)"), "t12", 12),
        # First read in each arm; the second arm changes its own, which what
        # reads the name after the if would not see.
        (
            (
                "if m:",
                "    upper = a.tones[(1, 2)]",
                "else:",
                "    upper = a.tones[(1, 2)]",
                "    shift_phase(upper, 0.5)",
                "drive(a, square(8, 0.5), tone=upper)",
            ),
            "upper",
            10,
        ),
        (
            (
                "if m:",
                "    upper = a.tones[(1, 2)]",
                "else:",
                "    upper = a.tones[(1, 2)]",
                "    set_frequency(upper, 4.75e9)",
                "drive(a, square(8, 0.5), tone=upper)",
            ),
            "upper",
            10,
        ),
    ],
)
@pytest.mark.parametrize("passes", [None, ()])
def test_a_tone_of_another_transition_the_arms_bind_differently_is_named(body, held, line, passes):
    # These compiled, and played one path's tone on both paths.
    options = {} if passes is None else {"passes": passes}
    with pytest.raises(pulse.CompilationError) as raised:
        _events(_kernel(*body), **options)
    assert str(raised.value).startswith(f"after this if, {held} holds one tone when")
    assert raised.value.location.line == line


@pytest.mark.parametrize(
    "body",
    [
        # Swapped names, then the line's own tone.
        (
            *_UPPER,
            "if m:",
            "    t12, u12 = u12, t12",
            "drive(a, square(8, 0.5), tone=a.tones[(1, 2)])",
        ),
        # Swapped and swapped back.
        (
            *_UPPER,
            "if m:",
            "    t12, u12 = u12, t12",
            "    drive(a, square(8, 0.5), tone=u12)",
            "    t12, u12 = u12, t12",
            "drive(a, square(8, 0.5), tone=t12)",
        ),
        # First read in an arm, read again after.
        (
            "if m:",
            "    drive(a, square(8, 0.5), tone=a.tones[(1, 2)])",
            "drive(a, square(8, 0.5), tone=a.tones[(1, 2)])",
        ),
        # First read in each arm, under one name, and used after.
        (
            "if m:",
            "    upper = a.tones[(1, 2)]",
            "else:",
            "    upper = a.tones[(1, 2)]",
            "drive(a, square(8, 0.5), tone=upper)",
        ),
        # First read in each arm, then read again in an arm of a later if.
        (
            "if m:",
            "    upper = a.tones[(1, 2)]",
            "else:",
            "    upper = a.tones[(1, 2)]",
            "m2 = readout(r, square(40, 0.05))",
            "if m2:",
            "    upper = a.tones[(1, 2)]",
            "drive(a, square(8, 0.5), tone=upper)",
        ),
        # First read in each arm; the first arm shifts its phase.
        (
            "if m:",
            "    upper = a.tones[(1, 2)]",
            "    shift_phase(upper, 0.5)",
            "else:",
            "    upper = a.tones[(1, 2)]",
            "drive(a, square(8, 0.5), tone=upper)",
        ),
        # First read in each arm; the second arm plays with its own in a loop
        # that is compiled as one loop.
        (
            "if m:",
            "    upper = a.tones[(1, 2)]",
            "else:",
            "    upper = a.tones[(1, 2)]",
            "    for i in range(12):",
            "        drive(a, square(8, 0.5), tone=upper)",
            "drive(a, square(8, 0.5), tone=upper)",
        ),
        # Bound in one arm to another read of the same tone.
        (
            "t12 = a.tones[(1, 2)]",
            "if m:",
            "    t12 = a.tones[(1, 2)]",
            "drive(a, square(8, 0.5), tone=t12)",
        ),
    ],
)
def test_tones_of_another_transition_that_are_one_tone_on_both_paths_compile(body):
    events = _events(_kernel(*body))
    assert events[-1] == ("drive", 0, events[-1][2], events[-1][2] + 8)
    assert _events(_kernel(*body), passes=()) == events


def test_two_tones_read_before_the_if_are_two_tones_at_one_frequency_too():
    # Qudit 1's 1-2 transition at qudit 0's 1-2 frequency: the swapped names
    # still hold two tones (what one arm does to one is not done to the
    # other), as they do on a system where the frequencies differ.
    from cudaq_pulse.quantum_systems import PhysicalQudit

    system = pulse.quantum_system(
        name="one-upper-frequency",
        qudits={
            0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-300e6),
            1: PhysicalQudit(1, 4.9e9, 50.0, 30.0, anharmonicity_hz=-200e6),
        },
    )
    swapped = (*_UPPER, "if m:", "    t12, u12 = u12, t12", "drive(a, square(8, 0.5), tone=t12)")
    # One read before the if, shifted in the arm that then reads the other.
    shifted = (
        "t12 = a.tones[(1, 2)]",
        "if m:",
        "    pass",
        "else:",
        "    shift_phase(t12, 0.5)",
        "    t12 = b.tones[(1, 2)]",
        "drive(a, square(8, 0.5), tone=t12)",
    )
    # The same, with the tone first read in the arms of an earlier if.
    later = (
        "if m:",
        "    t12 = a.tones[(1, 2)]",
        "else:",
        "    t12 = a.tones[(1, 2)]",
        "m2 = readout(r, square(40, 0.05))",
        "if m2:",
        "    pass",
        "else:",
        "    shift_phase(t12, 0.5)",
        "    t12 = b.tones[(1, 2)]",
        "drive(a, square(8, 0.5), tone=t12)",
    )
    for body in (swapped, shifted, later):
        for options in ({}, {"passes": ()}):
            with pytest.raises(pulse.CompilationError) as raised:
                pulse.compile(_kernel(*body), quantum_system=system, **options).lower(
                    *system.qudits[0:2]
                )
            assert str(raised.value).startswith("after this if, t12 holds one tone when")


def test_a_tone_played_in_many_ifs_is_looked_at_once():
    # Each if's result leads back to the one before it through both arms:
    # walked once per path, 28 ifs took a quarter of a minute to decode.
    played = [
        line
        for i in range(28)
        for line in (
            f"m{i} = readout(r, square(40, 0.05))",
            f"if m{i}:",
            "    drive(a, square(8, 0.5), tone=t)",
        )
    ]
    body = (
        "if m:",
        "    u = a.tones[(1, 2)]",
        "else:",
        "    u = a.tones[(1, 2)]",
        "t = a.tones[(1, 2)]",
        *played,
        "mz = readout(r, square(40, 0.05))",
        "if mz:",
        "    u = a.tones[(1, 2)]",
        "drive(a, square(8, 0.5), tone=u)",
    )
    assert len(_events(_kernel(*body), passes=())) == 1 + 2 * 28 + 2


def test_a_name_bound_to_another_line_in_one_arm_is_named():
    with pytest.raises(pulse.CompilationError) as raised:
        _events(_kernel("if m:", "    a = b", "wait(a, 100)"))
    assert str(raised.value).startswith("after this if, a holds one line when")


def test_a_name_each_arm_binds_to_its_own_line_is_named_when_used_after():
    kernel = _kernel(
        "if m:", "    target = b", "else:", "    target = a", "drive(target, square(16, 0.1))"
    )
    with pytest.raises(pulse.CompilationError) as raised:
        _events(kernel)
    assert str(raised.value).startswith("after this if, target holds one line when")


@pytest.mark.parametrize(
    ("body", "events"),
    [
        # A name for the arm's own line, used in the arms only: the verify
        # pass refused this, as the name's two lines were yielded twice.
        (
            (
                "if m:",
                "    target = b",
                "    drive(target, square(16, 0.1))",
                "else:",
                "    target = a",
                "    drive(target, square(16, 0.1))",
            ),
            [("readout", 0, 0, 40), ("drive", 1, 40, 56), ("drive", 0, 40, 56)],
        ),
        (
            (
                "if m:",
                "    a, b = b, a",
                "    drive(a, square(16, 0.1))",
                "    drive(b, square(16, 0.1))",
            ),
            [("readout", 0, 0, 40), ("drive", 1, 40, 56), ("drive", 0, 40, 56)],
        ),
        (
            (
                "line = a",
                "if m:",
                "    for line in lines:",
                "        drive(line, square(16, 0.1))",
                "drive(b, square(8, 0.5))",
            ),
            [
                ("readout", 0, 0, 40),
                ("drive", 0, 48, 64),
                ("drive", 1, 40, 56),
                ("drive", 1, 56, 64),
            ],
        ),
        # Bound differently and not used after.
        (("if m:", "    a, b = b, a"), [("readout", 0, 0, 40)]),
        (("if m:", "    target = b", "else:", "    target = a"), [("readout", 0, 0, 40)]),
    ],
)
def test_a_name_bound_differently_and_not_used_after_the_if_compiles(body, events):
    assert _events(_kernel(*body)) == events
    assert _events(_kernel(*body), passes=()) == events


@pytest.mark.parametrize(
    ("body", "events"),
    [
        # The same rebinding in both arms.
        (
            (
                "if m:",
                "    a, b = b, a",
                "    drive(a, square(16, 0.1))",
                "else:",
                "    a, b = b, a",
                "drive(a, square(8, 0.5))",
            ),
            [("readout", 0, 0, 40), ("drive", 1, 40, 56), ("drive", 1, 56, 64)],
        ),
        # Swapped and swapped back before the arm ends.
        (
            (
                "if m:",
                "    a, b = b, a",
                "    drive(a, square(16, 0.1))",
                "    a, b = b, a",
                "drive(a, square(8, 0.5))",
            ),
            [("readout", 0, 0, 40), ("drive", 1, 40, 56), ("drive", 0, 48, 56)],
        ),
        # Rebound to the same line under another name.
        (
            (
                "same = a",
                "if m:",
                "    a = same",
                "    drive(a, square(16, 0.1))",
                "drive(a, square(8, 0.5))",
            ),
            [("readout", 0, 0, 40), ("drive", 0, 40, 56), ("drive", 0, 56, 64)],
        ),
        # A loop variable over the lines in one arm.
        (
            (
                "if m:",
                "    for line in lines:",
                "        drive(line, square(16, 0.1))",
                "drive(a, square(8, 0.5))",
            ),
            [
                ("readout", 0, 0, 40),
                ("drive", 0, 40, 56),
                ("drive", 1, 48, 64),
                ("drive", 0, 56, 64),
            ],
        ),
    ],
)
def test_names_that_hold_one_line_on_both_paths_compile_as_before(body, events):
    assert _events(_kernel(*body)) == events


def test_a_name_no_longer_one_line_stays_so_through_a_later_if():
    # The second if leaves the name as the first did; nothing reads it.
    quiet = _kernel(
        *_SWAP, "m2 = readout(r, square(40, 0.05))", "if m2:", "    drive(lines[0], square(8, 0.5))"
    )
    assert len(_events(quiet)) == 4
    # Read after the second if: the error is at the first.
    with pytest.raises(pulse.CompilationError) as raised:
        _events(
            _kernel(
                *_SWAP, "m2 = readout(r, square(40, 0.05))", "if m2:", "    a = a", "wait(a, 8)"
            )
        )
    assert str(raised.value).startswith("after this if, a holds one line when")


_ROUND = "m2 = readout(r, square(40, 0.05))"
_CARRIED = (
    "holds one line when the measurement is set and another when it is not (its arms bind it "
    "differently, as a, b = b, a in one arm does), so the loop cannot carry it from one "
    "iteration to the next; do in each arm what follows on that arm's line, or bind it to the "
    "same line in both arms; when no iteration reads it before binding it, a larger "
    "unroll_threshold= compiles the loop one iteration at a time"
)


@pytest.mark.parametrize(
    "body",
    [
        # A name for the arm's own line, used in the arms only.
        (
            "for i in range(6):",
            f"    {_ROUND}",
            "    if m2:",
            "        target = b",
            "        drive(target, square(16, 0.1))",
            "    else:",
            "        target = a",
            "        drive(target, square(16, 0.1))",
        ),
        # Names of the body, swapped in one arm and used there.
        (
            "for i in range(6):",
            f"    {_ROUND}",
            "    first, second = a, b",
            "    if m2:",
            "        first, second = second, first",
            "        drive(first, square(16, 0.1))",
        ),
    ],
)
def test_a_name_of_a_loops_body_bound_differently_by_its_arms_compiles(body):
    # The loop as one scf.for (it was refused there, though no operation was
    # on the name) gives what its iterations written out give.
    rolled = _events(_kernel(*body), passes=())
    assert rolled == _events(_kernel(*body), passes=(), unroll_threshold=10_000)
    assert len(_events(_kernel(*body))) > 1


@pytest.mark.parametrize(
    ("body", "held", "line"),
    [
        # Swapped in one arm; no iteration operates on the names.
        (("for i in range(12):", f"    {_ROUND}", "    if m2:", "        a, b = b, a"), "a", 12),
        # No one line before the loop, bound in the body before it is read.
        (
            (*_SWAP, "for i in range(12):", "    a = lines[0]", "    drive(a, square(8, 0.5))"),
            "a",
            10,
        ),
    ],
)
def test_a_name_from_before_a_loop_is_not_carried_as_different_lines(body, held, line):
    # Which iterations read the name is not known when the loop is compiled
    # as one loop: it is refused there, and compiles one iteration at a time.
    with pytest.raises(pulse.CompilationError) as raised:
        _events(_kernel(*body))
    assert str(raised.value).startswith(f"after this if, {held} {_CARRIED}")
    assert raised.value.location.line == line
    assert len(_events(_kernel(*body), unroll_threshold=10_000)) > 1


@pytest.mark.parametrize(
    ("body", "so", "line"),
    [
        # Read at the top of the next iteration.
        (
            (
                "for i in range(12):",
                "    drive(a, square(8, 0.5))",
                f"    {_ROUND}",
                "    if m2:",
                "        a, b = b, a",
            ),
            "so the loop cannot carry it from one iteration to the next",
            13,
        ),
        # Read after the if, in the body.
        (
            (
                "for i in range(12):",
                f"    {_ROUND}",
                "    if m2:",
                "        a, b = b, a",
                "    drive(a, square(8, 0.5))",
            ),
            "so no operation after the if can be written on it",
            12,
        ),
        # Read after the loop.
        (
            (
                "for i in range(12):",
                f"    {_ROUND}",
                "    if m2:",
                "        a, b = b, a",
                "wait(a, 8)",
            ),
            "so the loop cannot carry it from one iteration to the next",
            12,
        ),
    ],
)
def test_a_name_bound_differently_in_a_loop_and_read_is_an_error_either_way(body, so, line):
    for options in ({}, {"unroll_threshold": 10_000}):
        with pytest.raises(pulse.CompilationError) as raised:
            _events(_kernel(*body), **options)
        text = str(raised.value)
        assert text.startswith("after this if, ")
        assert "holds one line when the measurement is set and another when it is not" in text
        assert raised.value.location.line == line
        if not options:
            assert so in text


@pytest.mark.parametrize(
    ("after", "message"),
    [
        (["assert False, a"], "assertion failed in kernel"),
        (
            ["m2 = readout(r, square(40, 0.05))", "pair = (a, r)", "if m2:", "    pair = (r, a)"],
            "'pair' would depend on a measurement result",
        ),
        (["drive(a)"], "drive()"),
    ],
)
def test_a_failure_that_is_not_the_names_keeps_its_own_error(after, message):
    # The kernel holds a name that is no one line; what fails is something else.
    with pytest.raises(pulse.CompilationError) as raised:
        _events(_kernel("if m:", "    a, b = b, a", *after))
    assert str(raised.value).startswith(message)
