# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Mistakes in a kernel are CompilationErrors that say what to write."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError

from tests._handles import Q
from tests._ir import python_ir

# ── Operands are checked when the op is emitted ──────────────────────


def test_swapped_line_and_tone_from_the_friction_log():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d.tone, gaussian(40, 0.1, 10.0), tone=d)

    message = (
        r"drive\(\): argument 1 must be a drive line, got a tone; "
        r"the line comes first and the tone by keyword: drive\(line, waveform\) "
        r"or drive\(line, waveform, tone=tone\)"
    )
    with pytest.raises(CompilationError, match=message):
        python_ir(k, Q[0])
    with pytest.raises(CompilationError, match=message):
        pulse.compile(k).lower(Q[0])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("drive(d, 0.5)", r"drive\(\): argument 2 must be a waveform, got float 0.5; pass a"),
        ("drive(r, square(8, 0.1))", r"argument 1 must be a drive line, got a readout line"),
        ("readout(d, square(8, 0.1))", r"argument 1 must be a readout line, got a drive line"),
        (
            "wait(d.tone, 8)",
            r"wait\(\): argument 1 must be a drive line or readout line, got a tone",
        ),
        ("shift_phase(d, 0.1)", r"shift_phase\(\): argument 1 must be a tone, got a drive line"),
        ("get_drive_line(3)", r"argument 1 must be a qudit, got int 3; pass one of the kernel"),
        ("wf_add(square(8, 0.1), 0.2)", r"wf_add\(\): argument 2 must be a waveform, got float"),
    ],
)
def test_wrong_operands_name_the_argument(body, message):
    namespace: dict = {}
    exec(
        f"def k(q):\n    d = get_drive_line(q)\n    r = get_readout_line(q)\n    {body}\n",
        namespace,
    )
    kernel = pulse.kernel(namespace["k"])
    with pytest.raises(CompilationError, match=message):
        pulse.compile(kernel).lower(Q[0])


def test_correct_operands_still_compile():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        wait(d, 8)
        wait(r, 8)
        drive(d, wf_scale(square(8, 0.1), 0.5))
        readout(r, square(8, 0.1))

    pulse.compile(k).lower(Q[0])


# ── sync() takes lines, or lists and tuples of them ──────────────────


def test_sync_flattens_lists_and_tuples_from_the_friction_log():
    @pulse.kernel
    def k(q0, q1, q2):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        d2 = get_drive_line(q2)
        sync([d0, d1])
        sync((d0, d1), d2)
        lines = [d0, [d1, d2]]
        sync(lines)
        drive(d2, square(8, 0.1))

    syncs = [op for op in python_ir(k, Q[0], Q[1], Q[2]).ops if op.kind == "pulse.sync"]
    assert [len(op.operands) for op in syncs] == [2, 3, 3]
    pulse.compile(k).lower(Q[0], Q[1], Q[2])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("sync(d0, d0.tone)", r"sync\(\): argument 2 must be a line, got a tone"),
        ("sync([d0, 3])", r"sync\(\): argument 1\[1\] must be a line, got int 3"),
        ("sync([d0, d0.tone])", r"argument 1\[1\] must be a line, got a tone; pass the"),
        ("sync(d0)", r"sync\(\) got one line, which has no other line to align with; pass two"),
        ("sync([d0])", r"sync\(\) got one line, .* as in sync\(d, r\), or remove the call"),
        ("sync()", r"sync\(\) got no lines; it aligns two or more lines, as in sync\(d, r\)"),
        ("sync([])", r"sync\(\) got no lines"),
        ("sync([d0], d0)", r"sync\(\): a line is passed more than once"),
    ],
)
def test_sync_rejects_anything_but_lines(body, message):
    namespace: dict = {}
    exec(
        f"def k(q0, q1):\n    d0 = get_drive_line(q0)\n    d1 = get_drive_line(q1)\n    {body}\n",
        namespace,
    )
    with pytest.raises(CompilationError, match=message):
        python_ir(pulse.kernel(namespace["k"]), Q[0], Q[1])


# ── Errors point at the kernel's source ──────────────────────────────


def _note(excinfo) -> str:
    return "\n".join(getattr(excinfo.value, "__notes__", ()))


def _drive_swapped(d):
    drive(d.tone, square(8, 0.1), tone=d)


def test_a_misspelled_op_from_the_friction_log_names_the_file_and_line():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drvie(d, gaussian(40, 0.1, 10.0))  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    with pytest.raises(
        CompilationError, match=r"unknown pulse op: drvie; did you mean drive\(\)\?"
    ) as e:
        pulse.compile(k).lower(Q[0])
    note = _note(e)
    assert f'File "{__file__}", line {k.__wrapped__.__code__.co_firstlineno + 3}, in k' in note
    assert "drvie(d, gaussian(40, 0.1, 10.0))" in note


def test_an_error_in_a_helper_shows_the_kernel_line_then_the_helper_line():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        _drive_swapped(d)

    with pytest.raises(CompilationError, match="argument 1 must be a drive line") as e:
        python_ir(k, Q[0])
    note = _note(e)
    assert note.index("_drive_swapped(d)") < note.index("drive(d.tone, square(8, 0.1), tone=d)")
    assert "in _drive_swapped" in note


def test_an_unknown_name_far_from_any_suggests_nothing():
    @pulse.kernel
    def k(q):
        zzzzqqq(q)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    with pytest.raises(CompilationError) as e:
        python_ir(k, Q[0])
    assert str(e.value) == "unknown pulse op: zzzzqqq"


def test_pulse_dot_drive_from_the_friction_log_points_at_ops():
    @pulse.kernel
    def k(q):
        d = pulse.get_drive_line(q)
        pulse.drive(d, pulse.gaussian(40, 0.1, 10.0))

    message = (
        r"module 'cudaq_pulse' has no attribute 'get_drive_line': the kernel language is in "
        r"cudaq_pulse.ops; write get_drive_line\(...\) or cudaq_pulse.ops.get_drive_line\(...\)"
    )
    with pytest.raises(CompilationError, match=message):
        python_ir(k, Q[0])

    @pulse.kernel
    def works(q):
        d = pulse.ops.get_drive_line(q)
        pulse.ops.drive(d, pulse.ops.gaussian(40, 0.1, 10.0))

    python_ir(works, Q[0])


def test_a_misspelled_attribute_suggests_the_right_one():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        pulse.ops.drvie(d, square(8, 0.1), d.tone)

    with pytest.raises(
        CompilationError, match=r"on module 'cudaq_pulse.ops'; did you mean 'drive'\?"
    ):
        python_ir(k, Q[0])


# ── assert ───────────────────────────────────────────────────────────


def _unrewritten(source: str):
    """A kernel from *source*, outside pytest's rewriting of assert."""
    namespace: dict = {}
    exec(source, namespace)
    return pulse.kernel(namespace["k"])


def test_assert_with_a_message_from_the_friction_log():
    k = _unrewritten(
        "def k(q, n_rounds: int):\n"
        "    assert n_rounds > 0, 'boom'\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(8 * n_rounds, 0.1))\n"
    )
    python_ir(k, Q[0], 2)
    with pytest.raises(CompilationError) as e:
        python_ir(k, Q[0], 0)
    assert str(e.value) == "assertion failed in kernel: boom"


def test_assert_messages_built_with_calls_and_bare_asserts():
    formatted = _unrewritten(
        "def k(n: int):\n    assert n > 0, f'{n} rounds, need {len([1, 2])}'\n"
    )
    bare = _unrewritten("def k(n: int):\n    assert n > 0\n")
    with pytest.raises(CompilationError) as e:
        python_ir(formatted, 0)
    assert str(e.value) == "assertion failed in kernel: 0 rounds, need 2"
    with pytest.raises(CompilationError) as e:
        python_ir(bare, 0)
    assert str(e.value) == "assertion failed in kernel"


def test_raise_still_names_the_exception():
    @pulse.kernel
    def k(n: int):
        if n < 1:
            raise ValueError("n must be positive")

    with pytest.raises(CompilationError, match=r"kernel raised ValueError\('n must be positive'\)"):
        python_ir(k, 0)


# ── Unsupported constructs are named, with what to write instead ─────


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("match (1, 2):\n        case (a, b):\n            pass", "match statements with sequence"),
        ("match {1: 2}:\n        case {1: v}:\n            pass", "use if/elif with =="),
        ("global G\n    G = 1", r"assignments to globals are not supported.*keep the value"),
        ("d.name = 'x'", r"attribute assignments \(obj.x = ...\) are not supported"),
        ("class C:\n        pass", "class definitions are not supported.*module level"),
        ("n = 0\n    while n < 2:\n        n += 1", "while loops are not supported"),
    ],
)
def test_unsupported_constructs_say_what_they_are(body, message):
    namespace: dict = {}
    exec(f"def k(q):\n    d = get_drive_line(q)\n    {body}\n", namespace)
    with pytest.raises(CompilationError, match=message) as e:
        python_ir(pulse.kernel(namespace["k"]), Q[0])
    assert "bytecode" not in str(e.value)
