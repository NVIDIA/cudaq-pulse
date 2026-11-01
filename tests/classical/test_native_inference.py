# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The native length inference against classical._infer: a length the
native code gives is the length classical._infer gives, for Bits and for
number lists, and the bodies most functions have take the native path."""

import inspect
import random
import re
import typing
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import infer_list_lengths
from cudaq_pulse.classical._infer import InferenceError, infer_lengths

from tests.classical import test_infer, test_infer_differential

LENGTHS = range(7)


def _python(fn, names, lengths, positions, count, bits):
    results = [(p, "pulse.Bits") for p in positions]
    try:
        return tuple(infer_lengths(fn, fn.__name__, names, lengths, results, count, bits))
    except InferenceError:
        return None


def _agrees(fn, names, lengths, positions, count) -> bool:
    """Whether the native code answered; asserts its answer is Python's."""
    got = infer_list_lengths(fn, names, lengths, positions, count)
    if got is None:
        return False
    for bits in (frozenset(lengths), frozenset()):
        python = _python(fn, names, lengths, positions, count, bits)
        assert python == got, f"{fn.__qualname__} {lengths}: native {got}, python {python}"
    return True


# ── Bodies the native code follows ──────────────────────────────────────────


def _flip(o: pulse.Bits) -> pulse.Bits:
    return [b ^ 1 for b in o]


def _syndrome(d: pulse.Bits) -> pulse.Bits:
    return [d[i] ^ d[i + 1] for i in range(len(d) - 1)]


def _corrections(s: pulse.Bits) -> pulse.Bits:
    fix = [0] * (len(s) + 1)
    for i in range(len(s)):
        if s[i]:
            fix[i + 1] ^= 1
    return fix


def _detectors(this: pulse.Bits, prev: pulse.Bits) -> pulse.Bits:
    return [a ^ b for a, b in zip(this, prev)]


def _halves(x: pulse.Bits) -> tuple[pulse.Bits, pulse.Bits, int]:
    h = len(x) // 2
    return x[:h], x[h:], h


def _padded(a: pulse.Bits) -> pulse.Bits:
    return [0] + a + [1]


def _scaled(xs: list[float], k: float) -> list[float]:
    return [k * x for x in xs]


def _doubled(xs: list[float]) -> list[float]:
    return xs * 2


def _trimmed(x: pulse.Bits) -> pulse.Bits:
    return x[1:-1:2]


@pytest.mark.parametrize(
    "fn",
    [_flip, _syndrome, _corrections, _detectors, _halves, _padded, _scaled, _doubled, _trimmed],
)
def test_common_bodies_take_the_native_path(fn):
    for n in range(1, 7):
        names, positions, count = _signature(fn)
        lengths = {name: n for name in names if name != "k"}
        assert _agrees(fn, names, lengths, positions, count)


_SOURCE = Path(__file__).resolve().parents[2] / "core" / "mlir" / "bindings" / "LengthInference.cpp"


@pytest.mark.skipif(not _SOURCE.is_file(), reason="needs the source tree")
def test_overflow_results_are_read_after_the_check():
    """An overflow builtin's result read in another argument of the same call
    is read before it is written where arguments are evaluated right to left
    (GCC on x86-64), so each one is the condition of its own `if`."""
    source = _SOURCE.read_text()
    calls = re.findall(r"(.{0,4})__builtin_\w+_overflow\(", source)
    assert calls
    assert all(before == "if (" for before in calls), calls


@pytest.mark.skipif(not _SOURCE.is_file(), reason="needs the source tree")
def test_every_native_overflow_check_is_the_condition_of_its_own_if():
    sources = sorted(_SOURCE.parents[1].rglob("*.[ch]*"))
    calls = [
        (path.name, before)
        for path in sources
        if path.suffix in (".cpp", ".h")
        for before in re.findall(r"(.{0,4})__builtin_\w+_overflow\(", path.read_text())
    ]
    assert len({name for name, _ in calls}) >= 2
    assert all(before == "if (" for _, before in calls), calls


def _signature(fn):
    """Parameter names, list-result positions and result count, from the hints."""
    names = tuple(inspect.signature(fn).parameters)
    ret = inspect.signature(fn).return_annotation
    results = typing.get_args(ret) if typing.get_origin(ret) is tuple else (ret,)
    positions = tuple(i for i, r in enumerate(results) if r is pulse.Bits or r == list[float])
    return names, positions, len(results)


def _list_parameters(fn):
    hints = inspect.signature(fn).parameters.values()
    return [p.name for p in hints if p.annotation is pulse.Bits or p.annotation == list[float]]


def test_inference_test_bodies_agree():
    answered = 0
    for fn in vars(test_infer).values():
        if not inspect.isfunction(fn) or fn.__module__ != test_infer.__name__:
            continue
        try:
            names, positions, count = _signature(fn)
        except (TypeError, ValueError):
            continue
        lists = _list_parameters(fn)
        if not positions or not lists:
            continue
        for n in LENGTHS:
            answered += _agrees(fn, names, dict.fromkeys(lists, n), positions, count)
    assert answered > 0


# ── Generated bodies ────────────────────────────────────────────────────────

PROGRAMS = 300
SEED = 20261030


def _native_statement(rng: random.Random, depth: int = 0) -> list[str]:
    """A statement of the part of Python the native code follows, or of one it
    must leave to classical._infer."""
    k = rng.randint(-2, 4)
    start = rng.choice(["", "1", "-2", "n", "len(x) // 2"])
    stop = rng.choice(["", "-1", "n", "5"])
    step = rng.choice(["", "1", "2", "-1", "-2"])
    number = rng.choice(
        ["len(a)", "len(x) - 1", "min(len(a), 3)", "max(len(x), len(y))", "abs(n - 4)"]
        + ["int(len(y))", "len(range(n))", "-n", "~n", "n % 3", "n // 2", "2 ** 3", "n << 1"]
        + ["n >> 1", "n & 3", "n | 1", "n ^ 2", "len(a) * 2 - 1"]
    )
    times = rng.choice(["n", "(n + 1)", str(k), "len(y)", "bool(x)", "int(True)", "(len(x) > 2)"])
    forms = [
        f"a = a[{start}:{stop}:{step}]",
        f"n = {number}",
        f"a = [0] * {times}",
        "a = [v * 2.5 for v in a]",
        "a = [v for v in range(n)]",
        "a = [u + v for u, v in zip(a, x)]",
        "a = list(range(min(n, 4)))",
        "a = list(zip(a, y))",
        "a = (a + [1, 2])[1:]",
        "a = [*a, 0]",
        "a = [sum(a), max(a) if a else 0, abs(sum(x))]",
        "t = (a, len(a))\n(a, n) = t",
        "a = a if len(a) > n else a + [0]",
        "a = a if x and y else [0]",
        "a = a if a[0] == 1 else a[:]",
        "a = a if (n is None) else a + [n]",
        "if 1 in a:\n    a = a[:]",
        "if not a:\n    a = [1]",
        "if a is not None:\n    a = a + [k for k in a]",
        "s = 1.5 / (n or 1)",
        "for u in x:\n    if u:\n        s = u * 2",
    ]
    if depth < 2:
        body = "\n".join(
            "    " + line
            for _ in range(rng.randint(1, 2))
            for line in _native_statement(rng, depth + 1)
        )
        test = rng.choice(["<", "<=", "==", "!=", ">", ">="])
        count = rng.choice(["n", "len(x)", "3"])
        forms += [
            f"if len(a) {test} {k}:\n{body}",
            f"for i in range({count}):\n    s = x[i % (len(x) or 1)] if x else 0",
        ]
    return rng.choice(forms).split("\n")


def _native_program(rng: random.Random, index: int):
    lines = ["a = list(x)", "n = 2", "s = 0"]
    for _ in range(rng.randint(2, 6)):
        lines += _native_statement(rng)
    body = "\n".join("    " + line for line in lines)
    source = f"def native_{index}(x, y):\n{body}\n    return a\n"
    namespace: dict[str, typing.Any] = {}
    exec(compile(source, f"<native {index}>", "exec"), namespace)  # noqa: S102 - generated test bodies
    return namespace[f"native_{index}"]


@pytest.mark.parametrize("grammar", ["inference", "native"])
def test_generated_bodies_agree(grammar):
    rng = random.Random(SEED)
    answered = total = 0
    for index in range(PROGRAMS):
        if grammar == "inference":
            fn, _ = test_infer_differential._program(rng, index)
        else:
            fn = _native_program(rng, index)
        for nx in LENGTHS:
            lengths = {"x": nx, "y": (nx * 3 + index) % 5}
            answered += _agrees(fn, ("x", "y"), lengths, (0,), 1)
            total += 1
    # Enough bodies take the native path for the comparison to mean something;
    # the inference grammar is mostly helpers and tests of values.
    assert answered > {"inference": 0.05, "native": 0.15}[grammar] * total, (answered, total)
