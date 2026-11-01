# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Inferred list lengths checked against running the body: generated
functions of the list operations inference follows (and some it must
refuse), each inferred for every input length and run on random bits. An
inferred length that differs from the length the body returns is a wrong
schema on the wire."""

import random

import pytest
from cudaq_pulse.classical._infer import InferenceError, LengthRules, infer_lengths

SEED = 20261029
PROGRAMS = 300
LENGTHS = range(7)
RUNS = 4  # random contents per length


def _pad(xs, n):
    return xs + [0] * n


def _halve(xs):
    if len(xs) > 1:
        return xs[: len(xs) // 2]
    return list(xs)


def _grow(xs, k):
    for _ in range(k):
        xs.append(1)


HELPERS = {"_pad": _pad, "_halve": _halve, "_grow": _grow}


def _statement(rng: random.Random, depth: int = 0) -> list[str]:
    k = rng.randint(0, 3)
    simple = [
        f"a = a[{rng.choice(['', '1', '-1', 'n', '-n'])}:{rng.choice(['', '2', '-1', 'len(x)'])}]",
        "a = a + y",
        f"a = [0] * (len(x) + {k})",
        "a = [b ^ 1 for b in a]",
        "a = [b for b, c in zip(a, y)]",
        "a = list(reversed(a))",
        "a = sorted(a)",
        f"a = a[::{k + 1}]",
        "a.append(0)",
        "a.extend(y)",
        "a = [*a, *x]",
        "a = a * 2",
        f"a.insert({k}, 1)",
        "n = len(a) // 2",
        f"n = len(y) + {k}",
        "a = a[:n]",
        f"a = _pad(a, {k})",
        "a = _halve(a)",
        f"_grow(a, {k})",
        "a = [sum(a)] * len(x)",
        "a = [len(a)] + a",
        "if a:\n    a.pop()",
        "if len(a) > 2:\n    del a[0]",
        "a = [b for b in x if b]",  # the values decide: refused
        "a = a[: a[0]] if a else a",  # the values decide: refused
        "while len(a) % 3:\n    a.append(0)",
        "while a and a[-1]:\n    a.pop()",  # the values decide: refused
    ]
    if depth < 2:
        body = _block(rng, depth + 1)
        other = _block(rng, depth + 1)
        simple += [
            f"if len(x) > {k}:\n" + _indent(body) + "\nelse:\n" + _indent(other),
            "for _ in range(len(y) % 3):\n" + _indent(body),
            f"for i in range({k}):\n" + _indent(body),
        ]
    return rng.choice(simple).split("\n")


def _block(rng: random.Random, depth: int) -> list[str]:
    return [line for _ in range(rng.randint(1, 2)) for line in _statement(rng, depth)]


def _indent(lines: list[str]) -> str:
    return "\n".join("    " + line for line in lines)


def _program(rng: random.Random, index: int):
    lines = ["a = list(x)", "n = 1"]
    for _ in range(rng.randint(2, 6)):
        lines += _statement(rng)
    source = f"def generated_{index}(x, y):\n" + _indent(lines) + "\n    return a\n"
    namespace = dict(HELPERS)
    exec(compile(source, f"<generated {index}>", "exec"), namespace)  # noqa: S102 - generated test bodies
    return namespace[f"generated_{index}"], source


def _executed(fn, rng: random.Random, nx: int, ny: int) -> set[int] | None:
    """The lengths the body returns on random bits, or None when it raises."""
    seen = set()
    for _ in range(RUNS):
        x = [rng.randint(0, 1) for _ in range(nx)]
        y = [rng.randint(0, 1) for _ in range(ny)]
        try:
            seen.add(len(fn(x, y)))
        except (IndexError, ValueError, ZeroDivisionError):
            return None
    return seen


def _inferred(rules: LengthRules | None, fn, nx: int, ny: int) -> int | None:
    lengths = {"x": nx, "y": ny}
    try:
        if rules is None:
            return infer_lengths(
                fn, fn.__name__, ("x", "y"), lengths, [(0, "pulse.Bits")], 1, frozenset("xy")
            )[0]
        return rules.lengths(fn.__name__, lengths)[0]
    except InferenceError:
        return None


@pytest.mark.parametrize("cached", [False, True], ids=["each version", "reused runs"])
def test_inferred_lengths_match_running_the_body(cached):
    rng = random.Random(SEED)
    inferred = refused = 0
    for index in range(PROGRAMS):
        fn, source = _program(rng, index)
        rules = LengthRules(fn, ("x", "y"), [(0, "pulse.Bits")], 1, frozenset("xy"))
        for nx in LENGTHS:
            ny = (nx * 5 + index) % 4
            got = _inferred(rules if cached else None, fn, nx, ny)
            ran = _executed(fn, rng, nx, ny)
            if got is None:
                refused += 1
                continue
            inferred += 1
            if ran is not None:
                assert ran == {got}, f"x={nx}, y={ny}: inferred {got}, ran {sorted(ran)}\n{source}"
    # Most generated bodies are ones inference follows; a few it refuses.
    assert inferred > 0.6 * (inferred + refused), (inferred, refused)
    assert refused > 0
