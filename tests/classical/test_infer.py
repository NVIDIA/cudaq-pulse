"""List result lengths read from a classical function's bytecode: one
family of instructions per test, the lengths a version gets, and the
reason a version is rejected (with the line it points at)."""

import contextlib
import inspect
import sys

import cudaq_pulse as pulse
import pytest

K = 3
TABLE = {(0, 0): [0, 0, 0], (0, 1): [0, 0, 1], (1, 0): [1, 0, 0], (1, 1): [0, 1, 0]}
SHARED = [0, 0]


def _helper(bits):
    return sum(bits)


def _starred(*lists):
    return len(lists)


def _lengths(fn, **n):
    """The lengths of a version's results (1 for a scalar)."""
    return tuple(r.count for r in pulse.classical(fn, max_latency_ns=1).specialize(**n).results)


def _rejected(fn, match, **n):
    declared = pulse.classical(fn, max_latency_ns=1)
    with pytest.raises(pulse.CompilationError, match=match) as raised:
        declared.specialize(**n)
    return str(raised.value)


def _line(fn, text):
    """The line of *fn*'s source holding *text*."""
    lines, start = inspect.getsourcelines(fn)
    return start + next(i for i, s in enumerate(lines) if text in s)


# ── Loads, stores, arithmetic ───────────────────────────────────────────────


def _arith(x: pulse.Bits) -> pulse.Bits:
    m = len(x) * 3 // 2 - 1
    return [0] * (m % 4 + 1)


def _globals_and_constants(x: pulse.Bits) -> pulse.Bits:
    return x[:K] + [0] * len(SHARED)


def _unary(x: pulse.Bits) -> pulse.Bits:
    return [0] * -(-len(x)) + [1] * abs(-2)


@pytest.mark.parametrize(
    ("fn", "n", "expected"),
    [
        (_arith, 5, (3,)),
        (_arith, 4, (2,)),
        (_globals_and_constants, 5, (5,)),
        (_globals_and_constants, 2, (4,)),
        (_unary, 3, (5,)),
    ],
)
def test_numbers_from_lengths_follow_arithmetic(fn, n, expected):
    assert _lengths(fn, x=n) == expected


# ── Building and slicing lists ──────────────────────────────────────────────


def _slices(x: pulse.Bits) -> tuple[pulse.Bits, pulse.Bits, pulse.Bits, pulse.Bits]:
    return x[1:], x[::2], x[-2:], x[1:3]


def _concat(a: pulse.Bits, b: pulse.Bits) -> pulse.Bits:
    return [0] + a + b + [1]


def _repeat(a: pulse.Bits) -> pulse.Bits:
    return a * 2


def _unpack(x: pulse.Bits) -> pulse.Bits:
    first, *rest = x
    return rest


def test_slices_concatenation_repetition_and_unpacking():
    assert _lengths(_slices, x=5) == (4, 3, 2, 2)
    assert _lengths(_slices, x=2) == (1, 1, 2, 1)
    assert _lengths(_concat, a=3, b=4) == (9,)
    assert _lengths(_repeat, a=3) == (6,)
    assert _lengths(_unpack, x=4) == (3,)


# ── Branches ────────────────────────────────────────────────────────────────


def _pad_even(x: pulse.Bits) -> pulse.Bits:
    if len(x) % 2:
        return x + [0]
    return x


def _ternary(x: pulse.Bits) -> pulse.Bits:
    return x[:2] if len(x) > 3 else x


def _boolop(x: pulse.Bits, k: int) -> pulse.Bits:
    n = k and 2 or 3
    return [1] * n


def _two_returns(x: pulse.Bits) -> pulse.Bits:
    if x[0]:
        return x
    return x[1:]


def _match(x: pulse.Bits) -> pulse.Bits:
    match len(x):
        case 3:
            return x
        case _:
            return x + [0]


def test_branches_the_lengths_decide_take_one_path():
    assert _lengths(_pad_even, x=5) == (6,)
    assert _lengths(_pad_even, x=6) == (6,)
    assert _lengths(_ternary, x=5) == (2,)
    assert _lengths(_ternary, x=3) == (3,)
    assert _lengths(_match, x=3) == (3,)
    assert _lengths(_match, x=4) == (5,)


def test_branches_the_values_decide_are_rejected_at_their_line():
    message = _rejected(_two_returns, "different lengths", x=4)
    top, bottom = _line(_two_returns, "return x\n"), _line(_two_returns, "return x[1:]")
    assert f"(4 at line {top}, 3 at line {bottom})" in message
    message = _rejected(_boolop, "depending on its test", x=2)
    assert "`k and 2 or 3` has value 2 or 3" in message


# ── Loops ───────────────────────────────────────────────────────────────────


def _append(data: pulse.Bits) -> pulse.Bits:
    flip = sum(data) * 2 > len(data)
    out = []
    for b in data:
        out.append(b ^ flip)
    return out


def _corrections(syndrome: pulse.Bits) -> pulse.Bits:
    fix = [0] * (len(syndrome) + 1)
    for i in range(len(syndrome)):
        if syndrome[i] and (i == 0 or not syndrome[i - 1]):
            fix[i + 1] ^= 1
    return fix


def _nested(a: pulse.Bits, b: pulse.Bits) -> pulse.Bits:
    out = []
    for x in a:
        for y in b:
            out.append(x ^ y)
    return out


def _counted_while(x: pulse.Bits) -> pulse.Bits:
    out = []
    i = 0
    while i < len(x):
        out.append(x[i])
        i += 1
    return out


def _break(x: pulse.Bits) -> pulse.Bits:
    out = [0]
    for b in x:
        if b > 5:
            break
    else:
        out = out + [1]
    return out


def _until_zero(x: pulse.Bits) -> pulse.Bits:
    out = []
    i = 0
    while i < len(x) and x[i]:
        out.append(1)
        i += 1
    return out


def _append_if(x: pulse.Bits) -> pulse.Bits:
    out = []
    for b in x:
        if b:
            out.append(b)
    return out


def test_loops_that_move_by_the_same_amount_each_pass():
    assert _lengths(_append, data=7) == (7,)
    assert _lengths(_corrections, syndrome=4) == (5,)
    assert _lengths(_nested, a=3, b=2) == (6,)
    assert _lengths(_counted_while, x=4) == (4,)
    assert _lengths(_break, x=3) == (2,)  # a bit is never above 5


def test_loops_the_values_decide_are_rejected_at_their_line():
    message = _rejected(_until_zero, "changes inside the `while` loop", x=4)
    assert f"at line {_line(_until_zero, 'while i')}, which runs a number of times" in message
    message = _rejected(_append_if, "depending on the if", x=4)
    assert (
        f"`out` has length 1 or 0 depending on the if at line {_line(_append_if, 'if b')}"
        in message
    )


# ── Comprehensions, generators, builtins ────────────────────────────────────


def _comprehension(data: pulse.Bits) -> pulse.Bits:
    return [data[i] ^ data[i + 1] for i in range(len(data) - 1)]


def _zip(this_round: pulse.Bits, previous: pulse.Bits) -> pulse.Bits:
    return [a ^ b for a, b in zip(this_round, previous)]


def _generator(x: pulse.Bits) -> pulse.Bits:
    return list(b ^ 1 for b in x)


def _map(x: pulse.Bits) -> pulse.Bits:
    return list(map(lambda b: 1 - b, x))


def _builtins(x: pulse.Bits, xs: list[int]) -> pulse.Bits:
    return sorted(x) + list(reversed(x)) + [0] * max(len(x), len(xs), 2) + [1] * min(len(x), 1)


def _ones(x: pulse.Bits) -> pulse.Bits:
    return [b for b in x if b]


def _filter(x: pulse.Bits) -> pulse.Bits:
    return list(filter(None, x))


def test_comprehensions_generators_and_builtins():
    assert _lengths(_comprehension, data=5) == (4,)
    assert _lengths(_zip, this_round=8, previous=8) == (8,)
    assert _lengths(_generator, x=3) == (3,)
    assert _lengths(_map, x=3) == (3,)
    assert _lengths(_builtins, x=3, xs=5) == (12,)


def test_a_condition_on_the_values_is_rejected_with_the_comprehension():
    message = _rejected(_ones, "is a comprehension with a condition", x=4)
    assert "`[b for b in x if b]` is a comprehension with a condition (`if b`)" in message
    assert f"(line {_line(_ones, 'return')})" in message
    _rejected(_filter, r"`filter\(None, x\)` keeps only some items", x=4)


# ── Counting set bits ───────────────────────────────────────────────────────


def _padded(syndrome: pulse.Bits) -> tuple[int, list[int]]:
    where = [i for i, s in enumerate(syndrome) if s]
    return len(where), where + [0] * (len(syndrome) - len(where))


def _ones_then_zeros(x: pulse.Bits) -> pulse.Bits:
    return [b for b in x if b] + [0] * (len(x) - sum(x))


def _ints_then_zeros(xs: list[int]) -> list[int]:
    return [b for b in xs if b] + [0] * (len(xs) - sum(xs))


def test_what_the_count_of_set_bits_cancels_is_a_length():
    assert _lengths(_padded, syndrome=6) == (1, 6)
    assert _lengths(_ones_then_zeros, x=5) == (5,)
    # The sum of integers is not a count of items.
    _rejected(_ints_then_zeros, "depends on values", xs=5)


def _lookup(syndrome: pulse.Bits) -> pulse.Bits:
    return TABLE[tuple(syndrome)]


def test_a_module_table_of_equal_lists_gives_their_length():
    assert _lengths(_lookup, syndrome=2) == (3,)


# ── What could change a list ────────────────────────────────────────────────


def _passed(syndrome: pulse.Bits) -> pulse.Bits:
    w = _starred(syndrome)
    return [w > len(syndrome) // 2] * len(syndrome)


def _passed_by_keyword(x: pulse.Bits) -> pulse.Bits:
    out = [0, 0]
    _helper(bits=out)
    return out


def _in_a_container(x: pulse.Bits) -> pulse.Bits:
    out = [0]
    box = [out]
    box[0].append(1)
    return out


def _closure(x: pulse.Bits) -> pulse.Bits:
    out = []

    def add(v):
        out.append(v)

    for b in x:
        add(b)
    return out


def _nonlocal(x: pulse.Bits) -> pulse.Bits:
    n = 1

    def bump():
        nonlocal n
        n = 5

    bump()
    return [0] * n


def _shared(x: pulse.Bits) -> pulse.Bits:
    SHARED.append(1)
    return list(SHARED)


def _aliased(x: pulse.Bits) -> pulse.Bits:
    a = list(x)
    b = a
    b.append(1)
    return a


def test_a_list_something_unseen_could_change_has_no_length():
    message = _rejected(_passed, "is passed to `_starred\\(\\)`", syndrome=6)
    assert f"at line {_line(_passed, 'w = ')}, which could change its length" in message
    _rejected(_passed_by_keyword, "is passed to `_helper\\(\\)`", x=2)
    _rejected(_in_a_container, "could change it", x=2)
    _rejected(_closure, "`out` is used by a nested function", x=2)
    _rejected(_nonlocal, "`n` is used by a nested function", x=2)
    _rejected(_shared, "`SHARED` is a module-level variable the function changes", x=2)
    assert _lengths(_aliased, x=3) == (4,)


# ── Plain Python helpers ────────────────────────────────────────────────────


def _pad(xs, n=2):
    return xs + [0] * n


def _twice(xs):
    return [*xs, *xs]


def _grow_if_long(xs):
    if len(xs) > 3:
        xs.append(1)


def _same(xs):
    return xs


def _first_ones(xs):
    return xs[: xs[0]]


def _halves(xs):
    return _halves(xs[1:]) if xs else []


def _majority(x: pulse.Bits) -> pulse.Bits:
    w = _helper(x)
    return [w > len(x) // 2] * len(x)


def _nested_helpers(x: pulse.Bits) -> pulse.Bits:
    return _pad(_twice(x))


def _changed_by_a_helper(x: pulse.Bits) -> pulse.Bits:
    out = list(x)
    _grow_if_long(out)
    _grow_if_long(out)
    return out


def _returned_alias(x: pulse.Bits) -> pulse.Bits:
    out = _same(list(x))
    out.append(0)
    return out


def _value_dependent_helper(x: pulse.Bits) -> pulse.Bits:
    return _first_ones(x)


def _recursive_helper(x: pulse.Bits) -> pulse.Bits:
    return _halves(x)


def _swapped(x: pulse.Bits) -> pulse.Bits:
    return _pad(x)


@pytest.mark.parametrize(
    ("fn", "n", "expected"),
    [
        (_majority, 6, (6,)),
        (_nested_helpers, 3, (8,)),
        (_changed_by_a_helper, 3, (3,)),
        (_changed_by_a_helper, 4, (6,)),
        (_returned_alias, 2, (3,)),
    ],
)
def test_lengths_follow_plain_python_helpers(fn, n, expected):
    assert _lengths(fn, x=n) == expected


def test_a_helper_reason_names_the_call():
    message = _rejected(_value_dependent_helper, "in `_first_ones\\(x\\)`, `xs\\[0\\]`", x=3)
    assert f"(line {_line(_value_dependent_helper, 'return _first_ones')})" in message
    _rejected(_recursive_helper, "`_halves\\(xs\\[1:\\]\\)` calls itself", x=3)


def test_a_version_rereads_the_helpers_it_called(monkeypatch):
    declared = pulse.classical(_swapped, max_latency_ns=1)
    assert declared.specialize(x=3).results[0].count == 5
    monkeypatch.setattr(sys.modules[__name__], "_pad", _twice)
    assert declared.specialize(x=4).results[0].count == 8


# ── Exceptions and with ─────────────────────────────────────────────────────


def _try(x: pulse.Bits, k: int) -> pulse.Bits:
    out = [0]
    try:
        out.append(1)
        if k:
            raise ValueError
        out.append(2)
    except ValueError:
        pass
    return out


def _try_finally(x: pulse.Bits) -> pulse.Bits:
    out = []
    try:
        out.append(1)
    finally:
        out.append(2)
    return out


def _with(x: pulse.Bits) -> pulse.Bits:
    with contextlib.nullcontext():
        out = [b for b in x]
    return out


def _with_suppress(x: pulse.Bits, k: int) -> pulse.Bits:
    out = []
    with contextlib.suppress(ValueError):
        out.append(1)
        if k:
            raise ValueError
        out.append(2)
    return out


def test_exceptions_follow_the_edges_to_their_handler():
    assert _lengths(_try_finally, x=2) == (2,)
    assert _lengths(_with, x=3) == (3,)
    _rejected(_try, "`out` has length", x=2)
    _rejected(_with_suppress, "`out` has length", x=2)


# ── Parameters the values decide ────────────────────────────────────────────


def _head(x: pulse.Bits, k: int) -> pulse.Bits:
    return x[:k]


def test_a_number_parameter_is_known_only_at_run_time():
    message = _rejected(_head, "`k` is a number parameter, known only when the function runs", x=5)
    assert f"(line {_line(_head, 'return')})" in message
    assert message.endswith(
        "Write it in plain Python, or give it a fixed size: -> pulse.Length[pulse.Bits, 8]."
    )


# ── One run for every version ───────────────────────────────────────────────

WIDTH = 2


def _widened(x: pulse.Bits) -> pulse.Bits:
    return x + [0] * WIDTH + [0] * len(SHARED)


def _versions(fn, name, counts):
    declared = pulse.classical(fn, max_latency_ns=1)
    found = [tuple(r.count for r in declared.specialize(**{name: n}).results) for n in counts]
    return found, declared._length_rules.runs


def test_a_body_runs_once_and_its_lengths_serve_every_version():
    assert _versions(_corrections, "syndrome", range(1, 9)) == ([(n + 1,) for n in range(1, 9)], 1)
    assert _versions(_arith, "x", range(1, 9)) == (
        [((n * 3 // 2 - 1) % 4 + 1,) for n in range(1, 9)],
        1,
    )
    assert _versions(_padded, "syndrome", [4, 2, 7]) == ([(1, 4), (1, 2), (1, 7)], 1)


def test_a_loop_the_lengths_end_runs_for_each_version():
    assert _versions(_counted_while, "x", range(1, 5)) == ([(n,) for n in range(1, 5)], 4)


def test_a_changed_global_runs_the_body_again():
    global WIDTH
    declared = pulse.classical(_widened, max_latency_ns=1)
    assert declared.specialize(x=3).results[0].count == 7
    assert declared.specialize(x=4).results[0].count == 8
    WIDTH = 3
    SHARED.append(0)
    try:
        assert declared.specialize(x=5).results[0].count == 11
    finally:
        WIDTH = 2
        SHARED.pop()
    assert declared.specialize(x=6).results[0].count == 10
    assert declared._length_rules.runs == 2


def test_a_rejected_version_keeps_no_rule():
    declared = pulse.classical(_until_zero, max_latency_ns=1)
    for n in (4, 5):
        with pytest.raises(pulse.CompilationError, match="changes inside the `while` loop"):
            declared.specialize(x=n)
    assert declared._length_rules.runs == 2
    assert declared._length_rules.rules == []


def test_inference_reads_no_source():
    import linecache

    read = []
    getlines = linecache.getlines
    linecache.getlines = lambda *a, **k: read.append(1) or getlines(*a, **k)
    try:
        assert _lengths(_corrections, syndrome=3) == (4,)
        assert _lengths(_padded, syndrome=4) == (1, 4)
    finally:
        linecache.getlines = getlines
    assert read == []
