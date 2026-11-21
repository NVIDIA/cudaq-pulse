# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical functions a controller computes: the export's ``functions``.

A body in the subset a ``Processor`` runs with ``fixed_cost_only`` lowers to
a straight-line program of run-time expressions, which
``controller.evaluate_function`` computes as the controller must. Each
construct is checked against the Python body on random inputs, to the last
bit of float64; so are the shipped calibration estimators, the functions of
examples/21 and what ``controller_usage`` counts of them. A body outside the
subset is refused with its line and construct.
"""

from __future__ import annotations

import json
import math
import random
import runpy
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse._controller_functions import ARITHMETIC_TERMS, lower
from cudaq_pulse.classical import Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.quantum_systems import transmon_surface17

_ROOT = Path(__file__).resolve().parents[1]
_CLASSES = ("add", "mul", "div", "sqrt", "exp", "log", "trig")
QS = transmon_surface17()
Q = QS.qudits


def _program(fn):
    return lower(fn).entry


def _same(a, b):
    """Equal to the bit, a NaN equal to a NaN, and of one type."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (a != a and b != b)
    return type(a) is type(b) and a == b


def _agrees(fn, draws, count=400, seed=5):
    """*fn*'s program gives what its body gives, on *count* draws of inputs."""
    program = _program(fn)
    rng = random.Random(seed)
    checked = 0
    for _ in range(count):
        args = draws(rng)
        try:
            expected = fn(*args)
        except (ArithmeticError, ValueError):
            continue  # where Python raises, the program gives IEEE 754's value
        expected = expected if isinstance(expected, tuple) else (expected,)
        got = controller.evaluate_function(program, args)
        assert len(got) == len(expected)
        for want, have in zip(expected, got, strict=True):
            assert _same(float(want) if isinstance(have, float) else want, have), (args, want, have)
        checked += 1
    assert checked > count // 2
    return program


def _number(rng):
    return rng.choice(
        [rng.uniform(-5, 5), rng.uniform(-1e4, 1e4), rng.uniform(1e-6, 1e-3), 0.0, -0.0, 1.0]
    )


def _terms(program):
    """The ops of every term of *program*, with repeats."""
    found = []

    def walk(e):
        if isinstance(e, dict) and "op" in e:
            found.append(e["op"])
            for x in e["of"]:
                walk(x)

    for binding in program["body"]["let"]:
        walk(binding["value"])
    for result in program["body"]["results"]:
        walk(result)
    return found


# ── Each construct against its Python body ──────────────────────────────────


@pulse.classical(max_latency_ns=100)
def _arithmetic(x: float, y: float, n: int, m: int) -> tuple[float, float, float, int]:
    a = x * y - x / 3.0 + y * -2.5
    b = (x + n) * (y - m) / (x * x + 1.0)
    c = n * m - (n + 7) * 3
    d = a % 1.75 + b**2 + abs(x) ** 0.5 + x**3
    e = n // (m * m + 1) + n % 5 + m**3 - -n
    return a, d + c, b + float(n) / (m * m + 1), e + int(x * 10.0)


def test_arithmetic_agrees_with_the_body():
    program = _agrees(
        _arithmetic,
        lambda r: (_number(r), _number(r), r.randint(-50, 50), r.randint(-50, 50)),
    )
    ops = set(_terms(program))
    assert {"add", "sub", "mul", "div", "mod", "pow", "floor_div", "neg", "abs"} <= ops
    assert {"to_float", "to_int"} <= ops


@pulse.classical(max_latency_ns=100)
def _branches(x: float, y: float, n: int, bit: pulse.Length[pulse.Bits, 2]) -> tuple[float, int]:
    if x > y:
        z = x - y
        k = n
    elif bit[0] and not bit[1]:
        z = y * 2.0
        k = n + 1
    else:
        z = 0.5
        k = 3
    w = z if 0.0 < z <= 10.0 else -z
    if n == 7 or x != x:
        return w, k
    flag = (x < 0.0 and y < 0.0) or bit[1]
    if flag:
        w += 1.0
    return w * (1.0 if bit[0] else -1.0), k * 2


def test_branches_and_early_returns_agree_with_the_body():
    program = _agrees(
        _branches,
        lambda r: (
            r.choice([_number(r), math.nan]),
            _number(r),
            r.choice([7, r.randint(-9, 9)]),
            [r.randint(0, 1), r.randint(0, 1)],
        ),
    )
    assert "select" in _terms(program)


def _scaled(x, scale=2.0, offset=0.0):
    t = x * scale
    if t < 0.0:
        return -t + offset
    return t + offset


def _pair(a, b):
    return a + b, a - b


@pulse.classical(max_latency_ns=100)
def _helpers(x: float, y: float, bits: pulse.Length[pulse.Bits, 5]) -> tuple[float, float, float]:
    total = 0
    for b in bits:
        total += b
    weighted = 0.0
    for i, b in enumerate(bits):
        weighted += i * b * x
    acc = 1.0
    for _ in range(4):
        acc = acc * 0.5 + _scaled(y, offset=1.0)
    s, d = _pair(x, _scaled(x, 3.0))
    return weighted + total, acc, s * d + max(x, y, 0.25) - min(x, y) + len(bits)


@pulse.classical(max_latency_ns=100)
def _first_above(xs: pulse.Length[list[float], 4], limit: float) -> tuple[float, int]:
    count = 0
    for i, x in enumerate(xs):
        count += 1
        if x > limit:
            return x, i * 10 + count
    return limit, count


def test_a_return_inside_a_loop_agrees_with_the_body():
    _agrees(_first_above, lambda r: ([r.uniform(-1, 1) for _ in range(4)], r.uniform(-1, 1)))


def test_loops_and_helpers_agree_with_the_body():
    program = _agrees(
        _helpers, lambda r: (_number(r), _number(r), [r.randint(0, 1) for _ in range(5)])
    )
    names = [b["name"] for b in program["body"]["let"]]
    # Unrolled, with each binding its own name; a helper's locals carry its name.
    assert len(set(names)) == len(names)
    assert any(name.startswith("_scaled.") for name in names)
    assert '{"argument": 2, "index": 4}' in json.dumps(program)


@pulse.classical(max_latency_ns=100)
def _elementary(x: float, y: float) -> tuple[float, float, float, float]:
    a = math.exp(x / 4.0) + math.log(abs(y) + 1e-3) + math.sqrt(abs(x))
    b = math.sin(x) * math.cos(y) + math.atan2(y, x)
    c = math.fabs(x - y) + math.pi * math.e
    d = x**y if x > 0.0 else 0.0
    return a, b, c, d + math.sqrt(x * x + y * y)


@pulse.classical(max_latency_ns=100)
def _rounding(x: float, y: float) -> tuple[int, int, bool]:
    special = math.isnan(x) or math.isinf(y) or not math.isfinite(x * 1e300)
    return math.floor(x), math.ceil(y), special


def test_elementary_functions_agree_with_the_body():
    def draws(r):
        y = math.inf if r.random() < 0.1 else r.choice([r.uniform(-30, 30), 0.0, -0.0])
        return r.choice([r.uniform(-30, 30), 0.0, -0.0]), y

    ops = set(_terms(_agrees(_elementary, draws))) | set(_terms(_agrees(_rounding, draws)))
    assert {"exp", "log", "sqrt", "sin", "cos", "atan2", "pow", "abs", "floor", "ceil"} <= ops


@pulse.classical(max_latency_ns=100)
def _ints_and_bools(n: int, m: int, bit: pulse.Length[pulse.Bits, 1]) -> tuple[int, bool, float]:
    k = n * 3 + bit[0] - (m > n) + abs(n - m)
    return k, bool(k) and n < m < 40 or bit[0] == 1, (n > 0) * 1.5


def test_integers_and_bools_agree_with_the_body():
    _agrees(_ints_and_bools, lambda r: (r.randint(-60, 60), r.randint(-60, 60), [r.randint(0, 1)]))


# ── The shipped estimators ──────────────────────────────────────────────────


@pulse.classical(max_latency_ns=100)
def _adaptive_t1(
    shape: float, rate: float, excited: int, wait: float, e10: float, e01: float
) -> tuple[float, float, float]:
    shape, rate = cal.adaptive_t1_update(shape, rate, excited, wait, e10, e01)
    return shape, rate, cal.adaptive_t1_wait(shape, rate, 0.51)


@pulse.classical(max_latency_ns=100)
def _binary_search(
    mean: float, sigma: float, outcome: int, wait: float
) -> tuple[float, float, float, float]:
    mean, sigma = cal.binary_search_update(mean, sigma, outcome, wait)
    wait = cal.binary_search_wait(sigma)
    return mean, sigma, wait, cal.binary_search_phase(mean, wait)


@pulse.classical(max_latency_ns=100)
def _decay(
    p0: float, p1: float, p3: float, spacing: float, shots: int
) -> tuple[float, float, bool]:
    return cal.decay_time(p0, p1, p3, spacing, shots)


@pulse.classical(max_latency_ns=100)
def _phase(
    pm: float, p0: float, pp: float, shots: int, mean: float
) -> tuple[float, float, float, float]:
    theta, sigma = cal.sparse_phase(pm, p0, pp, shots)
    return (
        theta,
        sigma,
        cal.pi_train_amplitude(0.15, theta, 21),
        cal.ramsey_detuning(mean, theta, 2000.0),
    )


@pulse.classical(max_latency_ns=100)
def _closed_forms(a: float, b: float, c: float, t: float) -> tuple[float, float, float, float]:
    return (
        cal.three_point_decay(a, b, c, t),
        cal.three_point_phase(a, b, c, t, 0.4),
        cal.t1_wait(a + 2.0, b + 2.0, 1.3),
        cal.survival_probability(t, a) + cal.ramsey_probability(t, b, c * 1e5),
    )


@pulse.classical(max_latency_ns=100)
def _rabi(a: float, b: float) -> float:
    return cal.rabi_probability(a, b)


@pulse.classical(max_latency_ns=100)
def _ones(bits: pulse.Length[pulse.Bits, 40]) -> float:
    return cal.count_ones(bits) / len(bits)


def _probability(r):
    return r.uniform(0.0, 1.0)


@pytest.mark.parametrize(
    ("fn", "draws"),
    [
        (
            _adaptive_t1,
            lambda r: (
                r.uniform(1, 50),
                r.uniform(1e3, 1e6),
                r.randint(0, 1),
                r.uniform(10, 4e5),
                r.uniform(0, 0.2),
                r.uniform(0, 0.2),
            ),
        ),
        (
            _binary_search,
            lambda r: (
                r.uniform(-1e6, 1e6),
                r.uniform(1e3, 2e6),
                r.choice([-1, 1]),
                r.uniform(10, 2e4),
            ),
        ),
        (
            _decay,
            lambda r: (_probability(r), _probability(r), _probability(r), r.uniform(10, 1e4), 50),
        ),
        (
            _phase,
            lambda r: (_probability(r), _probability(r), _probability(r), 50, r.uniform(-1e5, 1e5)),
        ),
        (
            _closed_forms,
            lambda r: (_probability(r), _probability(r), _probability(r), r.uniform(0.1, 5.0)),
        ),
        (_rabi, lambda r: (r.uniform(-7, 7), r.uniform(0.5, 1.5))),
        (_ones, lambda r: ([r.randint(0, 1) for _ in range(40)],)),
    ],
    ids=[
        "adaptive_t1",
        "binary_search",
        "decay_time",
        "sparse_phase",
        "closed_forms",
        "rabi",
        "count",
    ],
)
def test_the_shipped_estimators_lower_and_agree(fn, draws):
    _agrees(fn, draws, count=600)


def test_the_adaptive_tracker_follows_its_body_through_fifty_shots():
    # The papers' run: prior (3, 450 us), assignment errors 0.11 and 0.14,
    # true T1 160 us; each shot's answer feeds the next, as on the controller.
    program = _program(_adaptive_t1)
    rng = random.Random(11)
    for _ in range(20):
        python = computed = (3.0, 450_000.0, cal.adaptive_t1_wait(3.0, 450_000.0, 0.51))
        for _ in range(50):
            p = 0.14 + (1 - 0.11 - 0.14) * math.exp(-python[2] / 160_000.0)
            bit = int(rng.random() < p)
            python = _adaptive_t1(python[0], python[1], bit, python[2], 0.11, 0.14)
            computed = controller.evaluate_function(
                program, [computed[0], computed[1], bit, computed[2], 0.11, 0.14]
            )
            assert computed == python


# ── Examples 20 and 21, and what the export holds ───────────────────────────


def _example(name):
    path = _ROOT / "examples" / name
    if not path.exists():
        pytest.skip("needs the source tree's examples/")
    return runpy.run_path(str(path), run_name="example")


def test_example_21_exports_its_estimators_and_a_backend_can_run_them():
    ex = _example("21_calibration_protocol.py")
    compiled = pulse.compile(
        ex["binary_search"], quantum_system=QS, control_system=ex["control_system"]
    )
    lowered = compiled.lower(Q[0], 2)
    exported = lowered.export_to_controller()
    assert list(exported).index("functions") == list(exported).index("manifest") + 1
    (name,) = exported["functions"]
    entry = exported["functions"][name]
    assert name == "search_step__2"
    assert entry["arguments"] == ["bits:2", "f64", "f64", "f64"]
    assert entry["results"] == ["f64"] * 4
    assert entry["number_format"] == "float64"
    json.dumps(entry, allow_nan=False)
    # A backend that answers from the export takes every branch the body does.
    found = []

    def from_export(bits, mean, sigma, wait):
        found.append(1)
        return controller.evaluate_function(entry, [bits, mean, sigma, wait])

    options = {"dt_ns": 0.25, "shots_count": 1, "seed": 3, "branches": "sampled"}
    plain = pulse.simulate(lowered, **options)
    served = pulse.simulate(lowered, **options, handlers={"search_step": from_export})
    assert found
    assert [b.answers("search_step") for b in plain.branches] == [
        b.answers("search_step") for b in served.branches
    ]


def test_example_21_lowers_every_estimator_of_its_chain():
    ex = _example("21_calibration_protocol.py")
    compiled = pulse.compile(
        ex["calibration_chain"], quantum_system=QS, control_system=ex["control_system"]
    )
    lowered = compiled.lower(Q[0], 40_000.0)
    exported = lowered.export_to_controller()
    called = {fn["name"] for fn in exported["manifest"]["functions"]}
    assert set(exported["functions"]) == called
    assert len(called) == 7
    # The structured format holds the same table, and unroll keeps it.
    structured = lowered.export_to_controller(version=2)
    assert structured["functions"] == exported["functions"]
    assert controller.unroll(structured)["functions"] == exported["functions"]


def test_usage_counts_are_the_programs_terms():
    ex = _example("21_calibration_protocol.py")
    compiled = pulse.compile(
        ex["calibration_chain"], quantum_system=QS, control_system=ex["control_system"]
    )
    lowered = compiled.lower(Q[0], 40_000.0)
    rows = {
        row.function: row
        for c in lowered.controller_usage.controllers
        for p in c.processors
        for row in p.functions
    }
    exported = lowered.export_to_controller()["functions"]
    compared = 0
    for name, entry in exported.items():
        made = lower(next(fn for fn in lowered.classical if fn.name == name))
        terms = _terms(entry)
        # Each arithmetic term and elementary function is one operation counted.
        assert made.counted == sum(op in ARITHMETIC_TERMS for op in terms)
        if "count_ones" in json.dumps(entry):
            # An intrinsic is counted by its controller_cost, not its body.
            continue
        ops = rows[name].cost.ops
        assert {k: made.classes.get(k, 0) for k in _CLASSES} == {
            k: getattr(ops, k) for k in _CLASSES
        }
        compared += 1
    assert compared >= 3


def test_example_20s_estimator_is_refused_with_its_lines():
    ex = _example("20_adaptive_calibration.py")
    estimator = ex["frequency_estimator"].specialize(bit=1)
    with pytest.raises(pulse.UnsupportedError) as error:
        lower(estimator)
    text = str(error.value)
    assert "frequency_update line" in text and "an exception" in text
    assert "ramsey_setting line" in text and "IsNot" in text
    # On a link no processor serves it runs as before, and the export has no table.
    compiled = pulse.compile(
        ex["binary_search"], quantum_system=QS, control_system=ex["control_system"]
    )
    assert "functions" not in compiled.lower(Q[0], 1).export_to_controller()


# ── Refusals ────────────────────────────────────────────────────────────────


def _hidden_index(x: list[float], k: int) -> float:
    return x[k]


@pulse.classical(max_latency_ns=100)
def _index(x: pulse.Length[list[float], 3], k: int) -> float:
    return x[k]


@pulse.classical(max_latency_ns=100)
def _tangent(x: float) -> float:
    return math.tan(x)


@pulse.classical(max_latency_ns=100)
def _power(n: int, m: int) -> int:
    return n**m


@pulse.classical(max_latency_ns=100)
def _floor_division(x: float) -> float:
    return x // 2.5


@pulse.classical(max_latency_ns=100)
def _bitwise(n: int) -> int:
    return n & 3


@pulse.classical(max_latency_ns=100)
def _while(x: float) -> float:
    while x > 1.0:
        x = x / 2.0
    return x


@pulse.classical(max_latency_ns=100)
def _one_path(x: float) -> float:
    if x > 0:
        y = x
    return y  # noqa: F821


@pulse.classical(max_latency_ns=100)
def _point(z: complex) -> float:
    return 2.0


@pytest.mark.parametrize(
    ("fn", "text"),
    [
        (_index, "_index line"),
        (_tangent, "math.tan"),
        (_power, "an integer power"),
        (_floor_division, "a floor division of floats"),
        (_bitwise, "the operator BitAnd"),
        (_while, "a while loop"),
        (_one_path, "`y` is assigned on one path"),
        (_point, "the argument z (iq:1)"),
    ],
)
def test_a_body_outside_the_subset_is_refused_with_its_line_and_construct(fn, text):
    with pytest.raises(pulse.UnsupportedError, match="can't be lowered") as error:
        lower(fn)
    assert text in str(error.value)
    assert fn.name in str(error.value)


def test_the_index_refusal_names_the_line():
    with pytest.raises(pulse.UnsupportedError) as error:
        lower(_index)
    line = _index.implementation.__code__.co_firstlineno + 2  # under its decorator
    assert f"_index line {line}: an index known only when the function runs" in str(error.value)


def _system(fixed_cost_only):
    processor = pulse.Processor(
        links=["controller"], fixed_cost_only=fixed_cost_only, max_concurrent_calls=64
    )
    return pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[processor])],
        links={"controller": Link(fixed_ns=4, feedback_ns=8)},
    )


@pulse.classical(max_latency_ns=300, link="controller")
def _tangent_on_controller(x: float) -> float:
    return math.tan(x)


@pulse.classical(max_latency_ns=300, link="controller")
def _halve(x: float) -> float:
    return x * 0.5


@pulse.kernel
def _calls_tangent(q):
    d = get_drive_line(q)
    x = _tangent_on_controller.submit(1.5).result()
    wait(d, x * pulse.ns, bounds=(10 * pulse.ns, 100 * pulse.ns))


@pulse.kernel
def _calls_halve(q):
    d = get_drive_line(q)
    x = _halve.submit(1.5).result()
    wait(d, x * pulse.ns, bounds=(10 * pulse.ns, 100 * pulse.ns))


def test_the_export_refuses_a_body_a_fixed_cost_processor_cannot_run():
    lowered = pulse.compile(_calls_tangent, quantum_system=QS, control_system=_system(True)).lower(
        Q[0]
    )
    with pytest.raises(pulse.UnsupportedError, match="math.tan"):
        lowered.export_to_controller()
    # Another processor runs what it can: the function has no entry.
    lowered = pulse.compile(_calls_tangent, quantum_system=QS, control_system=_system(False)).lower(
        Q[0]
    )
    assert "functions" not in lowered.export_to_controller()


def test_only_programs_with_such_functions_change():
    on_host = pulse.ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})
    lowered = pulse.compile(_calls_halve, quantum_system=QS, control_system=on_host).lower(Q[0])
    assert "functions" not in lowered.export_to_controller()
    lowered = pulse.compile(_calls_halve, quantum_system=QS, control_system=_system(True)).lower(
        Q[0]
    )
    exported = lowered.export_to_controller()
    assert exported["functions"]["_halve"]["body"] == {
        "let": [],
        "results": [{"op": "mul", "type": "float", "of": [{"argument": 0}, {"const": 0.5}]}],
    }


@pulse.kernel
def _two_lines(q, p):
    a = get_drive_line(q)
    b = get_drive_line(p)
    x = _halve.submit(1.5).result()
    shift_phase(a.tone, x)
    drive(a, square(20 * pulse.ns, 0.1))
    drive(b, square(20 * pulse.ns, 0.1))


def test_each_controller_exports_the_functions_its_processors_compute():
    processor = pulse.Processor(links=["controller"], fixed_cost_only=True)
    system = pulse.ControlSystem(
        controllers=[
            pulse.Controller(drive_lines=[0], processors=[processor]),
            pulse.Controller(drive_lines=[1]),
        ],
        links={"controller": Link(fixed_ns=4, feedback_ns=8)},
    )
    lowered = pulse.compile(_two_lines, quantum_system=QS, control_system=system).lower(Q[0], Q[1])
    first = lowered.export_to_controller(controller=0)
    assert set(first["functions"]) == {"_halve"}
    assert "functions" not in lowered.export_to_controller(controller=1)


def test_a_body_is_lowered_once_however_many_calls_and_exports(monkeypatch):
    import cudaq_pulse._controller_functions as functions

    @pulse.classical(max_latency_ns=300, link="controller")
    def step(x: float) -> float:
        return x * 0.5 + 1.0

    @pulse.kernel
    def many(q, n: int):
        d = get_drive_line(q)
        for _ in range(n):
            shift_phase(d.tone, step.submit(1.0).result())
            drive(d, square(20 * pulse.ns, 0.1))

    runs = []
    real = functions._lower
    monkeypatch.setattr(functions, "_lower", lambda fn: runs.append(fn) or real(fn))
    compiled = pulse.compile(many, quantum_system=QS, control_system=_system(True))
    for n in (2, 8, 32):
        exported = compiled.lower(Q[0], n).export_to_controller()
        assert len(exported["calls"]) == n
        exported = compiled.lower(Q[0], n).export_to_controller(version=2)
    assert len(runs) == 1
    # The program's size is the body's, not the calls'.
    assert len(_terms(exported["functions"]["step"])) == 2


# ── evaluate_function and the new terms ─────────────────────────────────────


def test_evaluate_function_checks_its_program_and_arguments():
    program = _program(_halve)
    assert controller.evaluate_function(program, [3.0]) == (1.5,)
    with pytest.raises(pulse.InvalidValueError, match="takes 1 arguments"):
        controller.evaluate_function(program, [])
    with pytest.raises(pulse.InvalidValueError, match="float64"):
        controller.evaluate_function({**program, "number_format": "float32"}, [3.0])
    with pytest.raises(pulse.InvalidValueError, match="only a function's body has"):
        controller.evaluate_expression({"argument": 0})


def _term(op, kind, *of):
    return {"op": op, "type": kind, "of": [{"const": v} for v in of]}


def test_the_float_terms_give_ieee_values_where_python_raises():
    evaluate = controller.evaluate_expression
    assert evaluate(_term("log", "float", 0.0)) == -math.inf
    assert math.isnan(evaluate(_term("log", "float", -1.0)))
    assert math.isnan(evaluate(_term("sqrt", "float", -1.0)))
    assert evaluate(_term("exp", "float", 1e4)) == math.inf
    assert evaluate(_term("pow", "float", -10.0, 401.0)) == -math.inf
    assert evaluate(_term("pow", "float", -0.0, -3.0)) == -math.inf
    assert math.isnan(evaluate(_term("pow", "float", -8.0, 1 / 3)))
    assert math.isnan(evaluate(_term("sin", "float", math.inf)))
    assert evaluate(_term("floor", "float", -2.5)) == -3.0
    assert evaluate(_term("ceil", "float", math.inf)) == math.inf
    assert evaluate(_term("abs", "float", -0.0)) == 0.0
    assert evaluate(_term("atan2", "float", 1.0, -1.0)) == math.atan2(1.0, -1.0)
    # Python's %: the sign of the divisor.
    assert evaluate(_term("mod", "float", -1.0, 3.0)) == 2.0
    assert evaluate(_term("mod", "int", -7, 3)) == 2
    assert math.isnan(evaluate(_term("mod", "float", 1.0, 0.0)))
    with pytest.raises(pulse.ExecutionError, match="mod of 1 by 0"):
        evaluate(_term("mod", "int", 1, 0))
    with pytest.raises(pulse.InvalidValueError, match="a term of floats"):
        evaluate(_term("exp", "int", 1))
