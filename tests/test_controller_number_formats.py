# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The arithmetic a controller computes a classical function in: a
function's ``number_format``, ``controller.evaluate_function`` in float32
and fixed point, and ``simulate(..., controller_arithmetic=True)``, which
answers calls with the exported programs."""

from __future__ import annotations

import math
import random

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import controller
from cudaq_pulse._controller_functions import lower
from cudaq_pulse.classical import Link
from cudaq_pulse.quantum_systems import transmon_surface17

QS = transmon_surface17()
Q = QS.qudits
_ULP = 2.0**-28  # fixed:4.28


@pytest.mark.parametrize("text", ["float16", "fixed:0.8", "fixed:4.50", "fixed:4", "fixed:-1.3", 7])
def test_a_number_format_is_checked_where_it_is_declared(text):
    with pytest.raises(pulse.InvalidValueError, match="number_format must be"):

        @pulse.classical(max_latency_ns=100, number_format=text)
        def f(x: float) -> float:
            return x


@pulse.classical(max_latency_ns=100, number_format="float32")
def _single(x: float, y: float, z: float, w: float) -> float:
    return (x * y + z) / w - math.sqrt(abs(x)) * 0.1


def test_float32_rounds_every_float_as_binary32_does():
    program = lower(_single).entry
    assert program["number_format"] == "float32"
    rng = random.Random(3)
    f32 = np.float32
    for _ in range(500):
        x, y, z, w = (rng.uniform(-1e3, 1e3) for _ in range(4))
        a, b, c, d = f32(x), f32(y), f32(z), f32(w)
        want = (a * b + c) / d - np.sqrt(np.abs(a)) * f32(0.1)
        (got,) = controller.evaluate_function(program, [x, y, z, w])
        assert got == float(want)
    (big,) = controller.evaluate_function(program, [1e30, 1e30, 0.0, 1e-30])
    assert big == math.inf


@pulse.classical(max_latency_ns=100, number_format="fixed:4.28")
def _fixed(x: float, y: float) -> tuple[float, float, float, float]:
    return x + y, x * y, x / y, math.log(x)


def test_fixed_point_rounds_to_even_and_saturates():
    program = lower(_fixed).entry
    largest = 8.0 - _ULP

    def run(x, y):
        return controller.evaluate_function(program, [x, y])

    # Arguments round to the grid, ties to even.
    assert run(1.5 * _ULP, 0.0)[0] == 2 * _ULP
    assert run(2.5 * _ULP, 0.0)[0] == 2 * _ULP
    # Saturation at both ends, and of infinities.
    assert run(7.5, 0.75)[0] == largest
    assert run(-7.5, -0.75)[0] == -8.0
    assert run(100.0, 1.0)[0] == largest
    # A product is exact, then rounded: 3 ulp * 0.5 is 1.5 ulp, to even 2 ulp.
    assert run(3 * _ULP, 0.5)[1] == 2 * _ULP
    assert run(4.0, 4.0)[1] == largest
    # A division by zero saturates; 0 / 0 is 0. log(0) saturates, log(-1) is 0.
    assert run(1.0, 0.0)[2] == largest
    assert run(-1.0, 0.0)[2] == -8.0
    assert run(0.0, 0.0)[2] == 0.0
    assert run(0.0, 1.0)[3] == -8.0
    assert run(-1.0, 1.0)[3] == 0.0
    assert run(2.0, 3.0)[2] == round(2 / 3 / _ULP) * _ULP
    assert run(2.0, 3.0)[3] == round(math.log(2.0) / _ULP) * _ULP


@pulse.classical(max_latency_ns=100, number_format="fixed:4.28")
def _tracker_fixed(survived: int, p: float, wait: float) -> tuple[float, float]:
    p = p + 0.02 * (survived - p)
    return p, -wait / math.log(p)


@pulse.classical(max_latency_ns=100)
def _tracker(survived: int, p: float, wait: float) -> tuple[float, float]:
    p = p + 0.02 * (survived - p)
    return p, -wait / math.log(p)


def test_a_fixed_point_t1_tracker_stays_near_float64_over_ten_thousand_updates():
    # Survival after a wait of 0.5 (in units of 10 us) on a qubit of T1 = 2.
    fixed, exact = lower(_tracker_fixed).entry, lower(_tracker).entry
    rng = random.Random(7)
    p_fixed = p_exact = p_body = 0.5
    worst = 0.0
    for _ in range(10_000):
        survived = int(rng.random() < math.exp(-0.5 / 2.0))
        p_fixed, t1_fixed = controller.evaluate_function(fixed, [survived, p_fixed, 0.5])
        p_exact, t1_exact = controller.evaluate_function(exact, [survived, p_exact, 0.5])
        p_body, _ = _tracker(survived, p_body, 0.5)
        assert p_exact == p_body
        worst = max(worst, abs(p_fixed - p_exact))
        assert p_fixed == round(p_fixed / _ULP) * _ULP
    # The rounding shows, and stays within a few ulp per update over the run
    # (the update contracts the error by 0.98 each time).
    assert 0 < worst < 100 * _ULP
    assert abs(t1_fixed - t1_exact) < 1e-5
    assert 1.4 < t1_exact < 3.2


def test_the_float64_program_is_the_python_body():
    program = lower(_tracker).entry
    rng = random.Random(1)
    for _ in range(200):
        args = [rng.randint(0, 1), rng.uniform(0.05, 0.95), rng.uniform(0.1, 2.0)]
        assert controller.evaluate_function(program, args) == _tracker(*args)


def test_a_version_keeps_its_templates_number_format():
    @pulse.classical(max_latency_ns=100, number_format="fixed:8.16")
    def count(bits: pulse.Bits) -> float:
        total = 0.0
        for b in bits:
            total += b * 0.5
        return total

    version = count.specialize(bits=3)
    assert version.number_format == "fixed:8.16"
    assert lower(version).entry["number_format"] == "fixed:8.16"
    assert controller.evaluate_function(lower(version).entry, [[1, 0, 1]]) == (1.0,)


# ── Simulating with the controller's arithmetic ─────────────────────────────


@pulse.classical(max_latency_ns=300, link="controller", number_format="fixed:4.12")
def _coarse(bits: pulse.Length[pulse.Bits, 1], x: float) -> float:
    return x / 3.0 + bits[0] * 0.1


@pulse.kernel
def _shot(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(20 * pulse.ns, math.pi / 40))
    sync(d, r)
    m = readout(r, square(200 * pulse.ns, 0.05))
    sync(d, r)
    shift_phase(d.tone, _coarse.submit([m], 1.0).result())
    drive(d, square(20 * pulse.ns, 0.1))


def _system():
    processor = pulse.Processor(links=["controller"], fixed_cost_only=True)
    return pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[processor])],
        links={"controller": Link(fixed_ns=4, feedback_ns=8)},
    )


def test_simulate_can_answer_with_the_controllers_arithmetic():
    lowered = pulse.compile(_shot, quantum_system=QS, control_system=_system()).lower(Q[0])
    entry = lowered.export_to_controller()["functions"]["_coarse"]
    assert entry["number_format"] == "fixed:4.12"
    body = pulse.simulate(lowered, dt_ns=0.25)
    computed = pulse.simulate(lowered, dt_ns=0.25, controller_arithmetic=True)
    for plain, fixed in zip(body.branches, computed.branches, strict=True):
        bit = int(plain.levels[0] != 0)
        (answer,) = plain.answers("_coarse")
        (rounded,) = fixed.answers("_coarse")
        assert answer == 1.0 / 3.0 + bit * 0.1
        assert rounded == controller.evaluate_function(entry, [[bit], 1.0])[0]
        assert rounded != answer and abs(rounded - answer) <= 2.0**-12
    # A handler given still answers.
    given = pulse.simulate(
        lowered, dt_ns=0.25, controller_arithmetic=True, handlers={"_coarse": lambda b, x: 0.25}
    )
    assert {b.answers("_coarse")[0] for b in given.branches} == {0.25}


def test_controller_arithmetic_needs_a_function_a_processor_computes():
    on_host = pulse.ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})
    lowered = pulse.compile(_shot, quantum_system=QS, control_system=on_host).lower(Q[0])
    with pytest.raises(pulse.UnsupportedError, match="no processor"):
        pulse.simulate(lowered, dt_ns=0.25, controller_arithmetic=True)
