# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Processors: the classical compute of a controller, and what a program
uses of it (``Lowered.controller_usage``)."""

from __future__ import annotations

import json

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._controller_cost import analyze_callable
from cudaq_pulse._errors import DeviceDefinitionError
from cudaq_pulse.classical import Bits, Float
from cudaq_pulse.controller import ControllerLimitError

KRINNER = "transmon_krinner_17q"


def _parity(bits):
    acc = 0
    for i in range(4):
        acc = acc ^ bits[i]
    return [acc]


def _twice(x):
    return x + x


def _scaled(x):
    # A divide: by a constant it would be a multiply.
    return _twice(x) / x


def _sign(x):
    if x > 0.0:
        return 1.0
    return -1.0


parity = pulse.classical(args=(Bits(4),), results=(Bits(1),), max_latency_ns=60, link="cores")(
    _parity
)
scaled = pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=400, link="cores")(
    _scaled
)
sign = pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=400, link="gpu")(_sign)


@pulse.kernel
def corrected(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    fix = parity.submit([m, m, m, m]).result()
    sync(r, d)
    if fix[0]:
        drive(d, square(20, 0.1), t)


@pulse.kernel
def both(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    fix = parity.submit([m, m, m, m]).result()
    scaled.submit(1.0).result()
    sign.submit(1.0).result()
    sync(r, d)
    if fix[0]:
        drive(d, square(20, 0.1), t)


def _qpu(*processors, model="box.a", links=("cores", "gpu")):
    box = pulse.Controller(
        drive_lines=[0, 1], readout_lines=[0, 1], processors=list(processors), model=model
    )
    system = pulse.ControlSystem(controllers=[box], links={name: pulse.Link() for name in links})
    return pulse.device({"quantum_system": KRINNER, "layout": [0, 1], "control_system": system})


def _lower(kernel, *processors):
    qpu = _qpu(*processors)
    return pulse.compile(kernel, device=qpu).lower(qpu.qudits[0])


def _fails(kernel, *processors):
    with pytest.raises(ControllerLimitError) as raised:
        _lower(kernel, *processors)
    return str(raised.value)


# ── Definitions ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "error", "text"),
    [
        ({"links": "cores"}, pulse.InvalidTypeError, "list of link names or ids"),
        ({"links": ["co res"]}, pulse.InvalidValueError, "names a link"),
        ({"links": [-1]}, pulse.InvalidTypeError, "ids >= 0"),
        ({"links": ["a", "a"]}, pulse.InvalidValueError, "twice"),
        ({"clock_ghz": 0}, pulse.InvalidValueError, "clock_ghz must be a positive number"),
        ({"cycles_per_op": {"dvi": 2}}, pulse.InvalidValueError, "did you mean 'div'"),
        ({"cycles_per_op": {"div": -1}}, pulse.InvalidValueError, "whole number of cycles"),
        ({"cycles_per_op": {"div": True}}, pulse.InvalidValueError, "whole number of cycles"),
        ({"unsupported": "div"}, pulse.InvalidTypeError, "list of operations"),
        ({"unsupported": ["lop"]}, pulse.InvalidValueError, "did you mean 'loop'"),
        (
            {"unsupported": ["div"], "cycles_per_op": {"div": 9}},
            pulse.InvalidValueError,
            "supported or not",
        ),
        ({"max_concurrent_calls": 0}, pulse.InvalidValueError, "max_concurrent_calls"),
        ({"max_ops_total": 0}, pulse.InvalidValueError, "max_ops_total"),
        ({"max_state_bytes_per_function": 1.5}, pulse.InvalidValueError, "whole number"),
        ({"fixed_cost_only": 1}, pulse.InvalidTypeError, "True or False"),
        ({"dispatch_keys": {"f": -1}}, pulse.InvalidValueError, "0 to 0xFFFFFFFF"),
        ({"dispatch_keys": {"f": 2**32}}, pulse.InvalidValueError, "0 to 0xFFFFFFFF"),
        ({"dispatch_keys": {"f": True}}, pulse.InvalidValueError, "0 to 0xFFFFFFFF"),
        ({"dispatch_keys": {"a b": 1}}, pulse.InvalidValueError, "keyed by function name"),
        ({"dispatch_keys": {"f": 1, "g": 1}}, pulse.InvalidValueError, "both 'f' and 'g'"),
    ],
)
def test_a_processor_checks_its_fields(fields, error, text):
    with pytest.raises(error, match=text):
        pulse.Processor(**fields)


def test_a_processor_is_frozen_hashable_and_reads_back_its_dict():
    cores = pulse.Processor(
        links=["cores"],
        cycles_per_op={"div": 20, "loop": 2},
        unsupported=["trig", "trig"],
        dispatch_keys={"_parity": 7},
    )
    assert cores.links == ("cores",) and cores.unsupported == ("trig",) and cores.loops
    assert not pulse.Processor(unsupported=["loop"]).loops
    data = json.loads(json.dumps(cores.as_dict()))
    assert pulse.Processor(**data) == cores
    assert hash(pulse.Processor(**data)) == hash(cores)
    with pytest.raises(TypeError):
        cores.dispatch_keys["other"] = 1  # type: ignore[index]


def test_a_controller_computes_on_its_processors_or_with_its_own_fields_not_both():
    cores = pulse.Processor(links=["cores"])
    box = pulse.Controller(processors=[cores, {"links": ["gpu"]}])
    assert box.processors == (cores, pulse.Processor(links=("gpu",)))
    with pytest.raises(pulse.InvalidValueError, match="move its compute_links"):
        pulse.Controller(processors=[cores], max_ops=10)
    with pytest.raises(pulse.InvalidValueError, match="move its compute_links"):
        pulse.Controller(processors=[cores], compute_links=("cores",))
    with pytest.raises(pulse.InvalidTypeError, match="list of pulse.Processor"):
        pulse.Controller(processors=cores)  # type: ignore[arg-type]
    with pytest.raises(pulse.InvalidValueError, match=r"processors\[0\]: .*'link'"):
        pulse.Controller(processors=[{"link": ["cores"]}])
    # A controller without processors writes what it did before.
    assert "processors" not in pulse.Controller().as_dict()


def test_a_link_belongs_to_one_processor_of_the_system():
    a = pulse.Controller(drive_lines=[0], processors=[pulse.Processor(links=["cores"])])
    b = pulse.Controller(drive_lines=[1], processors=[pulse.Processor(links=[0])])
    with pytest.raises(DeviceDefinitionError) as raised:
        pulse.ControlSystem(controllers=[a, b], links={"cores": pulse.Link()})
    assert "controllers[1].processors[0].links[0]: is also a link of controllers[0]" in str(
        raised.value
    )
    c = pulse.Controller(processors=[pulse.Processor(links=["coers"])])
    with pytest.raises(DeviceDefinitionError, match="did you mean 'cores'"):
        pulse.ControlSystem(controllers=[c], links={"cores": pulse.Link()})


# ── What a function costs on a processor ─────────────────────────────────────


def test_a_body_records_its_loops_unrolled_size_and_where_it_uses_each_operation():
    cost = analyze_callable(_parity, {"bits": 4})
    # Rolled: the xor and the load, then the counter and the jump.
    assert cost.size == 4 and cost.iterations == 4 and cost.unrolled_size == 8
    assert cost.ops == {"add": 8, "branch": 4, "load": 4}
    line = _parity.__code__.co_firstlineno
    assert cost.uses == {"add": f"_parity line {line + 3}", "load": f"_parity line {line + 3}"} | {
        "loop": f"_parity line {line + 2}"
    }
    # An inlined call is not a branch, and a helper's uses are its own lines.
    cost = analyze_callable(_scaled, {})
    assert set(cost.uses) == {"add", "div"}
    assert cost.uses["add"].startswith("_twice line")
    assert analyze_callable(_sign, {}).uses["branch"].startswith("_sign line")


def test_a_processor_prices_a_loop_by_its_loop_op_or_runs_it_unrolled():
    cost = analyze_callable(_parity, {"bits": 4})
    # Rolled: 4 xors, 4 loads and 4 iterations.
    assert cost.processor_cycles({}, loops=True) == 12
    assert cost.processor_cycles({"loop": 3}, loops=True) == 20
    assert cost.processor_cycles({"loop": 3}, loops=False) == 8
    assert cost.processor_cycles({"add": 2}, loops=True) == 16
