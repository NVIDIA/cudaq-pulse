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
from cudaq_pulse.controller import ControllerLimitError

TRANSMON_17Q = "transmon_17q"


def _parity(bits: pulse.Length[pulse.Bits, 4]) -> pulse.Bits:
    acc = 0
    for i in range(4):
        acc = acc ^ bits[i]
    return [acc]


def _twice(x):
    return x + x


def _scaled(x: float) -> float:
    # A divide: by a constant it would be a multiply.
    return _twice(x) / x


def _sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    return -1.0


parity = pulse.classical(_parity, max_latency_ns=60, link="cores")
scaled = pulse.classical(_scaled, max_latency_ns=400, link="cores")
sign = pulse.classical(_sign, max_latency_ns=400, link="gpu")


@pulse.kernel
def corrected(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    fix = parity.submit([m, m, m, m]).result()
    sync(r, d)
    if fix[0]:
        drive(d, square(20, 0.1))


@pulse.kernel
def both(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    fix = parity.submit([m, m, m, m]).result()
    scaled.submit(1.0).result()
    sign.submit(1.0).result()
    sync(r, d)
    if fix[0]:
        drive(d, square(20, 0.1))


def _qpu(*processors, model="box.a", links=("cores", "gpu")):
    box = pulse.Controller(
        drive_lines=[0, 1], readout_lines=[0, 1], processors=list(processors), model=model
    )
    system = pulse.ControlSystem(controllers=[box], links={name: pulse.Link() for name in links})
    return pulse.device(
        {"quantum_system": TRANSMON_17Q, "layout": [0, 1], "control_system": system}
    )


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
    with pytest.raises(pulse.InvalidValueError, match="dispatch_keys to a pulse.Processor"):
        pulse.Controller(processors=[cores], dispatch_keys={"_parity": 7})
    # A controller without processors writes what it did before.
    assert {"processors", "dispatch_keys"}.isdisjoint(pulse.Controller().as_dict())


def test_a_controller_computing_on_its_own_fields_states_its_dispatch_keys(tmp_path):
    box = pulse.Controller(compute_links=("cores",), max_ops=40, dispatch_keys={"_parity": 7})
    assert box.dispatch_keys == {"_parity": 7}
    assert pulse.Controller(**json.loads(json.dumps(box.as_dict()))) == box
    with pytest.raises(pulse.InvalidValueError, match="gives 7 to both '_a' and '_b'"):
        pulse.Controller(dispatch_keys={"_a": 7, "_b": 7})
    with pytest.raises(pulse.InvalidValueError, match=r"Controller.dispatch_keys\['_a'\] is -1"):
        pulse.Controller(dispatch_keys={"_a": -1})
    path = tmp_path / "box.toml"
    path.write_text(
        'model = "acme.box"\nversion = "1"\ncompute_links = ["cores"]\n\n'
        "[dispatch_keys]\n_parity = 7\n"
    )
    loaded = pulse.controller(str(path), drive_lines=[0])
    assert loaded.dispatch_keys == {"_parity": 7} and loaded.model == "acme.box"


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


# ── Checks ───────────────────────────────────────────────────────────────────


def test_a_function_that_fits_its_processor_compiles_and_is_reported():
    cores = pulse.Processor(links=["cores"], clock_ghz=1.0, max_ops_per_function=8)
    usage = _lower(corrected, cores).controller_usage
    assert usage.within_limits and usage.off_board == ()
    (row,) = usage.controllers
    (processor,) = row.processors
    assert (row.controller, row.model, processor.index, processor.clock_ghz) == (0, "box.a", 0, 1.0)
    (function,) = processor.functions
    assert (function.function, function.ops, function.unrolled) == ("_parity", 4, False)
    assert function.compute_cycles == 12 and function.compute_ns == 12.0
    assert function.slack_ns == 48.0 and function.calls_per_shot == 1
    assert processor.ops == 4 and processor.calls_per_shot == 1
    assert processor.bytes_per_shot == function.request_bytes + function.answer_bytes
    text = str(usage)
    assert text.startswith("Controller usage: within limits\n")
    assert "controller 0 (box.a), processor 0 at 1 GHz (links: cores)" in text
    assert "_parity" in text and "4 / 8" in text and "12 / 60 ns" in text


def test_a_processor_without_loop_holds_the_unrolled_body_and_says_so():
    cores = pulse.Processor(links=["cores"], unsupported=["loop"], max_ops_per_function=6)
    text = _fails(corrected, cores)
    assert (
        "_parity holds 8 operations with its loops unrolled (processors[0] of controllers[0] "
        "(box.a) does not support 'loop'); processors[0] of controllers[0] (box.a) holds 6 per "
        "function (max_ops_per_function)"
    ) in text
    fits = pulse.Processor(links=["cores"], unsupported=["loop"])
    (function,) = _lower(corrected, fits).controller_usage.controllers[0].processors[0].functions
    assert (function.ops, function.unrolled, function.compute_cycles) == (8, True, 8)


def test_an_operation_a_processor_does_not_support_is_named_with_its_line():
    cores = pulse.Processor(links=["cores"], unsupported=["load", "div"], max_concurrent_calls=2)
    text = _fails(both, cores)
    line = _parity.__code__.co_firstlineno + 3
    assert f"_parity uses load (_parity line {line}), which processors[0]" in text
    assert "_scaled uses div (_scaled line" in text
    # Branches are if and else only: an inlined call needs none.
    _lower(corrected, pulse.Processor(links=["cores"], unsupported=["branch"]))


def test_totals_and_per_shot_limits_count_every_function_of_a_processor():
    cores = pulse.Processor(
        links=["cores"],
        max_concurrent_calls=2,
        max_ops_total=6,
        max_state_bytes_total=20,
        max_calls_per_shot=1,
        max_bytes_per_shot=100,
    )
    text = _fails(both, cores)
    label = "processors[0] of controllers[0] (box.a)"
    assert f"{label}: its classical functions hold 7 operations; it holds 6 in all" in text
    assert "(max_ops_total)" in text and "(max_state_bytes_total)" in text
    assert f"{label}: one shot makes 2 calls to it; it takes 1 (max_calls_per_shot)" in text
    assert "(max_bytes_per_shot)" in text
    # Each limit left out is no limit: the program fits.
    fits = _lower(both, pulse.Processor(links=["cores"], max_concurrent_calls=2))
    assert fits.controller_usage.controllers[0].processors[0].ops == 7


def test_the_error_carries_the_usage_report_with_by_how_much_each_limit_is_exceeded():
    cores = pulse.Processor(
        links=["cores"],
        clock_ghz=0.1,
        max_concurrent_calls=2,
        max_ops_total=6,
        max_calls_per_shot=1,
    )
    with pytest.raises(ControllerLimitError) as raised:
        _lower(both, cores)
    error = raised.value
    report = error.usage
    assert report is not None and not report.within_limits
    assert report.controllers[0].processors[0].ops == 7
    label = "processors[0] of controllers[0] (box.a)"
    exceeded = {(e.where, e.limit): (e.value, e.most, e.over, e.unit) for e in report.exceeded}
    assert exceeded[(label, "max_ops_total")] == (7, 6, 1, "operations")
    assert exceeded[(label, "max_calls_per_shot")] == (2, 1, 1, "calls")
    value, most, over, unit = exceeded[("_parity", "max_latency_ns")]
    assert (most, unit) == (60.0, "ns") and over == value - 60.0 > 0
    assert len(report.exceeded) == len(report.violations)
    assert error.violations == report.violations
    text = str(error)
    assert "(max_ops_total): 1 operations over" in text
    assert (
        f"{label}: one shot makes 2 calls to it; it takes 1 (max_calls_per_shot): 1 calls over"
        in text
    )
    assert text.endswith(str(report).split("\nExceeds:")[0])
    assert text.count("max_ops_total") == 1


def test_a_call_over_a_processor_limit_carries_the_report_of_its_values():
    @pulse.kernel
    def waits(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        fix = parity.submit([m, m, m, m]).result()
        wait(d, tau)
        sync(r, d)
        if fix[0]:
            drive(d, square(20, 0.1))

    qpu = _qpu(pulse.Processor(links=["cores"], max_ops_per_function=2))
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(waits, device=qpu).lower(qpu.qudits[0], 8.0)
    (exceeded,) = raised.value.usage.exceeded
    assert (exceeded.where, exceeded.limit, exceeded.over) == ("_parity", "max_ops_per_function", 2)
    # Limits of a system without processors have no usage report.
    box = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], max_pulses=1)
    plain = pulse.device(
        {
            "quantum_system": TRANSMON_17Q,
            "layout": [0, 1],
            "control_system": pulse.ControlSystem(controllers=[box], links={"cores": pulse.Link()}),
        }
    )
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(corrected, device=plain).lower(plain.qudits[0])
    assert raised.value.usage is None


def test_calls_computing_at_once_are_held_to_max_concurrent_calls():
    text = _fails(both, pulse.Processor(links=["cores"]))
    assert "2 calls compute on it at once; it computes 1 (max_concurrent_calls)" in text
    assert "at once 2 / 2" in str(
        _lower(both, pulse.Processor(links=["cores"], max_concurrent_calls=2)).controller_usage
    )


def test_a_slow_processor_misses_a_function_deadline():
    slow = pulse.Processor(links=["cores"], clock_ghz=0.1, cycles_per_op={"load": 4})
    text = _fails(corrected, slow)
    assert "_parity computes for about 240 ns on processors[0]" in text
    assert "(24 cycles on its longest path), beyond its max_latency_ns=60" in text


def test_a_fixed_cost_processor_rejects_a_body_whose_cost_is_not_fixed():
    def unbounded(x: float) -> float:
        while x > 1.0:
            x = x / 2.0
        return x

    loose = pulse.classical(unbounded, max_latency_ns=400, link="cores")

    @pulse.kernel
    def calls(q):
        d = get_drive_line(q)
        loose.submit(3.0).result()
        drive(d, square(20, 0.1))

    cores = pulse.Processor(links=["cores"], fixed_cost_only=True)
    assert "unbounded is not fixed-cost, which processors[0]" in _fails(calls, cores)
    assert "a while loop" in _fails(calls, cores)
    _lower(calls, pulse.Processor(links=["cores"]))


def test_functions_on_no_processor_and_externals_run_off_board():
    @pulse.classical.external(max_latency_ns=400, link="cores")
    def table(x: float) -> float:
        return float()

    @pulse.kernel
    def calls(q):
        d = get_drive_line(q)
        sign.submit(1.0).result()
        table.submit(1.0).result()
        drive(d, square(20, 0.1))

    usage = _lower(calls, pulse.Processor(links=["cores"])).controller_usage
    assert usage.off_board == ("_sign", "table")
    assert usage.controllers[0].processors[0].functions == ()
    assert "Off the processors: _sign, table" in str(usage)


def test_a_float_parameter_defers_the_processor_check_to_the_call():
    @pulse.kernel
    def waits(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        fix = parity.submit([m, m, m, m]).result()
        wait(d, tau)
        sync(r, d)
        if fix[0]:
            drive(d, square(20, 0.1))

    qpu = _qpu(pulse.Processor(links=["cores"], max_ops_per_function=2))
    compiled = pulse.compile(waits, device=qpu)
    with pytest.raises(ControllerLimitError, match="_parity holds 4 operations"):
        compiled.lower(qpu.qudits[0], 8.0)


def test_a_system_without_processors_reports_every_function_off_board():
    qpu = pulse.device(
        {
            "quantum_system": TRANSMON_17Q,
            "layout": [0, 1],
            "control_system": pulse.ControlSystem(
                controllers=[pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])],
                links={"cores": pulse.Link()},
            ),
        }
    )
    usage = pulse.compile(corrected, device=qpu).lower(qpu.qudits[0]).controller_usage
    assert usage.controllers == () and usage.off_board == ("_parity",) and usage.within_limits


# ── Descriptions and system files ────────────────────────────────────────────


def test_a_description_states_its_processors_and_their_dispatch_keys(tmp_path):
    path = tmp_path / "cores.toml"
    path.write_text(
        'model = "acme.cores"\nversion = "1"\n\n'
        '[[processors]]\nlinks = ["cores"]\nclock_ghz = 0.5\nunsupported = ["loop", "trig"]\n'
        "max_concurrent_calls = 2\n"
        "[processors.cycles_per_op]\ndiv = 20\n"
        "[processors.dispatch_keys]\n_parity = 7\n_scaled = 8\n\n"
        '[[processors]]\nlinks = ["gpu"]\n'
    )
    box = pulse.controller(str(path), drive_lines=[0, 1], readout_lines=[0, 1])
    cores, gpu = box.processors
    assert cores.dispatch_keys == {"_parity": 7, "_scaled": 8}
    assert cores.cycles_per_op == {"div": 20}
    assert cores.unsupported == ("loop", "trig") and gpu.links == ("gpu",)
    system = pulse.ControlSystem(
        controllers=[box], links={"cores": pulse.Link(), "gpu": pulse.Link()}
    )
    qpu = pulse.device({"quantum_system": TRANSMON_17Q, "layout": [0, 1], "control_system": system})
    data = json.loads(json.dumps(qpu.as_dict()))
    assert data["control_system"]["controllers"][0]["processors"][0]["dispatch_keys"] == {
        "_parity": 7,
        "_scaled": 8,
    }
    again = pulse.device(data)
    assert again.control_system.controllers[0].processors == box.processors
    lowered = pulse.compile(both, device=again).lower(again.qudits[0])
    assert lowered.controller_usage.controllers[0].processors[0].clock_ghz == 0.5
    keys = {c.compiled_name: c.dispatch_key for c in lowered.classical_report()}
    assert (keys["_parity"], keys["_scaled"]) == (7, 8)


def test_a_bad_processor_in_a_system_file_is_a_problem_with_its_path():
    system = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[pulse.Processor(links=["cores"])])],
        links={"cores": pulse.Link()},
    )
    qpu = pulse.device({"quantum_system": TRANSMON_17Q, "layout": [0, 1], "control_system": system})
    data = qpu.as_dict()
    data["control_system"]["controllers"][0]["processors"][0]["unsupported"] = ["dvi"]
    with pytest.raises(DeviceDefinitionError, match="did you mean 'div'"):
        pulse.device(data)
