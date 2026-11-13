# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Links with one instance per controller (``Link(per_controller=True)``),
and the programs of controllers a program does not use."""

from __future__ import annotations

import json
import tomllib

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._errors import DeviceDefinitionError
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import resources

from tests._handles import ladder_target


def _decode(m: pulse.Length[pulse.Bits, 1]) -> float:
    if m[0]:
        return 0.1
    return 0.0


decode = pulse.classical(_decode, max_latency_ns=200, link="decoder")


@pulse.classical(max_latency_ns=40, link="decoder")
def unread(x: float) -> float:
    return x


@pulse.kernel
def two_rounds(a, b):
    ra = get_readout_line(a)
    da = get_drive_line(a)
    rb = get_readout_line(b)
    db = get_drive_line(b)
    ma = readout(ra, square(40, 0.1))
    mb = readout(rb, square(40, 0.1))
    drive(da, square(8, decode.submit([ma]).result()))
    drive(db, square(8, decode.submit([mb]).result()))


@pulse.kernel
def crossing(a, b):
    ra = get_readout_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(40, 0.1))
    drive(db, square(8, decode.submit([m]).result()))


@pulse.kernel
def with_unread_call(a, b):
    two_rounds(a, b)
    unread.submit(0.1).result()


def _controllers(processors=False):
    def processor():
        return [pulse.Processor(links=["decoder"])] if processors else []

    return [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], processors=processor()),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], processors=processor()),
        pulse.Controller(),
    ]


def _device(per_controller, processors=False, slots=1):
    link = Link(fixed_ns=10, slots=slots, per_controller=per_controller)
    system = pulse.ControlSystem(controllers=_controllers(processors), links={"decoder": link})
    return pulse.device(ladder_target(4), control_system=system)


def _lower(kernel, qpu, schedule="asap"):
    return pulse.compile(kernel, device=qpu, schedule=schedule).lower(qpu.qudits[0], qpu.qudits[2])


# ── Declaring one ─────────────────────────────────────────────────────────────


def test_a_link_is_shared_unless_declared_per_controller():
    assert Link().per_controller is False
    assert "per_controller" not in Link().cost()
    assert Link(per_controller=True).cost()["per_controller"] is True
    with pytest.raises(pulse.InvalidTypeError, match="per_controller"):
        Link(per_controller="yes")  # type: ignore[arg-type]


def test_a_system_file_states_a_per_controller_link_in_json_and_toml():
    qpu = _device(True)
    data = json.loads(json.dumps(qpu.as_dict()))
    assert data["control_system"]["links"]["decoder"]["per_controller"] is True
    again = pulse.Device.from_dict(data)
    assert again.control_system.links["decoder"].per_controller
    table = tomllib.loads(
        "[[control_system.controllers]]\ndrive_lines = [0, 1]\nreadout_lines = [0, 1]\n"
        "[[control_system.controllers]]\ndrive_lines = [2, 3]\nreadout_lines = [2, 3]\n"
        "[control_system.links.decoder]\nfixed_ns = 10\nslots = 1\n"
        "per_controller = true\n"
    )
    loaded = pulse.Device.from_dict({"quantum_system": ladder_target(4)} | table)
    assert loaded.control_system.links["decoder"] == Link(fixed_ns=10, slots=1, per_controller=True)
    shared = _device(False).as_dict()["control_system"]["links"]["decoder"]
    assert "per_controller" not in shared


# ── Slots, scheduling and exports ─────────────────────────────────────────────


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_each_controller_has_its_own_slots_of_a_per_controller_link(schedule):
    shared = _lower(two_rounds, _device(False), schedule).classical_report()
    # One slot for both controllers: the second call waits for the first.
    assert [c.controller for c in shared] == [None, None]
    assert shared[1].submit_cycles >= shared[0].due_cycles
    local = _lower(two_rounds, _device(True), schedule).classical_report()
    assert sorted(c.controller for c in local) == [0, 1]
    assert local[0].submit_cycles == local[1].submit_cycles
    assert local[0].due_cycles - local[0].submit_cycles == shared[0].due_cycles - (
        shared[0].submit_cycles
    )


def test_each_controller_exports_the_calls_of_its_own_instance():
    lowered = _lower(two_rounds, _device(True))
    for index in (0, 1):
        program = lowered.export_controller(controller=index)
        assert [c["controller"] for c in program["calls"]] == [index]
        calls = [e for e in program["events"] if e["kind"] in ("submit", "answer")]
        assert [e["controller"] for e in calls] == [index, index]
        assert all(e["qudit"] == -1 for e in calls)
    links = resources(lowered).links
    assert [(u.controller, u.peak_slots_used, u.calls_per_shot) for u in links] == [
        (0, 1, 1),
        (1, 1, 1),
    ]
    report = resources(lowered)
    assert report.violations == () and "decoder@1" in str(report)


def test_a_shared_link_exports_without_an_instance():
    program = _lower(two_rounds, _device(False)).export_controller(controller=0)
    assert all("controller" not in c for c in program["calls"])
    assert all("controller" not in e for e in program["events"])
    assert [u.controller for u in resources(_lower(two_rounds, _device(False))).links] == [None]


def test_a_call_over_a_per_controller_link_runs_on_one_controller():
    with pytest.raises(RuntimeError, match=r"over link 'decoder', which has one instance per "):
        _lower(crossing, _device(True))
    with pytest.raises(RuntimeError, match=r"controllers\[0\] and controllers\[1\].*drop "):
        _lower(crossing, _device(True))
    # Shared, the same call crosses the link between them.
    assert _lower(crossing, _device(False)).classical_report()[0].controller is None


def test_a_call_whose_controller_is_not_known_keeps_the_link_shared():
    report = _lower(with_unread_call, _device(True)).classical_report()
    assert [c.controller for c in report] == [None, None, None]
    shared = _lower(with_unread_call, _device(False)).classical_report()
    assert [c.submit_cycles for c in report] == [c.submit_cycles for c in shared]
    # No call crosses between controllers, so none waits out their alignment.
    assert all(a.due_cycles <= b.due_cycles for a, b in zip(report, shared, strict=True))
    calls = _lower(with_unread_call, _device(True)).export_controller(controller=1)["calls"]
    assert [c["function"] for c in calls] == ["_decode", "_decode"]


def test_one_controller_schedules_a_per_controller_link_as_a_shared_one():
    single = [pulse.Controller(drive_lines=range(4), readout_lines=range(4))]
    reports = []
    for per in (False, True):
        link = Link(fixed_ns=10, slots=1, per_controller=per)
        system = pulse.ControlSystem(controllers=single, links={"decoder": link})
        qpu = pulse.device(ladder_target(4), control_system=system)
        reports.append(
            [(c.submit_cycles, c.due_cycles) for c in _lower(two_rounds, qpu).classical_report()]
        )
    assert reports[0] == reports[1]


# ── Processors ────────────────────────────────────────────────────────────────


def test_each_controller_places_a_per_controller_link_on_its_own_processor():
    with pytest.raises(DeviceDefinitionError, match="is also a link of controllers"):
        _device(False, processors=True)
    lowered = _lower(two_rounds, _device(True, processors=True))
    usage = lowered.controller_usage
    assert [row.controller for row in usage.controllers] == [0, 1]
    for row in usage.controllers:
        (processor,) = row.processors
        assert [(f.function, f.calls_per_shot) for f in processor.functions] == [("_decode", 1)]
    assert usage.within_limits and usage.off_board == ()


def test_two_processors_of_one_controller_cannot_both_hold_its_instance():
    two = pulse.Controller(
        drive_lines=[0, 1],
        processors=[pulse.Processor(links=["decoder"]), pulse.Processor(links=["decoder"])],
    )
    with pytest.raises(DeviceDefinitionError, match="per controller"):
        pulse.ControlSystem(controllers=[two], links={"decoder": Link(per_controller=True)})


# ── Controllers a program does not use ────────────────────────────────────────


@pulse.classical(max_latency_ns=40, link="controller")
def gain(x: float) -> float:
    return x


@pulse.kernel
def driven(q):
    d = get_drive_line(q)
    drive(d, square(8, gain.submit(0.1).result()))


@pulse.kernel
def calls_only(q):
    gain.submit(0.1).result()


def _three():
    system = pulse.ControlSystem(controllers=_controllers(), links={"controller": Link()})
    return pulse.device(ladder_target(4), control_system=system)


def test_a_controller_the_program_does_not_use_exports_an_empty_program():
    qpu = _three()
    lowered = pulse.compile(driven, device=qpu).lower(qpu.qudits[0])
    used = lowered.export_controller(controller=0)
    for index in (1, 2):
        program = lowered.export_controller(controller=index)
        assert program.keys() == used.keys()
        assert program["lines"] == program["events"] == program["calls"] == []
        assert program["qudits"] == [] and program["schedule"] == used["schedule"]
        json.dumps(program)
        emulation = lowered.export_emulation(controller=index)
        assert emulation["calls"] == [] and emulation["events"] == []


def test_a_program_of_calls_only_is_exported_by_the_controller_making_them():
    qpu = _three()
    lowered = pulse.compile(calls_only, device=qpu).lower(qpu.qudits[0])
    program = lowered.export_controller()
    assert [c["function"] for c in program["calls"]] == ["gain"]
    assert [e["kind"] for e in program["events"]] == ["submit", "answer"]
    assert lowered.export_controller(controller=1)["calls"] == []
