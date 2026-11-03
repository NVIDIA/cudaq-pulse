# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Control systems: controllers, their wiring, links and system files."""

from __future__ import annotations

import io
import json
import tomllib
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._errors import (
    DeviceDefinitionError,
    InvalidValueError,
    UnsupportedError,
    UpgradeNotice,
)
from cudaq_pulse._system import installed_descriptions
from cudaq_pulse.classical import Link
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits
from cudaq_pulse.targets.base import (
    Coupling,
    DispersiveReadout,
    MicrowaveDrive,
    Qubit,
    TargetDescription,
)
from cudaq_pulse.targets.models import SimulationModel

TRANSMON_17Q = "transmon_17q"


@pulse.kernel
def drive_and_branch(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    d1 = get_drive_line(r)
    drive(d, gaussian(16, 0.5, 4.0))
    sync(d, ro)
    m = readout(ro, square(100, 0.05))
    sync(ro, d1)
    if m:
        drive(d1, square(20, 0.3))


@pulse.classical(max_latency_ns=400, link="feedback")
def flip(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def corrected(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    m = readout(ro, square(100, 0.05))
    fix = flip.submit([m]).result()
    sync(ro, d)
    if fix[0]:
        drive(d, square(20, 0.3))


def _qpu(*controllers, links=None, layout=(0, 1, 2, 3), **config):
    system = pulse.ControlSystem(controllers=list(controllers), links=links or {})
    return pulse.device(
        {"quantum_system": TRANSMON_17Q, "layout": list(layout), "control_system": system, **config}
    )


def _two_boxes(**second):
    return (
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], model="box.a"),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], model="box.b", **second),
    )


# ── Definitions ──────────────────────────────────────────────────────────────


def test_a_controller_checks_its_wiring_fields():
    with pytest.raises(TypeError, match="carrier index"):
        pulse.Controller(drive_lines=["q0"])
    with pytest.raises(ValueError, match="twice"):
        pulse.Controller(readout_lines=[1, 1])
    with pytest.raises(ValueError, match="did you mean 'acquire'"):
        pulse.Controller(supports=["play", "aquire"])
    with pytest.raises(ValueError, match="negative"):
        pulse.Controller(clock_skew_cycles=-1)
    box = pulse.Controller(drive_lines=range(3), supports=["play", pulse.Capability.SYNC])
    assert box.drive_lines == (0, 1, 2)
    assert box.supports == (pulse.Capability.PLAY, pulse.Capability.SYNC)
    assert box.supports_capability(pulse.Capability.SYNC)
    assert not box.supports_capability(pulse.Capability.BRANCH)
    assert pulse.Controller().supports_capability(pulse.Capability.BRANCH)
    assert hash(box) == hash(pulse.Controller(drive_lines=[0, 1, 2], supports=["play", "sync"]))
    assert isinstance(box, ControllerLimits)


def test_a_control_system_reports_every_wiring_problem_at_once():
    a = pulse.Controller(drive_lines=[0, 1], peer_latency_cycles={0: 4})
    b = pulse.Controller(drive_lines=[1, 2])
    stranger = pulse.Controller()
    with pytest.raises(DeviceDefinitionError) as raised:
        pulse.ControlSystem(
            controllers=[a, b], links={"feedback": pulse.Link(controllers=[stranger])}
        )
    paths = [p.path for p in raised.value.problems]
    assert paths == [
        "controllers[1].drive_lines[0]",
        "controllers[0].peer_latency_cycles",
        "links.feedback.controllers[0]",
    ]
    assert "also on controllers[0]" in str(raised.value)
    with pytest.raises(TypeError, match="make it a pulse.Controller"):
        pulse.ControlSystem(controllers=[ControllerLimits()])
    with pytest.raises(ValueError, match="keyed by name"):
        pulse.ControlSystem(controllers=[a], links={0: pulse.Link()})


def test_wiring_names_carriers_of_the_quantum_system():
    box = pulse.Controller(drive_lines=[0, 40], readout_lines=[41], acquirers={42: 0})
    with pytest.raises(DeviceDefinitionError) as raised:
        _qpu(box)
    assert [p.path for p in raised.value.problems] == [
        "control_system.controllers[0].drive_lines[1]",
        "control_system.controllers[0].readout_lines[0]",
        "control_system.controllers[0].acquirers",
    ]
    assert "it has qudits 0..16" in str(raised.value)
    assert isinstance(raised.value, ValueError)


def test_the_device_holds_its_control_system():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4))
    qpu = _qpu(box)
    assert qpu.control_system.controllers == (box,)
    assert qpu.quantum_system is qpu.target
    assert qpu.qudits[1] == qpu.qudit_handles[1]
    other = _qpu(pulse.Controller(drive_lines=range(4), readout_lines=range(3)))
    assert qpu.fingerprint == other.fingerprint and qpu != other
    legacy = pulse.device({"target": TRANSMON_17Q, "layout": [0, 1, 2, 3]})
    assert legacy.control_system is None and legacy.fingerprint == qpu.fingerprint
    with pytest.raises(pulse.DeviceDefinitionError, match="target: is the former name"):
        pulse.device({"target": TRANSMON_17Q, "quantum_system": TRANSMON_17Q})


# ── rcp and the controllers' feedback latency ────────────────────────────────


def _schedule(machine):
    lowered = pulse.compile(drive_and_branch, schedule="rcp", machine=machine).lower(
        *pulse.device("transmon_17q").qudits[:2]
    )
    events = [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule]
    return events, lowered.mlir()


def test_rcp_waits_the_feedback_latency_of_the_controllers():
    by_model = _schedule(MachineModel(readout_latency_cycles=50))
    system = pulse.ControlSystem(controllers=[pulse.Controller(feedback_latency_cycles=50)])
    assert system._machine.readout_latency_cycles == 50
    assert _schedule(system) == by_model
    assert _schedule(MachineModel()) != by_model
    # Of several, the slowest that plays readouts, and on another controller
    # the start uncertainty of both too: one cycle each without a reference.
    a, b = _two_boxes(feedback_latency_cycles=50)
    slow = pulse.Controller(drive_lines=[5], feedback_latency_cycles=900)
    assert pulse.ControlSystem(controllers=[a, b, slow])._machine.readout_latency_cycles == 52
    shared = pulse.ControlSystem(controllers=[a, b, slow], reference_clock_ghz=0.5)
    assert shared._machine.readout_latency_cycles == 50


# ── Compiling for a control system ───────────────────────────────────────────


def test_compile_uses_the_device_control_system():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4), max_events=1)
    qpu = _qpu(box)
    with pytest.raises(ControllerLimitError, match="at most 1"):
        pulse.compile(drive_and_branch, device=qpu)(qpu.qudits[0], qpu.qudits[1])
    # As with the same limits given to a machine model.
    with pytest.raises(ControllerLimitError, match="at most 1"):
        pulse.compile(
            drive_and_branch, machine=MachineModel(controller=ControllerLimits(max_events=1))
        ).lower(qpu.qudits[0], qpu.qudits[1])
    with pytest.raises(TypeError, match="MachineModel or a ControlSystem"):
        pulse.compile(drive_and_branch, machine=box)


def test_a_line_on_no_controller_does_not_compile():
    qpu = _qpu(*_two_boxes(), layout=range(6))
    with pytest.raises(
        ControllerLimitError, match="the drive line of qudit 4\\) is on no controller"
    ):
        pulse.compile(drive_and_branch, device=qpu).lower(qpu.qudits[0], qpu.qudits[4])
    pulse.compile(drive_and_branch, schedule="rcp", device=qpu).lower(qpu.qudits[0], qpu.qudits[3])


def test_each_controller_is_held_to_its_own_limits():
    a, b = _two_boxes(max_events=0 + 1)
    qpu = _qpu(a, b)
    compiled = pulse.compile(drive_and_branch, schedule="rcp", device=qpu)
    # Two events on box.a, one on box.b.
    compiled.lower(qpu.qudits[0], qpu.qudits[2])
    a, b = _two_boxes()
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], model="box.a", max_events=1)
    qpu = _qpu(a, b)
    with pytest.raises(ControllerLimitError, match=r"controllers\[0\] \(box.a\): 2 drives"):
        pulse.compile(drive_and_branch, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])


def test_one_controller_on_its_own_clock_is_timed_as_on_a_machine_model():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4), clock_ghz=1.0)
    # Without a declared program clock, the controller's is the program's.
    assert _qpu(box).clock_ghz == 1.0
    qpu = _qpu(box, clock_ghz=2.0)
    by_system = pulse.compile(drive_and_branch, device=qpu).lower(qpu.qudits[0], qpu.qudits[1])
    machine = MachineModel(controller=ControllerLimits(clock_ghz=1.0))
    by_model = pulse.compile(drive_and_branch, machine=machine).lower(qpu.qudits[0], qpu.qudits[1])
    system, model = by_system.export_controller(), by_model.export_controller()
    assert system["clock_ghz"] == 1.0 and system["program_clock_ghz"] == qpu.clock_ghz
    assert system["events"] == model["events"]

    @pulse.kernel
    def odd(q):
        d = get_drive_line(q)
        drive(d, square(15, 0.1))

    with pytest.raises(ControllerLimitError, match="line d0 moves in timesteps of 2 cycles"):
        pulse.compile(odd, device=qpu).lower(qpu.qudits[0])


def _outputs(lowered):
    return (
        lowered.mlir(),
        json.dumps(lowered.export_controller(), sort_keys=True),
        json.dumps(lowered.export_emulation(), sort_keys=True),
    )


def test_line_delays_and_a_reference_clock_are_the_control_systems():
    delays = {("drive", 0): 2.3, ("readout", 0): 1.0}
    box = pulse.Controller(
        drive_lines=range(4), readout_lines=range(4), clock_ghz=1.0, max_fine_delay_ns=1.0
    )
    system = pulse.ControlSystem(controllers=[box], reference_clock_ghz=0.5, line_delays_ns=delays)
    assert system._machine.line_delays_ns == delays and system._machine.reference_clock_ghz == 0.5
    timed = pulse.device(
        {"quantum_system": TRANSMON_17Q, "layout": [0, 1, 2, 3], "control_system": system}
    )
    by_system = pulse.compile(drive_and_branch, device=timed).lower(
        timed.qudits[0], timed.qudits[1]
    )
    machine = MachineModel(controller=box, reference_clock_ghz=0.5, line_delays_ns=delays)
    by_model = pulse.compile(drive_and_branch, machine=machine).lower(
        timed.qudits[0], timed.qudits[1]
    )
    assert _outputs(by_system) == _outputs(by_model)
    assert by_system.export_controller()["reference_clock_ghz"] == 0.5
    lines = {(line["role"], line["qudit"]): line for line in by_system.export_controller()["lines"]}
    assert lines[("drive", 0)]["fine_delay_ns"] == pytest.approx(0.3)


def test_new_line_delays_of_a_control_system_schedule_again_to_the_same_bytes():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4))
    kernel = pulse.kernel(drive_and_branch.__pulse_kernel_fn__)
    for k, delays in enumerate(({("drive", 0): 1.0}, {("drive", 1): 4.0, ("readout", 0): 1.5})):
        system = pulse.ControlSystem(controllers=[box], line_delays_ns=delays)
        qpu = pulse.device(
            {"quantum_system": TRANSMON_17Q, "layout": [0, 1], "control_system": system}
        )
        again = pulse.compile(kernel, device=qpu).lower(qpu.qudits[0], qpu.qudits[1])
        assert (again.metrics.trace_ms == 0.0) == (k > 0)
        fresh = pulse.kernel(drive_and_branch.__pulse_kernel_fn__)
        full = pulse.compile(fresh, device=qpu).lower(qpu.qudits[0], qpu.qudits[1])
        assert _outputs(again) == _outputs(full)


def test_a_control_systems_line_delays_and_reference_clock_are_checked():
    with pytest.raises(ValueError, match=r"ControlSystem.line_delays_ns\[\('drive', 0\)\] is -1"):
        pulse.ControlSystem(line_delays_ns={("drive", 0): -1.0})
    with pytest.raises(ValueError, match="ControlSystem.reference_clock_ghz must be positive"):
        pulse.ControlSystem(reference_clock_ghz=0.0)
    with pytest.raises(
        DeviceDefinitionError, match="delays qudit 4's drive line, which no controller plays"
    ):
        pulse.ControlSystem(controllers=list(_two_boxes()), line_delays_ns={("drive", 4): 1.0})
    system = pulse.ControlSystem(
        controllers=[pulse.Controller()], line_delays_ns={("readout", 40): 1.0}
    )
    with pytest.raises(DeviceDefinitionError, match="the qudit is not in the quantum system"):
        pulse.device({"quantum_system": TRANSMON_17Q, "control_system": system})


def test_a_control_system_takes_its_times_as_times():
    ns, us = pulse.ns, pulse.us
    lines = {"drive_lines": range(4), "readout_lines": range(4)}
    box = pulse.Controller(max_fine_delay_ns=1 * ns, **lines)
    assert box == pulse.Controller(max_fine_delay_ns=1.0, **lines)
    by_time = pulse.ControlSystem(
        controllers=[box],
        links={"decoder": Link(fixed_ns=0.1 * us, feedback_ns=20 * ns, per_controller=True)},
        line_delays_ns={("drive", 0): 2.5 * ns, ("readout", 0): 0.001 * us},
    )
    by_number = pulse.ControlSystem(
        controllers=[box],
        links={"decoder": Link(fixed_ns=100, feedback_ns=20, per_controller=True)},
        line_delays_ns={("drive", 0): 2.5, ("readout", 0): 1.0},
    )
    assert by_time == by_number
    assert [ns.__class__ for ns in by_time.line_delays_ns.values()] == [float, float]
    qpus = [
        pulse.device({"quantum_system": TRANSMON_17Q, "layout": [0, 1], "control_system": system})
        for system in (by_time, by_number)
    ]
    assert qpus[0].as_dict() == qpus[1].as_dict()
    with pytest.raises(ValueError, match=r"ControlSystem.line_delays_ns\[\('drive', 0\)\] is -1"):
        pulse.ControlSystem(line_delays_ns={("drive", 0): -1 * ns})


@pytest.mark.parametrize(
    ("name", "value"),
    [("clock_skew_cycles", 4 * pulse.ns), ("peer_latency_cycles", {1: 4 * pulse.ns})],
)
def test_a_controllers_cycles_are_its_own_not_a_time(name, value):
    with pytest.raises(
        pulse.InvalidTypeError, match=f"Controller.{name} is a whole number of the controller's own"
    ):
        pulse.Controller(**{name: value})


def test_several_controllers_on_their_own_clocks_schedule_on_their_grids():
    qpu = _qpu(*_two_boxes(clock_ghz=1.0), clock_ghz=2.0)
    lowered = pulse.compile(drive_and_branch, schedule="rcp", device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    starts = [e.start for e in lowered.schedule]
    assert starts and all(start % 2 == 0 for start in starts)


def test_a_program_needing_a_capability_a_controller_lacks_does_not_compile():
    a, b = _two_boxes(supports=["play"])
    qpu = _qpu(a, b)
    compiled = pulse.compile(drive_and_branch, device=qpu)
    with pytest.raises(ControllerLimitError) as raised:
        compiled.lower(qpu.qudits[0], qpu.qudits[2])
    text = str(raised.value)
    # The branch plays on box.b, which forwards nothing but must branch and
    # sync; box.a forwards the outcome.
    assert "controllers[1] (box.b): the drive on line 2 (qudit 2)" in text
    assert "needs 'branch'" in text and "needs 'sync'" in text
    full = pulse.Controller(
        drive_lines=[0, 1],
        readout_lines=[0, 1],
        supports=[c for c in pulse.Capability if c != "forward_outcome"],
    )
    qpu = _qpu(full, b)
    with pytest.raises(
        ControllerLimitError, match=r"controllers\[0\]: the readout .* needs 'forward_outcome'"
    ):
        pulse.compile(drive_and_branch, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # Both on one controller: nothing to forward.
    pulse.compile(drive_and_branch, device=qpu).lower(qpu.qudits[0], qpu.qudits[1])


@pulse.classical(max_latency_ns=400, link="feedback")
def settle(x: float) -> float:
    return x


@pulse.kernel
def plays(q):
    d = get_drive_line(q)
    drive(d, square(20, 0.1))


@pulse.kernel
def acquires(q):
    r = get_readout_line(q)
    readout(r, square(100, 0.05))


@pulse.kernel
def waits_for_an_answer(q):
    d = get_drive_line(q)
    wait(d, settle.submit(8.0).result(), bounds=(0, 800))
    drive(d, square(20, 0.1))


@pulse.kernel
def syncs(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(20, 0.1))
    sync(d, r)
    readout(r, square(100, 0.05))


@pulse.kernel
def turns_by_an_outcome(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    phase = 0.1
    if m:
        phase = 0.5
    shift_phase(d.tone, phase)
    drive(d, square(20, 0.1))


@pulse.kernel
def turns_by_a_parameter(q, phase: float):
    d = get_drive_line(q)
    shift_phase(d.tone, phase)
    drive(d, square(20, 0.1))


@pulse.kernel(unroll_threshold=0)
def loops(q):
    d = get_drive_line(q)
    for _ in range(4):
        drive(d, square(20, 0.1))


# For each capability, a program that needs it and the qudits it runs on:
# two on one controller, or (0, 2) across two.
_NEEDING = {
    pulse.Capability.PLAY: (plays, (0,)),
    pulse.Capability.ACQUIRE: (acquires, (0,)),
    pulse.Capability.WAIT: (waits_for_an_answer, (0,)),
    pulse.Capability.SYNC: (syncs, (0,)),
    pulse.Capability.CALL: (corrected, (0, 1)),
    pulse.Capability.ANSWER: (corrected, (0, 1)),
    pulse.Capability.BRANCH: (drive_and_branch, (0, 1)),
    pulse.Capability.RUNTIME_FRAME_UPDATE: (turns_by_an_outcome, (0,)),
    pulse.Capability.FORWARD_OUTCOME: (drive_and_branch, (0, 2)),
    pulse.Capability.LOOP: (loops, (0,)),
}


def test_every_capability_has_a_program_that_needs_it():
    assert set(_NEEDING) == set(pulse.Capability)


@pytest.mark.parametrize("capability", list(pulse.Capability))
def test_a_controller_lacking_only_one_capability_rejects_what_needs_it(capability):
    kernel, placed = _NEEDING[capability]
    others = [c for c in pulse.Capability if c != capability]
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], supports=others)
    b = pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3])
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    qudits = [qpu.qudits[i] for i in placed]
    full = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])
    complete = _qpu(full, b, links={"feedback": pulse.Link(controllers=[full])})
    pulse.compile(kernel, schedule="rcp", device=complete).lower(
        *[complete.qudits[i] for i in placed]
    )
    if capability == pulse.Capability.LOOP:
        # Without it, the controller plays the loop unrolled.
        assert pulse.compile(kernel, device=qpu).lower(*qudits).export_controller()
        return
    with pytest.raises(ControllerLimitError, match=f"controllers\\[0\\]: .* needs '{capability}'"):
        pulse.compile(kernel, device=qpu).lower(*qudits)


def test_a_tone_a_parameter_sets_needs_no_runtime_frame_update():
    others = [c for c in pulse.Capability if c != pulse.Capability.RUNTIME_FRAME_UPDATE]
    qpu = _qpu(pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], supports=others))
    compiled = pulse.compile(turns_by_a_parameter, device=qpu)
    assert compiled.lower(qpu.qudits[0], 0.3).export_controller()["events"][0]["phase_rad"] == 0.3


@pulse.kernel(unroll_threshold=0)
def two_loops(q: pulse.Qudit, r: pulse.Qudit):
    d = get_drive_line(q)
    e = get_drive_line(r)
    for _ in range(30):
        drive(d, square(20, 0.1))
        drive(d, square(20, 0.2))
    for _ in range(10):
        drive(e, square(20, 0.1))


def test_a_controller_without_loop_gets_its_loops_unrolled_for_it_only():
    no_loop = [c for c in pulse.Capability if c != "loop"]
    a, b = _two_boxes(max_events=4)
    a = pulse.Controller(drive_lines=[0, 1], model="box.a", max_events=4, supports=no_loop)
    qpu = _qpu(a, b)
    assert "scf.for" in two_loops.lower(qpu.qudits[0], qpu.qudits[2]).mlir()
    # Without loop, box.a plays all 60 drives of its loop, naming it; box.b
    # holds its loop's body once.
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(two_loops, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    text = str(raised.value)
    assert "controllers[0] (box.a): 60 drives and readouts" in text
    assert "the loop of 30 iterations from cycle 0 plays 58 more" in text
    assert "box.b" not in text
    # Unrolling adds pulses, not waveforms: memory for one shape fits either way.
    for supports in (None, no_loop):
        held = pulse.Controller(drive_lines=[0, 1], waveform_samples=20, supports=supports)
        qpu = _qpu(held, b)
        pulse.compile(two_loops, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # With loop, a body over max_events is counted once and still fails.
    small = pulse.Controller(drive_lines=[0, 1], model="box.a", max_events=1)
    qpu = _qpu(small, b)
    with pytest.raises(ControllerLimitError, match="body counted once; the controller holds"):
        pulse.compile(two_loops, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # A plain ControllerLimits counts every iteration of both loops, as before.
    machine = MachineModel(controller=ControllerLimits(max_events=60))
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(two_loops, machine=machine).lower(*qpu.qudits[:2])
    text = str(raised.value)
    assert "70 drives and readouts; the controller plays at most 60" in text
    assert text.endswith(
        "the loop of 30 iterations from cycle 0 plays 58 more; "
        "the loop of 10 iterations from cycle 1000 plays 9 more"
    )


def test_a_call_sends_and_answers_only_controllers_its_link_reaches():
    a, b = _two_boxes()
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    compiled = pulse.compile(corrected, device=qpu)
    compiled.lower(qpu.qudits[0], qpu.qudits[1])
    with pytest.raises(ControllerLimitError) as raised:
        compiled.lower(qpu.qudits[2], qpu.qudits[0])
    assert "over link 'feedback' sends the measurements of controllers[1] (box.b)" in str(
        raised.value
    )
    with pytest.raises(ControllerLimitError, match="answers controllers\\[1\\]"):
        compiled.lower(qpu.qudits[0], qpu.qudits[2])


def test_runtime_qudits_are_checked_against_the_wiring_per_call():
    @pulse.kernel
    def pulse_one(q: pulse.Qudit):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))

    qpu = _qpu(*_two_boxes(), layout=range(6))
    compiled = pulse.compile(pulse_one, device=qpu)
    compiled(qpu.qudits[3])
    with pytest.raises(ControllerLimitError, match="qudit 5\\) is on no controller"):
        compiled(qpu.qudits[5])


# ── System files ─────────────────────────────────────────────────────────────


def _round_trip(qpu):
    data = qpu.as_dict()
    assert data["format"] == "cudaq-pulse.system/1"
    assert data["written_by"].startswith("cudaq-pulse ")
    back = pulse.device(json.loads(json.dumps(data)))
    assert back == qpu and back.fingerprint == qpu.fingerprint
    assert back.as_dict() == data
    return data


def test_a_qpu_round_trips_through_its_system_file():
    a, b = _two_boxes(
        peer_latency_cycles={0: 12}, supports=["play", "acquire", "sync"], backend="toy"
    )
    qpu = _qpu(
        a, b, links={"feedback": pulse.Link(fixed_ns=40.0, controllers=[b])}, layout=range(6)
    )
    data = _round_trip(qpu)
    assert data["quantum_system"] == TRANSMON_17Q and data["calibration"] == "transmon_17q"
    system = data["control_system"]
    assert system["links"]["feedback"]["controllers"] == [1]
    assert system["controllers"][1]["peer_latency_cycles"] == {"0": 12}
    assert system["controllers"][1]["supports"] == ["play", "acquire", "sync"]


def test_a_qudit_s_drive_and_readout_round_trip_through_the_system_file():
    readout = DispersiveReadout(frequency_hz=7.1e9, linewidth_hz=1e6, error_1to0=0.02)
    qubits = {
        0: Qubit(
            0, 5.0e9, -2e8, 50.0, 30.0, drive=MicrowaveDrive(x_amplitude=0.3), readout=readout
        ),
        1: Qubit(1, 5.1e9, -2e8, 50.0, 30.0),
    }
    dev = pulse.device({"quantum_system": TargetDescription(name="ro", qubits=qubits)})
    qudits = _round_trip(dev)["quantum_system"]["qudits"]
    assert qudits[0]["readout"]["kind"] == "dispersive"
    assert qudits[0]["readout"]["frequency_hz"] == 7.1e9
    assert qudits[0]["drive"] == {"kind": "microwave", **dict.fromkeys(MicrowaveDrive._fields)} | {
        "x_amplitude": 0.3
    }
    assert qudits[1]["readout"] is None and qudits[1]["drive"] is None
    other = Qubit(0, 5.0e9, -2e8, 50.0, 30.0, readout=readout.replace(linewidth_hz=2e6))
    moved = pulse.device(
        {"quantum_system": TargetDescription(name="ro", qubits={**qubits, 0: other})}
    )
    assert moved.fingerprint != dev.fingerprint


def test_a_system_file_refuses_an_unknown_component_kind_or_field():
    data = pulse.device(
        {
            "quantum_system": TargetDescription(
                name="ro", qubits={0: Qubit(0, 5.0e9, -2e8, 50.0, 30.0)}
            )
        }
    ).as_dict()
    qudit = data["quantum_system"]["qudits"][0]
    qudit["readout"] = {"kind": "fluorescence"}
    qudit["drive"] = {"kind": "microwave", "x_amp": 0.2}
    with pytest.raises(DeviceDefinitionError) as found:
        pulse.device(data)
    message = str(found.value)
    assert "qudits[0].readout.kind" in message and "the kinds are dispersive" in message
    assert "qudits[0].drive.x_amp" in message and "did you mean 'x_amplitude'" in message


def test_line_delays_and_a_reference_clock_round_trip_through_the_system_file():
    system = pulse.ControlSystem(
        controllers=list(_two_boxes()),
        reference_clock_ghz=0.25,
        line_delays_ns={("drive", 0): 1.5, ("readout", 3): 2.0},
    )
    qpu = pulse.device(
        {"quantum_system": TRANSMON_17Q, "layout": [0, 1, 2, 3], "control_system": system}
    )
    data = _round_trip(qpu)["control_system"]
    assert data["line_delays_ns"] == {"drive": {"0": 1.5}, "readout": {"3": 2.0}}
    assert data["reference_clock_ghz"] == 0.25
    assert "line_delays_ns" not in _qpu(*_two_boxes()).as_dict()["control_system"]
    data["line_delays_ns"] = {"drive": {"zero": 1.0}}
    with pytest.raises(DeviceDefinitionError, match="line_delays_ns.drive: is keyed by qudit"):
        pulse.device(
            {"quantum_system": TRANSMON_17Q, "layout": [0, 1, 2, 3], "control_system": data}
        )


def test_a_program_clock_round_trips_through_the_system_file():
    system = pulse.ControlSystem(controllers=[pulse.Controller(clock_ghz=1.0)])
    qpu = pulse.device(
        {
            "quantum_system": TRANSMON_17Q,
            "layout": [0, 1],
            "control_system": system,
            "clock_ghz": 4.0,
        }
    )
    assert qpu.clock_ghz == 4.0
    assert _round_trip(qpu)["clock_ghz"] == 4.0
    assert "clock_ghz" not in _round_trip(_qpu(pulse.Controller(), layout=[0, 1]))


def test_a_custom_quantum_system_is_written_inline_with_the_new_key_names():
    target = TargetDescription(
        name="pair",
        qubits={i: Qubit(i, 5e9 + i * 1e8, -2.5e8, 50.0, 40.0) for i in range(2)},
        couplings=[Coupling((0, 1), 3e6)],
    )
    qpu = pulse.device({"quantum_system": target})
    data = _round_trip(qpu)
    assert data["quantum_system"]["couplings"] == [
        {"qudits": [0, 1], "strength_hz": 3e6, "gate": None}
    ]
    assert [q["index"] for q in data["quantum_system"]["qudits"]] == [0, 1]
    assert "control_system" not in data


def test_a_system_file_reports_every_problem_with_its_line(tmp_path):
    qpu = _qpu(*_two_boxes())
    data = qpu.as_dict()
    data["control_system"]["controllers"][1]["drive_lines"] = [2, 30]
    data["control_system"]["controllers"][0]["issue_cycle"] = 2
    path = tmp_path / "qpu.json"
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(DeviceDefinitionError) as raised:
        pulse.device(path)
    unknown, wiring = raised.value.problems
    assert unknown.path == "control_system.controllers[0].issue_cycle"
    assert "did you mean 'issue_cycles'?" in str(raised.value)
    assert wiring.path == "control_system.controllers[1].drive_lines[1]" and wiring.value == 30
    text = path.read_text().splitlines()
    assert unknown.file == wiring.file == str(path)
    assert text[wiring.line - 1].strip().startswith("30")
    assert '"issue_cycle"' in text[unknown.line - 1]
    assert str(raised.value).splitlines()[1].strip().startswith(f"{path}:{unknown.line}:")


def test_a_newer_system_file_format_is_refused():
    with pytest.raises(DeviceDefinitionError, match="newer than this cudaq-pulse reads"):
        pulse.device({"format": "cudaq-pulse.system/2", "quantum_system": TRANSMON_17Q})
    with pytest.raises(DeviceDefinitionError, match="not a system file format"):
        pulse.device({"format": "something/1", "quantum_system": TRANSMON_17Q})


# ── Controller descriptions ──────────────────────────────────────────────────


def test_a_description_is_loaded_by_name_with_the_installation_wiring():
    box = pulse.controller("reference.controller", drive_lines=[0, 1], readout_lines=[0])
    assert isinstance(box, pulse.Controller)
    assert (box.model, box.version) == ("reference.controller", "1")
    assert box.drive_lines == (0, 1) and box.readout_lines == (0,)
    assert pulse.controller("reference.controller") is pulse.controller("reference.controller")
    with pytest.raises(KeyError, match="did you mean 'reference.controller'"):
        pulse.controller("reference.controler")
    with pytest.raises(TypeError, match="did you mean 'drive_lines'"):
        pulse.controller("reference.controller", drive_line=[0])


@pytest.mark.parametrize("name", sorted(installed_descriptions()))
def test_every_installed_description_loads_and_compiles(name):
    path = installed_descriptions()[name]
    assert "owner" in tomllib.loads(path.read_text(encoding="utf-8"))
    box = pulse.controller(name, drive_lines=range(4), readout_lines=range(4))
    assert box.model == name
    qpu = _qpu(box)
    pulse.compile(drive_and_branch, schedule="rcp", device=qpu).lower(qpu.qudits[0], qpu.qudits[1])
    # A description that names its backend opts in to the backend's checks.
    validate = getattr(pulse.backends, "validate", None)
    if box.backend is not None and validate is not None:
        validate(pulse.backends.get(box.backend), controller=box)


def test_a_description_file_reports_every_problem_with_its_line(tmp_path):
    path = tmp_path / "box.toml"
    path.write_text('model = "lab.box"\nissue_cycle = 2\ndrive_lines = [0]\n')
    with pytest.raises(DeviceDefinitionError) as raised:
        pulse.controller(path)
    problems = {p.path: p for p in raised.value.problems}
    assert problems["issue_cycle"].line == 2 and "issue_cycles" in problems["issue_cycle"].fix
    assert problems["drive_lines"].line == 3 and "installation" in problems["drive_lines"].message
    assert any("no version" in p.message for p in raised.value.problems)
    path.write_text('model = "lab.box"\nversion = "2"\nissue_cycles = 3\n')
    box = pulse.controller(path, drive_lines=[0])
    assert box.issue_cycles == 3 and box.model == "lab.box"


def test_a_system_file_from_another_description_version_is_noticed_once():
    qpu = _qpu(pulse.controller("reference.controller", drive_lines=range(4)))
    data = qpu.as_dict()
    data["control_system"]["controllers"][0]["version"] = "0-test"
    with pytest.warns(
        UpgradeNotice, match="version 0-test; the installed description is version 1"
    ):
        old = pulse.device(data)
    assert old.control_system.controllers[0].version == "0-test"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pulse.device(data)
        pulse.device(qpu.as_dict())


# ── doctor --system ──────────────────────────────────────────────────────────


def _doctor(path, *flags):
    from cudaq_pulse.doctor import main

    out = io.StringIO()
    return main(["--system", str(path), *flags], out=out), out.getvalue()


def test_doctor_checks_a_system_file(tmp_path):
    a, b = _two_boxes()
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    path = tmp_path / "qpu.json"
    path.write_text(json.dumps(qpu.as_dict(), indent=2))
    code, text = _doctor(path)
    assert code == 0, text
    assert f"{path}: ok, device {TRANSMON_17Q} with 4 qudits" in text
    assert "controllers[1]: box.b, drive lines 2..3" in text
    assert "link 'feedback': reaches controllers [0]" in text
    data = qpu.as_dict()
    data["control_system"]["controllers"][0]["backend"] = "no_such_backend"
    path.write_text(json.dumps(data, indent=2))
    code, text = _doctor(path)
    assert code == 1 and "backend no_such_backend (not installed)" in text
    assert "control_system.controllers[0].backend: backend 'no_such_backend' is not" in text
    code, text = _doctor(path, "--json")
    report = json.loads(text)
    assert code == 1 and report["device"]["control_system"]["links"] == {"feedback": [0]}
    data["control_system"]["controllers"][0]["readout_lines"] = [0, 99]
    path.write_text(json.dumps(data, indent=2))
    code, text = _doctor(path)
    assert code == 1 and f"{path}:" in text and "readout_lines[1]: qudit 99" in text


def test_a_system_file_keeps_the_simulation_model():
    target = TargetDescription(
        name="other-pair",
        qubits={i: Qubit(i, 5e9 + i * 1e8, 0.0, 50.0, 40.0) for i in range(2)},
        model=SimulationModel(levels={1: 3}),
    )
    qpu = pulse.device({"target": target})
    data = qpu.as_dict()
    assert data["quantum_system"]["model"] == {"kind": "generic", "levels": {"1": 3}}
    again = pulse.device(json.loads(json.dumps(data)))
    assert again.target.model == target.model and again == qpu

    @pulse.kernel
    def nothing(q):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))

    with pytest.raises(UnsupportedError, match="SimulationModel has no simulator"):
        pulse.simulate(pulse.compile(nothing).lower(again.qudits[0]))

    data["quantum_system"]["model"] = None
    assert pulse.device(data).target.model is None
    data["quantum_system"]["model"] = {"kind": "fluxonium"}
    with pytest.raises(DeviceDefinitionError, match="model.kind: must be one of transmon, generic"):
        pulse.device(data)


def test_a_system_file_names_a_model_it_cannot_hold():
    class Custom(SimulationModel):
        pass

    target = TargetDescription(
        name="custom-pair",
        qubits={i: Qubit(i, 5e9 + i * 1e8, 0.0, 50.0, 40.0) for i in range(2)},
        model=Custom,
    )
    with pytest.raises(InvalidValueError, match="Custom is not one"):
        pulse.device({"target": target}).as_dict()
