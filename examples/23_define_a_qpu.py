# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a QPU: its quantum system, its controllers and a feedback link.

Two controllers, loaded from an installed description with this
installation's wiring, each play the lines of two qudits. A classical
function on the "feedback" link takes a readout of one controller and
answers the other, which branches on it. Compiling checks the program
against the control system: every line on a controller, each controller's
limits and capabilities, and the link's reach. The QPU is saved as one
system file and loaded back.
"""

import json
import tempfile
from pathlib import Path

import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits

left = pulse.controller("reference.controller", drive_lines=[0, 1], readout_lines=[0, 1])
right = pulse.controller("reference.controller", drive_lines=[2, 3], readout_lines=[2, 3])
control = pulse.ControlSystem(
    controllers=[left, right],
    links={"feedback": pulse.Link(fixed_ns=40.0, feedback_ns=20.0, controllers=[left, right])},
)
qpu = pulse.device(
    {"quantum_system": "transmon_krinner_17q", "layout": [0, 1, 2, 3], "control_system": control}
)


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=400, link="feedback")
def correction(outcome):
    """Whether to flip the partner qudit: here, the outcome itself."""
    return [outcome[0]]


@pulse.kernel
def measure_and_correct(measured, partner):
    readout_line, readout_tone = get_readout_line(measured)
    drive_line, tone = get_drive_line(partner)
    outcome = readout(readout_line, square(200, 0.05), readout_tone)
    flip = correction.submit([outcome]).result()
    sync(readout_line, drive_line)
    if flip[0]:
        drive(drive_line, gaussian(40, 0.5, 10.0), tone)


compiled = pulse.compile(measure_and_correct, schedule="rcp", device=qpu)
# Read out on the left controller, correct on the right one.
lowered = compiled.lower(qpu.qudits[0], qpu.qudits[2])
for index in range(len(control.controllers)):
    program = lowered.export_controller(controller=index)
    events = [(e["kind"], e["line"], e["start_cycles"]) for e in program["events"]]
    print(f"controllers[{index}] events:", events)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "qpu.json"
    path.write_text(json.dumps(qpu.as_dict(), indent=2))
    assert pulse.device(path) == qpu
    print("saved and loaded", path.name, "format", json.loads(path.read_text())["format"])

    # A wiring mistake is reported with its place in the file.
    broken = qpu.as_dict()
    broken["control_system"]["controllers"][1]["drive_lines"] = [2, 3, 30]
    path.write_text(json.dumps(broken, indent=2))
    try:
        pulse.device(path)
    except pulse.DeviceDefinitionError as error:
        for problem in error.problems:
            print("problem:", problem.path, "line", problem.line, "-", problem.message)
