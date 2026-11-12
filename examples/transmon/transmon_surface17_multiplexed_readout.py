#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiplexed readout: one feedline reads out four qudits of transmon_surface17.

Demonstrates:
  - A shared line: ReadoutLine([9, 10, 11, 12], max_concurrent_tones=4), one
    feedline reaching the four ancillas of the X checks (X1-X4), each read out
    at its own frequency
  - get_readout_line(q) of a qudit on the feedline: one of its tones, for q
  - The compiler's check of the tones a line plays at once
  - Format /2 of the controller export: the feedline listed once, and each
    readout with its tone slot and qudit; format /1 has one qudit per line
"""

import cudaq_pulse as pulse
from cudaq_pulse.controller import ControllerLimitError, ReadoutLine

ANCILLAS = [9, 10, 11, 12]  # X1..X4


def quantum_system(feedline: ReadoutLine) -> pulse.QuantumSystem:
    controller = pulse.Controller(drive_lines=ANCILLAS, readout_lines=[feedline])
    return pulse.quantum_system(
        "transmon_surface17",
        layout=ANCILLAS,
        control_system=pulse.ControlSystem(controllers=[controller]),
    )


@pulse.kernel
def measure_ancillas(a, b, c, d):
    lines = [get_readout_line(a), get_readout_line(b), get_readout_line(c), get_readout_line(d)]
    sync(lines[0], lines[1], lines[2], lines[3])
    for line in lines:
        readout(line, square(400, 0.05))


feedline = ReadoutLine(ANCILLAS, max_concurrent_tones=4)
print(f"{feedline!r}: qudits {feedline.qudits}, {feedline.max_concurrent_tones} tones at once")
qs = quantum_system(feedline)

scheduled = pulse.compile(measure_ancillas, quantum_system=qs).lower(*qs.qudits[:4])
program = scheduled.export_to_controller(version=2)
for line in program["lines"]:
    print("line", line)
for event in program["events"]:
    if event["kind"] == "readout":
        frequency = event["tones"][0]["frequency_hz"] / 1e9
        print(
            f"  readout of qudit {event['qudits'][0]} on line {event['line']}, tone slot "
            f"{event['tone_slot']}, {frequency:.3f} GHz, from cycle {event['start_cycles']}"
        )

# A feedline that plays two tones at once cannot read out all four together.
two_tones = quantum_system(ReadoutLine(ANCILLAS, max_concurrent_tones=2))
try:
    pulse.compile(measure_ancillas, quantum_system=two_tones).lower(*two_tones.qudits[:4])
except ControllerLimitError as error:
    print("two tones at once:", error.violations[0])

# Format /1 gives every line one qudit.
try:
    scheduled.export_to_controller()
except pulse.UnsupportedError as error:
    print("format /1:", error)
