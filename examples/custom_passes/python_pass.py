# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A custom check written in Python, run on a compiled kernel's schedule.

``Lowered.schedule`` lists every drive, readout and wait with its
line, qubit and scheduled time, so a pass that inspects the schedule is a
plain Python function. This one enforces a hardware limit the scheduler
does not know about: at most ``max_drives`` drive lines may play at once.

Transforms of the IR itself are written in C++ as MLIR passes and loaded
into ``cudaq-pulse-opt`` as plugins; see ``cpp/DriveCount.cpp``.
"""

import cudaq_pulse as pulse

cal = {"x_duration": 20 * pulse.ns, "x_sigma": 5 * pulse.ns, "drag_beta": 0.25 * pulse.ns}


def concurrent_drive_violations(compiled: pulse.Lowered, max_drives: int) -> list[tuple[int, int]]:
    """Return (time, active drives) wherever more than *max_drives* overlap."""
    edges = []
    for event in compiled.schedule:
        if event.kind == "drive" and event.duration_cycles:
            edges.append((event.start, 1))
            edges.append((event.end, -1))
    active = 0
    violations = []
    # Ends sort before starts at the same instant, so back-to-back pulses
    # do not count as overlapping.
    for time, step in sorted(edges):
        active += step
        if active > max_drives:
            violations.append((time, active))
    return violations


@pulse.kernel
def parallel_x(qs):
    for i in range(4):
        line = get_drive_line(qs[i])
        drive(line, drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"]))


def main() -> None:
    args = [dev.qudit_handles[:4]]

    asap = pulse.compile(parallel_x, schedule="asap", device=dev).lower(*args)
    print("ASAP violations of a 2-drive limit:", concurrent_drive_violations(asap, 2))

    rcp = pulse.compile(
        parallel_x, schedule="rcp", control_system=pulse.ControlSystem(max_concurrent_drives=2)
    ).lower(*args)
    violations = concurrent_drive_violations(rcp, 2)
    print("RCP violations of a 2-drive limit:", violations)
    assert not violations


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    main()
