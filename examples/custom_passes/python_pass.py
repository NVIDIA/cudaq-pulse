# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A custom check written in Python, run on a compiled kernel's schedule.

``CompiledKernel.events()`` lists every drive, readout and wait with its
line, qubit and scheduled time, so a pass that inspects the schedule is a
plain Python function. This one enforces a hardware limit the scheduler
does not know about: at most ``max_drives`` drive lines may play at once.

Transforms of the IR itself are written in C++ as MLIR passes and loaded
into ``cudaq-pulse-opt`` as plugins; see ``cpp/DriveCount.cpp``.
"""

import cudaq_pulse as pulse


def concurrent_drive_violations(
    compiled: pulse.CompiledKernel, max_drives: int
) -> list[tuple[int, int]]:
    """Return (time, active drives) wherever more than *max_drives* overlap."""
    edges = []
    for event in compiled.events():
        if event.kind == "drive" and event.duration_vtu:
            edges.append((event.start_vtu, 1))
            edges.append((event.start_vtu + event.duration_vtu, -1))
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
        line, tone = get_drive_line(qs[i])
        drive(line, drag(40, 0.25, 10.0, 0.5), tone)


def main() -> None:
    args = [pulse.qvec_ref(4)]
    freq = {i: 5.0e9 + 0.1e9 * i for i in range(4)}

    asap = pulse.compile(parallel_x, args, qubit_freq_hz=freq, schedule="asap")
    print("ASAP violations of a 2-drive limit:", concurrent_drive_violations(asap, 2))

    rcp = pulse.compile(
        parallel_x,
        args,
        qubit_freq_hz=freq,
        schedule="rcp",
        machine=pulse.MachineModel(max_concurrent_drives=2),
    )
    violations = concurrent_drive_violations(rcp, 2)
    print("RCP violations of a 2-drive limit:", violations)
    assert not violations


if __name__ == "__main__":
    main()
