# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel scheduling policies in a surface-code-style stabilizer cycle.

One ancilla measures the parity of four data qubits, round after round:

  1. a layer of single-qubit pulses on the data qubits,
  2. four cross-resonance pulses from the ancilla, one per data qubit,
  3. the same single-qubit layer again,
  4. readout of the ancilla, with a conditional reset,
  5. dynamical decoupling (XY4) on the data qubits while they wait.

Each step is a kernel. ``single_qudit_layer`` and ``xy4`` are declared with
``schedule="asap"``, and ``measure_and_reset`` with ``schedule="alap"``:
each is scheduled on its own with that policy, then placed whole by the
cycle's ALAP schedule. So the layer's pulses always start together, the
decoupling sequence keeps its spacing, and the readout ends where the reset
begins, wherever the cycle puts them. ``cross_resonance`` sets no policy and
is scheduled together with the cycle.

``compile(..., flatten_schedules=True)`` schedules the same program as one
ALAP schedule, for comparison: the layer's pulses are then pushed apart, each
up to the cross-resonance pulse that needs its tone.
"""

import cudaq_pulse as pulse

rounds = 2
data_qudits = 4
cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "cr_duration": 100 * pulse.ns,
    "cr_width": 80 * pulse.ns,
    "readout_duration": 400 * pulse.ns,
    "dd_tau": 60 * pulse.ns,
}


@pulse.kernel(schedule="asap")
def single_qudit_layer(lines, tones):
    """A sqrt(X) on every data qubit, all starting together."""
    for i in range(len(lines)):
        drive(
            lines[i], drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"]), tone=tones[i]
        )


@pulse.kernel
def cross_resonance(control, target_tone):
    """Drive the control qubit at the target's frequency."""
    cr = gaussian_square(cal["cr_duration"], 0.05, cal["x_sigma"], cal["cr_width"])
    drive(control, cr, tone=target_tone)


@pulse.kernel(schedule="alap")
def measure_and_reset(drive_line, readout_line):
    """Read the ancilla out, then flip it back to 0 if it was 1."""
    sync(drive_line, readout_line)
    result = readout(readout_line, square(cal["readout_duration"], 0.05))
    if result:
        drive(drive_line, drag(cal["x_duration"], 0.5, cal["x_sigma"], cal["drag_beta"]))
    return result


@pulse.kernel(schedule="asap")
def xy4(line, tone, tau):
    """X, Y, X, Y pi pulses, each followed by an idle of tau."""
    for phase in (0.0, 1.5708, 0.0, 1.5708):
        shift_phase(tone, phase)
        drive(line, drag(cal["x_duration"], 0.5, cal["x_sigma"], cal["drag_beta"]), tone=tone)
        shift_phase(tone, -phase)
        wait(line, tau)


@pulse.kernel
def stabilizer_cycle(qudits):
    ancilla = get_drive_line(qudits[0])
    readout_line = get_readout_line(qudits[0])
    lines, tones = [], []
    for q in qudits[1:]:
        line = get_drive_line(q)
        lines.append(line)
        tones.append(line.tone)

    for _ in range(rounds):
        single_qudit_layer(lines, tones)
        for i in range(data_qudits):
            cross_resonance(ancilla, tones[i])
        single_qudit_layer(lines, tones)
        measure_and_reset(ancilla, readout_line)
        for i in range(data_qudits):
            xy4(lines[i], tones[i], cal["dd_tau"])


def first_round(compiled: pulse.ScheduledKernel) -> dict[int, list[tuple[str, int, int]]]:
    """The first round's (kind, start, duration) on each line."""
    per_line: dict[int, list[tuple[str, int, int]]] = {}
    for event in compiled.schedule:
        per_line.setdefault(event.line, []).append(
            (event.kind, event.start, event.end - event.start)
        )
    events_per_round = {line: len(events) // rounds for line, events in per_line.items()}
    return {line: events[: events_per_round[line]] for line, events in per_line.items()}


def report(label: str, compiled: pulse.ScheduledKernel) -> None:
    timeline = first_round(compiled)
    data_lines = sorted(timeline)[2:]
    makespan = max(e.end for e in compiled.schedule)
    print(f"\n  --- {label} ---")
    print(f"    makespan: {makespan} clock cycles")
    layer = [timeline[line][0][1] for line in data_lines]
    print(f"    first layer starts on the data lines: {layer}")
    readout = next(e for e in timeline[1] if e[0] == "readout")
    print(f"    ancilla readout: [{readout[1]}, {readout[1] + readout[2]}]")
    dd = [start for kind, start, _ in timeline[data_lines[0]][2:] if kind == "drive"]
    print(f"    XY4 pulses on the first data line: {dd}")


def main():
    qs = pulse.quantum_system("transmon_surface17")
    qudits = [qs.qudits[: 1 + data_qudits]]
    report(
        "per-kernel schedules",
        stabilizer_cycle.lower(*qudits),
    )
    report(
        "flattened into one ALAP schedule",
        pulse.compile(stabilizer_cycle, flatten_schedules=True).lower(*qudits),
    )


if __name__ == "__main__":
    main()
