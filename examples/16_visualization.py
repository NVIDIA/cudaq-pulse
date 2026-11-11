# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulse schedule visualization with matplotlib.

Shows how to use the viz module to plot per-line Gantt charts of
compiled pulse kernels — useful for debugging timing, identifying
idle gaps, and verifying sync alignment.

Plots need matplotlib, from the viz extra: pip install "cudaq-pulse[viz]".
"""

from collections.abc import Sequence

import cudaq_pulse as pulse

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "cr_duration": 150 * pulse.ns,
    "cr_width": 130 * pulse.ns,
    "idle": 100 * pulse.ns,
    "readout_duration": 500 * pulse.ns,
}


@pulse.kernel
def vis_demo(q0, q1, q2):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    d2 = get_drive_line(q2)
    r0 = get_readout_line(q0)
    short = drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"])
    drive(d0, short)
    drive(d1, short)
    drive(d2, short)
    sync(d0, d1, d2)
    drive(
        d0, gaussian_square(cal["cr_duration"], 0.05, cal["x_sigma"], cal["cr_width"]), tone=d1.tone
    )
    wait(d2, cal["idle"])
    drive(d1, short)
    sync(d0, d1, d2, r0)
    readout(r0, square(cal["readout_duration"], 0.05))


def print_ascii_timeline(events: Sequence[pulse.compiler.ScheduledEvent], width: int = 70) -> None:
    """Render a simple ASCII timeline for terminals without matplotlib."""
    max_t = max(ev.end for ev in events)
    for line in sorted({ev.line for ev in events}):
        row = [" "] * width
        for ev in (ev for ev in events if ev.line == line):
            start = int(ev.start / max_t * (width - 1))
            end = int(ev.end / max_t * (width - 1))
            char = "D" if ev.kind == "drive" else ("R" if ev.kind == "readout" else ".")
            for c in range(start, min(end + 1, width)):
                row[c] = char
        print(f"  line {line}: |{''.join(row)}|")
    print(f"  {'':>8} 0{' ' * (width - 8)}{max_t} cycles")


def main():
    args = list(qs.qudits[:3])
    for policy in ("asap", "alap"):
        scheduled = pulse.compile(vis_demo, schedule=policy, quantum_system=qs).lower(*args)
        print(f"\n=== {policy.upper()} schedule ===")
        print_ascii_timeline(scheduled.schedule)

    try:
        scheduled.timeline(title="ALAP schedule (3 qubits)")
        print("\n  matplotlib figure created successfully.")
        print("  Save it with scheduled.timeline().savefig('output.png').")
    except ImportError:
        print("\n  matplotlib not installed. Install it with the viz extra: cudaq-pulse[viz]")


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    main()
