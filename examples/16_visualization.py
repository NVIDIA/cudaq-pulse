# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pulse schedule visualization with matplotlib.

Shows how to use the viz module to plot per-line Gantt charts of
compiled pulse kernels — useful for debugging timing, identifying
idle gaps, and verifying sync alignment.

Plots need matplotlib, from the viz extra: pip install "cudaq-pulse[viz]".
"""

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
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    r0, rt0 = get_readout_line(q0)
    short = drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"])
    drive(d0, short, t0)
    drive(d1, short, t1)
    drive(d2, short, t2)
    sync(d0, d1, d2)
    drive(d0, gaussian_square(cal["cr_duration"], 0.05, cal["x_sigma"], cal["cr_width"]), t1)
    wait(d2, cal["idle"])
    drive(d1, short, t1)
    sync(d0, d1, d2, r0)
    readout(r0, square(cal["readout_duration"], 0.05), rt0)


def print_ascii_timeline(events: list[pulse.compiler.ScheduledEvent], width: int = 70) -> None:
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
    args = list(dev.qudit_handles[:3])
    for policy in ("asap", "alap"):
        compiled = pulse.compile(vis_demo, schedule=policy, device=dev).lower(*args)
        print(f"\n=== {policy.upper()} schedule ===")
        print_ascii_timeline(compiled.schedule)

    try:
        compiled.timeline(title="ALAP schedule (3 qubits)")
        print("\n  matplotlib figure created successfully.")
        print("  Save it with compiled.timeline().savefig('output.png').")
    except ImportError:
        print("\n  matplotlib not installed. Install it with the viz extra: cudaq-pulse[viz]")


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    main()
