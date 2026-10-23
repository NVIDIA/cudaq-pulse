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

Requires matplotlib: pip install matplotlib
"""

import cudaq_pulse as pulse


@pulse.kernel
def vis_demo(q0, q1, q2):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    r0, rt0 = get_readout_line(q0)
    short = drag(40, 0.25, 10.0, 0.5)
    drive(d0, short, t0)
    drive(d1, short, t1)
    drive(d2, short, t2)
    sync(d0, d1, d2)
    drive(d0, gaussian_square(300, 0.05, 10.0, 260), t1)
    wait(d2, 200)
    drive(d1, short, t1)
    sync(d0, d1, d2, r0)
    readout(r0, square(1000, 0.05), rt0)


def print_ascii_timeline(events: list[pulse.ScheduledEvent], width: int = 70) -> None:
    """Render a simple ASCII timeline for terminals without matplotlib."""
    max_t = max(ev.start_vtu + ev.duration_vtu for ev in events)
    for line in sorted({ev.line for ev in events}):
        row = [" "] * width
        for ev in (ev for ev in events if ev.line == line):
            start = int(ev.start_vtu / max_t * (width - 1))
            end = int((ev.start_vtu + ev.duration_vtu) / max_t * (width - 1))
            char = "D" if ev.kind == "drive" else ("R" if ev.kind == "readout" else ".")
            for c in range(start, min(end + 1, width)):
                row[c] = char
        print(f"  line {line}: |{''.join(row)}|")
    print(f"  {'':>8} 0{' ' * (width - 8)}{max_t} VTU")


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
        print("\n  matplotlib not installed. Install with: pip install matplotlib")


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    main()
