# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Side-by-side comparison of ASAP, ALAP, and RCP scheduling policies.

Compiles a 4-qubit kernel with varying drive durations and a sync under
each policy, then prints the resulting timelines.
"""

import cudaq_pulse as pulse


@pulse.kernel
def four_qubit(qs):
    lines, tones = [], []
    for i in range(4):
        line, tone = get_drive_line(qs[i])
        lines.append(line)
        tones.append(tone)
    for i in range(4):
        drive(lines[i], gaussian(40 + 20 * i, 0.25, 10.0), tones[i])
    sync(lines[0], lines[1], lines[2], lines[3])
    for i in range(4):
        drive(lines[i], gaussian(40 + 20 * i, 0.25, 10.0), tones[i])


def print_timeline(compiled: pulse.Lowered, label: str) -> int:
    """Pretty-print the scheduled drives and return the makespan."""
    print(f"\n  --- {label} ---")
    events = sorted(compiled.schedule, key=lambda e: (e.start_vtu, e.line))
    for ev in events:
        end = ev.start_vtu + ev.duration_vtu
        print(f"    [{ev.start_vtu:5d} - {end:5d}]  line {ev.line}  {ev.kind}")
    return max(e.start_vtu + e.duration_vtu for e in events)


def main():
    args = [dev.qudit_handles[:4]]
    machine = pulse.compiler.MachineModel(max_concurrent_drives=2, line_switch_penalty_vtu=5)
    totals = {}
    for policy in ("asap", "alap", "rcp"):
        compiled = pulse.compile(four_qubit, schedule=policy, machine=machine, device=dev).lower(
            *args
        )
        label = policy.upper()
        if policy == "rcp":
            label += f" (max {machine.max_concurrent_drives} concurrent drives)"
        totals[policy] = (print_timeline(compiled, label), compiled.metrics.schedule_ms)

    print("\n  Policy   Total VTU   Schedule ms")
    for policy, (total, ms) in totals.items():
        print(f"  {policy.upper():<8} {total:>9d}   {ms:>11.3f}")


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    main()
