# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side comparison of ASAP, ALAP, and RCP scheduling policies.

Compiles a 4-qubit kernel with varying drive durations and a sync under
each policy, then prints the resulting timelines.
"""

import cudaq_pulse as pulse

# Qubit i's pulses last base_duration + i * duration_step.
cal = {"base_duration": 20 * pulse.ns, "duration_step": 10 * pulse.ns, "sigma": 5 * pulse.ns}


@pulse.kernel
def four_qudit(qs):
    lines, tones = [], []
    for i in range(4):
        line = get_drive_line(qs[i])
        lines.append(line)
        tones.append(line.tone)
    durations = [cal["base_duration"] + i * cal["duration_step"] for i in range(4)]
    for i in range(4):
        drive(lines[i], gaussian(durations[i], 0.25, cal["sigma"]), tone=tones[i])
    sync(lines[0], lines[1], lines[2], lines[3])
    for i in range(4):
        drive(lines[i], gaussian(durations[i], 0.25, cal["sigma"]), tone=tones[i])


def print_timeline(compiled: pulse.ScheduledKernel, label: str) -> int:
    """Pretty-print the scheduled drives and return the makespan."""
    print(f"\n  --- {label} ---")
    events = sorted(compiled.schedule, key=lambda e: (e.start, e.line))
    for ev in events:
        end = ev.end
        print(f"    [{ev.start:5d} - {end:5d}]  line {ev.line}  {ev.kind}")
    return max(e.end for e in events)


def main():
    args = [qs.qudits[:4]]
    control_system = pulse.ControlSystem(max_concurrent_drives=2, line_switch_penalty_cycles=5)
    totals = {}
    for policy in ("asap", "alap", "rcp"):
        compiled = pulse.compile(
            four_qudit, schedule=policy, control_system=control_system, quantum_system=qs
        ).lower(*args)
        label = policy.upper()
        if policy == "rcp":
            label += f" (max {control_system.max_concurrent_drives} concurrent drives)"
        totals[policy] = (print_timeline(compiled, label), compiled.metrics.schedule_ms)

    print("\n  Policy   Total cycles  Schedule ms")
    for policy, (total, ms) in totals.items():
        print(f"  {policy.upper():<8} {total:>9d}   {ms:>11.3f}")


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    main()
