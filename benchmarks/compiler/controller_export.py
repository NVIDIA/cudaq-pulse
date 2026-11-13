# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost of checking and exporting large programs for a controller.

Three programs, each compiled with controller limits so that compiling
checks them:

- ``unrolled``: ``ROUNDS`` rounds of a many-line round (a drive and a frame
  change on every drive line, a sync, a readout on every readout line, a
  sync), unrolled while tracing;
- ``rolled``: the same round in a rolled loop;
- ``nest``: a small body in four nested rolled loops of ``TRIPS`` each.

For each, in a fresh process: compile time, export time and bytes of the
default export and of the loop-preserving one (when this cudaq-pulse has
it), and the peak resident memory of the whole run.

    PYTHONPATH=build/dev/python python benchmarks/compiler/controller_export.py
    ... --small     # a quick run at small sizes (CI)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

_CHILD = r"""
import inspect, json, resource, sys, time
import cudaq_pulse as pulse
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.quantum_systems import PhysicalQudit

name, lines, rounds, trips = sys.argv[1], *map(int, sys.argv[2:5])
dev = pulse.quantum_system(
    name="export-bench",
    clock_ghz=1.0,
    qudits={i: PhysicalQudit(i, 5.0e9 + 1.0e6 * i, 50.0, 30.0, anharmonicity_hz=-2.0e8) for i in range(lines)},
)
readers = lines // 2
limits = Controller(
    clock_ghz=1.0,
    waveform_memory_samples=1 << 20,
    max_tones_per_line=4,
    detectors={q: q for q in range(readers)},
    detector_recovery_cycles=4,
    max_pulses=10**8,
)
system = ControlSystem(controllers=[limits])


@pulse.kernel
def one_round(qs):
    drives = []
    for q in qs:
        d = get_drive_line(q)
        drive(d, gaussian(16, 0.1, 4.0))
        shift_phase(d.tone, 0.25)
        drives.append(d)
    probes = [get_readout_line(qs[q]) for q in range(readers)]
    sync(*drives, *probes)
    for r in probes:
        readout(r, square(40, 0.05))
    sync(*drives, *probes)


@pulse.kernel
def repeated(qs):
    for _ in range(rounds):
        one_round(qs)


@pulse.kernel
def nest(qs):
    d = get_drive_line(qs[0])
    r = get_readout_line(qs[0])
    e = get_drive_line(qs[1])
    for _ in range(trips):
        for _ in range(trips):
            for _ in range(trips):
                for _ in range(trips):
                    drive(d, gaussian(16, 0.1, 4.0))
                    drive(e, square(16, 0.1))
                    sync(d, r, e)
                    readout(r, square(40, 0.05))
                    sync(d, r, e)


kernel = nest if name == "nest" else repeated
threshold = 10**9 if name == "unrolled" else 0
handles = dev.qudits[0:lines]
t0 = time.perf_counter()
lowered = pulse.compile(kernel, control_system=system, unroll_threshold=threshold).lower(handles)
compile_s = time.perf_counter() - t0
t0 = time.perf_counter()
program = lowered.export_to_controller()
export_s = time.perf_counter() - t0
events = len(program["events"])
default_bytes = len(json.dumps(program))
del program
out = {
    "program": name, "events": events, "compile_s": compile_s,
    "export_s": export_s, "export_bytes": default_bytes,
}
if "version" in inspect.signature(lowered.export_to_controller).parameters:
    t0 = time.perf_counter()
    rolled = lowered.export_to_controller(version=2)
    out["rolled_export_s"] = time.perf_counter() - t0
    out["rolled_export_bytes"] = len(json.dumps(rolled))
scale = 1 if sys.platform == "darwin" else 1024
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20
print(json.dumps(out))
"""


def run(name: str, lines: int, rounds: int, trips: int) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, name, str(lines), str(rounds), str(trips)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(f"the {name} program's process exited with status {out.returncode}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--small", action="store_true", help="small sizes, for CI")
    parser.add_argument("--programs", default="unrolled,rolled,nest")
    args = parser.parse_args()
    lines, rounds, trips = (8, 40, 4) if args.small else (32, 6400, 16)
    for name in args.programs.split(","):
        print(json.dumps(run(name, lines, rounds, trips)), flush=True)


if __name__ == "__main__":
    main()
