# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the solution.

    python _child.py solution.py

It holds the public qubit only (no T1). lab.run compiles a kernel for it and sends the scorer, as
JSON lines, what the kernel puts on the drive and readout lines; the scorer answers with the shots.
"""

import importlib.util
import json
import os
import sys

import cudaq_pulse as pulse
import numpy as np
from _qubit import mystery_system
from cudaq_pulse.controller import sample_waveform

channel = os.fdopen(os.dup(sys.stdout.fileno()), "w")  # the scorer's line; prints go to stderr
sys.stdout = sys.stderr
PUBLIC = mystery_system(0.0)  # T1 left blank (0 is no decay)


def tell(message):
    channel.write(json.dumps(message) + "\n")
    channel.flush()


def numbers(event):
    samples = np.asarray(
        sample_waveform(event["waveform"], event["duration_cycles"]), dtype=complex
    )
    return {
        "start_cycles": int(event["start_cycles"]),
        "re": [float(x) for x in samples.real],
        "im": [float(x) for x in samples.imag],
        "frequency_hz": float(event["frequency_hz"]),
        "phase_rad": float(event["phase_rad"]),
    }


class Lab:
    system, qubit = PUBLIC, PUBLIC.qudits[0]

    @staticmethod
    def run(kernel, *args, shots_count):
        program = pulse.compile(kernel, quantum_system=PUBLIC).lower(*args)
        drives, readouts, others = [], [], []
        for event in program.export_to_controller()["events"]:
            if event["kind"] == "drive":
                drives.append(numbers(event))
            elif event["kind"] == "readout":
                shape, params = event["waveform"]["shape"], event["waveform"]["params"]
                readouts.append(
                    {
                        "shape": str(shape),
                        "start_cycles": int(event["start_cycles"]),
                        "duration_cycles": int(event["duration_cycles"]),
                        "re": float(params.get("amplitude", 0.0)),
                        "im": float(params.get("amplitude_imag", 0.0)),
                        "frequency_hz": float(event["frequency_hz"]),
                        "phase_rad": float(event["phase_rad"]),
                    }
                )
            elif event["kind"] not in ("wait", "sync", "frame_update") or event.get("conditions"):
                others.append(str(event["kind"]))
        tell(
            {
                "run": {
                    "shots_count": shots_count,
                    "drives": drives,
                    "readouts": readouts,
                    "others": others,
                }
            }
        )
        answer = json.loads(sys.stdin.readline())
        if "error" in answer:
            raise RuntimeError(answer["error"])
        return np.asarray(answer["records"], dtype=int).reshape(-1, 1)


spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
solution = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(solution)
    tell({"loaded": True})
except BaseException as error:
    tell({"error": f"{type(error).__name__}: {error}"})
    raise SystemExit(1) from None
for line in sys.stdin:
    if line.strip() != "QUBIT":
        break
    try:
        tell({"estimate": float(solution.estimate_t1(Lab()))})
    except Exception as error:
        tell({"error": f"{type(error).__name__}: {error}"})
