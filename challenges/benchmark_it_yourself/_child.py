# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_rb.py

Its qubit does not decay. lab.run compiles a kernel for it and sends the scorer, as JSON lines,
which calibrated pulses the kernel plays and when; the scorer answers with the populations.
"""

import importlib.util
import json
import os
import sys

import cudaq_pulse as pulse
from _qubit import benchmark_system, calibrated

channel = os.fdopen(os.dup(sys.stdout.fileno()), "w")  # the scorer's line; prints go to stderr
sys.stdout = sys.stderr
PUBLIC = benchmark_system(0.0, 0.0)  # 0 is no decay
x90, y90, _, measurement = (
    pulse.compile(calibrated, quantum_system=PUBLIC)
    .lower(PUBLIC.qudits[0])
    .export_to_controller()["events"]
)
NAMES = {
    json.dumps(x90["waveform"], sort_keys=True): "X90",
    json.dumps(y90["waveform"], sort_keys=True): "Y90",
}


def tell(message):
    channel.write(json.dumps(message) + "\n")
    channel.flush()


class Result:
    def __init__(self, populations):
        self._populations = tuple(populations)

    def level_population(self, level):
        return self._populations[level]

    def level_populations(self, qudit=None):
        return self._populations


class Lab:
    system, qubit = PUBLIC, PUBLIC.qudits[0]

    @staticmethod
    def run(kernel, *args):
        program = pulse.compile(kernel, quantum_system=PUBLIC).lower(*args)
        pulses, readouts, others = [], [], []
        for event in program.export_to_controller()["events"]:
            if event["kind"] == "drive":
                name = NAMES.get(json.dumps(event["waveform"], sort_keys=True))
                if name is None or event["frequency_hz"] != x90["frequency_hz"]:
                    others.append("drive pulse that is not the calibrated X90 or Y90")
                else:
                    pulses.append([int(event["start_cycles"]), name, float(event["phase_rad"])])
            elif event["kind"] == "readout":
                if event["waveform"] != measurement["waveform"]:
                    others.append("readout that is not the calibrated one")
                readouts.append(int(event["start_cycles"]))
            elif event["kind"] not in ("wait", "sync", "frame_update") or event.get("conditions"):
                others.append(str(event["kind"]))
        tell({"run": {"pulses": pulses, "readouts": readouts, "others": others}})
        answer = json.loads(sys.stdin.readline())
        if "error" in answer:
            raise ValueError(answer["error"])
        return Result(answer["populations"])


spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
submission = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(submission)
    tell({"loaded": True})
except BaseException as error:
    tell({"error": f"{type(error).__name__}: {error}"})
    raise SystemExit(1) from None
for line in sys.stdin:
    if not line.startswith("SEED "):
        break
    try:
        tell({"estimate": float(submission.error_per_gate(Lab(), int(line.split()[1])))})
    except Exception as error:
        tell({"error": f"{type(error).__name__}: {error}"})
