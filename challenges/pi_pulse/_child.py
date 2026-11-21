# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run by score.py in a process of its own: the only code that loads the submission.

    python _child.py my_pulse.py QUDIT

Compiles the submission's ``pi_pulse`` for the qudit and prints, as one line of JSON, what the
kernel puts on the drive line: each pulse's start, samples, frequency and phase.
"""

import importlib.util
import json
import sys

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.controller import sample_waveform

spec = importlib.util.spec_from_file_location("submission", sys.argv[1])
assert spec is not None and spec.loader is not None
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
system = pulse.quantum_system("transmon_surface17")
program = pulse.compile(submission.pi_pulse, quantum_system=system).lower(system.qudit(sys.argv[2]))
pulses, others, total = [], [], 0
for event in program.export_to_controller()["events"]:
    total = max(total, int(event["end_cycles"]))
    if event["kind"] == "drive":
        samples = np.asarray(
            sample_waveform(event["waveform"], event["duration_cycles"]), dtype=complex
        )
        pulses.append(
            {
                "start_cycles": int(event["start_cycles"]),
                "re": [float(x) for x in samples.real],
                "im": [float(x) for x in samples.imag],
                "frequency_hz": float(event["frequency_hz"]),
                "phase_rad": float(event["phase_rad"]),
            }
        )
    elif event["kind"] not in ("wait", "frame_update"):
        others.append(str(event["kind"]))
message = {"pulses": pulses, "others": others, "total_cycles": total}
sys.stdout.write("PULSES " + json.dumps(message) + "\n")
