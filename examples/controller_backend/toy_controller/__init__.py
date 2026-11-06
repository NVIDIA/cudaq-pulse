# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A toy controller backend: the smallest thing ``cudaq_pulse.backends`` loads.

The toy controller runs at 1 GHz, issues one instruction every 2 cycles,
idles no less than 2 cycles and has one readout detector per pair of
qudits. Its program is text, one instruction per drive or readout:
``<line> <start> <kind> <shape> <amplitude> <frequency>``; running it
replays the instructions in time order and returns them with the outcome
a readout reports, which is always 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cudaq_pulse import Controller
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import FORMAT

__all__ = ["ToyBackend", "ToyResult"]


@dataclass(frozen=True)
class ToyResult:
    """The instructions played, in time order, and each shot's outcomes."""

    played: tuple[tuple[int, int, str], ...]
    outcomes: tuple[tuple[int, ...], ...]


class ToyBackend:
    name = "toy"
    formats = (FORMAT,)

    def __init__(self, qudits: int = 8) -> None:
        self.qudits = qudits

    def machine_model(self) -> MachineModel:
        limits = Controller(
            clock_ghz=1.0,
            instruction_cycles=2,
            min_idle_cycles=2,
            detectors={q: q // 2 for q in range(self.qudits)},
            detector_recovery_cycles=10,
            supports_timeouts=False,
        )
        return MachineModel(max_concurrent_readouts=1, controller=limits)

    def emit(self, program: Mapping[str, Any]) -> str:
        if program.get("format") != FORMAT:
            raise ValueError(f"expected a {FORMAT} program, got {program.get('format')!r}")
        lines = []
        for event in program["events"]:
            if event["kind"] in ("wait", "sync"):
                continue
            if event["kind"] not in ("drive", "readout") or event["conditions"]:
                raise ValueError(f"the toy controller cannot play {event['kind']} event {event}")
            amplitude = event["amplitude"] or {"re": 0.0}
            lines.append(
                f"{event['line']} {event['start_cycles']} {event['kind']} "
                f"{event['waveform']['shape']} {amplitude['re']:.6g} "
                f"{event['frequency_hz'] or 0.0:.6g}"
            )
        return "\n".join(lines) + "\n"

    def run(self, artifact: Any, *, shots: int = 1) -> ToyResult:
        played = []
        for text in str(artifact).splitlines():
            line, start, kind = text.split()[:3]
            played.append((int(start), int(line), kind))
        played.sort()
        readouts = sum(1 for _, _, kind in played if kind == "readout")
        return ToyResult(
            played=tuple((line, start, kind) for start, line, kind in played),
            outcomes=tuple((0,) * readouts for _ in range(shots)),
        )


if __name__ == "__main__":
    import cudaq_pulse as pulse
    from cudaq_pulse import backends
    from cudaq_pulse.targets import Qubit, TargetDescription

    # Installed, the backend is found by its entry point; here it is
    # registered in the process.
    backends.register("toy", ToyBackend)
    backend = backends.get("toy")
    target = TargetDescription(
        name="toy-device",
        qubits={0: Qubit(index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=0, t2_star_us=0)},
    )
    dev = pulse.device({"target": target, "clock_ghz": 1.0})
    cal = {
        "x_duration": 40 * pulse.ns,
        "x_sigma": 10 * pulse.ns,
        "readout_duration": 200 * pulse.ns,
    }

    @pulse.kernel
    def drive_and_measure(q):
        drive_line = get_drive_line(q)
        readout_line = get_readout_line(q)
        drive(drive_line, gaussian(cal["x_duration"], 0.5, cal["x_sigma"]))
        sync(drive_line, readout_line)
        readout(readout_line, square(cal["readout_duration"], 0.05))

    compiled = pulse.compile(drive_and_measure, device=dev, machine=backend.machine_model())
    program = backend.emit(compiled.lower(dev.qudit_handles[0]).export_controller())
    print(program, end="")
    print(backend.run(program, shots=2))
    backends.unregister("toy")
