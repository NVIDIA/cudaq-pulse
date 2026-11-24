# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random kernels that mix drive, readout and coupling lines schedule the
same with their loops rolled and unrolled, under every scheduling policy."""

from __future__ import annotations

import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.ops import (
    drive,
    gaussian,
    get_coupling_line,
    get_drive_line,
    get_readout_line,
    readout,
    square,
    sync,
    wait,
)
from cudaq_pulse.quantum_systems import Coupling, ExchangeControl, FluxTuning, PhysicalQudit

_TUNING = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.0)
QS = pulse.quantum_system(
    name="fuzzed",
    qudits={
        i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, 50.0, 30.0, anharmonicity_hz=-3.0e8, tuning=_TUNING)
        for i in range(3)
    },
    couplings=[
        Coupling(
            (0, 1),
            4.0e6,
            control=ExchangeControl(
                unit="flux_quantum", amplitudes=(-0.4, 0.0, 0.4), exchange_hz=(9e6, 4e6, 9e6)
            ),
        )
    ],
)
POLICIES = ("asap", "alap", "rcp", "alap_rcp")
# Lines 0-2 drive, 3-5 readout, 6-8 each qudit's coupling line, 9 the
# coupling line of the pair (a, b).
DRIVES, READOUTS, COUPLINGS = range(0, 3), range(3, 6), range(6, 10)
SEEDS = 48


def _op(rng: random.Random) -> tuple:
    kind = rng.choice(["drive", "drive", "couple", "couple", "couple", "wait", "sync", "readout"])
    if kind == "drive":
        return ("drive", rng.choice(DRIVES), rng.choice([8, 16, 24]), rng.choice([0.1, 0.2]))
    if kind == "couple":
        return ("couple", rng.choice(COUPLINGS), rng.choice([4, 10, 30]), rng.choice([-0.2, 0.3]))
    if kind == "wait":
        return ("wait", rng.randrange(10), rng.choice([2, 6, 12]))
    if kind == "readout":
        return ("readout", rng.choice(READOUTS), rng.choice([20, 40]))
    return ("sync", tuple(sorted(rng.sample(range(10), rng.choice([2, 2, 3])))))


def _program(seed: int) -> tuple:
    rng = random.Random(seed)
    program: list[tuple] = []
    for _ in range(rng.randrange(3, 7)):
        if rng.random() < 0.6:
            body = tuple(_op(rng) for _ in range(rng.randrange(1, 5)))
            program.append(("loop", rng.choice([2, 3, 7]), body))
        else:
            program.append(_op(rng))
    return tuple(program)


def _kernel(program: tuple):
    @pulse.kernel
    def fuzzed(a, b, c):
        lines = [
            get_drive_line(a),
            get_drive_line(b),
            get_drive_line(c),
            get_readout_line(a),
            get_readout_line(b),
            get_readout_line(c),
            get_coupling_line(a),
            get_coupling_line(b),
            get_coupling_line(c),
            get_coupling_line(a, b),
        ]
        for item in program:
            if item[0] == "loop":
                for _ in range(item[1]):
                    for inner in item[2]:
                        if inner[0] == "drive":
                            drive(lines[inner[1]], gaussian(inner[2], inner[3], inner[2] / 4))
                        elif inner[0] == "couple":
                            drive(lines[inner[1]], square(inner[2], inner[3]))
                        elif inner[0] == "wait":
                            wait(lines[inner[1]], inner[2])
                        elif inner[0] == "readout":
                            readout(lines[inner[1]], square(inner[2], 0.05))
                        elif len(inner[1]) == 2:
                            sync(lines[inner[1][0]], lines[inner[1][1]])
                        else:
                            sync(lines[inner[1][0]], lines[inner[1][1]], lines[inner[1][2]])
            elif item[0] == "drive":
                drive(lines[item[1]], gaussian(item[2], item[3], item[2] / 4))
            elif item[0] == "couple":
                drive(lines[item[1]], square(item[2], item[3]))
            elif item[0] == "wait":
                wait(lines[item[1]], item[2])
            elif item[0] == "readout":
                readout(lines[item[1]], square(item[2], 0.05))
            elif len(item[1]) == 2:
                sync(lines[item[1][0]], lines[item[1][1]])
            else:
                sync(lines[item[1][0]], lines[item[1][1]], lines[item[1][2]])

    return fuzzed


def _schedule(kernel, policy: str, threshold: int):
    compiled = pulse.compile(kernel, quantum_system=QS, schedule=policy, unroll_threshold=threshold)
    lowered = compiled.lower(QS.qudits[0], QS.qudits[1], QS.qudits[2])
    events = [
        (e.kind, e.line, e.qudit, e.start_cycles, e.duration_cycles) for e in lowered.schedule
    ]
    return events, lowered.duration, lowered.mlir()


@pytest.mark.parametrize("policy", POLICIES)
def test_rolled_and_unrolled_kernels_schedule_alike(policy):
    rolled_loops = coupling_in_loops = 0
    for seed in range(SEEDS):
        program = _program(seed)
        kernel = _kernel(program)
        rolled, rolled_duration, text = _schedule(kernel, policy, 0)
        unrolled, unrolled_duration, flat = _schedule(kernel, policy, 10**6)
        assert rolled == unrolled, (seed, program)
        assert rolled_duration == unrolled_duration, (seed, program)
        assert "scf.for" not in flat
        if "scf.for" in text:
            rolled_loops += 1
            coupling_in_loops += "-> (!pulse.coupling_line" in text or (
                ", !pulse.coupling_line" in text.split("scf.for", 1)[1].split("\n", 1)[0]
            )
        # Every event is scheduled, and each line's events follow one another.
        ends: dict[int, int] = {}
        for _kind, line, _qudit, start, duration in rolled:
            assert start is not None and duration is not None
            assert start >= ends.get(line, 0), (seed, line)
            ends[line] = start + duration
    # The comparison is of loops that stayed rolled, coupling lines among
    # what they carry.
    assert rolled_loops >= SEEDS // 2, rolled_loops
    assert coupling_in_loops >= SEEDS // 4, coupling_in_loops
