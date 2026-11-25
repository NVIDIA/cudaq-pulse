# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle replacement beside coupling lines: a pulse on a coupling line is
time on each qudit the line reaches, as the qudit's own readout is, so an
idle window ends where it starts and the rest is filled."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._idle_windows import gaps, read
from cudaq_pulse.ops import (
    drive,
    get_coupling_line,
    get_drive_line,
    square,
    sync,
    wait,
)
from cudaq_pulse.quantum_systems import Coupling, PhysicalQudit
from cudaq_pulse.quantum_systems.tuning import ExchangeControl, FluxTuning

_TUNING = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.0)
_EXCHANGE = ExchangeControl(
    unit="flux_quantum", amplitudes=(-0.4, 0.0, 0.4), exchange_hz=(9.0e6, 4.0e6, 9.0e6)
)
QS = pulse.quantum_system(
    name="tuned",
    qudits={
        i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, 50.0, 30.0, anharmonicity_hz=-3.0e8)
        for i in range(2)
    },
    couplings=[Coupling((0, 1), 4.0e6)],
    tunings={0: _TUNING, 1: _TUNING},
    controls={(0, 1): _EXCHANGE},
)
A, D = QS.qudits[0], QS.qudits[1]


@pulse.kernel
def echo(qudit, window):
    line = get_drive_line(qudit)
    lead = (window - 40) // 2
    wait(line, lead)
    drive(line, square(40, 0.1))
    wait(line, window - 40 - lead)


@pulse.kernel
def mixed(a, d):
    ca = get_coupling_line(a)
    da, dd = get_drive_line(a), get_drive_line(d)
    drive(da, square(40, 0.1))
    drive(ca, square(400, 0.1))
    drive(dd, square(400, 0.1))
    sync(da, dd, ca)
    drive(da, square(40, 0.1))


def _rule(fill=echo, **settings):
    return pulse.IdleReplacement(fill, **({"threshold": 100 * pulse.ns} | settings))


def _idled(kernel, schedule="asap", **settings):
    compiled = pulse.compile(kernel, quantum_system=QS, schedule=schedule)
    return compiled.replace_idle(_rule(**settings)).lower(A, D)


def _reports(lowered):
    return [(w.qudit, w.start, w.end, w.filled, w.before, w.result) for w in lowered.idle_windows]


def _windows(lowered):
    return [(w.qudit, w.start, w.end, w.rounds, w.filled) for w in lowered.idle_windows]


def _on_drive_line(lowered, qudit):
    """The events on the drive line of *qudit*, as (kind, start, end)."""
    lines = lowered.lines
    return [
        (e.kind, e.start_cycles, e.start_cycles + e.duration_cycles)
        for e in lowered.schedule
        if (lines[e.line].role, lines[e.line].qudits) == ("drive", (qudit,))
    ]


def test_a_window_ends_where_a_coupling_line_of_the_qudit_starts_to_play():
    # Qudit a's drive line is free from 40 to 400, and its coupling line
    # plays from 0 to 400: the qudit does not idle, and nothing is written.
    for schedule in ("asap", "alap"):
        lowered = _idled(mixed, schedule, leading=True)
        plain = pulse.compile(mixed, quantum_system=QS, schedule=schedule).lower(A, D)
        assert [w for w in _windows(lowered) if w[0] == 0] == []
        assert _on_drive_line(lowered, 0) == _on_drive_line(plain, 0)

    @pulse.kernel
    def later(a, d):
        ca = get_coupling_line(a)
        da, dd = get_drive_line(a), get_drive_line(d)
        drive(da, square(40, 0.1))
        drive(dd, square(1000, 0.1))
        wait(ca, 600)
        drive(ca, square(200, 0.1))
        sync(da, dd, ca)
        drive(da, square(40, 0.1))

    # The coupling line waits (the qudit idles) and then plays 600 to 800:
    # the window from 40 ends at 600, and the rest, 800 to 1000, is another.
    lowered = _idled(later)
    assert _windows(lowered) == [(0, 40, 600, 1, True), (0, 800, 1000, 1, True)]
    # The window names what it ends at as the kernel wrote it.
    assert lowered.idle_windows[0].before == "drive() on a coupling line"
    # The lanes of rcp place it alike, and the proof of the filled program
    # (nothing of the kernel moved, the coupling pulse included) holds.
    assert _windows(_idled(later, "rcp")) == _windows(lowered)
    written = [span for span in _on_drive_line(lowered, 0) if span[0] == "drive"]
    assert written == [
        ("drive", 0, 40),
        ("drive", 300, 340),
        ("drive", 880, 920),
        ("drive", 1000, 1040),
    ]
    assert not any(600 < end and start < 800 for _, start, end in written)


def test_the_line_of_a_pair_is_time_on_both_of_its_qudits():
    @pulse.kernel
    def pair(a, d):
        p = get_coupling_line(a, d)
        da, dd = get_drive_line(a), get_drive_line(d)
        drive(da, square(40, 0.1))
        drive(dd, square(40, 0.1))
        wait(p, 300)
        drive(p, square(200, 0.1))
        wait(p, 500)
        sync(da, dd, p)
        drive(da, square(40, 0.1))
        drive(dd, square(40, 0.1))

    lowered = _idled(pair)
    assert _windows(lowered) == [
        (0, 40, 300, 1, True),
        (0, 500, 1000, 1, True),
        (1, 40, 300, 1, True),
        (1, 500, 1000, 1, True),
    ]


def test_a_coupling_pulse_in_a_rolled_loop_cuts_the_window_of_every_round():
    @pulse.kernel
    def rounds(a, d):
        ca = get_coupling_line(a)
        da, dd = get_drive_line(a), get_drive_line(d)
        for _ in range(5):
            drive(da, square(40, 0.1))
            drive(ca, square(200, 0.1))
            drive(dd, square(900, 0.1))
            sync(da, dd, ca)

    lowered = _idled(rounds)
    assert "scf.for" in lowered.mlir()
    # Each round: the drive line is free from 40, the coupling line plays
    # to 200, the round ends at 900.
    assert _windows(lowered) == [(0, 200, 900, 5, True)]


def test_a_sync_meets_where_a_coupling_line_it_names_is_free():
    @pulse.kernel
    def meets(a, d):
        da, dd = get_drive_line(a), get_drive_line(d)
        cd = get_coupling_line(d)
        drive(da, square(40, 0.1))
        drive(cd, square(700, 0.1))
        sync(da, cd)
        drive(dd, square(900, 0.1))
        sync(da, dd)
        drive(da, square(40, 0.1))

    # The coupling line of qudit d is the last to be free, at 700: the sync
    # meets there and cuts the gap of qudit a's drive line there.
    plain = pulse.compile(meets, quantum_system=QS, schedule="asap").lower(A, D)
    lines, items = read(plain._module)
    assert lines == [("drive", 0), ("drive", 1), ("coupling", 1)]
    found = gaps(lines, items)
    cuts = [
        (a, b)
        for gap in found[0]
        for execution in gap.executions
        for _, a, b, what in execution.inner
        if what == "sync"
    ]
    assert (700, 700) in cuts
    assert _windows(_idled(meets)) == [(0, 40, 700, 1, True), (0, 700, 900, 1, True)]


def _gaps(kernel, *args):
    lowered = pulse.compile(kernel, quantum_system=QS, schedule="asap").lower(*args)
    lines, items = read(lowered._module)
    return lowered, {
        qudit: [
            (
                e.start,
                e.end,
                e.idle(),
                sorted((what, a, b) for _, a, b, what in e.inner),
                sorted(e.holes),
                sorted(e.repeats),
                e.uneven,
                e.first,
                e.last,
                e.rounds,
            )
            for gap in found
            for e in gap.executions
        ]
        for qudit, found in gaps(lines, items).items()
    }


def test_a_coupling_line_that_never_plays_leaves_the_gaps_as_they_are():
    @pulse.kernel
    def waits(a, d, with_line):
        da, dd = get_drive_line(a), get_drive_line(d)
        if with_line:
            wait(get_coupling_line(a), 300)
        for _ in range(4):
            drive(da, square(40, 0.1))
            drive(dd, square(900, 0.1))
            sync(da, dd)
        drive(da, square(40, 0.1))

    # The module with the line is read by the walk of a module that holds
    # a coupling line; a wait on the line is not time on the qudit.
    held, with_line = _gaps(waits, A, D, True)
    plain, without = _gaps(waits, A, D, False)
    assert "pulse.coupling_lines" in held.mlir() and "pulse.coupling_lines" not in plain.mlir()
    assert with_line == without and len(with_line[0]) > 1


def test_a_fill_does_not_fetch_a_coupling_line():
    @pulse.kernel
    def tuning_fill(qudit, window):
        drive(get_coupling_line(qudit), square(window, 0.1))

    @pulse.kernel
    def plain(a, d):
        da, dd = get_drive_line(a), get_drive_line(d)
        drive(da, square(40, 0.1))
        drive(dd, square(400, 0.1))
        sync(da, dd)
        drive(da, square(40, 0.1))

    compiled = pulse.compile(plain, quantum_system=QS, schedule="asap")
    with pytest.raises(pulse.CompilationError) as raised:
        compiled.replace_idle(_rule(tuning_fill)).lower(A, D)
    message = str(raised.value)
    assert "fill tuning_fill() fetches a coupling line" in message
    assert "drives and waits on its own qudit's drive line" in message


def test_a_readout_and_a_coupling_pulse_each_end_a_window_of_the_qudit():
    from cudaq_pulse.ops import get_readout_line, readout

    @pulse.kernel
    def both(a, d):
        ca = get_coupling_line(a)
        da, dd, ra = get_drive_line(a), get_drive_line(d), get_readout_line(a)
        drive(da, square(40, 0.1))
        drive(dd, square(2000, 0.1))
        wait(ca, 300)
        drive(ca, square(200, 0.1))
        wait(ra, 900)
        readout(ra, square(300, 0.2))
        sync(da, dd, ca, ra)
        drive(da, square(40, 0.1))

    reports = _reports(_idled(both))
    assert [row[:4] for row in reports] == [
        (0, 40, 300, True),
        (0, 500, 900, True),
        (0, 1200, 2000, True),
    ]
    assert reports[0][4] == "drive() on a coupling line"
    assert reports[1][4].startswith("readout() at line ")


def test_a_coupling_line_that_plays_in_every_round_of_a_loop_leaves_the_window_alone():
    @pulse.kernel
    def looped(a, d):
        ca = get_coupling_line(a)
        da = get_drive_line(a)
        drive(da, square(40, 0.1))
        for _ in range(1000):
            drive(ca, square(100, 0.1))
            wait(ca, 400)
        sync(da, ca)
        drive(da, square(40, 0.1))

    lowered = _idled(looped)
    plain = pulse.compile(looped, quantum_system=QS, schedule="asap").lower(A, D)
    assert lowered.mlir() == plain.mlir()
    (window,) = lowered.idle_windows
    assert (window.start, window.end, window.filled) == (499600, 500000, False)
    assert window.result == (
        "left alone: a loop reads the qudit out, plays a coupling line that reaches it, or "
        "plays its tone in every round meanwhile"
    )


def test_a_coupling_pulse_in_an_arm_of_a_branch_leaves_its_windows_alone():
    from cudaq_pulse.ops import get_readout_line, readout

    @pulse.kernel
    def armed(a, d):
        ca = get_coupling_line(a)
        da, rd = get_drive_line(a), get_readout_line(d)
        drive(da, square(40, 0.1))
        m = readout(rd, square(300, 0.2))
        if m:
            drive(ca, square(200, 0.1))
        wait(ca, 600)
        sync(da, ca, rd)
        drive(da, square(40, 0.1))

    reports = _reports(_idled(armed))
    assert [row[:4] for row in reports] == [(0, 40, 300, False), (0, 500, 1100, False)]
    assert {row[5] for row in reports} == {"left alone: in or next to an arm of a branch"}
