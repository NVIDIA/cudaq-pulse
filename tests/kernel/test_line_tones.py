# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A line's tone, ``line.tone``, and its transitions' tones, ``line.tones``."""

from __future__ import annotations

import json
import math
from typing import ClassVar

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.quantum_systems import (
    MicrowaveDrive,
    PhysicalQudit,
    SimulationModel,
    TransmonModel,
)

from tests._handles import Q, rebuilt


def _same_program(a, b, *args):
    """Both kernels lower to the same MLIR and export, but for their names."""
    one, two = pulse.compile(a).lower(*args), pulse.compile(b).lower(*args)
    rename = {f"@{a.__name__}": "@k", f"@{b.__name__}": "@k"}

    def text(lowered):
        mlir = lowered.mlir()
        for old, new in rename.items():
            mlir = mlir.replace(old, new)
        return mlir

    assert text(one) == text(two)
    export_one = json.dumps(one.export_to_controller(), sort_keys=True)
    export_two = json.dumps(two.export_to_controller(), sort_keys=True)
    assert export_one.replace(a.__name__, "k") == export_two.replace(b.__name__, "k")


def _tones_of(lowered):
    return [line for line in lowered.mlir().splitlines() if "= pulse.tone " in line]


@pulse.kernel
def default_tones(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(40, 0.1, 10.0))
    shift_phase(d.tone, 0.5)
    set_frequency(d.tone, 5.01e9)
    drive(d, gaussian(40, 0.1, 10.0))
    sync(d, r)
    readout(r, square(100, 0.2))


@pulse.kernel
def explicit_tones(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(40, 0.1, 10.0))
    shift_phase(d.tones[(0, 1)], 0.5)
    set_frequency(d.tone, 5.01e9)
    drive(d, gaussian(40, 0.1, 10.0), tone=d.tones[(0, 1)])
    sync(d, r)
    readout(r, square(100, 0.2), tone=r.tones[(0, 1)])


def test_the_default_tone_and_the_explicit_tone_make_the_same_program():
    _same_program(default_tones, explicit_tones, Q[0])


def test_a_name_for_the_tone_follows_the_ops_that_change_it():
    @pulse.kernel
    def named(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        t = d.tone
        drive(d, gaussian(40, 0.1, 10.0))
        shift_phase(t, 0.5)
        set_frequency(t, 5.01e9)
        drive(d, gaussian(40, 0.1, 10.0), tone=t)
        sync(d, r)
        readout(r, square(100, 0.2))

    _same_program(named, default_tones, Q[0])


def test_another_line_s_tone_retunes_a_drive():
    @pulse.kernel
    def by_keyword(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        drive(d0, square(40, 0.1), tone=d1.tone)
        drive(d1, square(40, 0.1))

    mlir = pulse.compile(by_keyword).lower(Q[0], Q[1]).mlir()
    # d0 plays at d1's tone, which d1's drive then continues from.
    assert "pulse.drive %line, %2, %tone_2 " in mlir
    assert "pulse.drive %line_1, %3, %updated_tone " in mlir


def test_another_transition_s_tone_starts_at_its_frequency():
    @pulse.kernel
    def second(q):
        d = get_drive_line(q)
        drive(d, square(40, 0.1))
        drive(d, square(40, 0.1), tone=d.tones[(1, 2)])
        shift_phase(d.tones[(1, 2)], 0.25)
        drive(d, square(40, 0.1), tone=d.tones[(1, 2)])

    lowered = pulse.compile(second).lower(Q[0])
    # One tone, made once: Q[0] is at 5 GHz with an anharmonicity of -300 MHz.
    (tone,) = _tones_of(lowered)
    assert "f64, f64 -> !pulse.tone" in tone
    assert "4.700000e+09" in lowered.mlir()
    assert "frame_phase_offset = 2.500000e-01" in lowered.mlir()


def test_tones_lists_the_transitions_of_the_qudit_s_levels():
    seen = {}

    @pulse.kernel
    def listed(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        seen["drive"] = list(d.tones)
        seen["readout"] = list(r.tones)
        seen["has"] = ((1, 2) in d.tones, (0, 3) in d.tones)
        drive(d, square(40, 0.1))

    lowered = pulse.compile(listed).lower(Q[0])
    assert seen == {
        "drive": [(0, 1), (0, 2), (1, 2)],
        "readout": [(0, 1)],
        "has": (True, False),
    }
    # Reading the keys makes no tone.
    assert _tones_of(lowered) == []


def _target(model):
    return pulse.quantum_system(
        name="tones-test",
        qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-200e6)},
        model=model,
    )


def test_a_missing_transition_is_a_key_error_listing_the_qudit_s():
    @pulse.kernel
    def missing(q):
        d = get_drive_line(q)
        drive(d, square(40, 0.1), tone=d.tones[(0, 3)])

    with pytest.raises(KeyError, match=r"no transition \(0, 3\).*\(0, 1\), \(0, 2\), \(1, 2\)"):
        pulse.compile(missing).lower(Q[0])
    dev = _target(TransmonModel(levels={0: 2}))
    with pytest.raises(pulse.NotFoundError, match=r"no transition \(0, 3\).*levels \(0, 1\), the"):
        pulse.compile(missing).lower(dev.qudits[0])

    @pulse.kernel
    def readout_transition(q):
        r = get_readout_line(q)
        readout(r, square(40, 0.1), tone=r.tones[(1, 2)])

    with pytest.raises(KeyError, match=r"a readout line has only its own tone"):
        pulse.compile(readout_transition).lower(Q[0])


def _readout_frequencies(lowered):
    return [
        e["frequency_hz"]
        for e in lowered.export_to_controller()["events"]
        if e["kind"] == "readout"
    ]


@pulse.kernel
def read_static(q, df: float):
    r = get_readout_line(q)
    readout(r, square(100, 0.1))
    shift_frequency(r.tone, df)
    readout(r, square(100, 0.1), tone=r.tones[(0, 1)])


@pulse.kernel
def read_runtime(q: pulse.Qudit, df: float):
    r = get_readout_line(q)
    readout(r, square(100, 0.1))
    shift_frequency(r.tone, df)
    readout(r, square(100, 0.1), tone=r.tones[(0, 1)])


def test_a_readout_line_plays_at_the_readout_frequency():
    # transmon_surface17's resonators are about 3 GHz above its qubits.
    dev = pulse.quantum_system("transmon_surface17")
    for handle in dev.qudits[[0, 9]]:
        qudit = dev.physical_qudits[handle.index]
        resonator = qudit.readout.frequency_hz
        assert abs(resonator - qudit.frequency_hz) > 1e9
        for kernel in (read_static, read_runtime):
            lowered = kernel.lower(handle, 2e6)
            assert _readout_frequencies(lowered) == [resonator, resonator + 2e6]


def test_a_readout_line_without_a_readout_frequency_plays_at_the_qudit_s():
    dev = _target(TransmonModel)
    q = dev.qudits[0]
    for kernel in (read_static, read_runtime):
        assert _readout_frequencies(kernel.lower(q, 1e6)) == [5.0e9, 5.0e9 + 1e6]


def test_the_frequency_tables_hold_the_program_s_qudits_not_the_device_s():
    # The first compile on a quantum system tabulates qudits 0 to the program's
    # highest; a later program on a higher qudit, static or runtime, extends
    # the tables, and one on a lower qudit reads them. New kernels, on a
    # quantum system of its own: no compile is shared with another test.
    dev = rebuilt(pulse.quantum_system("transmon_surface17"))
    static, runtime = (pulse.kernel(k.__wrapped__) for k in (read_static, read_runtime))
    resonator = {i: dev.physical_qudits[i].readout.frequency_hz for i in (0, 4, 9)}
    for kernel, index, count in (
        (static, 0, 1),
        (runtime, 4, 5),
        (static, 9, 10),
        (runtime, 0, 10),
    ):
        lowered = kernel.lower(dev.qudits[index], 1e6)
        assert _readout_frequencies(lowered) == [resonator[index], resonator[index] + 1e6]
        assert dev._tables[0] == count


def test_the_tones_of_a_line_are_read_only():
    @pulse.kernel
    def assign(q):
        d = get_drive_line(q)
        d.tones[(1, 2)] = d.tone

    with pytest.raises(pulse.CompilationError):
        pulse.compile(assign).lower(Q[0])

    @pulse.kernel
    def attribute(q):
        d = get_drive_line(q)
        drive(d, square(40, 0.1), tone=d.tone.tone)

    with pytest.raises(pulse.CompilationError, match="a tone has no attribute 'tone'"):
        pulse.compile(attribute).lower(Q[0])


class _UpperQudit(SimulationModel):
    """A model whose lines' own tones drive levels 1 and 2."""

    levels: ClassVar[int] = 3
    default_transition: ClassVar[tuple[int, int]] = (1, 2)


def test_the_default_transition_comes_from_the_model():
    @pulse.kernel
    def both(q):
        d = get_drive_line(q)
        drive(d, square(40, 0.1), tone=d.tones[(1, 2)])
        drive(d, square(40, 0.1), tone=d.tones[(0, 1)])

    dev = _target(_UpperQudit)
    lowered = pulse.compile(both).lower(dev.qudits[0])
    # (1, 2) is the line's own tone; (0, 1) is made, at the model's evenly
    # spaced 0-1 frequency.
    (tone,) = _tones_of(lowered)
    assert "5.000000e+09" in lowered.mlir()
    assert "pulse.drive %line, %1, %tone " in lowered.mlir()


def test_a_rolled_loop_that_reads_a_transition_s_tone_still_drives_it_each_time():
    @pulse.kernel
    def repeated(q):
        d = get_drive_line(q)
        for _ in range(4):
            drive(d, square(40, 0.1), tone=d.tones[(1, 2)])

    lowered = pulse.compile(repeated).lower(Q[0])
    assert len(_tones_of(lowered)) == 1
    assert lowered.duration == 4 * 40 * pulse.ns / 2


def test_driving_the_second_transition_moves_the_qudit_to_level_two():
    # 400 cycles at 2 GHz is 200 ns; H = amp X / 2 rotates 0-1 by amp * 200
    # and 1-2, whose matrix element is sqrt(2), by sqrt(2) times that.
    dev = pulse.quantum_system(
        name="tones-test",
        qudits={
            0: PhysicalQudit(
                0,
                5.0e9,
                0.0,
                0.0,
                anharmonicity_hz=-200e6,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
        },
    )
    one = math.pi / 200.0

    @pulse.kernel
    def ladder(q):
        d = get_drive_line(q)
        drive(d, square(400, one))
        drive(d, square(400, one / math.sqrt(2.0)), tone=d.tones[(1, 2)])

    rho = pulse.simulate(ladder, dev.qudits[0], levels=3).density_matrix
    assert rho[2, 2].real > 0.97


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "drive(d, square(40, 0.1), d.tone)",
            r"drive\(\) takes a line and a waveform, and a tone only by keyword: .*"
            r"drive\(line, waveform, tone=tone\) on another; got 3 positional arguments",
        ),
        (
            "readout(r, square(40, 0.1), r.tone)",
            r"readout\(\) takes .* a tone only by keyword: .* got 3 positional arguments",
        ),
        (
            "line, tone = d",
            r"cannot unpack a drive line into 2 names: a line is one value, not a pair; "
            r"write `line = get_drive_line\(q\)` and use line.tone for its tone",
        ),
        (
            "line, tone = r",
            r"cannot unpack a readout line into 2 names: .* write "
            r"`line = get_readout_line\(q\)` and use line.tone for its tone",
        ),
        (
            "for line in r:\n        wait(line, 8)",
            r"a line is one value, and its tone is line.tone",
        ),
    ],
)
def test_a_line_is_one_value_and_its_tone_is_passed_by_keyword(body, message):
    namespace: dict = {}
    exec(
        "from cudaq_pulse.ops import *\n"
        f"def k(q):\n    d = get_drive_line(q)\n    r = get_readout_line(q)\n    {body}\n",
        namespace,
    )
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.compile(pulse.kernel(namespace["k"])).lower(Q[0])
