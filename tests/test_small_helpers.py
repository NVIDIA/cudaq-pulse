# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A qudit by label, a coupling by its qudits, a program's duration and IR levels."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.compiler import IRLevel

TRANSMON_17Q = "transmon_17q"


def test_a_qudit_by_its_label():
    dev = pulse.device(TRANSMON_17Q)
    assert dev.qudit("D1") is dev.qudits[0]
    assert dev.qudit("Z4").index == 16
    with pytest.raises(pulse.NotFoundError, match=r"no qudit labelled 'D10'; did you mean 'D1'\?"):
        dev.qudit("D10")
    chip = pulse.device(TRANSMON_17Q, layout=[4, 5])
    assert chip.qudit("D6") is chip.qudits[1]
    with pytest.raises(pulse.NotFoundError, match="'D1' is carrier 0 of .* layout .* leaves out"):
        chip.qudit("D1")
    generic = pulse.device("transmon_generic:2")
    assert generic.qudit("Q1") is generic.qudits[1]
    with pytest.raises(pulse.NotFoundError, match="its labels are Q0, Q1$"):
        generic.qudit("D1")


def test_the_close_name_suggested_is_the_one_difflib_picks():
    import difflib
    import random

    from cudaq_pulse._errors import _closest

    rng = random.Random(7)
    letters = "abcdeo_xyzé\0"

    def word():
        return "".join(rng.choices(letters, k=rng.randint(0, 9)))

    for _ in range(3000):
        candidates = [word() for _ in range(rng.randint(0, 12))]
        name = word()
        if candidates and rng.random() < 0.4:
            name = "".join(ch for ch in rng.choice(candidates) if rng.random() > 0.2) + name[:1]
        close = difflib.get_close_matches(name, candidates, n=1)
        assert _closest(name, iter(candidates)) == (close[0] if close else None), (name, candidates)


def test_a_coupling_by_its_qudits():
    target = pulse.device(TRANSMON_17Q)._description
    first = target.couplings[0]
    a, b = first.qudits
    assert target.coupling(a, b) is first
    assert target.coupling(b, a) is first
    with pytest.raises(pulse.NotFoundError, match=r"qudits 0 and 1 of .* are not coupled; qudit 0"):
        target.coupling(0, 1)
    generic = pulse.device("transmon_generic:3")._description
    assert generic.coupling(2, 1).strength_hz > 0
    with pytest.raises(pulse.NotFoundError, match="qudit 0 couples to 1$"):
        generic.coupling(0, 2)


@pulse.kernel
def _rabi_then_wait(q, tau: float):
    d = get_drive_line(q)
    drive(d, gaussian(64, 0.1, 16.0))
    wait(d, tau)


def test_the_duration_of_a_program_is_a_time():
    q = pulse.device(TRANSMON_17Q).qudits[0]
    duration = _rabi_then_wait.lower(q, 6.0).duration
    assert duration.__class__ is pulse.Time
    assert duration == 35 * pulse.ns and float(duration) == 35.0
    compiled = pulse.compile(_rabi_then_wait)
    assert compiled.lower(q, 40.0).duration == 52 * pulse.ns


def test_the_ir_dump_takes_a_level():
    lowered = _rabi_then_wait.lower(pulse.device(TRANSMON_17Q).qudits[0], 6.0)
    assert "pulse.drive" in lowered.mlir() == lowered.mlir("pulse")
    assert "qop." in lowered.mlir("qop") and "pulse.drive" not in lowered.mlir("qop")
    assert "cudm." in lowered.mlir(IRLevel.CUDM)
    assert "llvm.func" in lowered.mlir("llvm")
    assert [str(level) for level in IRLevel] == ["pulse", "qop", "cudm", "llvm"]
    with pytest.raises(pulse.InvalidValueError, match="unknown IR level 'hir'; choose from pulse"):
        lowered.mlir("hir")
