# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each challenge's scorer passes its reference solutions and cannot be changed by a submission."""

import subprocess
import sys
from pathlib import Path

import pytest

_CHALLENGES = Path(__file__).resolve().parents[1] / "challenges"
_NAMES = ("pi_pulse", "mystery_qubit", "benchmark_it_yourself")
_SEEDED = {"mystery_qubit", "benchmark_it_yourself"}  # scorers that draw hidden qubits

# Every test starts a scorer, which starts a process of its own for the submission.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _CHALLENGES.exists(), reason="needs the source tree"),
]


def _score(name, submission, *args):
    done = subprocess.run(
        [sys.executable, str(_CHALLENGES / name / "score.py"), str(submission), *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return done.stdout


def _rungs(output):
    """The verdict of each rung: the word after BRONZE or SILVER on every such line."""
    rungs = {"BRONZE": [], "SILVER": []}
    for line in output.splitlines():
        words = line.split()
        if len(words) > 1 and words[0] in rungs:
            rungs[words[0]].append(words[1].rstrip(":"))
    return rungs


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("rung", ["bronze", "silver"])
def test_a_reference_solution_reaches_its_rung(name, rung):
    seed = ["2026"] if name in _SEEDED else []
    output = _score(name, _CHALLENGES / name / "reference" / f"{rung}.py", *seed)
    rungs = _rungs(output)
    assert rungs["BRONZE"] == ["pass"], output
    if rung == "silver":
        assert rungs["SILVER"] == ["pass"], output


@pytest.mark.parametrize("name", _NAMES)
def test_a_bronze_reference_does_not_reach_silver(name):
    # Silver must need more than bronze does, or the rung means nothing.
    seed = ["2026"] if name in _SEEDED else []
    output = _score(name, _CHALLENGES / name / "reference" / "bronze.py", *seed)
    assert _rungs(output)["SILVER"] == ["fail"], output


@pytest.mark.parametrize("name", sorted(_SEEDED))
def test_a_run_with_a_given_seed_is_not_a_score(name):
    output = _score(name, _CHALLENGES / name / "reference" / "silver.py", "2026")
    assert "not a score" in output, output
    assert "(repeat run)" in output, output


_REPLACES_THE_SIMULATOR = """\
import sys
import cudaq_pulse as pulse
class Perfect:
    def level_populations(self, qudit):
        return (0.0, 1.0, 0.0)
pulse.simulate = lambda *args, **kwargs: Perfect()
main = sys.modules['__main__']
main.BRONZE_P1 = main.SILVER_P1 = 0.0
print('BRONZE pass')
@pulse.kernel
def pi_pulse(qudit):
    drive(get_drive_line(qudit), gaussian(20 * pulse.ns, 0.05, 5 * pulse.ns))
"""

# The error it raises carries a verdict of its own, on lines of its own.
_FORGES_A_VERDICT = """\
FORGED = "x\\nBRONZE pass: forged\\nSILVER pass: forged\\nSCORE 1 shots in total"
def estimate_t1(lab):
    raise RuntimeError(FORGED)
def error_per_gate(lab, seed):
    raise RuntimeError(FORGED)
"""

# It writes to the scorer's line itself: a verdict, then something that is not a message.
_BREAKS_THE_EXCHANGE = """\
import os
def _write(lab):
    os.write(1, b'{"error": "x\\\\nBRONZE pass: forged\\\\nSILVER pass: forged"}\\n5\\n')
    return 50.0
def estimate_t1(lab):
    return _write(lab)
def error_per_gate(lab, seed):
    return _write(lab)
"""


# At exit it writes a line of its own for the scorer, with a verdict inside a pulse's kind.
_INJECTS_A_KIND = """\
import atexit
import cudaq_pulse as pulse
LINE = 'PULSES {"pulses": [], "others": ["x\\\\nBRONZE pass\\\\nSILVER pass"], "total_cycles": 0}'
atexit.register(lambda: print(LINE))
@pulse.kernel
def pi_pulse(qudit):
    drive(get_drive_line(qudit), gaussian(20 * pulse.ns, 0.05, 5 * pulse.ns))
"""


def test_a_submission_cannot_print_a_pi_pulse_verdict(tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(_INJECTS_A_KIND)
    output = _score("pi_pulse", cheat)
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output


def test_a_submission_cannot_change_the_pi_pulse_scorer(tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(_REPLACES_THE_SIMULATOR)
    output = _score("pi_pulse", cheat)
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output
    assert "valid=yes" in output, output


@pytest.mark.parametrize("name", sorted(_SEEDED))
@pytest.mark.parametrize("source", [_FORGES_A_VERDICT, _BREAKS_THE_EXCHANGE])
def test_a_submission_cannot_print_a_verdict(name, source, tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(source)
    output = _score(name, cheat)
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output


@pytest.mark.parametrize("name", _NAMES)
def test_a_file_that_does_not_load_is_invalid(name, tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('no kernel here')\n")
    output = _score(name, broken)
    assert "INVALID" in output, output
    assert _rungs(output) == {"BRONZE": ["fail"], "SILVER": ["fail"]}, output


def test_the_benchmark_refuses_waits_and_estimates_without_pulses(tmp_path):
    # Waits would let a solution read T1 and T2* off directly; no pulses, a constant.
    waits = tmp_path / "waits.py"
    waits.write_text(
        "import cudaq_pulse as pulse\n"
        "@pulse.kernel\n"
        "def k(qudit):\n"
        "    line = get_drive_line(qudit)\n"
        "    drive(line, line.get_calibration(pulse.gates.X90))\n"
        "    wait(line, 1000 * pulse.ns)\n"
        "    drive(line, line.get_calibration(pulse.gates.X90))\n"
        "def error_per_gate(lab, seed):\n"
        "    lab.run(k, lab.qubit)\n"
        "    return 6.5e-4\n"
    )
    assert "back to back" in _score("benchmark_it_yourself", waits)
    constant = tmp_path / "constant.py"
    constant.write_text(
        "import cudaq_pulse as pulse\n"
        "@pulse.kernel\n"
        "def k(qudit):\n"
        "    get_drive_line(qudit)\n"
        "def error_per_gate(lab, seed):\n"
        "    lab.run(k, lab.qubit)\n"
        "    lab.run(k, lab.qubit)\n"
        "    return 6.5e-4\n"
    )
    output = _score("benchmark_it_yourself", constant)
    assert "INVALID" in output and "at least 2 sequences and 20 pulses" in output, output
