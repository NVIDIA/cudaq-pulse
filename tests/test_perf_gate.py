# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""tools/perf-gate.py: reading Callgrind's dumps, the verdicts, and the
program sizes of its cases, which are the same on every platform."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cudaq_pulse
import pytest

_ROOT = Path(__file__).resolve().parents[1]

# The wheels' tests run on a copy of tests/ without the tools.
if not (_ROOT / "tools" / "perf-gate.py").exists():
    pytest.skip("tools/perf-gate.py is not in this tree", allow_module_level=True)


def _gate():
    spec = importlib.util.spec_from_file_location("perf_gate", _ROOT / "tools" / "perf-gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


gate = _gate()

# A dump as Callgrind 3.22 writes it with --cache-sim=yes (the format of
# its manual, "Callgrind Format Specification"): the header's summary:, then
# cost lines under ob=/fl=/fn= with positions relative to the last ("+4",
# "-700", "*"), trailing zero events left out, file names given once and
# then referred to by number, a call (calls= and its inclusive cost, which
# is not self cost) and an inlined file (fi=, back with fe=). The totals:
# are the sum of the self-cost lines, added up by hand.
_SUMMARY = "941000 380000 185000 401 1810 600 40 160 60"
_DUMP = f"""\
# callgrind format
version: 1
creator: callgrind-3.22.0
pid: 4242
cmd:  python3 benchmarks/perf_gate/cases.py --measure libperf_gate_marks.so
part: 3


desc: I1 cache: 32768 B, 64 B, 8-way associative
desc: D1 cache: 32768 B, 64 B, 8-way associative
desc: LL cache: 8388608 B, 64 B, 16-way associative
desc: Timerange: Basic block 18263114 - 18402551
desc: Trigger: Client Request: compile_bell

positions: line
events: Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw
summary: {_SUMMARY}


ob=(1) /usr/bin/python3.12
fl=(1) ./Python/ceval.c
fn=(1) _PyEval_EvalFrameDefault
812 600000 250000 120000 300 900 200 40 60 10
+4 200000 80000 30000 100 300
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfi=(2) ./string/../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S
cfn=(2) __memmove_avx_unaligned_erms
calls=1500 0
* 90000 30000 30000 0 600 400 0 100 50
fi=(3) ./Include/object.h
-700 50000 20000 5000 0 10
fe=(1)
+700 1000 0 0 1

ob=(2)
fl=(2)
fn=(2)
0 90000 30000 30000 0 600 400 0 100 50

totals: {_SUMMARY}
"""


def test_a_dump_gives_its_case_and_counts():
    dump = gate.parse_dump(_DUMP)
    assert dump.name == "compile_bell"
    assert dump.events["Ir"] == 941_000
    assert dump.events["DLmw"] == 60
    assert dump.summary == dump.events


def test_the_dump_at_exit_is_not_a_case():
    assert (
        gate.parse_dump(_DUMP.replace("Client Request: compile_bell", "Program termination"))
        is None
    )


def test_a_dump_after_instrumentation_stopped_has_a_zero_summary_and_its_totals_are_used():
    # Switching instrumentation off zeroes the running total the summary
    # is taken from, not the cost lines: the first CI run's dumps.
    dump = gate.parse_dump(_DUMP.replace(f"summary: {_SUMMARY}", "summary: 0"))
    assert dump.events["Ir"] == 941_000
    assert dump.summary["Ir"] == 0


def test_a_dump_whose_totals_do_not_add_up_or_are_missing_is_an_error():
    with pytest.raises(gate.GateError, match="differ from the sum"):
        gate.parse_dump(_DUMP.replace("+700 1000 0 0 1", "+700 1001 0 0 1"))
    with pytest.raises(gate.GateError, match="totals"):
        gate.parse_dump(_DUMP.replace(f"totals: {_SUMMARY}\n", ""))


def test_a_dump_may_give_any_number_of_positions():
    instr_line = """\
desc: Trigger: Client Request: submit
positions: instr line
events: Ir
summary: 27
fn=(1) f
0x401000 12 20
+3 * 7
jump=2 +9 *
+9 *
calls=1 0x402000 40
* * 500
totals: 27
"""
    assert gate.parse_dump(instr_line).events == {"Ir": 27}
    no_positions = "desc: Trigger: Client Request: submit\npositions:\nevents: Ir Dr\n"
    no_positions += "fn=(1) f\n20 4\n7\ntotals: 27 4\n"
    assert gate.parse_dump(no_positions).events == {"Ir": 27, "Dr": 4}


def test_the_cycle_estimate_weights_cache_misses():
    events = gate.parse_dump(_DUMP).events
    # 1,506,000 accesses, 2,811 L1 misses of which 260 miss the last level too.
    assert gate.estimated_cycles(events) == (1_506_000 - 2811) + 5 * (2811 - 260) + 35 * 260
    assert gate.estimated_cycles({"Ir": 10}) is None


def _runs(**instructions):
    return {n: gate.Measured([float(x)] * 2, [0.0] * 2) for n, x in instructions.items()}


_PLAUSIBLE = {
    "first_compile": 90e6,
    "compile_bell": 1.2e6,
    "compile_surface_d5": 11e6,
    "compile_calls_small": 1.1e6,
    "compile_calls_large": 24e6,
    "submit": 1500,
}


def test_plausible_counts_pass_the_checks():
    assert gate.implausible(_runs(**_PLAUSIBLE), set(_PLAUSIBLE)) == []


def test_zero_missing_or_too_small_counts_are_errors():
    problems = gate.implausible(_runs(**{**_PLAUSIBLE, "submit": 0}), set(_PLAUSIBLE) | {"x"})
    assert any(p.startswith("submit: 0, 0 instructions") for p in problems)
    assert "x: no instruction count measured" in problems
    small = gate.implausible(_runs(**{**_PLAUSIBLE, "compile_bell": 9e4}), set(_PLAUSIBLE))
    assert any(p.startswith("compile_bell:") for p in small)
    nan = gate.Measured([float("nan")], [0.0])
    assert gate.implausible({"submit": nan}, {"submit"})


def test_a_heavier_case_must_measure_well_above_a_lighter_one():
    problems = gate.implausible(
        _runs(**{**_PLAUSIBLE, "compile_calls_large": 1.5e6}), set(_PLAUSIBLE)
    )
    assert problems == [
        "compile_calls_large measured 1,500,000 instructions per repetition, not more than "
        "twice compile_calls_small's 1,100,000: the counts do not follow the work"
    ]
    names = set(BASELINE["cases"])
    assert {name for pair in gate.HEAVIER for name in pair} <= names


def _baseline(instructions=1_000_000, **sizes):
    return {"cases": {"c": {"instructions": instructions, "op_count": 15, **sizes}}}


def _verdict(measured, baseline=None, sizes=None):
    runs = gate.Measured(instructions=list(measured), estimated_cycles=[0.0] * len(measured))
    rows = gate.compare(baseline or _baseline(), {"c": sizes or {"op_count": 15}}, {"c": runs})
    return rows[0]


@pytest.mark.parametrize(
    ("measured", "verdict"),
    [
        (1_000_000, "ok"),
        (1_003_999, "ok"),
        (1_004_001, "slower"),
        (1_005_000, "slower"),
        (996_001, "ok"),
        (995_000, "faster"),
        (989_000, "improved"),
    ],
)
def test_the_verdict_follows_the_change_in_instructions(measured, verdict):
    assert _verdict([measured]).verdict == verdict


def test_the_lowest_of_identical_runs_is_compared_and_their_spread_reported():
    row = _verdict([1_000_100, 1_000_000])
    assert row.measured == 1_000_000
    assert row.spread == pytest.approx(1e-4)


def test_a_case_may_carry_its_own_tolerance():
    baseline = _baseline()
    baseline["cases"]["c"]["tolerance"] = 0.02
    assert _verdict([1_010_000], baseline).verdict == "ok"


def test_any_change_in_program_size_fails():
    row = _verdict([1_000_000], sizes={"op_count": 16})
    assert row.verdict == "sizes"
    assert row.sizes == "op_count 15 -> 16"


def test_new_removed_and_unmeasured_cases_fail():
    rows = gate.compare(_baseline(), {"d": {}}, {"d": gate.Measured([1.0], [1.0])})
    assert {r.name: r.verdict for r in rows} == {"c": "removed", "d": "new"}
    row = _verdict([1_000_000], {"cases": {"c": {"op_count": 15}}})
    assert row.verdict == "no baseline"
    assert set(gate.FAILING) >= {"slower", "improved", "sizes", "new", "removed", "no baseline"}


def test_without_instruction_counts_only_the_sizes_are_compared():
    rows = gate.compare(_baseline(), {"c": {"op_count": 15}}, None)
    assert rows[0].verdict == "ok"


def test_a_baseline_without_instruction_counts_keeps_the_previous_ones():
    previous = {"environment": {"python": "3.12.15"}, **_baseline(tolerance=0.01)}
    kept = gate.as_baseline({"python": "other"}, {"c": {"op_count": 16}}, None, previous)
    assert kept["environment"] == {"python": "3.12.15"}
    assert kept["cases"]["c"] == {"instructions": 1_000_000, "op_count": 16, "tolerance": 0.01}
    runs = {"c": gate.Measured([2_000_000.4, 2_000_100.0], [3_000_000.0, 3_000_001.0])}
    taken = gate.as_baseline({"python": "other"}, {"c": {"op_count": 16}}, runs, previous)
    assert taken["environment"] == {"python": "other"}
    assert taken["cases"]["c"]["instructions"] == 2_000_000
    assert taken["cases"]["c"]["estimated_cycles"] == 3_000_000


def test_in_bootstrap_mode_a_case_without_a_count_is_reported_and_passes():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    row = _verdict([1_000_000, 1_000_000], baseline)
    assert row.verdict == "bootstrap"
    assert row.measured == 1_000_000 and row.spread == 0
    assert "bootstrap" not in gate.FAILING


def test_in_bootstrap_mode_sizes_still_block():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    assert _verdict([1_000_000], baseline, sizes={"op_count": 16}).verdict == "sizes"


def test_bootstrap_mode_is_allowed_only_while_no_case_has_a_count():
    assert gate.baseline_error({"bootstrap": True, "cases": {"c": {"op_count": 1}}}) is None
    assert gate.baseline_error(_baseline()) is None
    error = gate.baseline_error({"bootstrap": True, **_baseline()})
    assert error is not None and "remove" in error
    assert gate.baseline_error({"bootstrap": "yes", "cases": {}}) is not None


def test_bootstrap_mode_is_an_error_on_main():
    bootstrapping = {"bootstrap": True, "cases": {"c": {"op_count": 1}}}
    assert "main" in gate.baseline_error(bootstrapping, frozen=True)
    assert gate.baseline_error({"bootstrap": False, **_baseline()}, frozen=True) is not None
    assert gate.baseline_error(_baseline(), frozen=True) is None


def test_a_baseline_count_of_zero_is_an_error():
    assert "positive" in gate.baseline_error(_baseline(instructions=0))
    assert "positive" in gate.baseline_error(_baseline(instructions=None))


def test_taking_a_measured_run_as_the_baseline_ends_bootstrap_mode():
    previous = {"bootstrap": True, "environment": {}, "cases": {"c": {"op_count": 15}}}
    runs = {"c": gate.Measured([1_000_000.0, 1_000_000.0], [2.0, 2.0])}
    taken = gate.as_baseline({"python": "3.12.15"}, {"c": {"op_count": 15}}, runs, previous)
    assert "bootstrap" not in taken
    assert gate.baseline_error(taken) is None
    sizes_only = gate.as_baseline({}, {"c": {"op_count": 16}}, None, previous)
    assert sizes_only["bootstrap"] is True


BASELINE = json.loads(gate.BASELINE.read_text())


def test_the_checked_in_baseline_is_bootstrapping_or_has_every_count():
    assert gate.baseline_error(BASELINE) is None
    if BASELINE.get("bootstrap") is not True:
        missing = [n for n, c in BASELINE["cases"].items() if "instructions" not in c]
        assert not missing, f"no instruction count for {missing}"


def test_the_program_sizes_match_the_baseline(tmp_path):
    env = gate._run_env(str(Path(cudaq_pulse.__file__).parents[1]), str(tmp_path))
    # Under the sanitizers the extension needs their runtime preloaded.
    for name in ("LD_PRELOAD", "ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "LSAN_OPTIONS"):
        if name in os.environ:
            env[name] = os.environ[name]
    sizes = gate._structural(sys.executable, env)
    expected = {
        name: {
            k: v
            for k, v in case.items()
            if k not in ("instructions", "estimated_cycles", "tolerance")
        }
        for name, case in BASELINE["cases"].items()
    }
    assert sizes == expected, "update with: python tools/perf-gate.py --update"


def test_the_workload_op_counts_agree_with_compile_time_json():
    workloads = json.loads((_ROOT / "benchmarks" / "baselines" / "compile_time.json").read_text())
    for name, entry in workloads.items():
        assert BASELINE["cases"][f"compile_{name}"]["op_count"] == entry["op_count"], name
