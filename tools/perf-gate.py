#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Deterministic performance gate: instruction counts and program sizes.

Runs the cases of ``benchmarks/perf_gate/cases.py`` and compares them with
``benchmarks/baselines/perf_gate.json``:

- program sizes (op count, packed program bytes, export bytes), measured
  natively; any difference fails.
- instructions executed per repetition (Callgrind's ``Ir``), measured under
  Valgrind on Linux x86-64; more than ``REGRESSION`` above the baseline
  fails, and more than ``IMPROVEMENT`` below it fails too, until the
  baseline is updated, so that the gain is kept. Callgrind's cache
  simulation, with a fixed cache geometry, also gives an estimate of cycles
  (``L1 hits + 5 x LL hits + 35 x memory accesses``); it is reported, not
  gated.

The instruction counts repeat exactly only in the environment they were
measured in: the baseline holds the one CI measures in (``environment``),
and a difference is reported with the comparison.

A baseline with ``"bootstrap": true`` and no instruction counts is in
bootstrap mode: the counts are measured and reported but do not fail the
run (sizes still do). Its measurements, taken as the baseline, drop the
field; the field next to any count is an error, so the gate cannot stay in
bootstrap mode once it has counts.

In every mode, the gate fails if the counts cannot be right: a case without
one, a count of zero or one too small for its case, or a heavier case not
measuring well above a lighter one (``implausible``). Such counts are never
written as a baseline; ``--dumps DIR`` keeps the dumps for inspection.

Usage:
    python tools/perf-gate.py                 # compare with the baseline
    python tools/perf-gate.py --update        # rewrite the baseline from this run
    python tools/perf-gate.py --processes 5   # also show the spread of 5 runs

``--out FILE`` writes this run's measurements in the baseline's format;
CI uploads it, so a baseline can be taken from a CI run (CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks" / "perf_gate" / "cases.py"
MARKS = ROOT / "benchmarks" / "perf_gate" / "marks.c"
BASELINE = ROOT / "benchmarks" / "baselines" / "perf_gate.json"
FORMAT = 1

# Fractions of the baseline's instructions per repetition.
REGRESSION = 0.004
IMPROVEMENT = 0.01

# Callgrind's cache model, fixed so the estimate does not depend on the
# runner's CPU: 32 KiB 8-way L1 instruction and data caches, 8 MiB 16-way
# last level, 64-byte lines.
CACHES = ("--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64")
CACHE_EVENTS = ("Ir", "Dr", "Dw", "I1mr", "D1mr", "D1mw", "ILmr", "DLmr", "DLmw")


@dataclass
class Measured:
    instructions: list[float] = field(default_factory=list)
    estimated_cycles: list[float] = field(default_factory=list)


def _environment(python: str) -> dict[str, str]:
    """What the instruction counts depend on besides the code."""
    probe = (
        "import json, platform; print(json.dumps({'python': platform.python_version(), "
        "'machine': platform.machine(), 'libc': ' '.join(platform.libc_ver())}))"
    )
    env = json.loads(subprocess.check_output([python, "-c", probe], text=True))
    compiler = shutil.which("c++") or shutil.which("cc")
    if compiler:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True)
        env["compiler"] = out.stdout.splitlines()[0] if out.stdout else ""
    if shutil.which("valgrind"):
        env["valgrind"] = subprocess.check_output(["valgrind", "--version"], text=True).strip()
    lock = json.loads((ROOT / "tools" / "mlir-toolchain.lock.json").read_text())
    env["mlir_toolchain"] = lock["release_tag"]
    return env


def _run_env(pythonpath: str, home: str) -> dict[str, str]:
    """A fixed environment: the process sees the same variables, and so the
    same stack layout, in every run."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": pythonpath,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        # One MLIR context for the whole run, made by the first compile: a
        # new one is otherwise made every 256 modules, inside whichever
        # case reaches that number.
        "CUDAQ_PULSE_CONTEXT_MAX_MODULES": str(1 << 40),
        "CUDAQ_PULSE_CONTEXT_MAX_WORDS": str(1 << 40),
        # NumPy's OpenBLAS otherwise starts a worker per CPU, whose waiting
        # for work Callgrind counts with the case that runs meanwhile.
        "OPENBLAS_NUM_THREADS": "1",
    }


def _structural(python: str, env: dict[str, str]) -> dict[str, dict[str, int]]:
    out = subprocess.run(
        [python, str(CASES), "--structural"], env=env, capture_output=True, text=True, cwd=ROOT
    )
    if out.returncode:
        sys.exit(f"perf gate: {CASES.name} --structural failed:\n{out.stderr}")
    return json.loads(out.stdout)


def _build_marks(directory: Path) -> Path:
    library = directory / "libperf_gate_marks.so"
    compiler = os.environ.get("CC") or shutil.which("cc") or "gcc"
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", "-o", str(library), str(MARKS)], check=True
    )
    return library


def _no_aslr() -> list[str]:
    """``setarch -R`` where the kernel lets this process turn off address
    randomization (a container's seccomp profile may not)."""
    setarch = shutil.which("setarch")
    if setarch is None:
        return []
    command = [setarch, platform.machine(), "-R"]
    probe = subprocess.run([*command, "true"], capture_output=True)
    return command if probe.returncode == 0 else []


class GateError(Exception):
    """Measurements that cannot be trusted: the gate fails in every mode."""


@dataclass
class Dump:
    name: str
    events: dict[str, int]
    summary: dict[str, int] | None


# A cost line starts with its positions (a number, "0x" address, or a
# "+n", "-n" or "*" relative to the last one) or, without positions, its
# first event.
_COST_LINE = re.compile(r"[0-9+*-]")


def parse_dump(text: str) -> Dump | None:
    """One Callgrind dump written by a ``perf_gate_stop`` client request, or
    None for any other dump.

    The events are the ``totals:`` line, which Callgrind writes as the sum
    of the dump's self-cost lines: the lines after ``calls=`` (a call's
    inclusive cost) and ``jump=``/``jcnd=`` are not self cost. The sum is
    checked against it. The header's ``summary:`` is kept for comparison."""
    trigger = re.search(r"^desc: Trigger: Client Request: (.+)$", text, re.MULTILINE)
    if trigger is None:
        return None
    name = trigger.group(1).strip()
    names: list[str] | None = None
    positions = 1  # "positions: line" when the header does not say
    summary = totals = None
    self_cost: list[int] = []
    not_self = False

    def costs(fields: list[str]) -> dict[str, int]:
        if names is None or len(fields) > len(names):
            raise GateError(f"dump for {name}: costs {fields} do not match its events: line")
        values = [int(v) for v in fields]
        return dict(zip(names, values + [0] * (len(names) - len(values))))

    for line in text.splitlines():
        if line.startswith("positions:"):
            positions = len(line.partition(":")[2].split())
        elif line.startswith("events:"):
            names = line.partition(":")[2].split()
            self_cost = [0] * len(names)
        elif line.startswith("summary:"):
            summary = costs(line.partition(":")[2].split())
        elif line.startswith("totals:"):
            totals = costs(line.partition(":")[2].split())
        elif line.startswith(("calls=", "jump=", "jcnd=")):
            not_self = True
        elif _COST_LINE.match(line):
            if not_self:
                not_self = False
                continue
            for i, value in enumerate(costs(line.split()[positions:]).values()):
                self_cost[i] += value
    if names is None or totals is None:
        raise GateError(f"dump for {name} has no events: or totals: line (truncated?)")
    if list(totals.values()) != self_cost:
        raise GateError(
            f"dump for {name}: totals {list(totals.values())} differ from the sum of its "
            f"cost lines {self_cost}"
        )
    return Dump(name, totals, summary)


def estimated_cycles(events: dict[str, int]) -> float | None:
    """``L1 hits + 5 x LL hits + 35 x memory accesses``, from a dump with
    cache simulation."""
    if not all(name in events for name in CACHE_EVENTS):
        return None
    accesses = events["Ir"] + events["Dr"] + events["Dw"]
    l1_misses = events["I1mr"] + events["D1mr"] + events["D1mw"]
    ll_misses = events["ILmr"] + events["DLmr"] + events["DLmw"]
    return (accesses - l1_misses) + 5 * (l1_misses - ll_misses) + 35 * ll_misses


def _callgrind(
    python: str, env: dict[str, str], marks: Path, directory: Path, prefix: list[str]
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """One Callgrind run of every case: its events per repetition, and
    notes on its dumps. The dumps and Valgrind's log stay in *directory*."""
    directory.mkdir(parents=True)
    command = [
        *prefix,
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        "--cache-sim=yes",
        *CACHES,
        f"--callgrind-out-file={directory}/callgrind.out",
        python,
        str(CASES),
        "--measure",
        str(marks),
    ]
    out = subprocess.run(command, env=env, capture_output=True, text=True, cwd=ROOT)
    (directory / "valgrind.log").write_text(out.stderr)
    if out.returncode:
        raise GateError(f"Callgrind run failed ({directory.name}):\n{out.stderr[-4000:]}")
    reps = json.loads(out.stdout.strip().splitlines()[-1])
    results: dict[str, dict[str, float]] = {}
    notes = []
    for path in sorted(directory.glob("callgrind.out*")):
        dump = parse_dump(path.read_text(errors="replace"))
        if dump is None:
            continue
        if dump.name not in reps or dump.name in results:
            raise GateError(f"unexpected Callgrind dump {path.name} for {dump.name}")
        n = reps[dump.name]
        cycles = estimated_cycles(dump.events)
        results[dump.name] = {
            "instructions": dump.events.get("Ir", 0) / n,
            "estimated_cycles": float("nan") if cycles is None else cycles / n,
        }
        if dump.summary is not None and dump.summary != dump.events:
            notes.append(
                f"{dump.name}: the dump's summary: Ir {dump.summary.get('Ir')} differs "
                f"from its totals: Ir {dump.events.get('Ir')} ({directory.name}/{path.name})"
            )
    missing = set(reps) - set(results)
    if missing:
        raise GateError(f"no Callgrind dump for {', '.join(sorted(missing))} ({directory.name})")
    return results, notes


def measure_instructions(
    python: str, env: dict[str, str], processes: int, workdir: Path, dumps: Path
) -> tuple[dict[str, Measured], list[str], bool]:
    """Each case's instructions per repetition in *processes* identical
    runs, made side by side (their dumps in *dumps*), notes on the dumps,
    and whether address randomization was off."""
    marks = _build_marks(workdir)
    prefix = _no_aslr()
    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        runs = list(
            pool.map(
                lambda k: _callgrind(python, env, marks, dumps / f"run{k}", prefix),
                range(processes),
            )
        )
    measured: dict[str, Measured] = {}
    notes = []
    for run, run_notes in runs:
        notes += run_notes
        for name, values in run.items():
            m = measured.setdefault(name, Measured())
            m.instructions.append(values["instructions"])
            m.estimated_cycles.append(values["estimated_cycles"])
    return measured, notes, bool(prefix)


# Instructions per repetition below which a count is wrong, not fast (a
# compile is well above a million), and pairs of cases whose first does
# several times the work of the second.
MINIMUM = 200
MINIMUM_COMPILE = 100_000
HEAVIER = (
    ("first_compile", "compile_bell"),
    ("compile_surface_d5", "compile_bell"),
    ("compile_calls_large", "compile_calls_small"),
)


def implausible(measured: dict[str, Measured], cases: set[str]) -> list[str]:
    """Why the instruction counts cannot be right: a case of *cases* without
    a count, a count of zero or one too small for its case, or a heavier
    case measuring less than twice a lighter one. Empty if plausible."""
    problems = [f"{name}: no instruction count measured" for name in sorted(cases - set(measured))]
    for name, m in sorted(measured.items()):
        least = MINIMUM_COMPILE if name.startswith(("compile_", "first_compile")) else MINIMUM
        if not m.instructions or not all(x >= least for x in m.instructions):
            problems.append(
                f"{name}: {', '.join(_num(x) for x in m.instructions) or 'no'} instructions "
                f"per repetition measured; a count below {least:,} is not plausible"
            )
    for heavy, light in HEAVIER:
        if heavy in measured and light in measured:
            h, lo = min(measured[heavy].instructions), min(measured[light].instructions)
            if not h >= 2 * lo or lo <= 0:
                problems.append(
                    f"{heavy} measured {_num(h)} instructions per repetition, not more than "
                    f"twice {light}'s {_num(lo)}: the counts do not follow the work"
                )
    return problems


@dataclass
class Row:
    name: str
    baseline: float | None
    measured: float | None
    spread: float | None
    cycles: float | None
    sizes: str
    verdict: str
    tolerance: float = REGRESSION

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.measured is None or not self.baseline:
            return None
        return self.measured / self.baseline - 1


def compare(
    baseline: dict[str, Any],
    sizes: dict[str, dict[str, int]],
    instructions: dict[str, Measured] | None,
) -> list[Row]:
    """One row per case of the baseline or of this run, with its verdict:
    ``ok``, ``faster`` (within ``IMPROVEMENT``) or a failure: ``slower``,
    ``improved`` (update the baseline), ``sizes``, ``new``, ``removed``,
    ``no baseline``. In bootstrap mode a case without an instruction count
    is ``bootstrap``, which passes."""
    cases = baseline.get("cases", {})
    bootstrap = baseline.get("bootstrap") is True
    rows = []
    for name in sorted(set(cases) | set(sizes)):
        expected = cases.get(name)
        if name not in sizes:
            rows.append(Row(name, None, None, None, None, "", "removed"))
            continue
        if expected is None:
            rows.append(Row(name, None, None, None, None, "", "new"))
            continue
        changed = [
            f"{key} {expected.get(key)} -> {value}"
            for key, value in sorted(sizes[name].items())
            if expected.get(key) != value
        ]
        changed += [
            f"{key} {expected[key]} -> none"
            for key in sorted(expected)
            if key not in ("instructions", "estimated_cycles", "tolerance")
            and key not in sizes[name]
        ]
        tolerance = float(expected.get("tolerance", REGRESSION))
        row = Row(
            name,
            expected.get("instructions"),
            None,
            None,
            None,
            "; ".join(changed) or "same",
            "sizes" if changed else "ok",
            tolerance,
        )
        if instructions is not None:
            m = instructions.get(name)
            if m is not None:
                low, high = min(m.instructions), max(m.instructions)
                row.measured = low
                row.spread = high / low - 1 if low else 0.0
                row.cycles = min(m.estimated_cycles)
            if row.baseline is None:
                if row.verdict == "ok":
                    row.verdict = "bootstrap" if bootstrap else "no baseline"
            elif row.verdict == "ok" and row.delta is not None:
                if row.delta > tolerance:
                    row.verdict = "slower"
                elif row.delta < -IMPROVEMENT:
                    row.verdict = "improved"
                elif row.delta < -tolerance:
                    row.verdict = "faster"
        rows.append(row)
    return rows


FAILING = {"slower", "improved", "sizes", "new", "removed", "no baseline"}


def baseline_error(baseline: dict[str, Any], frozen: bool = False) -> str | None:
    """Why the baseline is not allowed, or None: an instruction count that
    is not a positive number, a ``bootstrap`` field next to counts, or any
    ``bootstrap`` field on a *frozen* branch (main, which releases are cut
    from; tools/release-meta.py refuses it for a release tag too).

    Bootstrap mode (instruction counts measured and reported, not gated)
    is allowed only while no case has a count: taking a CI run's
    measurements as the baseline drops the field, so the gate cannot stay
    in bootstrap mode once it has counts."""
    cases = baseline.get("cases", {})
    bad = sorted(
        n
        for n, c in cases.items()
        if "instructions" in c
        and not (isinstance(c["instructions"], (int, float)) and c["instructions"] > 0)
    )
    if bad:
        return f"instruction counts must be positive numbers ({', '.join(bad)})"
    flag = baseline.get("bootstrap", False)
    if flag not in (True, False):
        return f"bootstrap must be true or false, not {flag!r}"
    if frozen and "bootstrap" in baseline:
        return (
            '"bootstrap" is not allowed on main: adopt a measured baseline before the '
            "freeze (CONTRIBUTING.md, The perf gate)"
        )
    counted = sorted(n for n, c in cases.items() if "instructions" in c)
    if flag and counted:
        return (
            "bootstrap is true but the baseline has instruction counts "
            f"({', '.join(counted[:3])}{', ...' if len(counted) > 3 else ''}); "
            'remove "bootstrap" so the counts are gated'
        )
    return None


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:+.2f}%"


def _spread(x: float) -> str:
    # Any difference at all shows: identical runs should not differ.
    return "0" if x == 0 else f"{100 * x:.3g}%"


def _num(x: float | None) -> str:
    return "" if x is None or x != x else f"{x:,.0f}"


def render(rows: list[Row], markdown: bool) -> str:
    header = (
        "case",
        "baseline Ir",
        "measured Ir",
        "change",
        "spread",
        "est. cycles",
        "sizes",
        "verdict",
    )
    table = [
        (
            r.name,
            _num(r.baseline),
            _num(r.measured),
            _pct(r.delta),
            "" if r.spread is None else _spread(r.spread),
            _num(r.cycles),
            r.sizes,
            r.verdict,
        )
        for r in rows
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(cells) + " |" for cells in table]
        return "\n".join(lines)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *table)]
    return "\n".join(
        "  ".join(
            str(c).ljust(w) if i in (0, 6) else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        ).rstrip()
        for cells in (header, *table)
    )


def as_baseline(
    environment: dict[str, str],
    sizes: dict[str, dict[str, int]],
    instructions: dict[str, Measured] | None,
    previous: dict[str, Any],
) -> dict[str, Any]:
    """This run in the baseline's format. Without instruction counts the
    previous baseline's are kept."""
    old = previous.get("cases", {})
    cases = {}
    for name in sorted(sizes):
        entry: dict[str, Any] = dict(sorted(sizes[name].items()))
        if instructions is not None and name in instructions:
            m = instructions[name]
            entry["instructions"] = round(min(m.instructions))
            cycles = min(m.estimated_cycles)
            if cycles == cycles:
                entry["estimated_cycles"] = round(cycles)
        else:
            for key in ("instructions", "estimated_cycles"):
                if key in old.get(name, {}):
                    entry[key] = old[name][key]
        if "tolerance" in old.get(name, {}):
            entry["tolerance"] = old[name]["tolerance"]
        cases[name] = entry
    if instructions is not None:
        return {"format": FORMAT, "environment": environment, "cases": cases}
    out: dict[str, Any] = {"format": FORMAT}
    if previous.get("bootstrap") is True:
        out["bootstrap"] = True
    out["environment"] = previous.get("environment", {})
    out["cases"] = cases
    return out


def _update_command() -> str:
    run = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if run and repo:
        return f"gh run download {run} -R {repo} -n perf-gate-baseline -D benchmarks/baselines"
    return "python tools/perf-gate.py --update   (on Linux x86-64, in CI's environment)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--out", type=Path, help="write this run in the baseline's format")
    parser.add_argument("--summary", type=Path, help="append a Markdown report to this file")
    parser.add_argument(
        "--processes", type=int, default=2, help="identical Callgrind runs, made side by side"
    )
    parser.add_argument("--pythonpath", default="build/dev/python")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--sizes-only", action="store_true", help="skip the instruction counts (no Valgrind)"
    )
    parser.add_argument(
        "--dumps", type=Path, help="keep Callgrind's dumps and logs in this new directory"
    )
    args = parser.parse_args()
    if args.dumps and args.dumps.exists() and any(args.dumps.iterdir()):
        parser.error(f"--dumps {args.dumps} is not empty")

    pythonpath = str((ROOT / args.pythonpath).resolve())
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    environment = _environment(args.python)
    # Instruction counts that cannot be right fail the gate in every mode.
    problems: list[str] = []
    dump_notes: list[str] = []

    with tempfile.TemporaryDirectory(prefix="perf-gate-") as tmp:
        workdir = Path(tmp)
        env = _run_env(pythonpath, str(workdir))
        # Byte-compile first, so no run compiles a module inside a case.
        purelib = subprocess.check_output(
            [args.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
        for directory in (pythonpath, str(ROOT / "benchmarks"), purelib):
            subprocess.run(
                [args.python, "-m", "compileall", "-q", directory],
                env={**env, "PYTHONDONTWRITEBYTECODE": ""},
                check=True,
                stdout=subprocess.DEVNULL,
            )
        sizes = _structural(args.python, env)
        instructions = None
        note = ""
        if args.sizes_only:
            note = "instruction counts skipped (--sizes-only)"
        elif shutil.which("valgrind") is None:
            note = "instruction counts skipped: valgrind is not installed"
        elif (platform.system(), platform.machine()) != ("Linux", "x86_64"):
            note = "instruction counts skipped: the baseline is for Linux x86-64"
        else:
            dumps = args.dumps.resolve() if args.dumps else workdir / "callgrind"
            try:
                instructions, dump_notes, no_aslr = measure_instructions(
                    args.python, env, args.processes, workdir, dumps
                )
            except GateError as e:
                problems.append(f"Callgrind: {e}")
                note = "instruction counts not measured"
            else:
                problems += implausible(instructions, set(sizes))
                note = "address randomization " + (
                    "off (setarch -R)" if no_aslr else "on: setarch -R is not permitted here"
                )

    rows = compare(baseline, sizes, instructions)
    # Counts that failed the checks are reported but never become a baseline.
    measured = as_baseline(environment, sizes, None if problems else instructions, baseline)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(measured, indent=2) + "\n")

    print(render(rows, markdown=False))
    if note:
        print(f"\n{note}")
    expected_env = baseline.get("environment", {})
    env_diff = [
        f"{key}: {expected_env.get(key)!r} -> {value!r}"
        for key, value in environment.items()
        if instructions is not None and expected_env.get(key) != value
    ]
    if env_diff:
        print("\nThe environment differs from the baseline's, so counts may differ with it:")
        for line in env_diff:
            print(f"  {line}")
    bootstrapped = [r for r in rows if r.verdict == "bootstrap"]
    # A case taken into the baseline must repeat exactly; once gated, a
    # spread of a quarter of its tolerance is worth a warning.
    unstable = [
        r
        for r in rows
        if r.spread is not None
        and (r.spread > 0 if r.verdict == "bootstrap" else r.spread > r.tolerance / 4)
    ]
    spread_lines = [
        f"{r.name} varied by {_spread(r.spread)} ({r.spread * (r.measured or 0):.3g} "
        f"instructions per repetition) between {args.processes} identical runs"
        + (
            ': give it its own "tolerance" in the baseline before it blocks'
            if r.verdict == "bootstrap"
            else ""
        )
        for r in unstable
    ]
    if len(dump_notes) > 3:
        dump_notes = [*dump_notes[:3], f"and {len(dump_notes) - 3} more dumps"]
    for line in spread_lines + dump_notes:
        print(f"::warning title=Perf gate::{line}")
    if bootstrapped and not problems:
        print(
            "\nBOOTSTRAP MODE: the baseline has no instruction counts for "
            f"{len(bootstrapped)} case(s); they are measured and reported, not gated."
            f"\nTo gate them, take this run as the baseline and commit it:\n  {_update_command()}"
        )
        if os.environ.get("GITHUB_ACTIONS"):
            print(
                "::warning title=Perf gate::bootstrap mode: instruction counts are not gated "
                "until this run's perf-gate-baseline artifact is committed"
            )

    failures = [r for r in rows if r.verdict in FAILING]
    if args.update:
        if problems:
            print("\nThe instruction counts are not valid; the baseline was not written:")
            for line in problems:
                print(f"  {line}")
            return 1
        args.baseline.write_text(json.dumps(measured, indent=2) + "\n")
        print(f"\nWrote {args.baseline}")
        if instructions is None:
            print("Only the program sizes were updated: " + note)
        return 0

    lines = list(problems)
    for r in failures:
        reason = {
            "slower": f"{_pct(r.delta)} instructions (limit +{100 * r.tolerance:.1f}%)",
            "improved": f"{_pct(r.delta)} instructions: update the baseline to keep the gain",
            "sizes": f"program sizes changed: {r.sizes}",
            "new": "not in the baseline",
            "removed": "in the baseline but no longer a case",
            "no baseline": "no instruction count in the baseline",
        }[r.verdict]
        lines.append(f"{r.name}: {reason}")
    invalid = baseline_error(baseline, frozen=os.environ.get("GITHUB_REF") == "refs/heads/main")
    if invalid:
        lines.append(f"{args.baseline.name}: {invalid}")
    if os.environ.get("GITHUB_ACTIONS"):
        dumps_hint = "the perf-gate-callgrind-dumps artifact has each run's dumps and Valgrind log"
    elif args.dumps:
        dumps_hint = f"each run's dumps and Valgrind log are in {args.dumps}"
    else:
        dumps_hint = "keep the dumps with --dumps DIR"
    if os.environ.get("GITHUB_ACTIONS"):
        for line in lines:
            print(f"::error title=Perf gate::{line}")
    if args.summary:
        with args.summary.open("a") as f:
            f.write("## Perf gate: instructions and program sizes\n\n")
            if problems:
                f.write(
                    "> [!CAUTION]\n"
                    "> **The instruction counts are not valid**, so this run fails in every "
                    "mode and its counts are not in its perf-gate-baseline artifact: "
                    + " ".join(f"{line}." for line in problems)
                    + f" To inspect: {dumps_hint}.\n\n"
                )
            elif bootstrapped:
                f.write(
                    "> [!WARNING]\n"
                    f"> **Bootstrap mode.** {len(bootstrapped)} case(s) have no instruction "
                    "count in the baseline: they are measured and reported below but not "
                    "gated. Program sizes are gated as always. To make the gate blocking, "
                    "take this run's measurements as the baseline and commit "
                    "`benchmarks/baselines/perf_gate.json` (this drops `bootstrap`):\n"
                    ">\n"
                    f"> ```\n> {_update_command()}\n> ```\n\n"
                )
            if spread_lines:
                f.write(
                    "> [!CAUTION]\n"
                    "> **Identical runs differed.** "
                    + " ".join(f"{line}." for line in spread_lines)
                    + "\n\n"
                )
            if dump_notes:
                f.write(
                    "> [!WARNING]\n"
                    "> **Callgrind's dump summaries differ from their totals** (the totals, "
                    "the sum of the cost lines, are used): "
                    + " ".join(f"{line}." for line in dump_notes)
                    + f" To inspect: {dumps_hint}.\n\n"
                )
            f.write(render(rows, markdown=True) + "\n\n")
            f.write(
                f"Fails above +{100 * REGRESSION:.1f}% or below -{100 * IMPROVEMENT:.1f}% "
                "instructions per repetition, or on any change in program size. "
                f"Spread: largest difference between {args.processes} identical runs.\n\n"
            )
            if note:
                f.write(f"{note}\n\n")
            if env_diff:
                f.write("Environment differs from the baseline's: " + "; ".join(env_diff) + "\n\n")
            if lines:
                f.write("**Failed:**\n\n" + "".join(f"- {line}\n" for line in lines) + "\n")
                if not problems:
                    f.write(
                        "If the change is intended, take this run as the baseline and "
                        f"commit it:\n\n```\n{_update_command()}\n```\n"
                    )
    if lines:
        print("\nPerf gate failed:")
        for line in lines:
            print(f"  {line}")
        if problems:
            print(f"\nThe instruction counts are not valid; {dumps_hint}.")
        else:
            print(f"\nIf the change is intended, update the baseline:\n  {_update_command()}")
        return 1
    if bootstrapped:
        print("\nPerf gate passed in bootstrap mode: instruction counts were not gated.")
    else:
        print("\nPerf gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
