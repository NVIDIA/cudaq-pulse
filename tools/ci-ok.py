#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether a workflow run did what its change needs.

Reads the ``needs`` context of the calling job, as JSON, from the ``NEEDS``
environment variable: each job it waits for, with its ``result`` and its
``outputs``.

``ci-ok.py summary`` is the ``ci-ok`` job of ``ci.yml``. The classifier
(``changes``, tools/ci-changes.py) says which jobs the change needs, and
``plan`` is that answer per job. The run passes when every job of the plan
succeeded and every other job was skipped. A job of the plan that was
skipped, cancelled or failed fails the run, and so does a classifier that
did not succeed or whose answers are not each ``true`` or ``false``: without
them there is no plan to hold the run to. ``GPU_ONLY=true`` (a
pull-request/<N> branch, which runs the gpu job alone) plans no job, and
``PERF_BASELINE=true`` (a manual run that takes the perf gate's baseline)
plans the classifier and the perf gate.

``ci-ok.py classifier NAME...`` is the first step of a job that another
workflow gates on the classifier (``docs.yml``, ``sanitizers.yml``): it fails
unless ``changes`` succeeded and each answer NAME is ``true`` or ``false``.
Such a job is skipped only when the answer is ``false``, so a classifier
that failed fails the job instead of skipping it.
"""

from __future__ import annotations

import json
import os
import sys

CLASSES = ("native", "python", "docs")


def classes(needs: dict, names: tuple[str, ...] = CLASSES) -> tuple[dict[str, bool], list[str]]:
    """The classifier's answers *names*, and what is wrong with them."""
    job = needs.get("changes")
    if not isinstance(job, dict):
        return {}, ["changes: the job is not waited for"]
    problems = []
    if job.get("result") != "success":
        problems.append(f"changes: {job.get('result')}, not success")
    outputs = job.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    for name in names:
        if outputs.get(name) not in ("true", "false"):
            problems.append(
                f"changes: its answer {name!r} is {outputs.get(name)!r}, not 'true' or 'false'"
            )
    if problems:
        return {}, problems
    answers = {name: outputs[name] == "true" for name in names}
    # Each class includes the ones after it (tools/ci-changes.py).
    for wider, narrower in zip(CLASSES, CLASSES[1:], strict=False):
        if answers.get(wider) and answers.get(narrower) is False:
            problems.append(f"changes: {wider} is true but {narrower} is false")
    return ({}, problems) if problems else (answers, [])


def plan(
    *, native: bool, python: bool, docs: bool, gpu_only: bool = False, perf_baseline: bool = False
) -> dict[str, bool]:
    """Whether each job of ``ci.yml`` that the summary waits for runs.

    The same conditions as the jobs' ``if:`` lines; tests/test_ci_changes.py
    holds the two together.
    """
    whole = not gpu_only and not perf_baseline
    return {
        "changes": not gpu_only,
        "lint": whole,
        "test": whole and python,
        "perf-gate": not gpu_only and python,
        "test-free-threaded": whole and python,
        "old-compilers": whole and native,
        "test-docs": whole and docs and not python,
    }


def summary(needs: dict, *, gpu_only: bool = False, perf_baseline: bool = False) -> list[str]:
    """What keeps the run of ``ci.yml`` from passing; nothing when it passes."""
    results = {
        job: need.get("result") if isinstance(need, dict) else None for job, need in needs.items()
    }
    if gpu_only:
        answers = dict.fromkeys(CLASSES, False)
        problems = []
    else:
        answers, problems = classes(needs)
    if problems:
        # No plan: say which jobs did not pass, beside the classifier.
        problems += [
            f"{job}: {result}"
            for job, result in results.items()
            if job != "changes" and result not in ("success", "skipped")
        ]
        return problems
    planned = plan(**answers, gpu_only=gpu_only, perf_baseline=perf_baseline)
    for job, wanted in planned.items():
        if job not in results:
            problems.append(f"{job}: in the plan, but the summary does not wait for it")
        elif wanted and results[job] != "success":
            problems.append(f"{job}: the change needs it, but it was {results[job]}")
        elif not wanted and results[job] != "skipped":
            problems.append(f"{job}: the change does not need it, but it was {results[job]}")
    problems += [f"{job}: not in the plan" for job in results if job not in planned]
    return problems


def main(argv: list[str]) -> int:
    if (
        not argv
        or argv[0] not in ("summary", "classifier")
        or (argv[0] == "summary") != (len(argv) == 1)
    ):
        print("usage: ci-ok.py summary | ci-ok.py classifier NAME...", file=sys.stderr)
        return 2
    if unknown := set(argv[1:]) - set(CLASSES):
        print(f"ci-ok.py: not an answer of the classifier: {sorted(unknown)}", file=sys.stderr)
        return 2
    try:
        needs = json.loads(os.environ.get("NEEDS", ""))
    except ValueError:
        needs = None
    if not isinstance(needs, dict):
        print("ci-ok.py: NEEDS is not the needs context as JSON", file=sys.stderr)
        return 2
    print(json.dumps(needs, indent=2))
    if argv[0] == "classifier":
        problems = classes(needs, tuple(argv[1:]))[1]
    else:
        problems = summary(
            needs,
            gpu_only=os.environ.get("GPU_ONLY") == "true",
            perf_baseline=os.environ.get("PERF_BASELINE") == "true",
        )
    for problem in problems:
        print(f"::error title=ci-ok::{problem}")
    if problems:
        print("not passed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
