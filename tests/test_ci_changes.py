# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which CI jobs a change runs: the classifier of tools/ci-changes.py, the
conditions the workflows put on their jobs, and the summary that holds a run
to them, tools/ci-ok.py (source tree only)."""

import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "ci-changes.py"
_OK = _ROOT / "tools" / "ci-ok.py"
_WORKFLOWS = _ROOT / ".github" / "workflows"
pytestmark = pytest.mark.skipif(not _TOOL.exists(), reason="needs the source tree's tools/")


def _classify(paths):
    spec = importlib.util.spec_from_file_location("ci_changes", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify(paths)


META, DOCS, PYTHON, NATIVE = (
    {"native": False, "python": False, "docs": False},
    {"native": False, "python": False, "docs": True},
    {"native": False, "python": True, "docs": True},
    {"native": True, "python": True, "docs": True},
)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        # The repository's own files: lint alone.
        (["CITATION.cff"], META),
        (["CODE_OF_CONDUCT.md", "SUPPORT.md", ".github/ISSUE_TEMPLATE/bug.yml"], META),
        # The pages, and the two files the pages or their tests read.
        (["docs/tutorials/coherence.rst"], DOCS),
        (["README.md"], DOCS),
        (["CHANGELOG.md", "CITATION.cff"], DOCS),
        # Files that tests read: the contributing guide, the license files.
        (["CONTRIBUTING.md"], DOCS),
        (["LICENSE", "LICENSES/Apache-2.0.txt", "NOTICE", "THIRD_PARTY_NOTICES"], DOCS),
        (["SECURITY.md"], DOCS),
        (["docs/_ext/a11y.py", "docs/_static/a11y.css"], DOCS),
        # The Python package, its tests and what runs them: a docstring too.
        (["core/frontend/cudaq_pulse/ops/__init__.py"], PYTHON),
        (["tests/test_units.py"], PYTHON),
        (["docs/index.rst", "tests/test_docs_a11y.py"], PYTHON),
        (["pyrightconfig.json", ".pre-commit-config.yaml"], PYTHON),
        (["tools/perf-gate.py", "benchmarks/perf_gate/cases.py"], PYTHON),
        (["tools/build-mlir-toolchain.sh", "tests/runtime/test_simulate.py"], PYTHON),
        # What the native build reads from those directories is native: the
        # pinned toolchain and its fetcher, the compiled fake of the GPU
        # runtime, compiled sources and CMake files, the build's
        # configuration and the pinned versions of its tools.
        (["tools/mlir-toolchain.lock.json"], NATIVE),
        (["tools/fetch-mlir-toolchain.py"], NATIVE),
        (["tests/runtime/fake_cudensitymat/fake_cudensitymat.cpp"], NATIVE),
        (["tests/runtime/fake_cudensitymat/include/cudensitymat.h"], NATIVE),
        (["tests/runtime/fake_cudensitymat/README"], NATIVE),
        (["tests/runtime/cudm_callback_table_test.cpp"], NATIVE),
        (["examples/custom_passes/cpp/CMakeLists.txt"], NATIVE),
        (["examples/custom_passes/cpp/DriveCount.cpp"], NATIVE),
        (["examples/custom_passes/cpp/FindPulse.cmake"], NATIVE),
        (["benchmarks/perf_gate/marks.c"], NATIVE),
        (["pyproject.toml"], NATIVE),
        (["uv.lock"], NATIVE),
        (["docs/index.rst", "uv.lock"], NATIVE),
        # The compiler, the runtime, the build and the workflows: everything.
        (["core/mlir/transforms/ScheduleAlap.cpp"], NATIVE),
        (["CMakeLists.txt"], NATIVE),
        (["test/Dialect/Pulse/ops.mlir"], NATIVE),
        ([".github/workflows/ci.yml"], NATIVE),
        ([".github/actions/changes/action.yml"], NATIVE),
        (["README.md", "core/mlir/bindings/bindings.cpp"], NATIVE),
        # What the tool cannot place runs everything; so does no list at all.
        (["a_new_directory/file.txt"], NATIVE),
        (["docs.txt"], NATIVE),
        (["README.md.orig"], NATIVE),
        ([], NATIVE),
        (["", "  "], NATIVE),
    ],
)
def test_a_change_is_classed_by_the_widest_thing_it_touches(paths, expected):
    assert _classify(paths) == expected


def test_every_top_level_entry_of_the_repository_is_placed_on_purpose():
    """A new top-level file or directory is native until someone places it;
    this test names the ones that are, so that a move is a decision."""
    classes = {
        name: _classify([name + "/x" if (_ROOT / name).is_dir() else name]) for name in _tracked()
    }
    assert {name for name, c in classes.items() if c == NATIVE} == {
        *(".clang-format", ".github", "CMakeLists.txt", "CMakePresets.json", "cmake", "core"),
        *("packaging", "pyproject.toml", "test", "uv.lock"),
    }
    assert _classify(["core/frontend/cudaq_pulse/compiler.py"]) == PYTHON
    assert _classify([".github/ISSUE_TEMPLATE/bug.yml"]) == META


def test_every_compiled_source_and_cmake_file_is_native():
    """Whatever directory holds it: the build of every compiler reads it."""
    if not shutil.which("git"):
        pytest.skip("needs git")
    out = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True)
    if out.returncode:
        pytest.skip("not a git checkout")
    suffixes = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".td", ".cmake")
    sources = [
        path
        for path in out.stdout.splitlines()
        if path.endswith(suffixes) or path.rsplit("/", 1)[-1] == "CMakeLists.txt"
    ]
    assert len(sources) > 50
    assert [path for path in sources if _classify([path]) != NATIVE] == []


def _tracked():
    out = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD"], cwd=_ROOT, capture_output=True, text=True
    )
    if out.returncode:
        pytest.skip("not a git checkout")
    return out.stdout.split()


def _jobs(workflow):
    """Each job's ``needs`` and ``if`` as text, read without a YAML parser:
    a job starts at a two-space key and holds its lines up to the next."""
    text = (_WORKFLOWS / workflow).read_text(encoding="utf-8")
    body = text[text.index("\njobs:\n") + len("\njobs:\n") :]
    jobs = {}
    starts = [m.start() for m in re.finditer(r"(?m)^  [A-Za-z][\w-]*:\n", body)]
    for start, end in zip(starts, [*starts[1:], len(body)], strict=True):
        part = body[start:end]
        name = part.split(":", 1)[0].strip()
        needs = re.search(r"(?m)^    needs: (.*)$", part)
        condition = re.search(r"(?ms)^    if: (.*?)\n    (?=[a-z-]+:)", part)
        jobs[name] = (
            needs.group(1) if needs else "",
            " ".join(condition.group(1).split()) if condition else "",
        )
    return jobs


# The class a job waits for; lint and the summary wait for none.
_CI = {
    "test": "python == 'true'",
    "perf-gate": "python == 'true'",
    "test-free-threaded": "python == 'true'",
    "old-compilers": "native == 'true'",
    "test-docs": "docs == 'true'",
}


def test_each_ci_job_runs_for_the_class_that_can_break_it():
    jobs = _jobs("ci.yml")
    assert set(_CI) | {"changes", "lint", "ci-ok", "gpu"} == set(jobs), (
        "a new job needs a class here"
    )
    for job, wanted in _CI.items():
        needs, condition = jobs[job]
        assert "changes" in needs, job
        assert f"needs.changes.outputs.{wanted}" in condition, (job, condition)
    # The pages' tests run alone only when no test job runs them anyway.
    assert "needs.changes.outputs.python != 'true'" in jobs["test-docs"][1]
    # Lint runs for every change.
    assert "needs.changes" not in jobs["lint"][1] and "changes" not in jobs["lint"][0]


def test_the_editor_types_check_runs_in_one_test_leg_from_a_fresh_environment():
    """It is steps of the test job, so the summary waits for it with the
    job: on the Linux x86-64, Python 3.12 leg, last, with the installed
    component in an environment outside the checkout."""
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "editor-types" not in _jobs("ci.yml")
    test = re.split(r"\n\n  (?=[#A-Za-z])", ci[ci.index("\n  test:\n") + 1 :])[0]
    steps = test.split("\n      - ")[1:]
    editor = [step for step in steps if step.startswith("name: Editor types, ")]
    assert len(editor) == 3 and editor == steps[-3:]
    leg = "if: matrix.platform == 'linux-x86_64' && matrix.python == '3.12'\n"
    assert all(leg in step and "continue-on-error" not in step for step in editor)
    install, environment, check = editor
    assert "cmake --install build/dev --component CudaqPulse --prefix installed" in install
    assert 'uv venv "$RUNNER_TEMP/user"' in environment
    assert 'cp -R installed/. "$site/"' in environment
    assert "PYTHONPATH" not in environment + check and "uv run" not in environment + check
    assert "python3 tools/check-editor-types.py" in check
    assert '--python "$RUNNER_TEMP/user/bin/python"' in check
    assert '--pyright "$RUNNER_TEMP/user/bin/pyright"' in check
    # The leg is in the matrix, and pyright is the version the tests use.
    assert '- { platform: linux-x86_64, runner: ubuntu-24.04, python: "3.12" }' in test
    version = re.search(r'"pyright==([\d.]+)"', (_ROOT / "pyproject.toml").read_text()).group(1)
    assert f"pyright=={version} " in environment


def test_the_perf_gate_runs_for_every_change_that_can_move_it():
    """The gate measures the compiler and the Python frontend: a change of
    either class runs it, and python is true for both."""
    assert _classify(["core/mlir/bindings/bindings.cpp"])["python"]
    assert _classify(["core/frontend/cudaq_pulse/kernel/_kernel.py"])["python"]
    assert _classify(["benchmarks/baselines/perf_gate.json"])["python"]
    assert _classify(["tools/perf-gate.py"])["python"]
    assert "needs.changes.outputs.python == 'true'" in _jobs("ci.yml")["perf-gate"][1]


def test_the_summary_waits_for_every_job_and_always_runs():
    jobs = _jobs("ci.yml")
    needs, condition = jobs["ci-ok"]
    waited = set(re.findall(r"[\w-]+", needs))
    # gpu runs elsewhere.
    assert waited == set(jobs) - {"ci-ok", "gpu"}
    assert condition.startswith("${{ always()")


def _ok():
    spec = importlib.util.spec_from_file_location("ci_ok", _OK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _needs(results, answers=None):
    """The ``needs`` context of a job that waits for *results*, with the
    classifier's *answers* as the outputs of ``changes``."""
    needs = {job: {"result": result, "outputs": {}} for job, result in results.items()}
    if answers is not None and "changes" in needs:
        needs["changes"]["outputs"] = answers
    return needs


def _run_ok(*args, needs, **env):
    """tools/ci-ok.py as the workflows run it."""
    needs = needs if isinstance(needs, str) else json.dumps(needs)
    return subprocess.run(
        [sys.executable, str(_OK), *args],
        env={"NEEDS": needs, **env},
        capture_output=True,
        text=True,
    )


def _summary(results, answers=None, **env):
    return _run_ok("summary", needs=_needs(results, answers), **env)


def _answers(widest):
    """The classifier's outputs for a change of the class *widest*."""
    order = ("none", "docs", "python", "native")
    return {
        name: "true" if order.index(widest) >= order.index(name) else "false"
        for name in ("native", "python", "docs")
    }


_ALL = ("changes", "lint", "test", "test-docs", "perf-gate", "test-free-threaded", "old-compilers")


def _results(ran, **other):
    """Every job of the summary: *ran* succeeded, the others were skipped."""
    return {**{job: "success" if job in ran else "skipped" for job in _ALL}, **other}


_PYTHON_JOBS = ("changes", "lint", "test", "perf-gate", "test-free-threaded")
_NATIVE_JOBS = (*_PYTHON_JOBS, "old-compilers")
_DOCS_JOBS = ("changes", "lint", "test-docs")


@pytest.mark.parametrize(
    ("widest", "results", "passes"),
    [
        # Each class of change, with the jobs it needs and no other.
        ("native", _results(_NATIVE_JOBS), True),
        ("python", _results(_PYTHON_JOBS), True),
        ("docs", _results(_DOCS_JOBS), True),
        ("none", _results(("changes", "lint")), True),
        # A job the change needs failed or was cancelled.
        ("python", _results(_PYTHON_JOBS, test="failure"), False),
        ("python", _results(_PYTHON_JOBS, **{"perf-gate": "cancelled"}), False),
        ("docs", _results(_DOCS_JOBS, lint="failure"), False),
        # A job the change needs was skipped: every native job, one of them,
        # the pages' tests, lint.
        ("native", _results(("changes", "lint")), False),
        ("native", _results(_PYTHON_JOBS), False),
        ("python", _results(("changes", "lint")), False),
        ("python", _results(_PYTHON_JOBS, **{"test-free-threaded": "skipped"}), False),
        ("docs", _results(("changes", "lint")), False),
        ("native", _results(_NATIVE_JOBS, lint="skipped"), False),
        # A job the change does not need was skipped, which is the plan; one
        # that ran although the plan leaves it out is a plan that is wrong.
        ("python", _results(_PYTHON_JOBS, **{"old-compilers": "skipped"}), True),
        ("python", _results(_NATIVE_JOBS), False),
        ("python", _results(_PYTHON_JOBS, **{"test-docs": "success"}), False),
        ("docs", _results(_DOCS_JOBS, **{"old-compilers": "failure"}), False),
    ],
)
def test_the_summary_passes_when_every_job_the_change_needs_succeeded(widest, results, passes):
    done = _summary(results, _answers(widest))
    assert (done.returncode == 0) == passes, done.stdout + done.stderr
    assert done.returncode in (0, 1)


@pytest.mark.parametrize(
    ("result", "answers"),
    [
        # The classifier did not succeed, whatever it answered.
        ("failure", _answers("native")),
        ("cancelled", _answers("docs")),
        ("skipped", _answers("none")),
        ("failure", {}),
        # It succeeded without an answer, or with one that is no answer.
        ("success", {}),
        ("success", {"native": "", "python": "", "docs": ""}),
        ("success", {"native": "true", "python": "true"}),
        ("success", {"native": "false", "python": "false", "docs": "True"}),
        ("success", {"native": "false", "python": "yes", "docs": "true"}),
        ("success", {"native": "false", "python": True, "docs": "true"}),
        # Each class includes the ones after it.
        ("success", {"native": "true", "python": "false", "docs": "true"}),
        ("success", {"native": "false", "python": "true", "docs": "false"}),
    ],
)
def test_the_summary_fails_without_a_classifier_that_succeeded_and_answered(result, answers):
    # Whatever the other jobs did: all skipped, or all passed.
    for ran in (("lint",), _ALL):
        done = _summary(_results(ran, changes=result), answers)
        assert done.returncode == 1, done.stdout + done.stderr
        assert "changes" in done.stderr


def test_the_summary_fails_on_a_needs_context_it_cannot_hold_to_the_plan():
    native = _answers("native")
    # A job of the plan that the summary does not wait for, and one it waits
    # for that the plan does not know.
    missing = _results(_NATIVE_JOBS)
    del missing["old-compilers"]
    assert _summary(missing, native).returncode == 1
    assert _summary(_results(_NATIVE_JOBS, deploy="success"), native).returncode == 1
    assert _summary(_results(_NATIVE_JOBS, deploy="skipped"), native).returncode == 1
    # No context at all.
    for needs in ("", "null", "[]", "{", '"success"'):
        assert _run_ok("summary", needs=needs).returncode == 2
    assert _run_ok("summary", needs={}).returncode == 1
    assert _run_ok("summary", needs={"changes": "success"}).returncode == 1
    assert _run_ok(needs={}).returncode == 2
    assert _run_ok("summary", "docs", needs={}).returncode == 2


def test_a_gpu_only_run_and_a_baseline_run_are_held_to_their_own_plans():
    # A pull-request/<N> branch runs the gpu job alone: nothing here ran.
    nothing = _results(())
    assert _summary(nothing, GPU_ONLY="true").returncode == 0
    assert _summary(nothing, GPU_ONLY="false").returncode == 1
    assert _summary(nothing).returncode == 1
    assert _summary(_results(("lint",)), GPU_ONLY="true").returncode == 1
    assert _summary(_results((), test="failure"), GPU_ONLY="true").returncode == 1
    # A manual run that takes the perf gate's baseline runs the gate alone.
    native = _answers("native")
    gate = _results(("changes", "perf-gate"))
    assert _summary(gate, native, PERF_BASELINE="true").returncode == 0
    assert _summary(gate, native, PERF_BASELINE="false").returncode == 1
    assert _summary(gate, native, PERF_BASELINE="").returncode == 1
    assert _summary(_results(("changes",)), native, PERF_BASELINE="true").returncode == 1
    assert _summary(_results(_NATIVE_JOBS), native, PERF_BASELINE="true").returncode == 1


def _condition(text):
    """A job's ``if:`` as a Python expression of the run's context and the
    classifier's answers; only the forms the workflow uses are read."""
    text = text.removeprefix(">- ")
    assert text.startswith("${{ ") and text.endswith(" }}"), text
    text = text[4:-3]
    for old, new in (
        ("!startsWith(github.ref, 'refs/heads/pull-request/')", "(not gpu_only)"),
        ("!inputs.perf_baseline", "(not perf_baseline)"),
        ("&&", "and"),
    ):
        text = text.replace(old, new)
    text = re.sub(r"needs\.changes\.outputs\.(\w+) == 'true'", r"\1", text)
    text = re.sub(r"needs\.changes\.outputs\.(\w+) != 'true'", r"(not \1)", text)
    assert re.fullmatch(r"[a-z_() ]+", text), f"a condition this test cannot read: {text}"
    return text


def test_the_plan_is_the_condition_of_each_job():
    """tools/ci-ok.py's plan and the jobs' ``if:`` lines say the same, for
    every kind of run and every class of change."""
    jobs = _jobs("ci.yml")
    conditions = {job: _condition(condition) for job, (_, condition) in jobs.items() if job in _ALL}
    assert set(conditions) == set(_ALL)
    plan = _ok().plan
    for gpu_only, perf_baseline, widest in itertools.product(
        (False, True), (False, True), ("none", "docs", "python", "native")
    ):
        answers = {name: value == "true" for name, value in _answers(widest).items()}
        context = {"gpu_only": gpu_only, "perf_baseline": perf_baseline}
        planned = plan(**answers, **context)
        assert set(planned) == set(_ALL)
        for job, condition in conditions.items():
            runs = eval(condition, {"__builtins__": {}}, {**answers, **context})
            assert planned[job] == runs, (job, condition, answers, context)


def test_the_summary_runs_the_tool_with_the_whole_context():
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("  ci-ok:\n") : ci.index("\n  gpu:\n")]
    assert "NEEDS: ${{ toJSON(needs) }}" in job
    assert "GPU_ONLY: ${{ startsWith(github.ref, 'refs/heads/pull-request/') }}" in job
    assert "PERF_BASELINE: ${{ inputs.perf_baseline }}" in job
    assert "run: python3 tools/ci-ok.py summary\n" in job
    assert "actions/checkout@" in job and "persist-credentials: false" in job


@pytest.mark.parametrize(
    ("workflow", "job", "wanted"),
    [
        ("docs.yml", "build", "docs"),
        ("sanitizers.yml", "sanitizers", "python"),
    ],
)
def test_the_other_workflows_skip_their_job_only_on_the_classifiers_word(workflow, job, wanted):
    """The job is a required check whose skipped run counts as passed: it is
    skipped when the classifier succeeded and said ``false``, and otherwise
    runs, its first step after the checkout failing unless the classifier
    succeeded and answered."""
    needs, condition = _jobs(workflow)[job]
    assert "changes" in needs
    assert condition == (
        "${{ !cancelled() && (needs.changes.result != 'success' || "
        f"needs.changes.outputs.{wanted} != 'false') }}}}"
    )
    text = (_WORKFLOWS / workflow).read_text(encoding="utf-8")
    steps = text[text.index(f"\n  {job}:\n") :]
    steps = re.findall(
        r"(?m)^      - (?:name|uses|run|id): (.*)$", steps[steps.index("    steps:") :]
    )
    assert steps[0].startswith("actions/checkout@")
    assert steps[1] == "The classifier succeeded and answered"
    assert f"run: python3 tools/ci-ok.py classifier {wanted}\n" in text


@pytest.mark.parametrize(
    ("result", "outputs", "passes"),
    [
        ("success", {"docs": "true"}, True),
        # The job's condition skips it for this answer; the step agrees.
        ("success", {"docs": "false"}, True),
        ("success", {"docs": "true", "python": ""}, True),
        ("failure", {"docs": "true"}, False),
        ("failure", {}, False),
        ("cancelled", {}, False),
        ("skipped", {}, False),
        ("success", {}, False),
        ("success", {"docs": ""}, False),
        ("success", {"docs": "1"}, False),
        ("success", {"python": "true"}, False),
    ],
)
def test_the_classifier_step_fails_unless_it_succeeded_and_answered(result, outputs, passes):
    done = _run_ok("classifier", "docs", needs={"changes": {"result": result, "outputs": outputs}})
    assert done.returncode == (0 if passes else 1), done.stdout + done.stderr
    assert _run_ok("classifier", "docs", needs={}).returncode == 1
    assert _run_ok("classifier", "pages", needs={}).returncode == 2


def _changes_action(tmp_path, before, after, *, event="pull_request", base=None):
    """The ``changes`` action's own script, run in a repository whose base
    commit holds the files *before* and whose head holds *after*; what it
    printed and what it wrote to the step's outputs."""
    if not all(shutil.which(tool) for tool in ("git", "bash", "python3")):
        pytest.skip("needs git, bash and python3")
    action = (_ROOT / ".github/actions/changes/action.yml").read_text(encoding="utf-8")
    script = action[action.index("      run: |\n") + len("      run: |\n") :]
    script = "\n".join(line[8:] for line in script.splitlines())
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(_TOOL, repo / "tools" / "ci-changes.py")

    def git(*args):
        identity = ("-c", "user.name=test", "-c", "user.email=test@example.com")
        done = subprocess.run(
            ["git", *identity, "-c", "commit.gpgsign=false", *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    def commit(files):
        for path in set(before) | set(after):
            (repo / path).unlink(missing_ok=True)
        for path, text in files.items():
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text(text)
        git("add", "-A")
        git("commit", "-q", "--allow-empty", "-m", "a commit")
        return git("rev-parse", "HEAD")

    git("init", "-q")
    first, head = commit(before), commit(after)
    outputs = tmp_path / "outputs"
    done = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
        cwd=repo,
        env={
            "PATH": os.environ["PATH"],
            "EVENT": event,
            "BASE": first if base is None else base,
            "GITHUB_SHA": head,
            "GITHUB_OUTPUT": str(outputs),
        },
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    answers = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    return {name: value == "true" for name, value in answers.items()}, done.stdout


_LONG = "".join(
    f"line {n} of a file long enough to be followed through a rename\n" for n in range(40)
)


@pytest.mark.parametrize(
    ("before", "after", "expected", "listed"),
    [
        # A file that the change adds.
        ({"README.md": "a"}, {"README.md": "a", "docs/new.rst": "b"}, DOCS, ["docs/new.rst"]),
        ({"README.md": "a"}, {"README.md": "a", "tests/test_new.py": "b"}, PYTHON, None),
        ({"README.md": "a"}, {"README.md": "a", "core/mlir/New.cpp": "b"}, NATIVE, None),
        ({"README.md": "a"}, {"README.md": "a", "a_new_directory/x": "b"}, NATIVE, None),
        # A file that it removes counts as the file did.
        ({"README.md": "a", "CITATION.cff": "b"}, {"README.md": "a"}, META, ["CITATION.cff"]),
        ({"README.md": "a", "docs/old.rst": "b"}, {"README.md": "a"}, DOCS, ["docs/old.rst"]),
        ({"README.md": "a", "tests/test_old.py": "b"}, {"README.md": "a"}, PYTHON, None),
        ({"README.md": "a", "test/old.mlir": "b"}, {"README.md": "a"}, NATIVE, None),
        # A file that it moves counts under both names: the tree it left lost one.
        (
            {"test/moved.mlir": _LONG},
            {"docs/moved.mlir": _LONG},
            NATIVE,
            ["docs/moved.mlir", "test/moved.mlir"],
        ),
        (
            {"tests/test_moved.py": _LONG},
            {"docs/test_moved.py": _LONG},
            PYTHON,
            ["docs/test_moved.py", "tests/test_moved.py"],
        ),
        ({"docs/a.rst": _LONG}, {"docs/b.rst": _LONG}, DOCS, ["docs/a.rst", "docs/b.rst"]),
        # A file that it edits; the widest class of several.
        ({"SUPPORT.md": "a"}, {"SUPPORT.md": "b"}, META, ["SUPPORT.md"]),
        ({"uv.lock": "a", "docs/a.rst": "a"}, {"uv.lock": "b", "docs/a.rst": "b"}, NATIVE, None),
        # A change of nothing has no path, and runs every job.
        ({"README.md": "a"}, {"README.md": "a"}, NATIVE, None),
    ],
)
def test_the_action_classes_what_a_pull_request_adds_removes_and_moves(
    tmp_path, before, after, expected, listed
):
    answers, printed = _changes_action(tmp_path, before, after)
    assert answers == expected
    if listed is not None:
        assert sorted(printed.split()[: len(listed)]) == listed


@pytest.mark.parametrize(
    ("event", "base"),
    [
        # A base that is not in the checkout, or that the event does not name.
        ("pull_request", "0" * 40),
        ("pull_request", "not-a-commit"),
        ("pull_request", ""),
        # Only a pull request is classed.
        ("push", None),
        ("workflow_dispatch", None),
    ],
)
def test_the_action_asks_for_every_job_when_it_cannot_compare(tmp_path, event, base):
    before, after = {"SUPPORT.md": "a"}, {"SUPPORT.md": "b"}
    answers, _ = _changes_action(tmp_path, before, after, event=event, base=base)
    assert answers == NATIVE


def _job_text(workflow, job):
    text = (_WORKFLOWS / workflow).read_text(encoding="utf-8")
    body = text[text.index(f"\n  {job}:\n") + 1 :]
    following = re.search(r"(?m)^  [A-Za-z][\w-]*:\n", body[len(job) + 4 :])
    return body[: len(job) + 4 + following.start()] if following else body


def test_the_sanitizer_job_to_require_fails_on_each_of_its_steps():
    """Nothing in the address+undefined row is allowed to fail: only the
    thread row may, and the job keeps the name that is required."""
    job = _job_text("sanitizers.yml", "sanitizers")
    assert "    name: sanitizers (${{ matrix.name }})\n" in job
    assert "- { name: address+undefined," in job
    assert re.findall(r"continue-on-error: (.*)", job) == ["${{ matrix.sanitizers == 'thread' }}"]
    assert "timeout " not in job
    for step in ("check-pulse-runtime", "check-pulse-mlir", "Python binding smoke check"):
        assert step in job


def test_the_sanitizer_smoke_check_names_tests_that_exist():
    job = _job_text("sanitizers.yml", "sanitizers")
    named = re.findall(r"(tests/[\w/]+\.py)::(\w+)", job)
    assert {path for path, _ in named} == {
        "tests/test_native_bindings.py",
        "tests/test_load.py",
        "tests/test_import.py",
    }
    assert 3 <= len(named) <= 8, "a smoke check, not the suite"
    for path, test in named:
        assert f"\ndef {test}(" in (_ROOT / path).read_text(encoding="utf-8"), (path, test)
    # A whole file or directory would not be a smoke check.
    code = "\n".join(line for line in job.splitlines() if not line.lstrip().startswith("#"))
    assert len(re.findall(r"(?<![\w/])tests\b", code)) == len(named)


def test_the_python_sweep_under_the_sanitizers_is_advisory():
    """It cannot fail the run or the required job, says so in its name, and
    leaves pull requests alone."""
    jobs = _jobs("sanitizers.yml")
    assert set(jobs) == {"changes", "sanitizers", "python-sweep"}
    assert jobs["sanitizers"][0] == "changes"
    needs, condition = jobs["python-sweep"]
    assert needs == "changes"
    assert "github.event_name != 'pull_request'" in condition
    sweep = _job_text("sanitizers.yml", "python-sweep")
    assert "    name: sanitizers (python sweep, advisory)\n" in sweep
    assert "\n    continue-on-error: true\n" in sweep
    assert '-m "not gpu and not slow"' in sweep


def test_the_nightly_hash_seeds_are_drawn_each_run_logged_and_counted():
    """Two seeds are drawn in the run, not written in the workflow; each has
    a job of its own that runs the suite under it and names it; and the
    nightly issue waits for both and gives the seeds."""
    text = (_WORKFLOWS / "property-nightly.yml").read_text(encoding="utf-8")
    jobs = _jobs("property-nightly.yml")
    # Every job but the report-only coverage counts toward the issue.
    waited = set(re.findall(r"[\w-]+", jobs["nightly-issue"][0]))
    assert waited == set(jobs) - {"nightly-issue", "coverage"}
    assert {"seeds", "hash-seeds"} <= waited
    assert jobs["hash-seeds"][0] == "seeds" and jobs["seeds"][0] == ""

    def job(name):
        return re.split(r"\n\n  (?=[#A-Za-z])", text[text.index(f"\n  {name}:\n") + 1 :])[0]

    draw = job("seeds")
    assert "shuf -i 1-4294967295 -n 2" in draw
    assert 'echo "first=${seeds[0]}"' in draw and 'echo "second=${seeds[1]}"' in draw
    assert "::notice title=PYTHONHASHSEED::" in draw and "$GITHUB_STEP_SUMMARY" in draw
    assert not re.search(r"PYTHONHASHSEED[=:] *\d", text), "a seed written in the workflow"
    suite = job("hash-seeds")
    for output in ("first", "second"):
        assert f"          - ${{{{ needs.seeds.outputs.{output} }}}}\n" in suite
    assert suite.split("\n    needs:")[0].endswith("PYTHONHASHSEED=${{ matrix.seed }})")
    assert "          PYTHONHASHSEED: ${{ matrix.seed }}\n" in suite
    assert 'if [ -z "$PYTHONHASHSEED" ]; then' in suite
    assert 'pytest tests -q -m "not gpu" -p no:cacheprovider' in suite
    assert "continue-on-error" not in draw + suite
    assert "fail-fast: false" in suite and re.search(r"(?m)^    timeout-minutes: \d+$", suite)
    # The test legs no longer run the suite a second and third time.
    assert "hash-seeds: true" not in text and "for seed in" not in text
    issue = job("nightly-issue")
    for name in ("first", "second"):
        assert f"{name.upper()}_SEED: ${{{{ needs.seeds.outputs.{name} }}}}\n" in issue
    assert "were given the hash seeds $FIRST_SEED and $SECOND_SEED" in issue
    assert issue.count('"$seeds" \\\n') == 1


def _module():
    spec = importlib.util.spec_from_file_location("ci_changes", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_test_reads_a_file_that_runs_lint_alone():
    """A file of META runs no test, so no test may read one."""
    names = [name.rstrip("/") for name in _module().META if not name.startswith(".github/")]
    names.remove(".gitignore")
    readers = {
        (path.name, name)
        for path in (_ROOT / "tests").rglob("*.py")
        if path.name != "test_ci_changes.py"
        for name in names
        if re.search(rf"[\"'/]{re.escape(name)}[\"']", path.read_text(encoding="utf-8"))
    }
    assert not readers


def test_the_pages_job_runs_every_test_that_reads_a_file_of_the_docs_class():
    """test-docs runs alone for a change of the docs class: each test file
    that names one of those files is in its list, or docs.yml runs it."""
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    listed = set(
        re.findall(r"tests/[\w/]+\.py", ci[ci.index("  test-docs:") : ci.index("  ci-ok:")])
    )
    listed |= set(
        re.findall(r"tests/[\w/]+\.py", (_WORKFLOWS / "docs.yml").read_text(encoding="utf-8"))
    )
    names = [name for name in _module().DOCS if name != "docs/"]
    pattern = "|".join(re.escape(name.rstrip("/")) for name in names)
    reads = re.compile(rf"[\"'/]({pattern})[\"'/]|\"docs\"|'docs'|/ \"docs\"|docs/[\w/]+\.rst")
    missing = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "tests").glob("*.py")
        if reads.search(path.read_text(encoding="utf-8"))
    } - listed
    assert not missing


def test_the_action_compares_a_pull_request_with_its_base_without_following_renames():
    action = (_ROOT / ".github/actions/changes/action.yml").read_text(encoding="utf-8")
    assert '"$EVENT" != pull_request ]] ||' in action
    assert "git diff --name-only --no-renames" in action
    assert action.count("ci-changes.py --all") == 1


def test_the_summary_has_the_required_name_only_where_it_is_the_whole_run():
    ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("  ci-ok:\n") :]
    assert "github.event_name == 'pull_request' || github.ref == 'refs/heads/main'" in job
    assert "!inputs.perf_baseline && 'ci-ok' ||" in job
    assert _summary(_results(())).returncode != 0
