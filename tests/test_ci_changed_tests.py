# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The test files a pull request runs above level 0: what
tools/ci-changed-tests.py selects from the changed paths and how far it lets
them run, what ``--changed-paths`` then selects in a small project of its
own, and how the workflow passes the paths (source tree only).

The tests of the rule itself are at level 0: a pull request that changes the
rule runs them whatever else it selects."""

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "ci-changed-tests.py"
pytestmark = pytest.mark.skipif(not _TOOL.exists(), reason="needs the source tree's tools/")


def _tool():
    spec = importlib.util.spec_from_file_location("ci_changed_tests", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TREE = {
    "tests/test_frames.py",
    "tests/kernel/test_new.py",
    "tests/test_ci_changes.py",
    "tests/test_ci_changed_tests.py",
    "tests/test_levels.py",
    "tests/test_packed_schema.py",
    "tests/test_perf_gate.py",
    "tests/test_tutorial_scripts.py",
    "tests/test_release_tools.py",
    "tests/conftest.py",
    "tests/_handles.py",
}


def _select(*paths, tree=_TREE):
    return _tool().select(paths, lambda path: path in tree)


@pytest.mark.level0
@pytest.mark.parametrize(
    ("paths", "tests", "removed", "shared"),
    [
        # A test file the change adds or edits: git lists both the same way.
        (["tests/test_frames.py"], ["tests/test_frames.py"], [], []),
        (["tests/kernel/test_new.py"], ["tests/kernel/test_new.py"], [], []),
        # One it removes; one it moves is listed under both names.
        (["tests/test_gone.py"], [], ["tests/test_gone.py"], []),
        (
            ["tests/test_old_name.py", "tests/kernel/test_new.py"],
            ["tests/kernel/test_new.py"],
            ["tests/test_old_name.py"],
            [],
        ),
        # What many tests share selects none of them.
        (["tests/_handles.py"], [], [], ["tests/_handles.py"]),
        (["tests/kernel/ir_corpus.py"], [], [], ["tests/kernel/ir_corpus.py"]),
        (["tests/physics/reference_values.json"], [], [], ["tests/physics/reference_values.json"]),
        (
            ["tests/runtime/cudm_runtime_smoke.cpp"],
            [],
            [],
            ["tests/runtime/cudm_runtime_smoke.cpp"],
        ),
        (["tests/kernel/conftest.py"], [], [], ["tests/kernel/conftest.py"]),
        # No test file, and nothing a test file watches.
        (
            ["core/frontend/cudaq_pulse/controller.py", "core/mlir/Pass.cpp", "pyproject.toml"],
            [],
            [],
            [],
        ),
        (["core/mlir/tests/test_x.py", "examples/tests/test_x.py", "test/test_x.py"], [], [], []),
        ([], [], [], []),
        (["", "  "], [], [], []),
        # Several, each once, in order.
        (
            [
                "tests/test_frames.py",
                "tests/_handles.py",
                "tests/test_frames.py",
                "benchmarks/a.py",
            ],
            ["tests/test_frames.py"],
            [],
            ["tests/_handles.py"],
        ),
    ],
)
def test_a_change_selects_the_test_files_it_adds_or_edits(paths, tests, removed, shared):
    selection = _select(*paths)
    assert list(selection.tests) == tests
    assert set(selection.tests.values()) <= {"changed"}
    assert (selection.removed, selection.shared) == (removed, shared)


@pytest.mark.level0
def test_a_path_is_selected_only_as_the_plain_name_of_a_test_file():
    for path in (
        # Names git quotes, names with shell syntax, and paths out of tests/.
        '"tests/test_caf\\303\\251.py"',
        "tests/test_frames.py; echo run",
        "tests/test_$(touch x).py",
        "tests/test_`id`.py",
        "tests/test_a b.py",
        "tests/test_frames.py\x00",
        "tests/../tests/test_frames.py",
        "tests/./test_frames.py",
        "/tests/test_frames.py",
        "./tests/test_frames.py",
        "tests//test_frames.py",
        "tests/test_frames.py::test_x",
        "tests/test_frames.pyc",
        "tests/test-frames.py",
        "tests/frames_test.py",
        "-p tests/test_frames.py",
        "--level=3",
    ):
        # Whatever the tree is said to hold.
        selection = _tool().select([path], lambda path: True)
        assert not selection.tests and not selection.removed, path


@pytest.mark.level0
@pytest.mark.parametrize(
    ("path", "tests"),
    [
        (
            ".github/workflows/ci.yml",
            ["test_ci_changed_tests", "test_ci_changes", "test_perf_gate", "test_release_tools"],
        ),
        (
            ".github/actions/changes/action.yml",
            ["test_ci_changed_tests", "test_ci_changes", "test_perf_gate", "test_release_tools"],
        ),
        ("tools/ci-changes.py", ["test_ci_changes", "test_release_tools"]),
        ("tools/ci-ok.py", ["test_ci_changes", "test_release_tools"]),
        ("tools/ci-changed-tests.py", ["test_ci_changed_tests", "test_release_tools"]),
        ("tools/perf-gate.py", ["test_perf_gate", "test_release_tools"]),
        ("benchmarks/perf_gate/cases.py", ["test_perf_gate"]),
        ("tools/check-packed-schema.py", ["test_packed_schema", "test_release_tools"]),
        ("tools/tutorial-scripts.py", ["test_release_tools", "test_tutorial_scripts"]),
        ("tests/_levels.py", ["test_ci_changed_tests", "test_levels"]),
        ("tests/levels.toml", ["test_levels"]),
        ("tests/conftest.py", ["test_ci_changed_tests", "test_levels"]),
        ("benchmarks/compiler/compile_time.py", []),
    ],
)
def test_a_change_of_ci_policy_selects_the_tests_that_hold_it(path, tests):
    selection = _select(path)
    assert list(selection.tests) == [f"tests/{name}.py" for name in tests]
    assert set(selection.tests.values()) == ({f"reads {path}"} if tests else set())
    # A policy test that the change edits is selected as changed, once.
    both = _select(path, "tests/test_ci_changes.py")
    assert both.tests["tests/test_ci_changes.py"] == "changed"
    # One that the tree does not hold is not selected.
    assert not _select(path, tree=set()).tests


@pytest.mark.level0
def test_every_test_that_reads_the_workflows_runs_when_they_change():
    watches = _tool().WATCHES
    readers = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "tests").rglob("test_*.py")
        if re.search(r"""["'/ ]\.github["'/_]""", path.read_text(encoding="utf-8"))
    }
    assert readers == {test for test, names in watches.items() if ".github/" in names}
    assert all((_ROOT / test).is_file() for test in watches)
    assert all((_ROOT / name).exists() for names in watches.values() for name in names)


@pytest.mark.level0
@pytest.mark.parametrize(
    ("counts", "reached"),
    [
        # Everything fits: level 1 and level 2 of each file.
        ({"a": {0: 3, 1: 80, 2: 3, 3: 9}, "b": {1: 42}}, {"a": 2, "b": 2}),
        ({"a": {}}, {"a": 2}),
        ({}, {}),
        # A sweep over the budget stays at the level it reached.
        ({"a": {1: 10, 2: 3667}}, {"a": 1}),
        ({"a": {2: 3667}, "b": {1: 80, 2: 3}}, {"a": 1, "b": 2}),
        ({"a": {1: 1001}}, {"a": 0}),
        ({"a": {1: 1000}}, {"a": 2}),
        # Level 1 of every file comes before level 2 of any, smallest first.
        ({"a": {1: 600, 2: 10}, "b": {1: 300, 2: 200}}, {"a": 2, "b": 1}),
        ({"a": {1: 600}, "b": {1: 500}, "c": {1: 400, 2: 50}}, {"a": 0, "b": 2, "c": 2}),
        # A file that did not get its level 1 does not get its level 2.
        ({"a": {1: 1001, 2: 1}, "b": {1: 1}}, {"a": 0, "b": 2}),
        # Many files: the budget is for all of them together.
        (
            {f"f{n:02}": {1: 90} for n in range(20)},
            {f"f{n:02}": 2 if n < 11 else 0 for n in range(20)},
        ),
    ],
)
def test_changed_files_run_to_level_2_within_a_budget_of_cases(counts, reached):
    tool = _tool()
    assert (tool.MAX_LEVEL, tool.MAX_CASES) == (2, 1000)
    got = tool.plan(counts, 0)
    assert got == reached
    added = sum(
        cases for file, per in counts.items() for n, cases in per.items() if 0 < n <= got[file]
    )
    assert added <= tool.MAX_CASES


@pytest.mark.level0
def test_a_run_at_a_higher_level_adds_only_the_levels_above_it():
    tool = _tool()
    assert tool.plan({"a": {1: 5000, 2: 10, 3: 4}}, 1) == {"a": 2}
    assert tool.plan({"a": {1: 5000, 2: 10, 3: 4}}, 2) == {"a": 2}
    assert tool.plan({"a": {3: 4}}, 3) == {"a": 3}


@pytest.mark.level0
def test_the_report_names_what_ran_and_what_was_left():
    tool = _tool()
    selection = _select(
        ".github/workflows/ci.yml",
        "tests/test_frames.py",
        "tests/kernel/test_new.py",
        "tests/test_gone.py",
        "tests/_handles.py",
    )
    counts = {file: {} for file in selection.tests}
    counts["tests/test_frames.py"] = {0: 2, 1: 85, 2: 2, 3: 1}
    counts["tests/kernel/test_new.py"] = {1: 4, 2: 3667}
    counts["tests/test_ci_changes.py"] = {1: 42}
    reached = tool.plan(counts, 0)
    lines = tool.report(selection, counts, reached, 0)
    text = "\n".join(lines)
    assert lines[0].startswith("test files of this change, run above level 0: to level 2, at most")
    assert re.search(
        r"tests/test_frames\.py +\(changed\) runs 85 of level 1, 2 of level 2; "
        r"left for the nightly run: 1 of level 3\n",
        text,
    )
    assert re.search(
        r"tests/kernel/test_new\.py +\(changed\) runs 4 of level 1; "
        r"over 1000 cases, left for the push to main: 3667 of level 2\n",
        text,
    )
    assert re.search(
        r"tests/test_ci_changes\.py +\(reads \.github/workflows/ci\.yml\) runs 42 of level 1\n",
        text,
    )
    assert re.search(r"tests/test_perf_gate\.py +\(reads .*\) runs nothing above level 0\n", text)
    assert "  removed: tests/test_gone.py\n" in text
    assert text.endswith("which run at their own level: tests/_handles.py")
    assert tool.report(_select("pyproject.toml"), {}, {}, 0)[1:] == ["  none"]


# The option in a small project of its own, as tests/test_levels.py has one.

_CONFTEST = """
from pathlib import Path
from tests import _levels

def pytest_addoption(parser):
    _levels.add_options(parser)

def pytest_configure(config):
    here = Path(__file__).parent
    _levels.register(config, here / "levels.toml", here)
"""
_INI = """
[pytest]
addopts = --strict-markers
markers =
    level0
    level1
    level2
    level3
"""
_TABLE = """
[level2.files]
"tests/test_sweep.py" = "a sweep"
[level3.tests]
"tests/test_new.py::test_wording" = "exact wording"
"""
_NEW = """
import pytest

@pytest.mark.level0
def test_contract():
    pass

def test_behaviour():
    pass

@pytest.mark.level2
def test_exhaustive():
    pass

def test_wording():
    pass
"""
_SWEEP = """
import pytest

@pytest.mark.parametrize("case", range(1001))
def test_swept(case):
    pass
"""


@pytest.fixture
def project(tmp_path):
    files = {
        "conftest.py": _CONFTEST,
        "pytest.ini": _INI,
        "levels.toml": _TABLE,
        "tests/test_new.py": _NEW,
        "tests/test_other.py": "def test_default():\n    pass\n",
        "tests/test_sweep.py": _SWEEP,
        "tests/helper.py": "",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(textwrap.dedent(text).lstrip())
    return tmp_path


def _pytest(project, *args, changed=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    for name in ("PYTEST_ADDOPTS", "FORCE_COLOR", "PY_COLORS"):
        env.pop(name, None)
    env["NO_COLOR"] = "1"
    if changed is not None:
        (project / "changed-paths.txt").write_bytes(changed)
        args = (*args, f"--changed-paths={project / 'changed-paths.txt'}")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _selected(project, *args, changed=None):
    run = _pytest(project, "--collect-only", *args, changed=changed)
    assert run.returncode == 0, run.stdout + run.stderr
    return {line.split("[")[0] for line in run.stdout.splitlines() if "::" in line}, run.stdout


_LEVEL_0 = {"tests/test_new.py::test_contract"}


def test_a_run_at_level_0_also_runs_the_changed_files_to_level_2(project):
    assert _selected(project, "--level", "0")[0] == _LEVEL_0
    changed = b"tests/test_new.py\ntests/test_sweep.py\ntests/helper.py\ntests/test_gone.py\n"
    selected, printed = _selected(project, "--level", "0", changed=changed)
    assert selected == _LEVEL_0 | {
        "tests/test_new.py::test_behaviour",
        "tests/test_new.py::test_exhaustive",
    }
    assert re.search(
        r"tests/test_new\.py +\(changed\) runs 1 of level 1, 1 of level 2; "
        r"left for the nightly run: 1 of level 3\n",
        printed,
    )
    assert re.search(
        r"tests/test_sweep\.py +\(changed\) runs nothing above level 0; "
        r"over 1000 cases, left for the push to main: 1001 of level 2\n",
        printed,
    )
    assert "  removed: tests/test_gone.py\n" in printed
    assert "which run at their own level: tests/helper.py\n" in printed
    # The files alone, at their own levels; and nothing without a level to add to.
    selected, _ = _selected(project, "--level", "1", changed=b"tests/test_new.py\n")
    assert selected == {
        "tests/test_other.py::test_default",
        "tests/test_new.py::test_contract",
        "tests/test_new.py::test_behaviour",
        "tests/test_new.py::test_exhaustive",
    }
    everything, printed = _selected(project, changed=b"tests/test_new.py\n")
    assert everything == _selected(project)[0] and "test files of this change" not in printed


def test_a_failing_test_of_level_1_in_a_changed_file_fails_the_run_at_level_0(project):
    (project / "tests" / "test_new.py").write_text(
        "def test_added_at_level_1():\n    assert False\n"
    )
    assert _pytest(project, "--level", "0").returncode == 5
    run = _pytest(project, "--level", "0", changed=b"core/mlir/Pass.cpp\ntests/test_new.py\n")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "test_added_at_level_1" in run.stdout
    # The lines are printed before the tests run and again beside the result.
    assert run.stdout.count("test files of this change, run above level 0") == 2
    # A change of another file leaves the test where its level puts it.
    run = _pytest(project, "--level", "0", changed=b"tests/test_other.py\n")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout


def test_paths_that_are_no_test_file_select_nothing(project):
    hostile = (
        b"tests/test_new.py; touch injected\n$(touch injected)\n`touch injected`\n"
        b"--level=3\n-p no:terminal\ntests/../tests/test_new.py\n/etc/passwd\n"
        b'"tests/test_caf\\303\\251.py"\n\xff\xfe tests/test_new.py\n\n'
    )
    selected, printed = _selected(project, "--level", "0", changed=hostile)
    assert selected == _LEVEL_0
    assert "  none\n" in printed
    assert not (project / "injected").exists()
    selected, _ = _selected(project, "--level", "0", changed=b"")
    assert selected == _LEVEL_0


def test_a_list_that_cannot_be_read_is_a_usage_error(project):
    run = _pytest(project, "--level", "0", f"--changed-paths={project / 'missing.txt'}")
    assert run.returncode == 4
    assert "--changed-paths cannot be read" in run.stderr


# How ci.yml passes the paths.


def _ci():
    return (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def _job(name):
    text = _ci()
    start = text.index(f"\n  {name}:\n")
    end = re.search(r"(?m)^  [A-Za-z][\w-]*:\n", text[start + len(name) + 5 :])
    return text[start : start + len(name) + 5 + end.start()]


@pytest.mark.level0
@pytest.mark.parametrize("job", ["test", "test-free-threaded"])
def test_each_test_job_runs_level_0_and_the_changed_files_on_a_pull_request(job):
    text = _job(job)
    options = re.search(r"PYTEST_ADDOPTS: >-\n((?:            .*\n)+)", text).group(1)
    options = " ".join(options.split())
    # Level 0 on a pull request with the list of paths, levels 0 to 2 otherwise.
    assert options == (
        "--durations=30 --level ${{ github.event_name == 'pull_request' && "
        "format('0 --changed-paths={0}/changed-paths.txt', runner.temp) || '2' }}"
    )
    assert text.count("PYTEST_ADDOPTS") == 1
    # The list is written before the tests, for a pull request, and the job
    # fails without it; no other event has a base to compare with.
    listing = text.index("      - name: Paths this pull request changes\n")
    assert listing < text.index("      - name: check-pulse (lit + pytest)\n")
    step = text[listing:].split("\n\n")[0]
    assert "        if: github.event_name == 'pull_request'\n" in step
    assert "continue-on-error" not in step
    assert "pull_request_target" not in _ci()


@pytest.mark.level0
@pytest.mark.parametrize("job", ["test", "test-free-threaded"])
def test_the_paths_are_listed_as_the_changes_action_lists_them(job):
    """One comparison for the classes of a change and for its test files; the
    base reaches the shell quoted, through the environment."""
    action = (_ROOT / ".github/actions/changes/action.yml").read_text(encoding="utf-8")
    command = 'git diff --name-only --no-renames "$BASE" "$GITHUB_SHA" --'
    assert command in action
    assert "BASE: ${{ github.event.pull_request.base.sha }}" in action
    step = _job(job)
    step = step[step.index("      - name: Paths this pull request changes\n") :]
    step = step.split("\n\n")[0] + "\n"
    assert "          BASE: ${{ github.event.pull_request.base.sha }}\n" in step
    assert f'        run: {command} > "$RUNNER_TEMP/changed-paths.txt"\n' in step
    assert "${{" not in step.split("        run: ")[1]


@pytest.mark.level0
def test_every_pull_request_runs_tests_of_threads_on_the_free_threaded_build():
    """Level 0 holds the tests of concurrent compiles and of the two caches
    that threads share; the rest of the file runs on the push to main."""
    text = (_ROOT / "tests" / "test_threads.py").read_text(encoding="utf-8")
    marked = re.findall(r"@pytest\.mark\.level0\ndef (test_\w+)\(", text)
    assert marked == [
        "test_concurrent_compiles_match_sequential_ones",
        "test_concurrent_first_calls_compile_once",
        "test_concurrent_first_loads_build_each_target_and_calibration_once",
    ]
    assert len(re.findall(r"(?m)^def test_\w+\(", text)) > len(marked)
