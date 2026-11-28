# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The files a wheel's tests get of the source tree (tests/_sources.py): the
list holds what the tests ask for, the layout holds the list, and a copy of
the list is a layout the check passes."""

import pytest

from tests import _sources

_LIST = "test-sources of [tool.cibuildwheel] in pyproject.toml"


@pytest.mark.level0
def test_the_layout_has_every_file_the_tests_of_a_wheel_read():
    assert _sources.problems() == []


@pytest.mark.level0
def test_a_file_that_is_not_listed_is_refused_by_name():
    # Under another name, so that the check does not read these calls.
    ask = _sources.source_file
    assert ask("benchmarks/compiler/compile_time.py").is_file()
    assert ask("tests/levels.toml").is_file()
    with pytest.raises(LookupError, match=r"tools/perf-gate\.py is not in test-sources"):
        ask("tools/perf-gate.py")
    # A listed name is a whole path, not the start of one or a way out of one.
    for path in ("tests_of_another_kind/a.py", "tests/../tools/perf-gate.py"):
        with pytest.raises(LookupError):
            ask(path)


@pytest.fixture
def tree(tmp_path):
    """A small source tree: two listed benchmarks, one of which loads the
    other, an unlisted one, and a test that asks for a listed file."""
    files = {
        "CMakeLists.txt": "",
        "pyproject.toml": "[tool.cibuildwheel]\n"
        'test-sources = ["tests", "pyproject.toml", "benchmarks/a.py", "benchmarks/data/b.py"]\n'
        f'test-command = ["{_sources.CHECK}", "python -m pytest"]\n'
        '[tool.cibuildwheel.linux]\nenvironment = { A = "1" }\n',
        "benchmarks/a.py": 'B = _load("benchmarks/data/b.py")\n',
        "benchmarks/data/b.py": "",
        "benchmarks/c.py": "",
        "tests/__init__.py": "",
        "tests/sub/test_reads.py": "from tests._sources import source_file\n"
        'A = source_file("benchmarks/a.py")\n',
    }
    root = tmp_path / "tree"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


@pytest.mark.level0
def test_a_copy_of_the_list_is_a_layout_the_check_passes(tree, tmp_path, capsys):
    assert _sources.problems(tree) == []
    copy = tmp_path / "copy"
    assert _sources.main(["copy", str(copy)], tree) == 0
    assert sorted(p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()) == [
        "benchmarks/a.py",
        "benchmarks/data/b.py",
        "pyproject.toml",
        "tests/__init__.py",
        "tests/sub/test_reads.py",
    ]
    assert not _sources.whole_tree(copy)
    assert _sources.main(["check"], copy) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.level0
def test_the_check_names_what_a_copy_lacks_and_what_a_test_asks_for(tree, tmp_path, capsys):
    copy = tmp_path / "copy"
    _sources.copy(copy, tree)
    (copy / "benchmarks" / "data" / "b.py").unlink()
    (copy / "tests" / "test_asks.py").write_text(
        "from tests._sources import source_file\n"
        'A = source_file("benchmarks/c.py")\n'
        'B = source_file("benchmarks/" + "a.py")\n'
    )
    assert _sources.problems(copy) == [
        f"benchmarks/data/b.py is in {_LIST} and is not in {copy}",
        f"tests/test_asks.py:2: benchmarks/c.py is not in {_LIST}",
        "tests/test_asks.py:3: source_file takes one literal path",
    ]
    assert _sources.main(["check"], copy) == 1
    assert "error: benchmarks/data/b.py is in test-sources" in capsys.readouterr().err


@pytest.mark.level0
def test_in_the_source_tree_a_listed_file_that_loads_an_unlisted_one_is_found(tree):
    loader = tree / "benchmarks" / "a.py"
    loader.write_text(loader.read_text() + 'C = _load("benchmarks/c.py")\n')
    assert _sources.problems(tree) == [
        f"benchmarks/a.py names benchmarks/c.py, which is not in {_LIST}"
    ]
    # The copy a wheel's tests get has no file to tell that by.
    (tree / "CMakeLists.txt").unlink()
    assert _sources.problems(tree) == []


@pytest.mark.level0
def test_every_test_command_of_the_wheels_begins_with_the_check(tree):
    project = tree / "pyproject.toml"
    project.write_text(
        project.read_text()
        + '[tool.cibuildwheel.macos]\ntest-command = "python -m pytest"\ntest-sources = ["tests"]\n'
        + '[[tool.cibuildwheel.overrides]]\nselect = "cp314t-*"\n'
        + f'test-command = ["python -m pytest", "{_sources.CHECK}"]\n'
    )
    assert _sources.problems(tree) == [
        f"[tool.cibuildwheel.macos] has test-sources of its own; {_LIST} is the one list",
        f"test-command of [tool.cibuildwheel.macos] does not begin with {_sources.CHECK!r}",
        f"test-command of [[tool.cibuildwheel.overrides]] 1 does not begin with {_sources.CHECK!r}",
    ]
