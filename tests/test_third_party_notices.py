# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THIRD_PARTY_NOTICES generation from linker maps, and the wheel content
check's license, notice, CUDA and GPL rules (source tree only)."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
pytestmark = pytest.mark.skipif(
    not (_TOOLS / "third-party-notices.py").exists(), reason="needs the source tree's tools/"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notices():
    return _load("third-party-notices")


@pytest.fixture(scope="module")
def checker():
    return _load("check-wheel-contents")


_LD64_MAP = """# Path: bin/cudaq-pulse-opt
# Arch: arm64
# Object files:
[  0] linker synthesized
[  1] core/mlir/tools/cudaq-pulse-opt/CMakeFiles/cudaq-pulse-opt.dir/cudaq-pulse-opt.cpp.o
[  2] lib/libCudaqPulsePulse.a[3](PulseOps.cpp.o)
[  3] /tc/lib/libMLIRIR.a[7](Operation.cpp.o)
[  4] /tc/lib/libLLVMSupport.a[12](xxhash.cpp.o)
[  5] /tc/lib/libLLVMSupport.a[13](MD5.cpp.o)
[  6] /sdk/usr/lib/libc++.tbd
# Sections:
# Address\tSize\tFile\tName
0x100000000\t0x10\t[  9] /tc/lib/libUnknown.a(not-an-input.o)
"""

_GNU_MAP = """Archive member included to satisfy reference by file (symbol)

/tc/lib/libLLVMSupport.a(blake3_sse2_x86-64_unix.S.o)
                              /tc/lib/libLLVMSupport.a(blake3.c.o) (_blake3_compress_in_place_sse2)
/usr/lib/gcc/x86_64-redhat-linux/14/libgcc.a(_udivti3.o)
                              build/x.o (__udivti3)
 .text          0x0000000000001000       0x40 lib/libnanobind-static.a(nb_func.cpp.o)
"""

_LLD_MAP = """     VMA      LMA     Size Align Out     In      Symbol
    1000     1000       20     4 /tc/lib/libLLVMSupport.a(regcomp.c.o):(.text)
"""


def _components(notices, tmp_path, *texts):
    maps = []
    for index, text in enumerate(texts):
        maps.append(tmp_path / f"{index}.map")
        maps[-1].write_text(text)
    return notices.linked_components(maps)


def test_ld64_maps_are_read_up_to_the_sections(notices, tmp_path):
    found, problems = _components(notices, tmp_path, _LD64_MAP)
    assert found == {"LLVM", "MLIR", "xxHash (LLVM Support)", "MD5 (LLVM Support)"}
    assert problems == []


def test_gnu_ld_and_lld_maps(notices, tmp_path):
    found, problems = _components(notices, tmp_path, _GNU_MAP, _LLD_MAP)
    assert found == {
        "LLVM",
        "BLAKE3 (LLVM Support)",
        "regex (LLVM Support)",
        "nanobind",
        "robin-map (in nanobind)",
    }
    assert problems == []


def test_unknown_archives_and_uncovered_sources_fail(notices, tmp_path):
    _, problems = _components(
        notices,
        tmp_path,
        "[  1] /usr/lib/libfoo.a[1](foo.o)\n"
        "[  2] /tc/lib/libLLVMSupport.a[2](UnicodeNameToCodepointGenerated.cpp.o)\n"
        "[  3] /tc/lib/libMLIRMPIToLLVM.a[3](MPIToLLVM.cpp.o)\n",
    )
    assert len(problems) == 3
    for expected in [
        "libfoo.a(foo.o), which tools/third-party-notices.py does not know",
        "UnicodeNameToCodepointGenerated.cpp.o), under the Unicode data license",
        "MPIToLLVM.cpp.o), under MPICH's license",
    ]:
        assert any(expected in problem for problem in problems), expected


def test_a_missing_map_fails(notices, tmp_path):
    _, problems = notices.linked_components([tmp_path / "absent.map"])
    assert problems == [f"{tmp_path / 'absent.map'}: no linker map; relink the binary"]


@pytest.mark.parametrize(
    "item, component",
    [
        ("cudaq_pulse.libs/libz-1a2b3c4d.so.1.3", "zlib"),
        ("cudaq_pulse/.dylibs/libzstd.1.dylib", "Zstandard"),
        ("cudaq_pulse.libs/libtinfo-0a1b.so.6", "ncurses"),
    ],
)
def test_grafted_libraries(notices, item, component):
    assert notices.grafted_components(["cudaq_pulse/__init__.py", item]) == ({component}, [])


def test_grafted_gpl_and_unknown_libraries_fail(notices):
    found, problems = notices.grafted_components(
        ["cudaq_pulse.libs/libreadline-1a2b.so.8", "cudaq_pulse/.dylibs/libmystery.dylib"]
    )
    assert found == set()
    assert problems == [
        f"grafts cudaq_pulse.libs/libreadline-1a2b.so.8, {notices.GPL_PROBLEM}",
        "grafts cudaq_pulse/.dylibs/libmystery.dylib, which tools/third-party-notices.py "
        "does not know",
    ]


def test_write_then_check(notices, tmp_path, capsys):
    (tmp_path / "a.map").write_text(_LD64_MAP)
    out = tmp_path / "NOTICES"
    args = ["--notices", str(out), "--map", str(tmp_path / "a.map")]
    assert notices.main(["write", *args]) == 0
    text = out.read_text()
    assert text == notices.render({"LLVM", "MLIR", "xxHash (LLVM Support)", "MD5 (LLVM Support)"})
    assert "Component: MD5 (LLVM Support)" in text
    assert (_ROOT / "LICENSES" / "LLVM-xxhash-LICENSE.txt").read_text().strip() in text
    assert notices.main(["check", *args]) == 0

    out.write_text(text.replace("Yann Collet", "someone"))
    assert notices.main(["check", *args]) == 1
    assert "is not what this tool generates" in capsys.readouterr().err

    out.write_text(notices.render({"LLVM", "MLIR", "xxHash (LLVM Support)"}))
    assert notices.main(["check", *args]) == 1
    assert "does not list MD5 (LLVM Support)" in capsys.readouterr().err


def test_check_only_notes_what_this_platform_does_not_link(notices, tmp_path, capsys):
    (tmp_path / "a.map").write_text(_LD64_MAP)
    out = tmp_path / "NOTICES"
    out.write_text(
        notices.render(notices.linked_components([tmp_path / "a.map"])[0] | {"nanobind"})
    )
    assert notices.main(["check", "--notices", str(out), "--map", str(tmp_path / "a.map")]) == 0
    assert "lists nanobind, which nothing given here links" in capsys.readouterr().out


def test_a_graft_needs_its_license_text(notices, tmp_path, capsys):
    wheel = tmp_path / "w.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cudaq_pulse.libs/libz-1a2b.so.1", b"")
    out = tmp_path / "NOTICES"
    assert notices.main(["write", "--notices", str(out), "--wheel", str(wheel)]) == 1
    assert "LICENSES/zlib-LICENSE.txt is missing" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "path, expected",
    [
        ("THIRD_PARTY_NOTICES", {"LLVM", "MLIR", "nanobind"}),
        ("packaging/cudaq-pulse-tools/THIRD_PARTY_NOTICES", {"LLVM", "MLIR"}),
    ],
)
def test_the_committed_notices_are_generated(notices, path, expected):
    text = (_ROOT / path).read_text()
    listed = notices.listed_components(text)
    assert expected <= listed
    assert text == notices.render(listed)


def test_the_driver_wheel_ships_the_license_texts_its_notices_need(notices):
    package = _ROOT / "packaging" / "cudaq-pulse-tools"
    listed = notices.listed_components((package / "THIRD_PARTY_NOTICES").read_text())
    needed = {Path(notices._BY_NAME[name].license_file).name for name in listed}
    shipped = {link.name for link in (package / "LICENSES").iterdir()}
    assert shipped == needed
    for link in (package / "LICENSES").iterdir():
        assert link.read_text() == (_ROOT / "LICENSES" / link.name).read_text()


_EXPRESSION = (
    "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND MIT "
    "AND Spencer-94 AND (CC0-1.0 OR Apache-2.0) AND Unicode-DFS-2015 AND LicenseRef-LLVM-MD5"
)


def _wheel(
    tmp_path,
    checker,
    *,
    extra=(),
    drop=(),
    notices_text=None,
    expression=_EXPRESSION,
    declared=None,
):
    dist_info = "cudaq_pulse-0.1.0.dist-info"
    files = {
        "__builtins__.pyi": "",
        "cudaq_pulse/__init__.py": "",
        "cudaq_pulse/py.typed": "",
        "cudaq_pulse/ops/__init__.py": "__all__ = []\n",
        "cudaq_pulse/_native/_cudaq_pulse_native.pyi": "",
        "cudaq_pulse/_native/_cudaq_pulse_native.abi3.so": "",
        "cudaq_pulse/targets/calibrations/transmon_17q.json": "{}",
    }
    for name in checker._LICENSES:
        source = _ROOT / name
        files[f"{dist_info}/licenses/{name}"] = (
            notices_text
            if name == "THIRD_PARTY_NOTICES" and notices_text is not None
            else source.read_text()
        )
    for item in extra:
        files[item] = ""
    for item in drop:
        del files[item]
    metadata = [
        "Metadata-Version: 2.4",
        "Name: cudaq-pulse",
        "Version: 0.1.0",
        "Requires-Dist: cudaq-pulse-tools==0.1.0",
    ]
    if expression:
        metadata.append(f"License-Expression: {expression}")
    metadata += [
        f"License-File: {f}" for f in (checker._LICENSES if declared is None else declared)
    ]
    files[f"{dist_info}/METADATA"] = "\n".join(metadata) + "\n\n"
    path = tmp_path / "cudaq_pulse-0.1.0-cp312-abi3-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def _problems(checker, path):
    return [problem.removeprefix(f"{path.name}: ") for problem in checker.check_wheel(path)]


def test_a_complete_wheel_passes(tmp_path, checker):
    assert _problems(checker, _wheel(tmp_path, checker)) == []


def test_our_own_gpu_runtime_is_not_a_cuda_library(tmp_path, checker):
    path = _wheel(tmp_path, checker, extra=["cudaq_pulse/lib/libcudm-runtime-cu12.so"])
    assert _problems(checker, path) == []


@pytest.mark.parametrize(
    "dropped",
    [
        "THIRD_PARTY_NOTICES",
        "LICENSE",
        "LICENSES/LLVM-LICENSE.txt",
    ],
)
def test_license_files_are_required_and_declared(tmp_path, checker, dropped):
    licenses = f"cudaq_pulse-0.1.0.dist-info/licenses/{dropped}"
    assert f"missing {licenses}" in _problems(checker, _wheel(tmp_path, checker, drop=[licenses]))
    undeclared = [f for f in checker._LICENSES if f != dropped]
    assert f"METADATA does not name License-File {dropped}" in _problems(
        checker, _wheel(tmp_path, checker, declared=undeclared)
    )


@pytest.mark.parametrize(
    "item",
    [
        "cudaq_pulse/lib/libcudart.so.12",
        "cudaq_pulse.libs/libcublasLt-1a2b.so.12",
        "cudaq_pulse.libs/libcudensitymat-1a2b.so.0",
        "cudaq_pulse/.dylibs/libcusolver.11.dylib",
        "cudaq_pulse.libs/libnvJitLink-1a2b.so.12",
        "cudaq_pulse/lib/libcudart_static.a",
        "cudaq_pulse.libs/libcutensornet-1a2b.so.2",
    ],
)
def test_cuda_libraries_are_rejected(tmp_path, checker, item):
    path = _wheel(tmp_path, checker, extra=[item])
    assert f"bundles {item}; CUDA libraries come from the extras" in _problems(checker, path)


@pytest.mark.parametrize(
    "item",
    [
        "cudaq_pulse.libs/libreadline-1a2b.so.8",
        "cudaq_pulse/.dylibs/libgmp.10.dylib",
        "cudaq_pulse.libs/libstdc++-1a2b.so.6.0.33",
        "cudaq_pulse/lib/libgcc_s.so.1",
    ],
)
def test_gpl_libraries_are_rejected_once(tmp_path, checker, item):
    problems = _problems(checker, _wheel(tmp_path, checker, extra=[item]))
    assert [p for p in problems if item in p] == [f"bundles {item}, a GPL or LGPL library"]


def test_grafted_libraries_must_be_listed(tmp_path, checker):
    path = _wheel(tmp_path, checker, extra=["cudaq_pulse.libs/libz-1a2b.so.1"])
    assert "THIRD_PARTY_NOTICES does not list zlib, which the wheel grafts" in _problems(
        checker, path
    )
    path = _wheel(tmp_path, checker, extra=["cudaq_pulse.libs/libmystery-1a2b.so.1"])
    assert (
        "grafts cudaq_pulse.libs/libmystery-1a2b.so.1, which tools/third-party-notices.py "
        "does not know" in _problems(checker, path)
    )


def test_notices_must_be_generated_and_list_llvm(tmp_path, checker):
    text = (_ROOT / "THIRD_PARTY_NOTICES").read_text()
    path = _wheel(tmp_path, checker, notices_text=text.replace("Wenzel Jakob", "W. J."))
    assert "THIRD_PARTY_NOTICES is not what tools/third-party-notices.py generates" in _problems(
        checker, path
    )
    path = _wheel(tmp_path, checker, notices_text="Third-party software notices\n")
    assert "THIRD_PARTY_NOTICES does not list LLVM" in _problems(checker, path)
    path = _wheel(tmp_path, checker, notices_text=text + "Component: Mystery\n")
    assert "THIRD_PARTY_NOTICES lists Mystery, which is not a known component" in _problems(
        checker, path
    )


def test_license_expression_names_every_listed_license_and_no_gpl(tmp_path, checker):
    path = _wheel(tmp_path, checker, expression=_EXPRESSION.replace(" AND LicenseRef-LLVM-MD5", ""))
    assert "License-Expression does not name LicenseRef-LLVM-MD5 (MD5 (LLVM Support))" in (
        _problems(checker, path)
    )
    path = _wheel(tmp_path, checker, expression=_EXPRESSION + " AND LGPL-2.1-only")
    assert any("names a GPL or LGPL license" in p for p in _problems(checker, path))
    path = _wheel(tmp_path, checker, expression="")
    assert "METADATA has no License-Expression" in _problems(checker, path)


def _tools_wheel(tmp_path, checker, *, extra=(), drop=(), expression=None, texts=None):
    dist_info = "cudaq_pulse_tools-0.1.0.dist-info"
    package = _ROOT / "packaging" / "cudaq-pulse-tools"
    files = {"cudaq_pulse_tools-0.1.0.data/scripts/cudaq-pulse-opt": "", f"{dist_info}/WHEEL": ""}
    for name in checker._TOOLS_LICENSES:
        files[f"{dist_info}/licenses/{name}"] = (package / name).read_text()
    for name, text in (texts or {}).items():
        files[f"{dist_info}/licenses/{name}"] = text
    for item in extra:
        files[item] = ""
    for item in drop:
        del files[item]
    if expression is None:
        expression = _EXPRESSION.replace(" AND BSD-3-Clause AND MIT", "")
    metadata = ["Metadata-Version: 2.4", "Name: cudaq-pulse-tools", "Version: 0.1.0"]
    metadata.append(f"License-Expression: {expression}")
    metadata += [f"License-File: {f}" for f in checker._TOOLS_LICENSES]
    files[f"{dist_info}/METADATA"] = "\n".join(metadata) + "\n\n"
    files[f"{dist_info}/RECORD"] = ""
    path = tmp_path / "cudaq_pulse_tools-0.1.0-py3-none-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name)
            if name.endswith("/cudaq-pulse-opt"):
                info.external_attr = 0o755 << 16
            archive.writestr(info, content)
    return path


def test_a_complete_driver_wheel_passes(tmp_path, checker):
    assert "THIRD_PARTY_NOTICES" in checker._TOOLS_LICENSES
    assert "LICENSES/LLVM-LICENSE.txt" in checker._TOOLS_LICENSES
    assert _problems(checker, _tools_wheel(tmp_path, checker)) == []


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE", "LICENSES/LLVM-LICENSE.txt"])
def test_both_wheels_carry_the_repositorys_license_texts(tmp_path, checker, name):
    package = _ROOT / "packaging" / "cudaq-pulse-tools"
    assert (package / name).read_bytes() == (_ROOT / name).read_bytes()
    path = _tools_wheel(tmp_path, checker, texts={name: (_ROOT / name).read_text() + "\n"})
    item = f"cudaq_pulse_tools-0.1.0.dist-info/licenses/{name}"
    assert f"{item} differs from the repository's {name}" in _problems(checker, path)


def test_the_driver_wheel_needs_its_notices(tmp_path, checker):
    notices_file = "cudaq_pulse_tools-0.1.0.dist-info/licenses/THIRD_PARTY_NOTICES"
    assert f"missing {notices_file}" in _problems(
        checker, _tools_wheel(tmp_path, checker, drop=[notices_file])
    )
    path = _tools_wheel(tmp_path, checker, expression=_EXPRESSION.replace(" AND Spencer-94", ""))
    assert "License-Expression does not name Spencer-94 (regex (LLVM Support))" in _problems(
        checker, path
    )


def test_the_driver_wheel_may_carry_only_listed_grafts(tmp_path, checker):
    problems = _problems(
        checker,
        _tools_wheel(
            tmp_path,
            checker,
            extra=[
                "cudaq_pulse_tools.libs/libz-1a2b.so.1",
                "cudaq_pulse_tools.libs/libcudart-1a2b.so.12",
                "cudaq_pulse_tools.libs/libgmp-1a2b.so.10",
            ],
        ),
    )
    assert "THIRD_PARTY_NOTICES does not list zlib, which the wheel grafts" in problems
    assert (
        "bundles cudaq_pulse_tools.libs/libcudart-1a2b.so.12; CUDA libraries come from the extras"
        in problems
    )
    assert "bundles cudaq_pulse_tools.libs/libgmp-1a2b.so.10, a GPL or LGPL library" in problems
    assert not any(p.startswith("contains cudaq_pulse_tools.libs/") for p in problems)
