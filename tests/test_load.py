# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.load``: a scheduled function another compiler built."""

import re

import cudaq_pulse as pulse
import pytest

MODULE = """
module {
  func.func private @other(%q: !pulse.qudit) attributes {pulse.schedule_state =
        #pulse.schedule_state<timed>} {
    return
  }
  func.func private @round(%a: !pulse.qudit, %b: !pulse.qudit) -> i1 attributes {made.by =
        "elsewhere", pulse.schedule_state = #pulse.schedule_state<timed>} {
    %da, %ta = pulse.get_drive_line %a {made.channel = 3 : i64} : (!pulse.qudit) ->
          (!pulse.drive_line, !pulse.tone)
    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %rb, %rt = pulse.get_readout_line %b : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %da1, %ta1 = pulse.drive %da, %w, %ta <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %db1, %tb1 = pulse.drive %db, %w, %tb <{duration_cycles = 16 : i64, start_cycles = 16 : i64}>
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %m = arith.constant 64 : i64
    %r = pulse.square %m, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rb1, %rt1, %out = pulse.readout %rb, %r, %rt, iq <{duration_cycles = 64 : i64, start_cycles =
          32 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line,
          !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %out : !pulse.measurement -> i1
    return %bit : i1
  }
}
"""


@pytest.fixture
def system():
    return pulse.quantum_system("transmon_surface17")


def events(program):
    return [
        (e["kind"], e["qudit"], e["start_cycles"], e["duration_cycles"])
        for e in program.export_to_controller()["events"]
    ]


def test_schedule_is_kept_and_qudits_are_bound_in_order(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[2], system.qudits[0]])
    assert isinstance(program, pulse.ScheduledKernel)
    assert program.qudits == (0, 2)
    assert events(program) == [
        ("drive", 2, 0, 16),
        ("drive", 0, 16, 16),
        ("readout", 0, 32, 64),
    ]


def test_foreign_attributes_and_results_are_dropped(system):
    program = pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    text = program.mlir()
    assert "made." not in text
    assert "func.func @main()" in text
    assert program.metrics.op_count == 3
    assert "3 ops" in repr(program)


@pulse.kernel
def _phase_folded_into_a_drive(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 3.0e6)
    drive(d, square(16, 0.1))
    shift_phase(d.tone, 0.3)
    drive(d, square(16, 0.1))


@pulse.kernel
def _phase_folded_into_fused_drives(q):
    d = get_drive_line(q)
    shift_phase(d.tone, 0.3)
    drive(d, square(16, 0.1))
    drive(d, square(16, 0.1))
    wait(d, 8)
    drive(d, gaussian(16, 0.1, 4.0))


@pulse.kernel
def _phase_folded_in_a_loop(q):
    d = get_drive_line(q)
    for _ in range(6):
        shift_phase(d.tone, 0.25)
        drive(d, square(16, 0.1))
        wait(d, 8)


@pulse.kernel
def _phase_folded_in_an_arm(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        shift_phase(d.tone, 0.5)
        drive(d, square(16, 0.1))
    drive(d, square(16, 0.1))


def _scheduled_text(lowered):
    """The scheduled assembly of a lowered kernel as a function of its
    qudits, which ``load`` takes, and the qudits in argument order."""
    text = lowered.mlir()
    allocated = re.findall(r"(%\w+) = pulse\.qudit_alloc \{qudit = (\d+) : i64\}", text)
    assert allocated
    text = re.sub(r"^ *%\w+ = pulse\.qudit_alloc .*\n", "", text, flags=re.M)
    for name, index in allocated:
        text, found = re.subn(
            rf"(pulse\.get_\w+_line) {re.escape(name)} \{{frequency_hz = [^}}]*\}}",
            rf"\1 %q{index}",
            text,
        )
        assert found
    arguments = ", ".join(f"%q{index}: !pulse.qudit" for _, index in allocated)
    assert "func.func @main()" in text and "pulse.qudit_alloc" not in text
    text = text.replace("func.func @main()", f"func.func @main({arguments})")
    text = re.sub(r"^module @\w+ attributes \{.*\} \{$", "module {", text, count=1, flags=re.M)
    return text, [int(index) for _, index in allocated]


def _both_exports(program):
    return [
        program.export_to_controller()["events"],
        program.export_to_controller(version=2)["events"],
    ]


@pytest.mark.level0
@pytest.mark.parametrize(
    ("kernel", "options"),
    [
        (_phase_folded_into_a_drive, {}),
        (_phase_folded_into_fused_drives, {}),
        (_phase_folded_in_a_loop, {}),
        (_phase_folded_in_an_arm, {}),
    ],
)
def test_a_phase_folded_into_a_drive_is_kept_as_in_the_compiled_kernel(system, kernel, options):
    lowered = pulse.compile(kernel, quantum_system=system, **options).lower(system.qudits[0])
    text, qudits = _scheduled_text(lowered)
    # The compiler folded the shift into the drive after it.
    assert "frame_phase_offset = " in text and "pulse.shift_phase" not in text
    program = pulse.load(text, entry="main", qudits=[system.qudits[i] for i in qudits])
    assert _both_exports(program) == _both_exports(lowered)
    folded = re.findall(r"frame_phase_offset = [^ ]+ : f64", lowered.mlir())
    assert re.findall(r"frame_phase_offset = [^ ]+ : f64", program.mlir()) == folded


_DRIVE = (
    "%da1, %ta1 = pulse.drive %da, %w, %ta <{duration_cycles = 16 : i64, start_cycles = 0 : i64}>"
)


def _with_on_the_first_drive(attributes):
    text = MODULE.replace(_DRIVE + " :", _DRIVE + " {" + attributes + "} :")
    assert text != MODULE
    return text


def _first_drive(program):
    [first, *_] = program.export_to_controller()["events"]
    assert first["kind"] == "drive"
    return first


def test_the_two_phases_of_a_drive_are_kept_and_exported(system):
    plain = _first_drive(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))
    assert plain["phase_rad"] == 0.0
    # A shift folded into the drive stays in the tone's frame; the drive's
    # own phase does not.
    text = _with_on_the_first_drive("frame_phase_offset = 0.25 : f64, phase_offset = 0.5 : f64")
    program = pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert "frame_phase_offset = 2.500000e-01 : f64, phase_offset = 5.000000e-01 : f64" in (
        program.mlir()
    )
    assert _first_drive(program)["phase_rad"] == 0.75
    only_own = _with_on_the_first_drive("phase_offset = 0.5 : f64")
    assert (
        _first_drive(pulse.load(only_own, entry="round", qudits=system.qudits[:2]))["phase_rad"]
        == 0.5
    )


@pytest.mark.parametrize(
    "attributes",
    [
        'frame_phase_offset = "0.25"',
        "frame_phase_offset = 1 : i64",
        "frame_phase_offset = 0.25 : f32",
        "frame_phase_offset = 0x7FF8000000000000 : f64",
        "phase_offset = 0x7FF0000000000000 : f64",
        "phase_offsets = 0.25 : f64",
        "frame_offset = 0.25 : f64",
    ],
)
def test_a_phase_that_is_not_a_finite_f64_on_a_drive_is_another_compilers(system, attributes):
    text = _with_on_the_first_drive(attributes)
    program = pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert "phase_offset" not in program.mlir() and "frame_offset" not in program.mlir()
    plain = pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    assert _both_exports(program) == _both_exports(plain)


def test_a_drives_phases_are_kept_on_a_drive_only(system):
    text = MODULE.replace(
        "start_cycles =\n          32 : i64}> :", "start_cycles =\n          32 : i64}> {PHASE} :"
    ).replace("%w = pulse.square %n, %amp, %zero :", "%w = pulse.square %n, %amp, %zero {PHASE} :")
    assert text.count("{PHASE}") == 2
    text = text.replace("{PHASE}", "{frame_phase_offset = 0.25 : f64, phase_offset = 0.5 : f64}")
    program = pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert "phase_offset" not in program.mlir()
    plain = pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    assert _both_exports(program) == _both_exports(plain)


def test_lines_get_the_frequencies_of_their_qudits(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[1], system.qudits[2]])
    exported = program.export_to_controller()
    drives = [e for e in exported["events"] if e["kind"] == "drive"]
    assert [e["frequency_hz"] for e in drives] == [system.frequencies[1], system.frequencies[2]]


def test_bytecode_is_accepted(system):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    bytecode = native.MLIRPipeline().parse_to_module(MODULE).serialize()[0]
    assert isinstance(bytecode, bytes) and bytecode.startswith(b"ML\xefR")
    program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert events(program) == events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))


# ── The dialect source a bytecode module states ──────────────────────────────
#
# Bytecode holds an operation's properties by position, in the shape of the
# writer's dialects: read with dialects of another shape, a start lands in a
# duration and nothing reports it. A writer states the source of its
# dialects in the producer string of its bytecode, as the package and
# cudaq-pulse-opt do, and load reads only bytecode that states this build's
# or none.

_KEY = b"cudaq-pulse-dialects="


def _source():
    from cudaq_pulse._dialect_source import DIALECT_SOURCE

    return DIALECT_SOURCE


def _bytecode(text):
    """*text* as the package writes it: a pickled program's module."""
    from cudaq_pulse._native import _cudaq_pulse_native as native

    return native.MLIRPipeline().parse_to_module(text).serialize()[0]


def _producer(bytecode):
    """The producer string of *bytecode*: it follows the magic bytes and the
    format's version, and a zero byte ends it."""
    assert bytecode[:4] == b"ML\xefR" and bytecode[4] & 1
    return bytecode[5 : bytecode.index(b"\0", 5)]


def _produced_by(bytecode, producer):
    """*bytecode* with another producer string."""
    return bytecode[:5] + producer + bytecode[5 + len(_producer(bytecode)) :]


def _stating(bytecode, source, tool="elsewhere 0.3"):
    """*bytecode* as another writer that states *source* writes it."""
    return _produced_by(bytecode, f"{tool} cudaq-pulse-dialects={source}".encode())


def _restated(bytecode, other):
    """*bytecode* that states this build's source, with *other* in its
    place in the header and every other byte as it was."""
    stated = _KEY + _source().encode()
    assert len(other) == len(_source()) and other != _source()
    assert _producer(bytecode).count(stated) == 1
    return bytecode.replace(stated, _KEY + other.encode(), 1)


def test_the_packages_bytecode_states_this_builds_dialect_source(system):
    bytecode = _bytecode(MODULE)
    assert _producer(bytecode).split()[-1] == _KEY + _source().encode()
    program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert events(program) == events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))


def test_the_drivers_bytecode_states_this_builds_dialect_source(system, tmp_path):
    """``cudaq-pulse-opt --emit-bytecode``, from assembly and from its own
    bytecode."""
    import shutil
    import subprocess
    from pathlib import Path

    built = Path(pulse.__file__).parent.parent.parent / "bin" / "cudaq-pulse-opt"
    driver = str(built) if built.exists() else shutil.which("cudaq-pulse-opt")
    if driver is None:
        pytest.skip("cudaq-pulse-opt is neither in the build tree nor on PATH")
    (tmp_path / "round.mlir").write_text(MODULE)
    expected = events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))
    given = "round.mlir"
    for written in ("first.mlirbc", "second.mlirbc"):
        subprocess.run(
            [driver, "--emit-bytecode", given, "-o", written],
            cwd=tmp_path,
            timeout=60,
            check=True,
        )
        bytecode = (tmp_path / written).read_bytes()
        producer = _producer(bytecode)
        assert producer.startswith(b"cudaq-pulse-opt ")
        assert producer.split()[-1] == _KEY + _source().encode()
        program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
        assert events(program) == expected
        with pytest.raises(pulse.CompilationError, match="states the dialect source"):
            pulse.load(
                _restated(bytecode, _another("digest")), entry="round", qudits=system.qudits[:2]
            )
        given = written


@pytest.mark.parametrize("tool", ["elsewhere 0.3", "", "a (b) c"])
def test_bytecode_that_states_this_builds_dialect_source_loads(system, tool):
    bytecode = _stating(_bytecode(MODULE), _source(), tool)
    program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert events(program) == events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))
    after = _stating(_bytecode(MODULE), _source() + " and more")
    assert events(pulse.load(after, entry="round", qudits=system.qudits[:2])) == events(program)


def _another(part):
    """This build's source with one part changed and its length kept: the
    digest, or the major, the minor or the patch of the LLVM version."""
    llvm, _, digest = _source().rpartition("-")
    if part == "digest":
        return llvm + "-" + ("0" if digest[0] != "0" else "1") + digest[1:]
    if part in ("major", "minor", "patch"):
        numbers = llvm.split(".")
        at = ("major", "minor", "patch").index(part)
        # Its last digit changed; what follows the patch's digits is kept.
        digits = len(numbers[at]) - len(numbers[at].lstrip("0123456789"))
        last = numbers[at][digits - 1]
        numbers[at] = (
            numbers[at][: digits - 1] + ("8" if last == "9" else "9") + numbers[at][digits:]
        )
        return ".".join(numbers) + "-" + digest
    return {"longer": _source() + "0", "shorter": _source()[:-1], "none": ""}[part]


@pytest.mark.level0
def test_bytecode_of_another_patch_release_of_the_same_llvm_minor_loads(system):
    """A writer built with another patch release of this build's LLVM minor,
    from the same dialect files, writes bytecode this build reads: the
    package's own bytecode with that source in its header loads as it
    does."""
    other = _another("patch")
    assert other != _source() and other.rpartition(".")[0] == _source().rpartition(".")[0]
    bytecode = _restated(_bytecode(MODULE), other)
    program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert events(program) == events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))


def _llvm_and_digest():
    llvm, _, digest = _source().rpartition("-")
    major, minor = llvm.split(".")[:2]
    return major, minor, digest


@pytest.mark.parametrize(
    "form",
    [
        "{major}.{minor}.4-{digest}",
        "{major}.{minor}.123-{digest}",
        "{major}.{minor}.8git-{digest}",
        "{major}.{minor}.0-rc1-{digest}",
        "{major}.{minor}.x.y-{digest}",
        "{major}.{minor}-{digest}",
        "{major}.{minor}.-{digest}",
        "{major}.{minor}.8-{major}.{minor}.8-{digest}",
    ],
)
def test_what_follows_the_llvm_minor_is_not_compared(system, form):
    major, minor, digest = _llvm_and_digest()
    stated = form.format(major=major, minor=minor, digest=digest)
    program = pulse.load(
        _stating(_bytecode(MODULE), stated), entry="round", qudits=system.qudits[:2]
    )
    assert len(events(program)) == 3


@pytest.mark.parametrize(
    "form",
    [
        "{major}-{digest}",
        "{major}.-{digest}",
        ".{minor}.8-{digest}",
        "{major}.{minor}git-{digest}",
        "{major}.{minor}-rc1-{digest}",
        "{major}.x.8-{digest}",
        "v{major}.{minor}.8-{digest}",
        "+{major}.{minor}.8-{digest}",
        "{major}.\u0661.8-{digest}",
        "{major}..{minor}-{digest}",
        "{major}.{minor}.8",
        "{major}.{minor}.8_{digest}",
        "{digest}",
        "-{digest}",
    ],
)
def test_a_stated_source_that_is_not_a_version_and_a_digest_is_refused(system, form):
    """A statement that cannot be read as an LLVM major and minor and a
    digest is a statement of another source, not of none: what is stated
    and cannot be held to this build's is not read."""
    major, minor, digest = _llvm_and_digest()
    stated = form.format(major=major, minor=minor, digest=digest)
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(_stating(_bytecode(MODULE), stated), entry="round", qudits=system.qudits[:2])
    assert "is not an LLVM version, a hyphen and a digest" in str(refused.value)


@pytest.mark.parametrize(
    ("form", "differs"),
    [
        ("0{major}.{minor}.8-{digest}", "LLVM versions differ in their major.minor"),
        ("{major}.0{minor}.8-{digest}", "LLVM versions differ in their major.minor"),
        ("{major}.{minor}0.8-{digest}", "LLVM versions differ in their major.minor"),
        ("{major}0.{minor}.8-{digest}", "LLVM versions differ in their major.minor"),
        ("{major}.{minor}.8-", "digests of the dialects' definition files differ"),
        ("{major}.{minor}.8-{digest}-", "digests of the dialects' definition files differ"),
        ("{major}.{minor}.8-{digest}0", "digests of the dialects' definition files differ"),
        ("{major}.{minor}.8-{other}", "digests of the dialects' definition files differ"),
        ("{major}0.{minor}.8-{other}", "digests of the dialects' definition files and"),
    ],
)
def test_a_version_or_a_digest_that_differs_as_written_is_refused(system, form, differs):
    major, minor, digest = _llvm_and_digest()
    other = digest.upper() if digest.upper() != digest else "x" + digest[1:]
    stated = form.format(major=major, minor=minor, digest=digest, other=other)
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(_stating(_bytecode(MODULE), stated), entry="round", qudits=system.qudits[:2])
    assert f": the {differs}" in str(refused.value)


@pytest.mark.level0
@pytest.mark.parametrize("part", ["digest", "major", "minor"])
def test_bytecode_of_another_dialect_source_is_refused(system, part):
    """The package's own bytecode, with another source in its header and
    every other byte as written."""
    other = _another(part)
    bytecode = _restated(_bytecode(MODULE), other)
    with pytest.raises(pulse.CompilationError) as refused:
        pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert refused.value.code == "wrong_argument"
    assert other in str(refused.value) and _source() in str(refused.value)


def test_another_dialect_source_is_refused_before_the_module_is_parsed(system):
    """The refusal reads the header alone: what follows it here is no
    module, and a module of another shape is not one to parse first."""
    header = f"ML\xefR\rsome tool cudaq-pulse-dialects={_another('digest')}".encode("latin-1")
    with pytest.raises(pulse.CompilationError, match="states the dialect source"):
        pulse.load(header + b"\0not a module", entry="round", qudits=system.qudits[:2])
    stated = f"ML\xefR\rsome tool cudaq-pulse-dialects={_source()}".encode("latin-1")
    with pytest.raises(pulse.CompilationError, match="cannot read the module"):
        pulse.load(stated + b"\0not a module", entry="round", qudits=system.qudits[:2])


@pytest.mark.parametrize(
    ("producer", "refused"),
    [
        ("a cudaq-pulse-dialects={mine} cudaq-pulse-dialects={other}", False),
        ("a cudaq-pulse-dialects={other} cudaq-pulse-dialects={mine}", True),
        ("a (cudaq-pulse-dialects={mine})", True),
        ("a cudaq-pulse-dialects={mine}\tb", False),
        ("my-cudaq-pulse-dialects={other}", True),
        ("a cudaq-pulse-dialects {other}", False),
    ],
)
def test_the_first_stated_source_counts_up_to_white_space(system, producer, refused):
    stated = producer.format(mine=_source(), other=_another("digest")).encode()
    module = _produced_by(_bytecode(MODULE), stated)
    if refused:
        with pytest.raises(pulse.CompilationError, match="states the dialect source"):
            pulse.load(module, entry="round", qudits=system.qudits[:2])
    else:
        assert len(events(pulse.load(module, entry="round", qudits=system.qudits[:2]))) == 3


@pytest.mark.level3
def test_a_refusal_names_both_dialect_sources_and_what_to_do(system):
    other = _another("digest")
    with pytest.raises(pulse.CompilationError) as refused:
        pulse.load(_stating(_bytecode(MODULE), other), entry="round", qudits=system.qudits[:2])
    assert str(refused.value) == (
        f"load: this bytecode states the dialect source {other} (its producer is "
        f"'elsewhere 0.3 cudaq-pulse-dialects={other}'), and this cudaq-pulse's is "
        f"{_source()}: the digests of the dialects' definition files differ; bytecode "
        "holds an operation's properties by position and is read only with the dialect "
        "source that wrote it: give the module as assembly, or write its bytecode with "
        "this source"
    )
    for part, differs in (
        ("minor", "the LLVM versions differ in their major.minor"),
        ("major", "the LLVM versions differ in their major.minor"),
        ("none", "the stated source is not an LLVM version, a hyphen and a digest"),
    ):
        with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
            pulse.load(
                _stating(_bytecode(MODULE), _another(part)),
                entry="round",
                qudits=system.qudits[:2],
            )
        assert f": {differs}; bytecode holds" in str(refused.value)


@pytest.mark.level3
@pytest.mark.parametrize("part", ["longer", "shorter", "none"])
def test_a_digest_that_is_longer_shorter_or_missing_is_refused(system, part):
    other = _another(part)
    bytecode = _stating(_bytecode(MODULE), other, "a tool")
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(memoryview(bytecode), entry="round", qudits=system.qudits[:2])
    assert f"(its producer is 'a tool cudaq-pulse-dialects={other}')" in str(refused.value)
    assert ("source with no value " in str(refused.value)) == (part == "none")


@pytest.mark.level3
def test_a_refusal_shows_the_start_of_a_long_producer_string(system):
    bytecode = _stating(_bytecode(MODULE), _another("digest"), "t" * 500)
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert "t" * 120 + "')" in str(refused.value) and "t" * 121 not in str(refused.value)
    bytecode = _stating(_bytecode(MODULE), "9" * 500 + "\x07")
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert f"dialect source {'9' * 64} (its" in str(refused.value)
    bytecode = _stating(_bytecode(MODULE), "9\x07")
    with pytest.raises(pulse.CompilationError, match="states the dialect source") as refused:
        pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert "dialect source '9\\x07' (its" in str(refused.value)


def test_bytecode_that_states_no_dialect_source_loads_as_before(system):
    """Bytecode from a writer that states none is read as it is given: the
    package's with the key written over, with the producer string an older
    build wrote, with one that names the source of something else and with
    none."""
    bytecode = _bytecode(MODULE)
    expected = events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))
    for module in (
        bytecode.replace(_KEY, b"x" * len(_KEY), 1),
        _produced_by(bytecode, b"cudaq-pulse"),
        _produced_by(bytecode, b"other-dialects=1.2-ab"),
        _produced_by(bytecode, b""),
    ):
        assert _KEY not in _producer(module)
        program = pulse.load(module, entry="round", qudits=system.qudits[:2])
        assert events(program) == expected


def test_assembly_states_no_dialect_source(system):
    """Assembly holds properties by name: text that names another source,
    as a string or as bytes, is read as any text is."""
    text = MODULE.replace(
        'made.by =\n        "elsewhere"',
        f'made.by =\n        "cudaq-pulse-dialects={_another("digest")}"',
    )
    assert text != MODULE
    expected = events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))
    for module in (text, text.encode(), b"ML" + text.encode()[2:]):
        if module[:2] == b"ML":
            with pytest.raises(pulse.CompilationError, match="cannot read the module"):
                pulse.load(module, entry="round", qudits=system.qudits[:2])
            continue
        assert events(pulse.load(module, entry="round", qudits=system.qudits[:2])) == expected


@pytest.mark.parametrize(
    "module",
    [
        b"ML\xefR",
        b"ML\xefR\r",
        b"ML\xefR\rcudaq-pulse-dialects=1",
        b"ML\xefR\0cudaq-pulse-dialects=1",
        b"ML\xefR\x02",
    ],
)
def test_bytecode_cut_short_in_its_header_is_left_to_the_parser(system, module):
    with pytest.raises(pulse.CompilationError, match="cannot read the module") as refused:
        pulse.load(module, entry="round", qudits=system.qudits[:2])
    assert "dialect source" not in str(refused.value)


@pytest.mark.parametrize("version", [b"\r", b"\x02\x00", b"\x00" + b"\x01" * 8])
def test_the_header_is_read_past_a_format_version_of_any_length(version):
    """A version of more than one byte is not MLIR's today; the producer
    string is found after it all the same."""
    from cudaq_pulse import _load

    stated = b"ML\xefR" + version + b"a tool " + _KEY
    assert _load._check_dialect_source(stated + _source().encode() + b"\0rest") is None
    assert _load._check_dialect_source(stated + _source().encode() + b" b\0rest") is None
    with pytest.raises(pulse.CompilationError, match="states the dialect source"):
        _load._check_dialect_source(stated + _another("digest").encode() + b"\0rest")


def test_a_loaded_program_is_written_with_this_builds_dialect_source(system):
    """The source is the writer's, not the program's: a program loaded from
    another writer's bytecode pickles as every program does, and its module
    as the package writes it states this build's source."""
    import pickle

    program = pulse.load(
        _stating(_bytecode(MODULE), _source()), entry="round", qudits=system.qudits[:2]
    )
    again = pickle.loads(pickle.dumps(program))
    assert events(again) == events(program)
    for written in (program._module.serialize()[0], again._module.serialize()[0]):
        assert _producer(written).split()[-1] == _KEY + _source().encode()


@pytest.mark.level3
def test_the_dialect_source_is_the_digest_of_the_dialects_definition_files(tmp_path):
    """What cmake/write_dialect_source.cmake writes, computed here as a writer
    built from a checkout computes it, and written as that writer includes
    it."""
    import hashlib
    import pathlib
    import shutil
    import subprocess

    import cudaq_pulse._pickling as staged
    from cudaq_pulse._build_config import LLVM_VERSION

    if not pathlib.Path(staged.__file__).is_symlink():
        pytest.skip("an installed package: its dialects were hashed when it was built")
    root = pathlib.Path(staged.__file__).resolve().parents[3]
    files = sorted(
        found.relative_to(root).as_posix()
        for folder in ("include/cudaq-pulse/Dialect", "dialects")
        for dialect in ("Pulse", "Classical")
        for found in (root / "core/mlir" / folder / dialect).rglob("*")
        if found.suffix in (".td", ".h", ".cpp") and found.is_file()
    )
    assert len(files) > 10
    lines = "".join(
        f"{file} {hashlib.sha256((root / file).read_bytes()).hexdigest()}\n" for file in files
    )
    if _source() != f"{LLVM_VERSION}-{hashlib.sha256(lines.encode()).hexdigest()[:16]}":
        pytest.fail("_dialect_source.py is stale: build the package again")
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("no cmake to write the source with")
    subprocess.run(
        [cmake, f"-DROOT={root}", f"-DLLVM_VERSION={LLVM_VERSION}"]
        + [f"-DOUTPUT={tmp_path / 'dialect_source.inc'}", "-P"]
        + [str(root / "cmake/write_dialect_source.cmake")],
        check=True,
    )
    written = (tmp_path / "dialect_source.inc").read_text().splitlines()
    assert written[-1] == f'"{_source()}"'


def test_a_function_that_selects_at_run_time_is_refused(system):
    text = MODULE.replace(
        ", pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %da,",
        ", pulse.schedule_state = #pulse.schedule_state<timed>, "
        "pulse.selected_values = 1 : i64} {\n    %da,",
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="selects values at run time"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_bytes_are_accepted(system):
    program = pulse.load(MODULE.encode(), entry="round", qudits=system.qudits[:2])
    assert len(events(program)) == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda kw: kw.update(entry="missing"), "no function named missing"),
        (lambda kw: kw.update(qudits=kw["qudits"][:1]), "takes 2 qudit"),
    ],
)
def test_wrong_entry_or_count(system, change, message):
    kw = dict(entry="round", qudits=list(system.qudits[:2]))
    change(kw)
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(MODULE, **kw)


def test_unscheduled_function_is_refused(system):
    text = MODULE.replace(
        ", pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %da,", "} {\n    %da,"
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="not scheduled"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_overlap_on_a_line_fails_verification(system):
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64")
    assert text != MODULE
    text = text.replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
    with pytest.raises(pulse.CompilationError, match="overlaps or precedes"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_qudits_must_be_distinct_and_of_one_system(system):
    with pytest.raises(pulse.InvalidValueError, match="twice"):
        pulse.load(MODULE, entry="round", qudits=[system.qudits[0], system.qudits[0]])
    other = pulse.quantum_system("transmon_surface17")
    if other is not system:
        with pytest.raises(pulse.InvalidValueError, match="more than one"):
            pulse.load(MODULE, entry="round", qudits=[system.qudits[0], other.qudits[1]])


def test_a_stated_clock_must_be_the_systems(system):
    period_fs = round(1e6 / system.clock_ghz)
    stated = (
        "module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = %d>} {"
    )
    same = MODULE.replace("module {", stated % period_fs, 1)
    assert len(events(pulse.load(same, entry="round", qudits=system.qudits[:2]))) == 3
    other = MODULE.replace("module {", stated % (2 * period_fs), 1)
    with pytest.raises(pulse.CompilationError, match="clock"):
        pulse.load(other, entry="round", qudits=system.qudits[:2])


T = "!pulse.drive_line, !pulse.tone"
TIMED = "attributes {pulse.schedule_state = #pulse.schedule_state<timed>}"


def _long(cycles):
    """The attributes of a function a call runs: its length and its schedule."""
    return f"attributes {{pulse.length_cycles = {cycles} : i64, " + TIMED[len("attributes {") :]


def _at(start, cycles, periods=None):
    """A call's timing."""
    period = "" if periods is None else f"{periods}, "
    return f"<{{duration_cycles = {cycles} : i64, {period}start_cycles = {start} : i64}}>"


CALLS = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func private @pair(%a: !pulse.drive_line, %at: !pulse.tone, %b: !pulse.drive_line, %bt:
        !pulse.tone) -> ({T}, {T}) {_long(16)} {{
    %a1, %at1 = pulse.call @x(%a, %at) {_at(0, 8)} : ({T}) -> ({T})
    %b1, %bt1 = pulse.call @x(%b, %bt) {_at(8, 8)} : ({T}) -> ({T})
    return %a1, %at1, %b1, %bt1 : {T}, {T}
  }}
  func.func @shot(%qa: !pulse.qudit, %qb: !pulse.qudit) {TIMED} {{
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %b, %bt = pulse.get_drive_line %qb : (!pulse.qudit) -> ({T})
    %a1, %at1, %b1, %bt1 = pulse.call @pair(%a, %at, %b, %bt) {_at(0, 16)}
        : ({T}, {T}) -> ({T}, {T})
    %b2, %bt2 = pulse.call @x(%b1, %bt1) {_at(40, 8)} : ({T}) -> ({T})
    %b3, %bt3, %a3, %at3 = pulse.call @pair(%b2, %bt2, %a1, %at1) {_at(100, 16)}
        : ({T}, {T}) -> ({T}, {T})
    return
  }}
}}
"""


def test_calls_with_a_stated_start_are_written_in_place(system):
    program = pulse.load(CALLS, entry="shot", qudits=system.qudits[:2])
    assert events(program) == [
        ("drive", 0, 0, 8),
        ("drive", 1, 8, 8),
        ("drive", 1, 40, 8),
        ("drive", 1, 100, 8),
        ("drive", 0, 108, 8),
    ]
    assert "call" not in program.mlir()


@pytest.mark.parametrize(
    "old, new, message",
    [
        (" " + _at(40, 8), "", "without stating when it starts"),
        (_at(0, 16), _at(0, 12), "is 16 cycles long"),
        (_at(40, 8), _at(12, 8), "overlaps or precedes"),
        (
            "pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %n",
            "made.by = 1 : i64} {\n    %n",
            "which is not scheduled",
        ),
    ],
)
def test_calls_that_cannot_be_written_in_place(system, old, new, message):
    text = CALLS.replace(old, new)
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_func_call_is_refused_with_the_call_to_write(system):
    text = CALLS.replace(f"pulse.call @x(%b1, %bt1) {_at(40, 8)}", "func.call @x(%b1, %bt1)")
    assert text != CALLS
    with pytest.raises(
        pulse.CompilationError, match="shot calls a function with func.call.*pulse.call"
    ):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_call_that_lasts_longer_than_its_function_holds_its_lines(system):
    # @x is 8 cycles; given 20 from 40, its line is free again at 60.
    longer = CALLS.replace(_at(40, 8), _at(40, 20))
    program = pulse.load(longer, entry="shot", qudits=system.qudits[:2])
    assert ("drive", 1, 40, 8) in events(program)
    late = longer.replace(_at(100, 16), _at(50, 16))
    with pytest.raises(pulse.CompilationError, match="overlaps or precedes"):
        pulse.load(late, entry="shot", qudits=system.qudits[:2])


@pytest.mark.parametrize(
    "timing",
    [
        "<{duration_cycles = 8 : i64, start_cycles = 9223372036854775807 : i64}>",
        "<{duration_cycles = 8 : i64, start_cycles = -5 : i64}>",
        "<{duration_cycles = 9223372036854775807 : i64, start_cycles = 40 : i64}>",
    ],
)
def test_a_calls_times_are_held_to_the_range_of_every_operation(system, timing):
    text = CALLS.replace(_at(40, 8), timing)
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match=r"shot calls x with .* outside 0 to 2\^50"):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_the_range_holds_for_nested_calls_periods_and_what_the_calls_write(system):
    far = 2**60
    nested = CALLS.replace(_at(8, 8), _at(far, 8))
    assert nested != CALLS
    with pytest.raises(pulse.CompilationError, match=r"pair calls x with .* outside 0 to 2\^50"):
        pulse.load(nested, entry="shot", qudits=system.qudits[:2])
    period = LOOP.replace("period_cycles = array<i64: 20>", f"period_cycles = array<i64: {far}>", 1)
    assert period != LOOP
    with pytest.raises(pulse.CompilationError, match=r"shot calls x with .* outside 0 to 2\^50"):
        pulse.load(period, entry="shot", qudits=system.qudits[:1])
    # Each start is in range; the drive the two calls write starts at their sum.
    half = 2**49
    summed = (
        CALLS.replace(_at(8, 8), _at(half, 8))
        .replace(_long(16), _long(half + 8))
        .replace(_at(0, 16), _at(0, half + 8))
        .replace(_at(100, 16), _at(2**50 - 8, half + 8))
        # Nothing else is wrong with it: the call between the two is after the first.
        .replace(_at(40, 8), _at(half + 16, 8))
    )
    with pytest.raises(
        pulse.CompilationError, match=r"outside 0 to 2\^50 cycles once its calls are written out"
    ):
        pulse.load(summed, entry="shot", qudits=system.qudits[:2])


def test_a_called_function_is_not_in_the_loaded_module(system):
    # Another compiler's attribute that names the function does not keep it.
    text = CALLS.replace(_long(8), _long(8)[:-1] + ", made.by = @x}")
    assert text != CALLS
    program = pulse.load(text, entry="shot", qudits=system.qudits[:2])
    assert program.mlir().count("func.func") == 1
    assert len(events(program)) == 5


def test_a_called_functions_recorded_length_is_where_its_operations_end(system):
    text = CALLS.replace(_long(8), _long(4))
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match="its operations end at 8"):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_function_that_calls_itself_is_refused(system):
    text = f"""
module {{
  func.func private @again(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(0)} {{
    %d1, %t1 = pulse.call @again(%d, %t) {_at(0, 0)} : ({T}) -> ({T})
    return %d1, %t1 : {T}
  }}
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @again(%d, %t) {_at(0, 0)} : ({T}) -> ({T})
    return
  }}
}}
"""
    with pytest.raises(pulse.CompilationError, match="calls itself"):
        pulse.load(text, entry="shot", qudits=system.qudits[:1])


# ── Classical functions ──────────────────────────────────────────────────────


def _links():
    from cudaq_pulse.classical import Link

    return pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=100)})


@pulse.classical(max_latency_ns=400, link="host")
def decide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits([bits[0]])


@pulse.kernel
def rounds(a, c):
    ro = get_readout_line(a)  # noqa: F821
    da = get_drive_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    for _ in range(2):
        drive(da, square(16, 0.5))  # noqa: F821
        sync(da, ro)  # noqa: F821
        m = readout(ro, square(100, 0.05))  # noqa: F821
        fix = decide.submit([m]).result()
        sync(dc, ro)  # noqa: F821
        if fix[0]:
            drive(dc, square(16, 0.5))  # noqa: F821


def _as_written(system, machine, kernel=None):
    """The compiled kernel and its module as another compiler would write it:
    a function of its qudits, with nothing of the target in it."""
    import re

    native = pulse.compile(kernel or rounds, quantum_system=system, control_system=machine).lower(
        system.qudits[0], system.qudits[2]
    )
    text = native.mlir()
    names = []
    for qudit in (0, 2):
        found = re.search(
            r"\n\s*(%\w+) = pulse\.qudit_alloc \{qudit = "
            + str(qudit)
            + r" : i64\} : !pulse\.qudit",
            text,
        )
        assert found is not None
        names.append(found.group(1))
        text = text.replace(found.group(0), "")
    text = text.replace(
        "func.func @main()",
        "func.func @main(" + ", ".join(f"{n}: !pulse.qudit" for n in names) + ")",
    )
    text = re.sub(r" \{frequency_hz = [^}]*\}", "", text)
    # Of its link, the module states the name: the link's cost is the target's.
    text = re.sub(
        r"classical\.links = \{.*?\}\}, ",
        'classical.links = {"0" = {name = "host"}}, ',
        text,
        count=1,
    )
    assert text.count("classical.links") == 1 and "classical.callable" in text
    return native, text


@pulse.kernel
def unread(a, c):
    ro = get_readout_line(a)  # noqa: F821
    da = get_drive_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    for _ in range(5000):
        drive(da, square(16, 0.5))  # noqa: F821
        sync(da, ro)  # noqa: F821
        m = readout(ro, square(100, 0.05))  # noqa: F821
        decide.submit([m])
        sync(dc, ro)  # noqa: F821


def test_a_report_refuses_the_readout_its_events_refuse_when_no_answer_is_used(system):
    """A report lists the program's events, up to the larger of 10,000 and
    1,000 per call, and that list refuses a readout in a mode the exports do
    not take yet. A report in which no answer is used makes no such list,
    and refuses the readout all the same."""
    machine = _links()
    native, text = _as_written(system, machine, unread)
    assert len(native.schedule) == 15_000
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    report = loaded.classical_report()
    assert report == native.classical_report() and len(report) == 5000
    assert {call.first_use_cycles for call in report} == {None}
    traced = pulse.load(
        text.replace(", iq <{", ", trace <{"), entry="main", qudits=qudits, control_system=machine
    )
    message = "readout mode trace is not supported by the exports yet; only iq is"
    with pytest.raises(NotImplementedError, match=message):
        traced.classical_report()


def _calls(program):
    exported = program.export_to_controller()
    return exported["calls"], exported["links"], exported["manifest"]


def test_classical_functions_are_read_from_the_module(system):
    machine = _links()
    native, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert events(loaded) == events(native)
    assert _calls(loaded) == _calls(native)
    assert len(_calls(loaded)[0]) == 2
    assert [fn.implementation for fn in loaded.classical] == [None]


@pulse.classical(max_latency_ns=600, latency_mode="timeout", timeout_ns=250, link="host")
def decide_timed(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits([bits[0]])


@pulse.kernel
def timed_rounds(a, c):
    ro = get_readout_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    m = readout(ro, square(100, 0.05))  # noqa: F821
    fix = decide_timed.submit([m]).result()
    sync(dc, ro)  # noqa: F821
    if fix[0]:
        drive(dc, square(16, 0.5))  # noqa: F821


def test_a_function_with_a_timeout_is_read_with_it(system):
    machine = _links()
    native, text = _as_written(system, machine, timed_rounds)
    assert "timeout_ns" in text
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert loaded.classical[0].timeout_ns == 250.0
    assert loaded.classical[0].latency_mode == "timeout"
    assert _calls(loaded) == _calls(native)


def test_a_given_classical_function_is_checked_and_its_latency_used(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=machine, classical=[decide]
    )
    assert loaded.classical[0].implementation is decide.implementation

    @pulse.classical(name="decide", max_latency_ns=900, link="host")
    def slow(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[slow])

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def wide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
        return pulse.Bits([bits[0], bits[0]])

    with pytest.raises(pulse.CompilationError, match=r"gives \(bits:1\).*gives \(bits:2\)"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[wide])

    @pulse.classical(max_latency_ns=400, link="host")
    def other(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named other"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[other])
    with pytest.raises(pulse.InvalidTypeError, match="takes classical functions"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=["decide"])


def test_the_links_are_the_control_systems(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=900)})
    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


LOOP = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func @shot(%qa: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %l1, %lt1 = pulse.call @x(%l, %lt) {_at(4, 8, "period_cycles = array<i64: 20>")}
          : ({T}) -> ({T})
      %l2, %lt2 = pulse.call @x(%l1, %lt1) {_at(12, 8, "period_cycles = array<i64: 20>")}
          : ({T}) -> ({T})
      scf.yield %l2, %lt2 : {T}
    }}
    return
  }}
}}
"""


def test_calls_in_a_loop_keep_the_loop_and_state_its_period(system):
    program = pulse.load(LOOP, entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [
        (4, 8),
        (12, 8),
        (24, 8),
        (32, 8),
        (44, 8),
        (52, 8),
    ]
    text = program.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.drive ") == 2
    assert "call" not in text


# ── What a review found ──────────────────────────────────────────────────────


def test_a_failed_load_carries_a_documented_code(system):
    with pytest.raises(pulse.CompilationError) as wrong:
        pulse.load(MODULE, entry="missing", qudits=system.qudits[:2])
    assert wrong.value.code == "wrong_argument"
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64").replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
    with pytest.raises(pulse.CompilationError) as overlap:
        pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert overlap.value.code == "other"


def test_a_function_with_a_qudit_of_its_own_is_refused(system):
    text = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    %own = pulse.qudit_alloc {qudit = 5 : i64} : !pulse.qudit\n"
        "    %n = arith.constant 16 : i64",
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="allocates a qudit"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


@pytest.mark.parametrize("limit", [{"max_instructions": 1000}, {"max_program_duration": 1e6}])
def test_program_limits_are_not_checked_yet_and_say_so(system, limit):
    machine = pulse.Controller(**limit)
    with pytest.raises(pulse.UnsupportedError, match="max_instructions or max_program_duration"):
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2], control_system=machine)


def test_a_slower_link_is_refused(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    # A link that takes longer to send on, with the same feedback: the answer
    # is still due in time as scheduled, and only the submission is too short.
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=5000, feedback_ns=100)})
    with pytest.raises(pulse.CompilationError, match=r"is given \d+ cycle\(s\).*it takes \d+"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


def test_every_submission_is_seen_by_the_check(system):
    from cudaq_pulse._load import _check_submissions
    from cudaq_pulse._native import _cudaq_pulse_native as native

    machine = _links()
    _, text = _as_written(system, machine)
    # The loader sees each submission, with the cycles its schedule gives it
    # and the cycles the classical check wrote that its link takes.
    rows = native.PulseModuleBuilder().load_submissions(text.encode())
    assert len(rows) == text.count("classical.submit ") == 1
    assert rows[0][0].startswith("decide") and rows[0][1] == rows[0][2] > 0
    _check_submissions(rows)
    # One the check wrote nothing on was not checked: it is refused.
    bare = re.sub(r"(, )?classical\.submit_cycles = \d+ : i64", "", text)
    assert bare != text
    rows = native.PulseModuleBuilder().load_submissions(bare.encode())
    assert [row[2] for row in rows] == [-1]
    with pytest.raises(pulse.CompilationError, match="no link time to check it against"):
        _check_submissions(rows)
    _check_submissions([("decide", 100, 100, [], False)])
    with pytest.raises(pulse.CompilationError, match="is given 99 cycle"):
        _check_submissions([("decide", 100, 100, [], False), ("decide", 99, 100, [], False)])


# What a module may not state for the program: answers, and a length that is
# not an answer's.

_LENGTH = (
    '  classical.callable @length__1 {args = ["bits:1"], device = 0 : i64, dispatch_key = '
    "3867861094 : i64, latency_max_ns = 5.0e+01 : f64, latency_min_ns = 0.0 : f64, "
    'latency_nominal_ns = 5.0e+01 : f64, mode = "bounded", results = ["f64"], submit = "eager"}'
)
_PLAYED = "!pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone"


def _after_an_answer(body, marks=""):
    """A function of one qudit that reads it out at cycles 0 to 100, sends
    the result to a classical function at 100 and takes its answer, an f64
    %ans, at 500 (the wait for it carrying *marks*); *body* follows, on the
    drive line %d and its tone %t, with a 16-cycle waveform %w."""
    return f"""
module attributes {{classical.links = {{"0" = {{name = "host"}}}}}} {{
{_LENGTH}
  func.func @main(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %long = arith.constant 100 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rw = pulse.square %long, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r, %rw, %rt, iq {_at(0, 100)}
        : !pulse.readout_line, !pulse.waveform, !pulse.tone
          -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l0 = classical.get_link {{device = 0 : i64}} : !classical.link
    %l1, %f = classical.submit %l0, @length__1(%m) {_at(100, 100)}
        : (!pulse.measurement) -> (!classical.link, !classical.future)
    %ans = classical.wait %f, @length__1 {_at(500, 0)}{marks} : f64
{body}
    return
  }}
}}
"""


@pytest.mark.level0
def test_answers_a_module_states_for_a_call_are_not_the_programs(system):
    body = f"""
    %wa = pulse.square %n, %ans, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %wa, %t {_at(500, 16)} : {_PLAYED}"""
    machine, qudits = _links(), system.qudits[:1]
    plain = pulse.load(_after_an_answer(body), entry="main", qudits=qudits, control_system=machine)
    drive = plain.export_to_controller()["events"][-1]
    # The amplitude is the answer's: the controller computes it as it runs.
    assert drive["kind"] == "drive" and drive["amplitude"] is None
    stated = _after_an_answer(body, " {classical.answers = array<f64: 9.0e-01>}")
    program = pulse.load(stated, entry="main", qudits=qudits, control_system=machine)
    assert "classical.answers" not in program.mlir()
    for version in (1, 2):
        assert program.export_to_controller(version=version) == plain.export_to_controller(
            version=version
        )


def _run_time_wait(length="%cycles", reserved=16, marks=None, then=None):
    """A wait of *length* cycles on the drive line, which states *reserved*
    cycles and carries *marks*, and a drive after it."""
    if marks is None:
        marks = (
            "classical.variable, pulse.max_cycles = 4000 : i64, pulse.min_cycles = 16 : i64, "
            "pulse.variable_wait"
        )
    return _after_an_answer(f"""
    %half = arith.constant 0.5 : f64
    %sum = arith.addf %ans, %half : f64
    %cycles = arith.fptosi %sum : f64 to i64
    %len = pulse.duration_from_int {length} : (i64) -> !pulse.duration
    %d0 = pulse.wait %d, %len {_at(500, reserved)} {{{marks}}}
        : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d1, %t1 = pulse.drive %d0, %w, %t {_at(500 + reserved if then is None else then, 16)}
        : {_PLAYED}""")


@pulse.classical(max_latency_ns=50, link="host")
def _cycles(bits: pulse.Bits) -> float:
    return 100.0 + 50.0 * bits[0]


@pulse.kernel
def _waits_for_an_answer(a, c):
    ro = get_readout_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    m = readout(ro, square(100, 0.05))  # noqa: F821
    wait(dc, _cycles.submit([m]).result(), bounds=(16, None))  # noqa: F821
    drive(dc, square(16, 0.5))  # noqa: F821


def test_a_compiled_wait_whose_length_is_an_answer_loads_with_its_mark(system):
    machine = _links()
    lowered, text = _as_written(system, machine, _waits_for_an_answer)
    assert "classical.variable" in text and "pulse.max_cycles" not in text
    program = pulse.load(
        text, entry="main", qudits=[system.qudits[0], system.qudits[2]], control_system=machine
    )
    assert "classical.variable" in program.mlir()
    for version in (1, 2):
        assert (
            program.export_to_controller(version=version)["events"]
            == lowered.export_to_controller(version=version)["events"]
        )


def test_a_wait_whose_length_is_an_answer_reserves_its_lowest_length(system):
    machine, qudits = _links(), system.qudits[:1]

    def load(text):
        return pulse.load(text, entry="main", qudits=qudits, control_system=machine)

    program = load(_run_time_wait())
    assert "classical.variable" in program.mlir()
    assert events(program)[-2:] == [("wait", 0, 500, 16), ("drive", 0, 516, 16)]
    # Less than its lowest length would start the drive inside the wait;
    # more is not how the scheduler places it.
    for reserved in (1, 400):
        with pytest.raises(pulse.CompilationError, match="reserves its lowest length") as refused:
            load(_run_time_wait(reserved=reserved))
        assert f"duration_cycles = {reserved} and pulse.min_cycles = 16" in str(refused.value)
        assert refused.value.code == "wrong_argument"
    # The mark alone states no lowest length to reserve.
    with pytest.raises(pulse.CompilationError, match="without a lowest length"):
        load(_run_time_wait(marks="classical.variable"))
    # Nor does a lowest length that is not a 64-bit integer.
    wide = "classical.variable, pulse.min_cycles = 16 : i32, pulse.variable_wait"
    with pytest.raises(pulse.CompilationError, match="without a lowest length"):
        load(_run_time_wait(marks=wide))


@pytest.mark.level0
def test_the_mark_of_a_run_time_wait_is_kept_on_such_a_wait_only(system):
    machine, qudits = _links(), system.qudits[:1]

    def load(text):
        return pulse.load(text, entry="main", qudits=qudits, control_system=machine)

    def drives(first, mark):
        return _after_an_answer(f"""
    %d1, %t1 = pulse.drive %d, %w, %t {_at(500, first)}{mark} : {_PLAYED}
    %d2, %t2 = pulse.drive %d1, %w, %t1 {_at(500 + first, 16)} : {_PLAYED}""")

    assert "classical.variable" in load(_run_time_wait()).mlir()
    # On a drive the mark is another compiler's attribute: the drive is held
    # to the 16 cycles its waveform takes.
    program = load(drives(16, " {classical.variable}"))
    assert "classical.variable" not in program.mlir()
    assert events(program)[-2:] == [("drive", 0, 500, 16), ("drive", 0, 516, 16)]
    for mark in ("", " {classical.variable}"):
        with pytest.raises(pulse.CompilationError, match="duration_cycles is 1, but the operation"):
            load(drives(1, mark))
    # So it is on a wait whose length no answer gives: a wait of 100 cycles.
    program = load(_run_time_wait(length="%long", reserved=100))
    assert "classical.variable" not in program.mlir()
    assert events(program)[-2:] == [("wait", 0, 500, 100), ("drive", 0, 600, 16)]
    with pytest.raises(pulse.CompilationError, match="duration_cycles is 16, but the operation"):
        load(_run_time_wait(length="%long"))


def test_a_given_bool_function_keeps_its_own_types(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def yes(bits: pulse.Bits) -> bool:
        return bool(bits[0])

    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[yes])
    assert [type(r).__name__ for r in loaded.classical[0].results] == [
        type(r).__name__ for r in yes.specialize(bits=1).results
    ]
    assert loaded.classical[0].base_name == "decide"


def test_a_function_is_given_for_its_versions_only():
    from cudaq_pulse._load import _given_functions

    row = ("decide__fast", 1, 0, ["bits:1"], ["bits:1"], (0.0, 400.0, 400.0), "bounded", "eager")
    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named decide"):
        _given_functions([decide], [row + (None, None, None)])
    version = ("decide__1",) + row[1:] + (None, None, None)
    assert set(_given_functions([decide], [version])) == {"decide__1"}
    lazy = version[:7] + ("lazy",) + version[8:]
    with pytest.raises(pulse.CompilationError, match="submit policy"):
        _given_functions([decide], [lazy])


# ── Links and dispatch keys are the control system's ─────────────────────────


def _host(**link):
    from cudaq_pulse.classical import Link

    return Link(**(link or {"fixed_ns": 50, "feedback_ns": 100}))


def _written(system):
    machine = _links()
    native, text = _as_written(system, machine)
    return native, text, [system.qudits[0], system.qudits[2]]


def test_a_link_is_found_by_its_name_not_its_number(system):
    native, text, qudits = _written(system)
    from cudaq_pulse.classical import Link

    moved = pulse.ControlSystem(links={"fast": Link(), "host": _host()})
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=moved)
    exported = loaded.export_to_controller()
    assert [link["name"] for link in exported["links"].values()] == ["host"]
    assert list(exported["links"]) == ["1"]
    assert [c["due_cycles"] for c in exported["calls"]] == [
        c["due_cycles"] for c in native.export_to_controller()["calls"]
    ]


def test_a_link_the_control_system_does_not_have_is_refused(system):
    _, text, qudits = _written(system)
    with pytest.raises(pulse.CompilationError, match="no link of that name") as none:
        pulse.load(text, entry="main", qudits=qudits)
    assert none.value.code == "control_system"
    other = pulse.ControlSystem(links={"nvqlink": _host()})
    with pytest.raises(pulse.CompilationError, match=r"link 'host'.*its links: 'nvqlink'"):
        pulse.load(text, entry="main", qudits=qudits, control_system=other)


@pytest.mark.parametrize("key", ["host", 0])
def test_links_gives_the_control_systems_name(system, key):
    native, text, qudits = _written(system)
    other = pulse.ControlSystem(links={"nvqlink": _host()})
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=other, links={key: "nvqlink"}
    )
    exported = loaded.export_to_controller()
    assert [link["name"] for link in exported["links"].values()] == ["nvqlink"]
    assert [
        (c["submit_cycles"], c["due_cycles"], c["taken_cycles"]) for c in exported["calls"]
    ] == [
        (c["submit_cycles"], c["due_cycles"], c["taken_cycles"])
        for c in native.export_to_controller()["calls"]
    ]


def test_links_is_checked(system):
    _, text, qudits = _written(system)
    machine = _links()
    with pytest.raises(pulse.InvalidValueError, match="'other', which is not a link"):
        pulse.load(
            text, entry="main", qudits=qudits, control_system=machine, links={"other": "host"}
        )
    with pytest.raises(pulse.InvalidTypeError, match="maps a link's name or id"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, links={"host": 0})
    unnamed = text.replace('classical.links = {"0" = {name = "host"}}, ', "")
    assert unnamed != text
    with pytest.raises(pulse.CompilationError, match="does not name its link 0"):
        pulse.load(unnamed, entry="main", qudits=qudits, control_system=machine)
    assert pulse.load(
        unnamed, entry="main", qudits=qudits, control_system=machine, links={0: "host"}
    ).classical


def test_a_controller_that_fixes_a_key_gives_it(system):
    _, text, qudits = _written(system)
    keys = pulse.Processor(links=["host"], dispatch_keys={"decide": 0x1234})
    fixed = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[keys])], links={"host": _host()}
    )
    function = pulse.load(
        text, entry="main", qudits=qudits, control_system=fixed
    ).export_to_controller()["manifest"]["functions"][0]
    assert function["dispatch_key"] == 0x1234
    assert "processors[0]" in function["dispatch_key_fixed_by"]
    unlisted = pulse.Processor(links=["host"], dispatch_keys={"other": 5})
    system_of = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[unlisted])], links={"host": _host()}
    )
    with pytest.raises(pulse.CompilationError, match="dispatch_keys have no 'decide'"):
        pulse.load(text, entry="main", qudits=qudits, control_system=system_of)


def test_a_reserved_dispatch_key_is_refused(system):
    _, text, qudits = _written(system)
    found = re.search(r"dispatch_key = (\d+) : i64", text)
    reserved = text.replace(
        found.group(0), f"dispatch_key = {0x7ED8BE82} : i64, dispatch_key_override"
    )
    with pytest.raises(pulse.CompilationError, match="0x7ed8be82.*rename it"):
        pulse.load(reserved, entry="main", qudits=qudits, control_system=_links())


# ── CUDA-Q QEC's decoder messages ────────────────────────────────────────────


def _decoder_rounds():
    from tests.classical.fake_cudaq_qec import qec

    enqueue = pulse.classical(qec.enqueue_syndromes, link="host")
    corrections = pulse.classical(qec.get_corrections, link="host", max_latency_ns=1000)

    @pulse.kernel
    def decoded(a, c):
        ra = get_readout_line(a)  # noqa: F821
        rc = get_readout_line(c)  # noqa: F821
        dc = get_drive_line(c)  # noqa: F821
        for _ in range(4):
            s0 = readout(ra, square(100, 0.05))  # noqa: F821
            s1 = readout(rc, square(100, 0.05))  # noqa: F821
            enqueue.submit(3, [s0, s1], 7)
            sync(dc, ra, rc)  # noqa: F821
        fix = corrections.submit(3, 1, True).result()
        sync(dc, ra, rc)  # noqa: F821
        if fix[0]:
            drive(dc, square(16, 0.5))  # noqa: F821

    return decoded, enqueue, corrections


def test_a_modules_decoder_messages_are_read_as_cudaq_qecs(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    native, text = _as_written(system, machine, kernel)
    assert "@enqueue_syndromes__3_2" in text and "@get_corrections__3_1" in text
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert events(loaded) == events(native)
    assert _calls(loaded) == _calls(native)
    # Rounds that only submit: four submissions with no answer, then one
    # answer, and the loop stays rolled.
    calls = _calls(loaded)[0]
    assert [c["function"] for c in calls] == ["enqueue_syndromes__3_2"] * 4 + [
        "get_corrections__3_1"
    ]
    assert [c["taken_cycles"] for c in calls[:4]] == [None] * 4
    assert loaded.mlir().count("scf.for") == 1
    # They are the decoding server's messages: its keys, routed by the
    # decoder id, and answered by the configured decoder.
    assert [(fn.name, fn._dispatch_key, fn._key_route) for fn in loaded.classical] == [
        (fn.name, fn._dispatch_key, fn._key_route) for fn in native.classical
    ]
    assert all(fn._key_route == 3 and fn.implementation is not None for fn in loaded.classical)


def test_cudaq_qec_functions_given_for_a_modules_messages_give_their_latency(system):
    from tests.classical.fake_cudaq_qec import qec

    machine = _links()
    kernel, enqueue, _ = _decoder_rounds()
    native, text = _as_written(system, machine, kernel)
    qudits = [system.qudits[0], system.qudits[2]]
    same = pulse.classical(qec.get_corrections, link="host", max_latency_ns=1000)
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=machine, classical=[enqueue, same]
    )
    assert _calls(loaded) == _calls(native)
    slow = pulse.classical(qec.get_corrections, link="host", max_latency_ns=5000)
    with pytest.raises(pulse.CompilationError, match="get_corrections__3_1.*before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[slow])
    other = pulse.classical(qec.reset_decoder, link="host")
    with pytest.raises(pulse.InvalidValueError, match="no classical function named reset_decoder"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[other])


def test_two_decoders_messages_share_the_servers_key(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    # A second decoder's version of the same message, declared and unused.
    found = re.search(r"\n\s*classical\.callable @get_corrections__3_1 [^\n]*", text)
    text = text.replace(found.group(0), found.group(0) + found.group(0).replace("__3_1", "__4_1"))
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert {fn.name: fn._key_route for fn in loaded.classical} == {
        "enqueue_syndromes__3_2": 3,
        "get_corrections__3_1": 3,
        "get_corrections__4_1": 4,
    }


def _decoder_text(system):
    machine = _links()
    kernel, enqueue, corrections = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    return machine, text, [system.qudits[0], system.qudits[2]], enqueue, corrections


def _one_message(system, name):
    """A module whose entry submits one decoder message, with every
    argument a constant of its own, and waits for it where it answers."""
    from cudaq_pulse.classical import _cudaqx

    from tests.classical.fake_cudaq_qec import qec

    machine = _links()
    declared = pulse.classical(
        getattr(qec, name), link="host", **({"max_latency_ns": 1000} if name[0] == "g" else {})
    )
    arguments = {
        "enqueue_syndromes": "3, [s0, s1], 7",
        "get_corrections": "3, 2, True",
        "reset_decoder": "3",
    }[name]
    source = f"""
@pulse.kernel
def one(a, c):
    ra = get_readout_line(a)
    rc = get_readout_line(c)
    s0 = readout(ra, square(100, 0.05))
    s1 = readout(rc, square(100, 0.05))
    sent = declared.submit({arguments})
    sync(ra, rc)
"""
    scope = {"pulse": pulse, "declared": declared}
    exec(source, scope)  # noqa: S102
    _, text = _as_written(system, machine, scope["one"])
    words = _cudaqx._WORDS[name]
    version = re.search(rf"classical\.submit %\w+, @({name}__\w+)\(([^)]*)\)", text)
    return machine, text, version.group(1), version.group(2).split(", ")[: len(words)], words


def _with_constant(text, old, value):
    """*text* with a new constant where one submission's operand *old* was."""
    other = "%another"
    line = re.search(r"(\n(\s*))(%\w+ = classical\.get_link)", text)
    changed = text.replace(
        line.group(0), f"{line.group(1)}{other} = arith.constant {value} : i64{line.group(0)}", 1
    )
    submit = re.search(r"classical\.submit %\w+, @\w+\(([^)]*)\)", changed)
    operands = submit.group(1).split(", ")
    assert operands.count(old) == 1
    return changed.replace(
        submit.group(0),
        submit.group(0).replace(
            submit.group(1), ", ".join(other if o == old else o for o in operands)
        ),
        1,
    )


RELEASED = ("enqueue_syndromes", "get_corrections", "reset_decoder")


def test_every_decoder_message_states_what_its_submissions_send():
    from cudaq_pulse.classical import _cudaqx

    assert set(_cudaqx._WORDS) == set(_cudaqx._MESSAGES)
    assert {m.name for m in _cudaqx._MESSAGES.values() if m.released} == set(RELEASED)
    for words in _cudaqx._WORDS.values():
        assert words[0] == ("decoder_id", "id")
        assert all(rule in ("id", "length", None, 0) for _, rule in words)


@pytest.mark.parametrize("name", RELEASED)
def test_a_decoder_message_loads_as_compile_sends_it(system, name):
    machine, text, version, operands, words = _one_message(system, name)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert [fn.name for fn in loaded.classical] == [version]
    # compile sends a constant wherever the message's description states one.
    constants = dict(re.findall(r"(%\w+) = arith\.constant (-?\d+) : i64", text))
    numbers = dict(zip(("id", "length"), map(int, version.split("__")[1].split("_")), strict=False))
    for operand, (_, rule) in zip(operands, words, strict=True):
        if rule is not None:
            assert int(constants[operand]) == (numbers[rule] if isinstance(rule, str) else rule)


def _every_word():
    from cudaq_pulse.classical import _cudaqx

    return [(name, at) for name in RELEASED for at in range(len(_cudaqx._WORDS[name]))]


@pytest.mark.parametrize("name, at", _every_word())
def test_every_word_of_a_decoder_message_is_held_to_its_description(system, name, at):

    machine, text, version, operands, words = _one_message(system, name)
    qudits = [system.qudits[0], system.qudits[2]]
    what, rule = words[at]
    changed = _with_constant(text, operands[at], 5)
    if rule is None:
        # The caller's value: any constant loads.
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)
        return
    said = "its name states" if isinstance(rule, str) else f"{name} sends 0"
    with pytest.raises(
        pulse.CompilationError,
        match=f"a submission to {version} sends the {what} 5, and {said}",
    ):
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)


def test_a_word_of_a_decoder_message_is_a_constant_64_bit_integer(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    sent = re.search(r"@get_corrections__3_1\((%\w+), [^)]*\) [^\n]*", text)
    line = re.search(r"(\n(\s*))(%\w+ = classical\.get_link)", text)
    # The decoder id as a sum the module computes: not a constant.
    computed = text.replace(
        line.group(0),
        f"{line.group(1)}%three = arith.constant 3 : i64"
        f"{line.group(1)}%sum = arith.addi %three, %three : i64{line.group(0)}",
        1,
    ).replace(sent.group(0), sent.group(0).replace(sent.group(1), "%sum", 1), 1)
    with pytest.raises(
        pulse.CompilationError,
        match="get_corrections__3_1 sends the decoder_id as a value that is not a constant",
    ):
        pulse.load(computed, entry="main", qudits=qudits, control_system=machine)
    # The decoder id as a 32-bit constant: the verifier takes any integer
    # width for an i64 argument, and compile writes 64 bits.
    assert ": (i64, i64, i64) ->" in sent.group(0)
    narrow = text.replace(
        line.group(0), f"{line.group(1)}%three = arith.constant 3 : i32{line.group(0)}", 1
    ).replace(
        sent.group(0),
        sent.group(0)
        .replace(sent.group(1), "%three", 1)
        .replace(": (i64, i64, i64) ->", ": (i32, i64, i64) ->"),
        1,
    )
    with pytest.raises(
        pulse.CompilationError,
        match="get_corrections__3_1 does not send the decoder_id as a 64-bit integer",
    ):
        pulse.load(narrow, entry="main", qudits=qudits, control_system=machine)


def test_an_ordinary_function_is_not_given_for_a_decoder_message(system):
    machine, text, qudits, _, _ = _decoder_text(system)

    def corrections(a: int, b: int, c: int) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([False])

    # A function cannot be declared under a message version's name.
    with pytest.raises(pulse.InvalidValueError, match="can't be named 'get_corrections__3_1'"):
        pulse.classical(corrections, name="get_corrections__3_1", max_latency_ns=10, link="host")
    # One that has the name all the same is not taken for the message.
    ordinary = pulse.classical(corrections, max_latency_ns=10, link="host")
    ordinary.name = "get_corrections__3_1"
    with pytest.raises(
        pulse.InvalidValueError,
        match="get_corrections__3_1 is CUDA-Q QEC's get_corrections; give pulse.classical",
    ):
        pulse.load(
            text,
            entry="main",
            qudits=qudits,
            control_system=machine,
            classical=[ordinary],
        )


def test_a_decoder_messages_name_is_never_an_ordinary_function(system):
    machine, text, qudits, _, corrections = _decoder_text(system)
    # With its own key: neither an ordinary function nor the message.
    own = text.replace(f"dispatch_key = {0x882D5BA1} : i64", "dispatch_key = 77 : i64")
    assert own != text
    for given in ((), (corrections,)):
        with pytest.raises(
            pulse.CompilationError,
            match="get_corrections__3_1 is named as CUDA-Q QEC's get_corrections and has the "
            "dispatch key 0x4d; the message's is 0x882d5ba1",
        ):
            pulse.load(own, entry="main", qudits=qudits, control_system=machine, classical=given)
    # A message that is in no release, whatever its key.
    for key in (0x7ED8BE82, 77):
        mapped = text.replace("enqueue_syndromes__3_2", "enqueue_syndromes_mapped__3_2").replace(
            f"dispatch_key = {0x7ED8BE82} : i64", f"dispatch_key = {key} : i64"
        )
        with pytest.raises(pulse.CompilationError, match="is in no CUDA-Q QEC release"):
            pulse.load(mapped, entry="main", qudits=qudits, control_system=machine)


def test_names_that_are_not_a_versions_name_are_ordinary_functions(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    # None of these is a name compile writes for a message: each is an
    # ordinary function, refused here for the decoding server's key.
    for name in (
        "enqueue_syndromes__3_02",
        "enqueue_syndromes__3",
        "enqueue_syndromes__3_2_1",
        "Enqueue_syndromes__3_2",
        f"enqueue_syndromes__{2**63}_2",
        "enqueue_syndromes__" + "9" * 5000 + "_2",
    ):
        renamed = text.replace("enqueue_syndromes__3_2", name)
        with pytest.raises(pulse.CompilationError, match="rename it"):
            pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)
    # A digit that is not ASCII (here ARABIC-INDIC DIGIT TWO, as a quoted
    # symbol) is not the name compile writes.
    renamed = text.replace("@enqueue_syndromes__3_2", '@"enqueue_syndromes__3_\\D9\\A2"')
    assert renamed != text
    with pytest.raises(pulse.CompilationError, match="rename it"):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)


def test_a_decoder_messages_answer_is_taken_from_its_submissions_own_future(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    # The future goes through a branch before it is waited on.
    found = re.search(
        r"\n(\s*)(%\w+), (%\w+) = classical\.submit (%[\w#]+), @get_corrections__3_1\(([^)]*)\) "
        r"(<\{[^\n]*)\n",
        text,
    )
    pad, link, future, old_link, operands, rest = found.groups()
    arm = f"classical.submit {old_link}, @get_corrections__3_1({operands}) {rest}"
    kinds = "!classical.link, !classical.future"
    through = (
        f"\n{pad}%flag = arith.constant true"
        f"\n{pad}{link}, {future} = scf.if %flag -> ({kinds}) {{"
        f"\n{pad}  %l_a, %f_a = {arm}"
        f"\n{pad}  scf.yield %l_a, %f_a : {kinds}"
        f"\n{pad}}} else {{"
        f"\n{pad}  %l_b, %f_b = {arm}"
        f"\n{pad}  scf.yield %l_b, %f_b : {kinds}"
        f"\n{pad}}}\n"
    )
    changed = text.replace(found.group(0), through, 1)
    refusal = "the future of a submission to get_corrections__3_1 is yielded or passed on"
    with pytest.raises(pulse.CompilationError, match=refusal):
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)
    # The same when the wait names an ordinary function of the module's.
    callable_ = re.search(r"\n\s*classical\.callable @get_corrections__3_1 [^\n]*", changed)
    mine = (
        callable_.group(0)
        .replace("@get_corrections__3_1", "@mine")
        .replace(
            f"dispatch_key = {0x882D5BA1} : i64, dispatch_key_override",
            "dispatch_key = 2122968502 : i64",  # the hash of its name
        )
    )
    assert "@mine" in mine and "dispatch_key = 2122968502" in mine
    renamed = changed.replace(callable_.group(0), callable_.group(0) + mine, 1)
    renamed = re.sub(
        r"classical\.wait (%\w+), @get_corrections__3_1", r"classical.wait \1, @mine", renamed
    )
    assert "classical.wait" in renamed and renamed != changed
    with pytest.raises(pulse.CompilationError, match=refusal):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)
    # The other direction: an ordinary function's submissions in the arms,
    # and their answer taken under the message's name, or under its own.
    # In a module that calls a decoder message no future is passed on.
    ordinary = "the future of a submission to mine is yielded or passed on; in a module that calls"
    for waited in (changed, renamed):
        reverse = waited.replace(callable_.group(0), callable_.group(0) + mine, 1)
        if "@mine" not in waited:
            waited = reverse
        arms = re.sub(
            r"(= classical\.submit %[\w#]+, )@get_corrections__3_1\(", r"\1@mine(", waited
        )
        assert arms.count("classical.submit") == waited.count("classical.submit")
        assert arms.count("@mine(") == 2 and "@enqueue_syndromes__3_2(" in arms
        with pytest.raises(pulse.CompilationError, match=ordinary):
            pulse.load(arms, entry="main", qudits=qudits, control_system=machine)


def test_a_loaded_programs_decoder_messages_pickle(system):
    import pickle

    machine, text, qudits, enqueue, corrections = _decoder_text(system)
    for given in ((), (enqueue, corrections)):
        loaded = pulse.load(
            text, entry="main", qudits=qudits, control_system=machine, classical=given
        )
        again = pickle.loads(pickle.dumps(loaded))
        assert _calls(again) == _calls(loaded)
        assert [(fn.name, fn.link, fn._dispatch_key, fn._key_route) for fn in again.classical] == [
            (fn.name, fn.link, fn._dispatch_key, fn._key_route) for fn in loaded.classical
        ]


def test_a_decoder_message_with_other_types_is_refused(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    assert 'args = ["i64", "i64", "u8"]' in text
    other = text.replace('args = ["i64", "i64", "u8"]', 'args = ["i64", "i64", "i64"]')
    qudits = [system.qudits[0], system.qudits[2]]
    with pytest.raises(
        pulse.CompilationError,
        match=r"^load: the module's get_corrections__3_1 is named as CUDA-Q QEC's "
        r"get_corrections, which takes \(i64, i64, u8\)",
    ):
        pulse.load(other, entry="main", qudits=qudits, control_system=machine)


def test_a_decoder_messages_key_under_another_name_is_refused(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    renamed = text.replace("get_corrections__3_1", "mine__3_1")
    qudits = [system.qudits[0], system.qudits[2]]
    with pytest.raises(pulse.CompilationError, match="mine__3_1 is dispatch key.*rename it"):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)


# ── A loaded module is untrusted input ───────────────────────────────────────

LEAF = f"""
  func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
"""


def _shot(cycles):
    return f"""
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @f0(%d, %t) {_at(0, cycles)} : ({T}) -> ({T})
    return
  }}
"""


def _chain(levels, calls):
    """@shot calls @f0; each @f{k} calls @f{k+1} *calls* times, one after
    the other; the last calls @leaf."""
    parts = ["module {", LEAF]
    for level in range(levels):
        callee = f"f{level + 1}" if level + 1 < levels else "leaf"
        length = 8 * calls ** (levels - level - 1)
        body, line, tone = [], "%d", "%t"
        for i in range(calls):
            body.append(
                f"    %d{i}, %t{i} = pulse.call @{callee}({line}, {tone}) "
                f"{_at(i * length, length)} : ({T}) -> ({T})"
            )
            line, tone = f"%d{i}", f"%t{i}"
        parts.append(
            f"  func.func private @f{level}(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) "
            f"{_long(calls * length)} {{\n"
            + "\n".join(body)
            + f"\n    return {line}, {tone} : {T}\n  }}"
        )
    parts += [_shot(8 * calls**levels), "}"]
    return "\n".join(parts)


def test_a_chain_of_calls_loads(system):
    program = pulse.load(_chain(3, 2), entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [
        (8 * i, 8) for i in range(8)
    ]


def test_calls_that_would_write_too_much_are_refused_before_any_is_written(system):
    # Each function calls the next twice: 2**21 leaves, written by none.
    with pytest.raises(pulse.CompilationError, match="write more than 1048576 operations"):
        pulse.load(_chain(21, 2), entry="shot", qudits=system.qudits[:1])


def test_calls_nested_too_deep_are_refused(system):
    with pytest.raises(pulse.CompilationError, match="nest deeper than 256"):
        pulse.load(_chain(300, 1), entry="shot", qudits=system.qudits[:1])
    assert pulse.load(_chain(200, 1), entry="shot", qudits=system.qudits[:1]).schedule


def _passes_on(name, callee):
    return (
        f"  func.func private @{name}(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) "
        f"{_long(8)} {{\n"
        f"    %d1, %t1 = pulse.call @{callee}(%d, %t) {_at(0, 8)} : ({T}) -> ({T})\n"
        f"    return %d1, %t1 : {T}\n  }}"
    )


def _two_chains(levels):
    """@shot calls @g0, the first of *levels* functions that each call the
    next, then @h0, the first of *levels* more, the last of which calls
    @g0."""
    parts = ["module {", LEAF]
    for k in range(levels):
        parts.append(_passes_on(f"g{k}", f"g{k + 1}" if k + 1 < levels else "leaf"))
    for k in range(levels):
        parts.append(_passes_on(f"h{k}", f"h{k + 1}" if k + 1 < levels else "g0"))
    parts.append(f"""  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @g0(%d, %t) {_at(0, 8)} : ({T}) -> ({T})
    %d2, %t2 = pulse.call @h0(%d1, %t1) {_at(8, 8)} : ({T}) -> ({T})
    return
  }}
}}""")
    return "\n".join(parts)


def test_calls_nested_too_deep_through_a_function_counted_before_are_refused(system):
    # @g0's 150 levels are counted for the first call; the second reaches
    # them below 150 more.
    with pytest.raises(pulse.CompilationError, match="the calls of shot nest deeper than 256"):
        pulse.load(_two_chains(150), entry="shot", qudits=system.qudits[:1])
    program = pulse.load(_two_chains(100), entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [(0, 8), (8, 8)]


def _nested_calls(functions, ifs):
    """@f0 .. each calls the next inside *ifs* nested branches on a
    measurement; the last calls @leaf."""
    parts = ["module {", LEAF]
    for k in range(functions):
        last = k + 1 == functions
        callee, args = ("leaf", "%d, %t") if last else (f"f{k + 1}", "%d, %t, %m")
        types = T if last else f"{T}, !pulse.measurement"
        body = [
            f"  func.func private @f{k}(%d: !pulse.drive_line, %t: !pulse.tone, "
            f"%m: !pulse.measurement) -> ({T}) {_long(8)} {{",
            "    %bit = pulse.is_set %m : !pulse.measurement -> i1",
        ]
        body += [f"    %r{i}:2 = scf.if %bit -> ({T}) {{" for i in range(ifs)]
        body.append(f"    %c0, %c1 = pulse.call @{callee}({args}) {_at(0, 8)} : ({types}) -> ({T})")
        body.append(f"    scf.yield %c0, %c1 : {T}")
        for i in reversed(range(ifs)):
            body.append(f"    }} else {{\n      scf.yield %d, %t : {T}\n    }}")
            if i:
                body.append(f"    scf.yield %r{i}#0, %r{i}#1 : {T}")
        body.append(f"    return %r0#0, %r0#1 : {T}\n  }}")
        parts.append("\n".join(body))
    parts.append(f"""  func.func @shot(%q: !pulse.qudit, %a: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %r, %rt = pulse.get_readout_line %a : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r, %w, %rt, iq
        <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.readout_line, !pulse.waveform, !pulse.tone
          -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %d1, %t1 = pulse.call @f0(%d, %t, %m) {_at(400, 8)} : ({T}, !pulse.measurement) -> ({T})
    return
  }}
}}""")
    return "\n".join(parts)


def test_calls_that_would_nest_regions_too_deep_are_refused_before_any_is_written(system):
    # Each function nests 100 branches, under the 254 a function may; written
    # in place of one another, three of them nest 300.
    with pytest.raises(
        pulse.CompilationError, match="the calls of shot nest regions deeper than 256"
    ):
        pulse.load(_nested_calls(3, 100), entry="shot", qudits=system.qudits[:2])
    program = pulse.load(_nested_calls(2, 100), entry="shot", qudits=system.qudits[:2])
    assert ("drive", 0, 400, 8) in events(program)


def _looped(period):
    """An entry with one drive that starts at 4 in a loop of three trips,
    *period* cycles apart. No call."""
    return f"""module {{
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %a, %at = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
      %l1, %lt1 = pulse.drive %l, %w, %lt
          {_at(4, 8, f"period_cycles = array<i64: {period}>")}
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
      scf.yield %l1, %lt1 : {T}
    }}
    return
  }}
}}"""


def test_an_operations_own_period_is_held_to_the_range(system):
    program = pulse.load(_looped(20), entry="shot", qudits=system.qudits[:1])
    assert max(e.start_cycles for e in program.schedule) == 44
    for period in (2**60, -20):
        with pytest.raises(
            pulse.CompilationError, match=r"start, duration or period is outside 0 to 2\^50"
        ):
            pulse.load(_looped(period), entry="shot", qudits=system.qudits[:1])


def test_the_last_trip_of_a_loop_starts_within_the_range(system):
    # Three trips of a drive that starts at 4: a period of 2**49 is in the
    # range, and the last trip starts at 2**50 + 4.
    with pytest.raises(
        pulse.CompilationError, match=r"shot has an operation in a loop whose last trip starts"
    ):
        pulse.load(_looped(2**49), entry="shot", qudits=system.qudits[:1])
    program = pulse.load(_looped(2**48), entry="shot", qudits=system.qudits[:1])
    assert max(e.start_cycles for e in program.schedule) == 2**49 + 4
    # The same for the drives calls write, which take the calls' periods.
    calls = LOOP.replace("period_cycles = array<i64: 20>", f"period_cycles = array<i64: {2**49}>")
    assert calls != LOOP
    with pytest.raises(
        pulse.CompilationError,
        match=r"last trip starts past 2\^50 cycles once its calls are written out",
    ):
        pulse.load(calls, entry="shot", qudits=system.qudits[:1])
    # The trips are read from constant bounds: a loop without them is
    # refused, wherever in it the timed operation is.
    summed = _looped(20).replace(
        "    %out:2 = scf.for %i = %c0 to %c3 ",
        "    %c2 = arith.constant 2 : index\n    %sum = arith.addi %c1, %c2 : index\n"
        "    %out:2 = scf.for %i = %c0 to %sum ",
    )
    assert summed != _looped(20)
    with pytest.raises(
        pulse.CompilationError, match="timed operation in a loop whose bounds are not constants"
    ):
        pulse.load(summed, entry="shot", qudits=system.qudits[:1])
    # Bounds whose difference overflows a trip count, and a loop that
    # counts unsigned, give no count to rely on either.
    wide = (
        _looped(20)
        .replace("%c0 = arith.constant 0 : index", f"%c0 = arith.constant {-(2**63)} : index")
        .replace("%c3 = arith.constant 3 : index", f"%c3 = arith.constant {2**63 - 1} : index")
        .replace("%c1 = arith.constant 1 : index", f"%c1 = arith.constant {2**62} : index")
    )
    unsigned = _looped(20).replace("scf.for %i = %c0", "scf.for unsigned %i = %c0")
    typed = re.sub(r"(%c[013] = arith\.constant \d) : index", r"\1 : i64", _looped(20)).replace(
        f"-> ({T}) {{\n      %w", f"-> ({T}) : i64 {{\n      %w"
    )
    for text in (wide, unsigned, typed):
        assert text != _looped(20)
        with pytest.raises(
            pulse.CompilationError, match="in a loop whose bounds are not constants"
        ):
            pulse.load(text, entry="shot", qudits=system.qudits[:1])
    # And a loop that is not an scf.for has no trips to count.
    whiled = _looped(20)
    head = whiled[whiled.index("    %out:2 = scf.for") : whiled.index("      %w = pulse.square")]
    whiled = whiled.replace(
        head,
        f"    %no = arith.constant false\n"
        f"    %out:2 = scf.while (%l = %a, %lt = %at) : ({T}) -> ({T}) {{\n"
        f"      scf.condition(%no) %l, %lt : {T}\n"
        f"    }} do {{\n"
        f"    ^bb0(%l: !pulse.drive_line, %lt: !pulse.tone):\n",
    ).replace("period_cycles = array<i64: 20>, ", "")
    with pytest.raises(pulse.CompilationError, match="neither a branch nor an scf.for loop"):
        pulse.load(whiled, entry="shot", qudits=system.qudits[:1])


# ── The number of events ─────────────────────────────────────────────────────

R = "!pulse.readout_line, !pulse.tone"


def _rounds(trips, mode="iq"):
    """An entry whose loop plays a drive and a readout in each of *trips*
    rounds, 80 cycles apart."""
    period = "period_cycles = array<i64: 80>"
    return f"""module {{
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %trips = arith.constant {trips} : index
    %n = arith.constant 8 : i64
    %long = arith.constant 64 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %a, %at = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> ({R})
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rw = pulse.square %long, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %out:4 = scf.for %i = %c0 to %trips step %c1
        iter_args(%l = %a, %lt = %at, %m = %r, %mt = %rt) -> ({T}, {R}) {{
      %l1, %lt1 = pulse.drive %l, %w, %lt {_at(0, 8, period)}
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
      %m1, %mt1, %bit = pulse.readout %m, %rw, %mt, {mode} {_at(8, 64, period)}
          : !pulse.readout_line, !pulse.waveform, !pulse.tone -> {R}, !pulse.measurement
      scf.yield %l1, %lt1, %m1, %mt1 : {T}, {R}
    }}
    return
  }}
}}"""


@pytest.mark.parametrize("trips", [0, 1, 5000, 5001, 60_000])
def test_the_op_count_is_the_length_of_the_schedule(system, trips):
    program = pulse.load(_rounds(trips), entry="shot", qudits=system.qudits[:1])
    schedule = program.schedule
    assert program.metrics.op_count == len(schedule) == 2 * trips
    # Up to 10,000 events the schedule is a list of them.
    assert isinstance(schedule, list) == (2 * trips <= 10_000)
    if trips:
        last = schedule[-1]
        assert (last.kind, last.start_cycles, last.duration_cycles) == (
            "readout",
            80 * (trips - 1) + 8,
            64,
        )


def test_the_events_are_counted_without_making_them(system):
    """What load() does in Python, and what it allocates, does not grow with
    the events of a loop: they are counted in the module, and no event is
    made to count them."""
    import tracemalloc

    from tests._work import python_work

    def loading(trips):
        text = _rounds(trips)
        return lambda: pulse.load(text, entry="shot", qudits=system.qudits[:1])

    few, many = loading(16), loading(4096)
    assert many().metrics.op_count == 8192
    assert python_work(many) == python_work(few)
    peaks = []
    for load in (few, many):
        tracemalloc.start()
        try:
            load()
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    # 8,192 listed events are over 2 MB.
    assert peaks[1] < peaks[0] + 100_000, peaks


def test_a_readout_the_listed_schedule_refuses_is_refused_at_load(system):
    """A schedule of up to 10,000 events is a list, which refuses a readout
    in a mode the exports do not take yet; counting the events refuses it
    as reading that list does, and no other program."""
    message = "readout mode trace is not supported by the exports yet; only iq is"
    for trips in (1, 5000):
        with pytest.raises(NotImplementedError, match=message):
            pulse.load(_rounds(trips, "trace"), entry="shot", qudits=system.qudits[:1])
    # A longer schedule is not a list, and a loop of no round lists nothing.
    for trips in (5001, 0):
        program = pulse.load(_rounds(trips, "trace"), entry="shot", qudits=system.qudits[:1])
        assert program.metrics.op_count == len(program.schedule) == 2 * trips


def _attribute(value):
    return MODULE.replace("module {", "module attributes {made.by = " + value + "} {", 1)


@pytest.mark.parametrize(
    "value",
    [
        "[" * 300 + "1" + "]" * 300,
        "[" + ", ".join(["affine_set<(d0) : (d0 - 1 >= 0, 9 - d0 >= 0)>"] * 300) + "]",
        "[" + ", ".join(["-1"] * 300) + "]",
        '"((([[[{{{<<< // \\" )))"',
    ],
    ids=["nested-lists", "many-comparisons", "negative-numbers", "brackets-in-a-string"],
)
def test_the_text_is_parsed_as_it_is_given(system, value):
    # Nothing is read before the parser reads it: what parses, loads.
    text = "// " + "-" * 300 + "\n// ((((" * 100 + "\n" + _attribute(value)
    assert events(pulse.load(text, entry="round", qudits=system.qudits[:2])) == events(
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    )


def test_a_function_not_valid_as_written_is_refused_with_the_reasons(system, monkeypatch):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    real = native.PulseModuleBuilder

    class Refusing:
        def __init__(self):
            self._real = real()

        def load_callables(self, text):
            return self._real.load_callables(text)

        def load_link_names(self, text):
            return self._real.load_link_names(text)

        def load_function(self, *arguments):
            raise RuntimeError(
                "load: round is not valid as a loaded function\n"
                "  loc(\"-\":3:5): 'pulse.drive' op the reason the verifier gave"
            )

    monkeypatch.setattr(native, "PulseModuleBuilder", Refusing)
    with pytest.raises(pulse.CompilationError) as refused:
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    assert refused.value.code == "wrong_argument"
    assert "is not valid as a loaded function" in str(refused.value)
    assert "the reason the verifier gave" in str(refused.value)


def _nested_ifs(count):
    lines = [
        "module {",
        f"  func.func @deep(%q: !pulse.qudit) {TIMED} {{",
        "    %c = arith.constant true",
    ]
    lines += ["    scf.if %c {"] * count + ["    }"] * count
    lines += ["    return", "  }", "}"]
    return "\n".join(lines)


def test_regions_nested_too_deep_are_refused_after_parsing(system):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    # As bytecode, which the scan before parsing does not read.
    def bytecode(count):
        return native.MLIRPipeline().parse_to_module(_nested_ifs(count)).serialize()[0]

    assert pulse.load(bytecode(200), entry="deep", qudits=system.qudits[:1]) is not None
    assert pulse.load(bytecode(254), entry="deep", qudits=system.qudits[:1]) is not None
    with pytest.raises(pulse.CompilationError, match="nests regions deeper than 256"):
        pulse.load(bytecode(255), entry="deep", qudits=system.qudits[:1])


def test_a_module_inside_the_module_is_refused(system):
    inner = MODULE.strip()
    nested = "module {\n" + inner.replace("module {", "module @inner {", 1) + "\n}\n"
    with pytest.raises(pulse.CompilationError, match="has symbols of its own"):
        pulse.load(nested, entry="round", qudits=system.qudits[:2])


def test_a_module_inside_a_function_is_refused(system):
    inside = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    builtin.module {\n    }\n    %n = arith.constant 16 : i64",
    )
    assert inside != MODULE
    with pytest.raises(pulse.CompilationError, match="has symbols of its own"):
        pulse.load(inside, entry="round", qudits=system.qudits[:2])


LOOPS = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func @shot(%qa: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %in:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%m = %l, %mt = %lt) -> ({T}) {{
        %m1, %mt1 = pulse.call @x(%m, %mt)
            <{{duration_cycles = 8 : i64, PERIODS, start_cycles = 4 : i64}}> : ({T}) -> ({T})
        scf.yield %m1, %mt1 : {T}
      }}
      scf.yield %in#0, %in#1 : {T}
    }}
    return
  }}
}}
"""


def test_a_call_states_one_period_per_loop_around_it(system):
    qudits = system.qudits[:1]
    two = LOOPS.replace("PERIODS", "period_cycles = array<i64: 100, 20>")
    assert [start for _, _, start, _ in events(pulse.load(two, entry="shot", qudits=qudits))] == [
        4,
        24,
        44,
        104,
        124,
        144,
    ]
    one = LOOPS.replace("PERIODS", "period_cycles = array<i64: 20>")
    fix = "one period per scf.for around the call, outermost first"
    with pytest.raises(
        pulse.CompilationError, match=rf"inside 2 scf.for loop\(s\) and states 1 period.*{fix}"
    ):
        pulse.load(one, entry="shot", qudits=qudits)
    outside = CALLS.replace(_at(40, 8), _at(40, 8, "period_cycles = array<i64: 9>"))
    assert outside != CALLS
    with pytest.raises(pulse.CompilationError, match=r"inside 0 scf.for loop\(s\) and states 1"):
        pulse.load(outside, entry="shot", qudits=system.qudits[:2])
    # A called function has no loop: its calls state no period of their own.
    within = CALLS.replace(_at(0, 8), _at(0, 8, "period_cycles = array<i64: 9>"))
    assert within != CALLS
    with pytest.raises(pulse.CompilationError, match="a called function has no loop"):
        pulse.load(within, entry="shot", qudits=system.qudits[:2])


def test_the_entry_is_a_function(system):
    machine = _links()
    _, text = _as_written(system, machine)
    name = re.search(r"classical\.callable @(\w+)", text).group(1)
    with pytest.raises(pulse.CompilationError, match=f"{name} is not a function"):
        pulse.load(text, entry=name, qudits=system.qudits[:2], control_system=machine)


def test_what_a_loaded_function_may_not_hold(system):
    qudits = system.qudits[:2]
    far = re.sub(r"start_cycles =\s+32 :\s+i64", f"start_cycles = {2**60} : i64", MODULE)
    assert far != MODULE
    with pytest.raises(pulse.CompilationError, match="outside 0 to 2\\^50 cycles"):
        pulse.load(far, entry="round", qudits=qudits)
    by_value = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    %fn = func.constant @other : (!pulse.qudit) -> ()\n    %n = arith.constant 16 : i64",
    )
    assert by_value != MODULE
    with pytest.raises(pulse.CompilationError, match="takes a function as a value"):
        pulse.load(by_value, entry="round", qudits=qudits)
    selects = _chain(1, 1).replace(
        f"func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) attributes {{",
        "func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> "
        f"({T}) attributes {{pulse.selected_values = 1 : i64, ",
    )
    assert "pulse.selected_values" in selects
    with pytest.raises(pulse.CompilationError, match="leaf selects values at run time"):
        pulse.load(selects, entry="shot", qudits=system.qudits[:1])


def test_attributes_on_results_and_arguments_are_dropped(system):
    text = MODULE.replace("-> i1 attributes", "-> (i1 {made.note = 1 : i64}) attributes").replace(
        "@round(%a: !pulse.qudit,", "@round(%a: !pulse.qudit {made.note = 2 : i64},"
    )
    assert text.count("made.note") == 2
    program = pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert "made." not in program.mlir()
    assert len(events(program)) == 3


def test_a_classical_function_named_main_is_refused(system):
    machine = _links()
    _, text = _as_written(system, machine)
    name = re.search(r"classical\.callable @(\w+)", text).group(1)
    key = re.search(r"dispatch_key = (\d+) : i64", text).group(0)
    # Under another name the function keeps its key, which it then states.
    clash = (
        text.replace(f"@{name}", "@main")
        .replace("func.func @main(", "func.func @entry(")
        .replace(key, key + ", dispatch_key_override")
    )
    with pytest.raises(pulse.CompilationError, match="classical function named main"):
        pulse.load(
            clash,
            entry="entry",
            qudits=[system.qudits[0], system.qudits[2]],
            control_system=machine,
        )


def test_a_link_no_classical_function_travels_on_is_refused(system):
    machine = _links()
    _, text = _as_written(system, machine)
    other = text.replace(
        "classical.get_link {device = 0 : i64}", "classical.get_link {device = 7 : i64}"
    )
    assert other != text
    with pytest.raises(pulse.CompilationError, match="gets link 7, which none"):
        pulse.load(
            other, entry="main", qudits=[system.qudits[0], system.qudits[2]], control_system=machine
        )


def test_a_load_beside_a_failing_load_on_another_thread_keeps_each_ones_reasons(system):
    import threading

    qudits = system.qudits[:2]
    broken = MODULE.replace("pulse.square %n, %amp, %zero", "pulse.square %n, %amp, %missing", 1)
    assert broken != MODULE
    failures, stop = [], threading.Event()

    def fails():
        while not stop.is_set():
            try:
                pulse.load(broken, entry="round", qudits=qudits)
                failures.append("the broken module loaded")
            except pulse.CompilationError as error:
                # Its own reason, with the parser's diagnostic, every time.
                if "cannot read the module" not in str(error) or "undeclared SSA value" not in str(
                    error
                ):
                    failures.append(str(error))

    def loads():
        for _ in range(300):
            if events(pulse.load(MODULE, entry="round", qudits=qudits))[0][0] != "drive":
                failures.append("a good module loaded wrong")

    workers = [threading.Thread(target=fails) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        loads()
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    assert failures == []


# A drive, a wait, then a frame change that nothing takes: the tone is not
# played again. Written as another compiler writes it: nothing says that the
# change is left over.
TRAILING = """
module {
  func.func @trailing(%q: !pulse.qudit) attributes {pulse.schedule_state =
        #pulse.schedule_state<timed>} {
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.1 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %m = arith.constant 40 : i64
    %length = pulse.duration_from_int %m : (i64) -> !pulse.duration
    %d2 = pulse.wait %d1, %length <{duration_cycles = 40 : i64, start_cycles = 16 : i64}> :
          (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %value = arith.constant VALUE : f64
    %t2 = pulse.CHANGE %t1, %value <{duration_cycles = 0 : i64, start_cycles = 56 : i64}> MARK:
          !pulse.tone, f64 -> !pulse.tone
    return
  }
}
"""


@pulse.kernel
def _then_shift_phase(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))
    wait(d, 40)
    shift_phase(d.tone, 0.4)


@pulse.kernel
def _then_set_phase(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))
    wait(d, 40)
    set_phase(d.tone, 0.4)


@pulse.kernel
def _then_shift_frequency(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))
    wait(d, 40)
    shift_frequency(d.tone, 3.0e6)


@pulse.kernel
def _then_set_frequency(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))
    wait(d, 40)
    set_frequency(d.tone, 5.002e9)


def _trailing(change, value, mark=""):
    return TRAILING.replace("CHANGE", change).replace("VALUE", value).replace("MARK", mark)


def _exports(program):
    """The events of both controller formats and the frames the tones end
    in, when the format gives them."""
    return [
        (exported["events"], exported.get("frame_state"))
        for exported in (
            program.export_to_controller(),
            program.export_to_controller(version=2),
        )
    ]


def _frame_updates(program):
    return [e for e in program.export_to_controller()["events"] if e["kind"] == "frame_update"]


@pytest.mark.level0
@pytest.mark.parametrize(
    ("kernel", "change", "value"),
    [
        (_then_shift_phase, "shift_phase", "0.4"),
        (_then_set_phase, "set_phase", "0.4"),
        (_then_shift_frequency, "shift_frequency", "3.0e6"),
        (_then_set_frequency, "set_frequency", "5.002e9"),
    ],
)
def test_a_frame_change_nothing_takes_is_a_frame_update_as_in_the_compiled_kernel(
    system, kernel, change, value
):
    compiled = pulse.compile(kernel, quantum_system=system).lower(system.qudits[0])
    program = pulse.load(_trailing(change, value), entry="trailing", qudits=[system.qudits[0]])
    [update] = _frame_updates(program)
    assert (update["op"], update["value"]) == (change, float(value))
    assert (update["start_cycles"], update["duration_cycles"]) == (56, 0)
    assert _exports(program) == _exports(compiled)
    assert program.metrics.op_count == len(program.schedule)
    # The module may say so itself, as the compiler's own text does: the same.
    marked = _trailing(change, value, "{pulse.unconsumed} ")
    again = pulse.load(marked, entry="trailing", qudits=[system.qudits[0]])
    assert _exports(again) == _exports(compiled)


def test_a_loaded_frame_change_stays_where_the_module_places_it(system):
    text = _trailing("shift_frequency", "3.0e6").replace(
        "start_cycles = 56 : i64", "start_cycles = 30 : i64"
    )
    program = pulse.load(text, entry="trailing", qudits=[system.qudits[0]])
    [update] = _frame_updates(program)
    assert (update["start_cycles"], update["duration_cycles"]) == (30, 0)
    assert "start_cycles = 30 : i64" in program.mlir()


def test_what_a_module_says_of_a_frame_change_nothing_takes_is_not_read(system):
    # The change is taken by the drive after it: marked or not, no frame update.
    text = MODULE.replace(
        "%da1, %ta1 = pulse.drive %da, %w, %ta <",
        "%shift = arith.constant 0.25 : f64\n"
        "    %tas = pulse.set_phase %ta, %shift <{duration_cycles = 0 : i64, start_cycles = 0 : "
        "i64}> MARK: !pulse.tone, f64 -> !pulse.tone\n"
        "    %da1, %ta1 = pulse.drive %da, %w, %tas <",
    )
    assert text != MODULE
    plain = pulse.load(text.replace("MARK", ""), entry="round", qudits=system.qudits[:2])
    for mark in ("{pulse.unconsumed} ", "{pulse.carried = 7 : i64} "):
        marked = pulse.load(text.replace("MARK", mark), entry="round", qudits=system.qudits[:2])
        assert _frame_updates(marked) == []
        assert _exports(marked) == _exports(plain)
        assert "pulse.unconsumed" not in marked.mlir() and "pulse.carried" not in marked.mlir()
    [drive, *_] = plain.export_to_controller()["events"]
    assert drive["phase_rad"] == 0.25


@pulse.kernel
def _in_a_loop(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 1.0e6)
    for _ in range(6):
        drive(d, square(16, 0.1))
        set_phase(d.tone, 0.25)
        wait(d, 8)


@pulse.kernel
def _in_nested_loops(q):
    d = get_drive_line(q)
    for _ in range(3):
        for _ in range(4):
            drive(d, square(16, 0.1))
            shift_frequency(d.tone, 1.0e6)
        wait(d, 24)


@pulse.kernel
def _in_a_loop_that_does_not_drive_its_tone(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(16, 0.1))
    for _ in range(4):
        drive(e, square(16, 0.1))
        shift_frequency(d.tone, 0.5e6)


@pulse.kernel
def _in_an_arm(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(16, 0.1))
        set_phase(d.tone, 0.5)


@pulse.kernel
def _before_a_branch_whose_then_drives(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    drive(d, square(16, 0.1))
    shift_frequency(d.tone, 2.0e6)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(16, 0.1))


@pulse.kernel
def _before_a_branch_whose_else_drives(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    shift_frequency(d.tone, 2.0e6)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        wait(d, 8)
    else:
        drive(d, square(16, 0.1))


@pulse.kernel
def _through_a_branch_in_a_loop(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(3):
        shift_frequency(d.tone, 1.0e6)
        m = readout(ro, square(100, 0.05))
        sync(d, ro)
        if m:
            drive(d, square(16, 0.1))


@pulse.kernel
def _on_a_tone_nothing_drives(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(e, square(16, 0.1))
    set_frequency(d.tone, 5.002e9)


@pulse.kernel
def _on_a_readout_tone(q):
    ro = get_readout_line(q)
    readout(ro, square(100, 0.05))
    shift_frequency(ro.tone, 1.0e6)


@pulse.kernel
def _all_taken(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 3.0e6)
    drive(d, square(16, 0.1))
    set_phase(d.tone, 0.3)
    drive(d, square(16, 0.1))


_MARK = (
    r"pulse\.(?:unconsumed|carried_in|carried|exit_updates|then_updates|else_updates)"
    r"(?: = (?:array<[^>]*>|\d+ : i64))?"
)


def _as_written_elsewhere(lowered):
    """The scheduled assembly of a lowered kernel as a function of its
    qudits, without what the compiler marked on its frame changes, loops
    and branches: what another compiler would write for the same program.
    With it, the qudits in argument order."""
    text = lowered.mlir()
    allocated = re.findall(r"(%\w+) = pulse\.qudit_alloc \{qudit = (\d+) : i64\}", text)
    assert allocated
    text = re.sub(r"^ *%\w+ = pulse\.qudit_alloc .*\n", "", text, flags=re.M)
    for name, index in allocated:
        text, found = re.subn(
            rf"(pulse\.get_\w+_line) {re.escape(name)} \{{frequency_hz = [^}}]*\}}",
            rf"\1 %q{index}",
            text,
        )
        assert found
    arguments = ", ".join(f"%q{index}: !pulse.qudit" for _, index in allocated)
    assert "func.func @main()" in text
    text = text.replace("func.func @main()", f"func.func @main({arguments})")
    text = re.sub(r"^module @\w+ attributes \{.*\} \{$", "module {", text, count=1, flags=re.M)
    marked = len(re.findall(_MARK, text))
    for one in (rf"{_MARK}, ", rf", {_MARK}", rf" \{{{_MARK}\}}"):
        text = re.sub(one, "", text)
    assert not re.search(_MARK, text) and "pulse.qudit_alloc" not in text
    return text, [int(index) for _, index in allocated], marked


@pytest.mark.level0
@pytest.mark.parametrize(
    ("kernel", "count", "options", "updates"),
    [
        (_in_a_loop, 1, {"unroll_threshold": 0}, 1),
        (_in_a_loop, 1, {}, 1),
        (_in_nested_loops, 1, {"unroll_threshold": 0}, 1),
        (_in_a_loop_that_does_not_drive_its_tone, 2, {"unroll_threshold": 0}, 4),
        (_in_an_arm, 1, {}, 1),
        (_before_a_branch_whose_then_drives, 1, {}, 1),
        (_before_a_branch_whose_else_drives, 1, {}, 1),
        (_through_a_branch_in_a_loop, 1, {"unroll_threshold": 0}, 3),
        (_on_a_tone_nothing_drives, 2, {}, 1),
        (_on_a_readout_tone, 1, {}, 1),
        (_all_taken, 1, {}, 0),
    ],
)
def test_a_loaded_module_has_the_frame_updates_of_the_compiled_kernel_it_is_the_text_of(
    system, kernel, count, options, updates
):
    lowered = pulse.compile(kernel, quantum_system=system, **options).lower(*system.qudits[:count])
    assert len(_frame_updates(lowered)) == updates
    text, qudits, marked = _as_written_elsewhere(lowered)
    # The compiler marked what it found, and none of it is in the text.
    assert bool(marked) == bool(updates)
    program = pulse.load(text, entry="main", qudits=[system.qudits[i] for i in qudits])
    assert _exports(program) == _exports(lowered)
    assert program.metrics.op_count == len(program.schedule) == len(lowered.schedule)
    # The marks are the compiler's: the loaded program reads as the kernel's.
    assert re.findall(_MARK, program.mlir()) == re.findall(_MARK, lowered.mlir())


@pytest.mark.level2
def test_a_loop_that_leaves_very_many_frame_changes_behind_loads(system):
    # One drive and 200,000 changes in a row per iteration, and nothing after
    # the loop: the loop lists them all.
    lines, tone = [], "%t1"
    for k in range(200_000):
        lines.append(
            f"      %c{k} = pulse.shift_phase {tone}, %amp <{{duration_cycles = 0 : i64, "
            "period_cycles = array<i64: 16>, start_cycles = 16 : i64}> : "
            "!pulse.tone, f64 -> !pulse.tone"
        )
        tone = f"%c{k}"
    text = (
        """
module {
  func.func @rounds(%q: !pulse.qudit) attributes {pulse.schedule_state =
        #pulse.schedule_state<timed>} {
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.1 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %c0_index = arith.constant 0 : index
    %c2_index = arith.constant 2 : index
    %c1_index = arith.constant 1 : index
    %de, %te = scf.for %i = %c0_index to %c2_index step %c1_index
          iter_args(%d0 = %d, %t0 = %t) -> (!pulse.drive_line, !pulse.tone) {
      %d1, %t1 = pulse.drive %d0, %w, %t0 <{duration_cycles = 16 : i64, period_cycles =
            array<i64: 16>, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform,
            !pulse.tone -> !pulse.drive_line, !pulse.tone
"""
        + "\n".join(lines)
        + f"""
      scf.yield %d1, {tone} : !pulse.drive_line, !pulse.tone
    }}
    return
  }}
}}
"""
    )
    program = pulse.load(text, entry="rounds", qudits=[system.qudits[0]])
    assert len(_frame_updates(program)) == 200_000


def test_a_loaded_module_for_frames_that_persist_gives_the_frame_each_tone_ends_in(system):
    from cudaq_pulse import Controller, ControlSystem

    persistent = ControlSystem(controllers=[Controller(persistent_frames=True)])
    lowered = pulse.compile(
        _then_set_frequency, quantum_system=system, control_system=persistent
    ).lower(system.qudits[0])
    text = _trailing("set_frequency", "5.002e9", "{pulse.unconsumed} ")
    program = pulse.load(
        text, entry="trailing", qudits=[system.qudits[0]], control_system=persistent
    )
    assert _frame_updates(program) == []
    [state] = program.export_to_controller()["frame_state"]
    assert state["frequency_hz"] == 5.002e9
    assert _exports(program) == _exports(lowered)


# ── Lines, tones and links through branches and loops ────────────────────────

_DRIVE_TYPES = f"!pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}"
_SUBMIT = "(!pulse.measurement) -> (!classical.link, !classical.future)"
_DECIDE = (
    '  classical.callable @decide__1 {args = ["bits:1"], device = 0 : i64, dispatch_key = '
    "1522877098 : i64, latency_max_ns = 4.0e+02 : f64, latency_min_ns = 0.0 : f64, "
    'latency_nominal_ns = 4.0e+02 : f64, mode = "bounded", results = ["bits:1"], submit = '
    '"eager"}\n'
)


def _two_lines(body, declarations="", attributes=""):
    """A function of two qudits: their drive lines and tones (%da, %ta and
    %db, %tb), a readout of the first at cycles 0 to 100 (%m, and %bit set
    by it), a 16-cycle waveform %w, and *body* after them."""
    return f"""
module{attributes} {{
{declarations}  func.func @main(%a: !pulse.qudit, %b: !pulse.qudit) {TIMED} {{
    %da, %ta = pulse.get_drive_line %a : (!pulse.qudit) -> ({T})
    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> ({T})
    %ra, %rta = pulse.get_readout_line %a : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %long = arith.constant 100 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rw = pulse.square %long, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %ra1, %rta1, %m = pulse.readout %ra, %rw, %rta, iq {_at(0, 100)}
        : !pulse.readout_line, !pulse.waveform, !pulse.tone
          -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
{body}
    return
  }}
}}
"""


def _branch(yielded):
    """One arm drives the first qudit's line; the other yields *yielded*;
    a drive follows on what the branch gives."""
    return _two_lines(f"""
    %x, %tx = scf.if %bit -> ({T}) {{
      %d1, %t1 = pulse.drive %da, %w, %ta {_at(100, 16)} : {_DRIVE_TYPES}
      scf.yield %d1, %t1 : {T}
    }} else {{
      scf.yield {yielded} : {T}
    }}
    %x1, %tx1 = pulse.drive %x, %w, %tx {_at(116, 16)} : {_DRIVE_TYPES}""")


@pytest.mark.level0
def test_a_line_that_leaves_a_branch_is_the_same_one_in_both_arms(system):
    qudits = system.qudits[:2]
    program = pulse.load(_branch("%da, %ta"), entry="main", qudits=qudits)
    assert events(program) == [("readout", 0, 0, 100), ("drive", 0, 100, 16), ("drive", 0, 116, 16)]
    # The export gives the drive after the branch one line: the arms name it.
    for yielded, named in (
        ("%db, %tb", "the drive line of qudit 0 and the drive line of qudit 1"),
        (
            "%da, %tb",
            "the tone of the drive line of qudit 0 and the tone of the drive line of qudit 1",
        ),
    ):
        with pytest.raises(pulse.CompilationError, match="scf.if whose arms give") as refused:
            pulse.load(_branch(yielded), entry="main", qudits=qudits)
        assert named in str(refused.value)
        assert refused.value.code == "wrong_argument"


def _made_tones(other):
    """A branch that gives a tone made with pulse.tone from one arm and
    *other* from the other, and a drive that takes it."""
    return _two_lines(f"""
    %f5 = arith.constant 5.0e9 : f64
    %f6 = arith.constant 6.0e9 : f64
    %u = pulse.tone %f5, %zero : f64, f64 -> !pulse.tone
    %v = pulse.tone %f6, %zero : f64, f64 -> !pulse.tone
    %tx = scf.if %bit -> (!pulse.tone) {{
      scf.yield %u : !pulse.tone
    }} else {{
      scf.yield {other} : !pulse.tone
    }}
    %x1, %tx1 = pulse.drive %da, %w, %tx {_at(100, 16)} : {_DRIVE_TYPES}""")


def test_a_tone_a_function_makes_is_one_tone_through_a_branch(system):
    qudits = system.qudits[:2]
    program = pulse.load(_made_tones("%u"), entry="main", qudits=qudits)
    [drive] = [e for e in program.export_to_controller()["events"] if e["kind"] == "drive"]
    assert drive["frequency_hz"] == 5.0e9
    # Another tone from the other arm would be a frequency the measured bit picks.
    for other, named in (
        ("%v", "a tone made by 'pulse.tone' and a tone made by 'pulse.tone'"),
        ("%ta", "a tone made by 'pulse.tone' and the tone of the drive line of qudit 0"),
    ):
        with pytest.raises(pulse.CompilationError, match="scf.if whose arms give") as refused:
            pulse.load(_made_tones(other), entry="main", qudits=qudits)
        assert named in str(refused.value)


def test_a_line_is_acquired_from_a_qudit_the_function_takes(system):
    qudits = system.qudits[:2]
    acquired = f"""
    %dq, %tq = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %x1, %tx1 = pulse.drive %dq, %w, %tq {_at(100, 16)} : {_DRIVE_TYPES}"""
    picked = "    %q = arith.select %bit, %a, %b : !pulse.qudit" + acquired
    with pytest.raises(pulse.CompilationError, match="main makes a qudit with 'arith.select'"):
        pulse.load(_two_lines(picked), entry="main", qudits=qudits)
    # Through a branch or a loop, even as the same qudit, the export would
    # not find the qudit of the line acquired after it.
    branch = """
    %q = scf.if %bit -> (!pulse.qudit) {
      scf.yield %b : !pulse.qudit
    } else {
      scf.yield OTHER : !pulse.qudit
    }"""
    for other in ("%a", "%b"):
        with pytest.raises(pulse.CompilationError, match="passes a qudit through an scf.if"):
            pulse.load(
                _two_lines(branch.replace("OTHER", other) + acquired), entry="main", qudits=qudits
            )
    loop = """
    %q = scf.for %i = %c0 to %c3 step %c1 iter_args(%k = %b) -> (!pulse.qudit) {
      scf.yield %k : !pulse.qudit
    }"""
    with pytest.raises(pulse.CompilationError, match="passes a qudit through an scf.for"):
        pulse.load(_two_lines(loop + acquired), entry="main", qudits=qudits)
    # Acquired from the qudit as the function takes it, the line is its own.
    head = f"    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> ({T})\n"
    direct = _two_lines(acquired.replace("%q ", "%b "))
    assert head in direct
    program = pulse.load(direct.replace(head, ""), entry="main", qudits=qudits)
    assert events(program)[-1] == ("drive", 1, 100, 16)


def _carried(yielded):
    """A loop of three trips that drives the line it carries first."""
    return _two_lines(f"""
    %l:4 = scf.for %i = %c0 to %c3 step %c1 iter_args(%x = %da, %tx = %ta, %y = %db, %ty = %tb)
        -> ({T}, {T}) {{
      %x1, %tx1 = pulse.drive %x, %w, %tx {_at(100, 16, "period_cycles = array<i64: 16>")}
          : {_DRIVE_TYPES}
      scf.yield {yielded} : {T}, {T}
    }}""")


@pytest.mark.level0
def test_a_line_a_loop_carries_is_the_same_one_in_every_trip(system):
    qudits = system.qudits[:2]
    program = pulse.load(_carried("%x1, %tx1, %y, %ty"), entry="main", qudits=qudits)
    assert events(program)[1:] == [("drive", 0, 100 + 16 * k, 16) for k in range(3)]
    # The second trip would drive the other qudit's line.
    with pytest.raises(pulse.CompilationError, match="scf.for that takes in") as refused:
        pulse.load(_carried("%y, %ty, %x1, %tx1"), entry="main", qudits=qudits)
    assert "the drive line of qudit 0 and yields, in its place, the drive line of qudit 1" in str(
        refused.value
    )
    # A tone alone is held to it too.
    tones = (
        "takes in the tone of the drive line of qudit 0 and yields, in its place, "
        "the tone of the drive line of qudit 1"
    )
    with pytest.raises(pulse.CompilationError, match=tones):
        pulse.load(_carried("%x1, %ty, %y, %tx1"), entry="main", qudits=qudits)
    # And a loop inside the loop, which swaps what the outer one carries.
    inner = _two_lines(f"""
    %l:4 = scf.for %i = %c0 to %c3 step %c1 iter_args(%x = %da, %tx = %ta, %y = %db, %ty = %tb)
        -> ({T}, {T}) {{
      %k:4 = scf.for %j = %c0 to %c3 step %c1 iter_args(%p = %x, %tp = %tx, %q = %y, %tq = %ty)
          -> ({T}, {T}) {{
        scf.yield %q, %tq, %p, %tp : {T}, {T}
      }}
      scf.yield %k#0, %k#1, %k#2, %k#3 : {T}, {T}
    }}""")
    with pytest.raises(pulse.CompilationError, match="scf.for that takes in the drive line of"):
        pulse.load(inner, entry="main", qudits=qudits)


@pytest.mark.parametrize(
    ("made", "what"),
    [
        (
            "%x = arith.select %bit, %da, %db : !pulse.drive_line",
            "a drive line with 'arith.select'",
        ),
        ("%tx = arith.select %bit, %ta, %tb : !pulse.tone", "a tone with 'arith.select'"),
        ("%tx = ub.poison : !pulse.tone", "a tone with 'ub.poison'"),
        ("%rx = ub.poison : !pulse.readout_line", "a readout line with 'ub.poison'"),
        ("%lx = ub.poison : !classical.link", "a link with 'ub.poison'"),
    ],
)
def test_a_line_comes_from_its_qudit_through_pulse_operations(system, made, what):
    with pytest.raises(pulse.CompilationError, match=f"main makes {what}") as refused:
        pulse.load(_two_lines("    " + made), entry="main", qudits=system.qudits[:2])
    assert refused.value.code == "wrong_argument"


def _two_links(other):
    """A submission on the link a branch gives: link 0 from one arm and
    *other* from the other."""
    # A second function, on the second link.
    second = _DECIDE.replace("@decide__1", "@amp__1").replace("device = 0", "device = 1")
    second = second.replace("1522877098", "2616160850")
    assert second.count("amp__1") == 1 and "device = 1" in second and "2616160850" in second
    return _two_lines(
        f"""
    %l0 = classical.get_link {{device = 0 : i64}} : !classical.link
    %l1 = classical.get_link {{device = 1 : i64}} : !classical.link
    %l = scf.if %bit -> (!classical.link) {{
      scf.yield %l0 : !classical.link
    }} else {{
      scf.yield {other} : !classical.link
    }}
    %l2, %f = classical.submit %l, @decide__1(%m) {_at(100, 100)} : {_SUBMIT}
    %ans = classical.wait %f, @decide__1 {_at(1200, 0)} : i1""",
        _DECIDE + second,
        ' attributes {classical.links = {"0" = {name = "host"}, "1" = {name = "side"}}}',
    )


def test_a_link_that_leaves_a_branch_is_the_same_one_in_both_arms(system):
    from cudaq_pulse.classical import Link

    machine = pulse.ControlSystem(
        links={name: Link(fixed_ns=50, feedback_ns=100) for name in ("host", "side")}
    )
    qudits = system.qudits[:2]
    program = pulse.load(_two_links("%l0"), entry="main", qudits=qudits, control_system=machine)
    assert [e[0] for e in events(program)] == ["readout", "submit", "answer"]
    with pytest.raises(pulse.CompilationError, match="as one result, link 0 and link 1"):
        pulse.load(_two_links("%l1"), entry="main", qudits=qudits, control_system=machine)


def test_a_line_is_followed_through_the_calls_written_in_place(system):
    drives = CALLS[CALLS.index("  func.func private @x") : CALLS.index("  func.func private @pair")]

    def text(other):
        return f"""
module {{
{drives}  func.func @shot(%qa: !pulse.qudit, %qb: !pulse.qudit) {TIMED} {{
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %b, %bt = pulse.get_drive_line %qb : (!pulse.qudit) -> ({T})
    %r, %rt = pulse.get_readout_line %qa : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r, %w, %rt, iq {_at(0, 8)}
        : !pulse.readout_line, !pulse.waveform, !pulse.tone
          -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %x, %tx = scf.if %bit -> ({T}) {{
      %a1, %at1 = pulse.call @x(%a, %at) {_at(8, 8)} : ({T}) -> ({T})
      scf.yield %a1, %at1 : {T}
    }} else {{
      {other}
    }}
    return
  }}
}}
"""

    qudits = system.qudits[:2]
    program = pulse.load(text(f"scf.yield %a, %at : {T}"), entry="shot", qudits=qudits)
    assert events(program) == [("readout", 0, 0, 8), ("drive", 0, 8, 8)]
    called = f"%b1, %bt1 = pulse.call @x(%b, %bt) {_at(8, 8)} : ({T}) -> ({T})"
    with pytest.raises(
        pulse.CompilationError, match="shot has an scf.if whose arms give"
    ) as refused:
        pulse.load(text(f"{called}\n      scf.yield %b1, %bt1 : {T}"), entry="shot", qudits=qudits)
    assert "the drive line of qudit 0 and the drive line of qudit 1" in str(refused.value)
