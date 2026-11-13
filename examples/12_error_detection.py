# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verification error detection: intentionally bad programs.

The kernel frontend rebinds every line and tone it advances, so most of
these mistakes cannot be written as a ``@pulse.kernel``. They can be written
as Pulse MLIR, though, by hand or by a custom pass, and the ``pulse-verify``
pass rejects them with a diagnostic per violation:

  - Linearity violations (a line consumed twice)
  - Backward time travel and overlap on one physical line
  - Non-positive waveform durations

The hand-written programs go through ``cudaq-pulse-opt``, the MLIR driver
installed with the package.
"""

import shutil
import subprocess

import cudaq_pulse as pulse

_PRELUDE = """
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
"""

_SQUARE = "pulse.square {dur}, %amp, %zero : i64, f64, f64 -> !pulse.waveform"
_DRIVE = (
    "pulse.drive {line}, {wf}, {tone}{attrs} : !pulse.drive_line, !pulse.waveform, "
    "!pulse.tone -> !pulse.drive_line, !pulse.tone"
)

programs = {
    "linearity: one drive line consumed twice": f"""
  %c40 = arith.constant 40 : i64
  %wf = {_SQUARE.format(dur="%c40")}
  %d1, %t1 = {_DRIVE.format(line="%d", wf="%wf", tone="%t", attrs="")}
  %d2, %t2 = {_DRIVE.format(line="%d", wf="%wf", tone="%t1", attrs="")}
""",
    "timing: a drive starts before its predecessor ends": f"""
  %c40 = arith.constant 40 : i64
  %wf = {_SQUARE.format(dur="%c40")}
  %d1, %t1 = {
        _DRIVE.format(
            line="%d",
            wf="%wf",
            tone="%t",
            attrs=" <{start_cycles = 0 : i64, duration_cycles = 40 : i64}>",
        )
    }
  %d2, %t2 = {
        _DRIVE.format(
            line="%d1",
            wf="%wf",
            tone="%t1",
            attrs=" <{start_cycles = 20 : i64, duration_cycles = 40 : i64}>",
        )
    }
""",
    "waveform: non-positive duration": f"""
  %c0 = arith.constant 0 : i64
  %wf = {_SQUARE.format(dur="%c0")}
  %d1, %t1 = {_DRIVE.format(line="%d", wf="%wf", tone="%t", attrs="")}
""",
}


def check(body: str) -> str:
    text = "func.func @main() {" + _PRELUDE + body + "  return\n}\n"
    tool = shutil.which("cudaq-pulse-opt")
    if tool is None:
        raise SystemExit(
            "cudaq-pulse-opt is not on PATH; it comes with cudaq-pulse, from the "
            f"cudaq-pulse-tools wheel: pip install cudaq-pulse-tools=={pulse.__version__}"
        )
    # Op verifiers check each op on its own as the text is parsed;
    # pulse-verify checks what needs more than one op, such as the value a
    # waveform's duration has.
    result = subprocess.run(
        [tool, "--pulse-verify", "-o", "/dev/null"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stderr.strip() if result.returncode else "verified"


@pulse.kernel
def zero_length(q0):
    d0 = get_drive_line(q0)
    duration = 0 * pulse.ns
    sigma = 5 * pulse.ns
    drive(d0, gaussian(duration, 0.1, sigma))


def main():
    for title, body in programs.items():
        print(f"=== {title} ===")
        print(check(body))
        print()

    print("=== the same check on a kernel ===")
    try:
        pulse.compile(zero_length, device=dev).lower(dev.qudits[0])
    except RuntimeError as error:
        print(error)


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    main()
