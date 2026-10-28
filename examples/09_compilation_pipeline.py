# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Full compilation pipeline walkthrough.

Demonstrates ``pulse.compile()``, which compiles a kernel ahead of its
call, and ``lower()``, which traces it for given arguments, runs the
optimization passes and the scheduler, and returns the MLIR module of the
call with its timing breakdown.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.compiler import DEFAULT_PASSES

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "idle": 50 * pulse.ns,
    "square_duration": 25 * pulse.ns,
}


@pulse.kernel
def pipeline_demo(qubit_0, qubit_1):
    """A 2-qubit program with redundancies for the pipeline to clean up.

    Includes:
      - A shift_phase before a drive (FOLD_PHASES folds it into the drive)
      - Adjacent waits (MERGE_WAITS, which main() adds to the default
        passes, merges them)
      - Adjacent same-amplitude square pulses (FUSE_PULSES merges them)
    """
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    shift_phase(tone_0, math.pi / 4)
    gaussian_envelope = gaussian(cal["x_duration"], 0.3, cal["x_sigma"])
    drive(drive_line_0, gaussian_envelope, tone_0)

    wait(drive_line_0, cal["idle"])
    wait(drive_line_0, cal["idle"])

    square_pulse_1 = square(cal["square_duration"], 0.2)
    drive(drive_line_0, square_pulse_1, tone_0)
    square_pulse_2 = square(cal["square_duration"], 0.2)
    drive(drive_line_0, square_pulse_2, tone_0)

    gaussian_envelope_2 = gaussian(cal["x_duration"], 0.3, cal["x_sigma"])
    drive(drive_line_1, gaussian_envelope_2, tone_1)


def main():
    compiled_kernel = pulse.compile(
        pipeline_demo, schedule="alap", device=dev, passes=(*DEFAULT_PASSES, pulse.Pass.MERGE_WAITS)
    ).lower(dev.qudit_handles[0], dev.qudit_handles[1])

    print("=== Compiled MLIR (first 30 lines) ===")
    for line in compiled_kernel.mlir().splitlines()[:30]:
        print(f"  {line}")

    metrics = compiled_kernel.metrics
    print("\n=== Compile metrics ===")
    print(f"  Trace     : {metrics.trace_ms:.3f} ms")
    print(f"  Passes    : {metrics.passes_ms:.3f} ms")
    print(f"  Schedule  : {metrics.schedule_ms:.3f} ms")
    print(f"  MLIR build: {metrics.ffi_ms:.3f} ms")
    print(f"  Total     : {metrics.total_ms:.3f} ms")

    # Lower to LLVM IR in process with the native MLIR binding.
    try:
        llvm_ir = compiled_kernel.mlir("llvm")
        print(f"\n=== LLVM IR ({len(llvm_ir)} chars, first 10 lines) ===")
        for line in llvm_ir.splitlines()[:10]:
            print(f"  {line}")
    except RuntimeError as error:
        print(f"\n(LLVM lowering failed: {error})")


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    main()
