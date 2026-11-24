// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-frame-changes %s | FileCheck %s
// RUN: not cudaq-pulse-opt --pulse-to-qop %s 2>&1 | FileCheck %s --check-prefix=QOP

// The passes beside the schedule passes, on a module that holds a coupling
// line. A frame change after a sync with a coupling line is placed when the
// coupling line is free, not when its own line is. The operator program has
// no term for a coupling line, so pulse-to-qop refuses the module.

// QOP: this program holds a coupling line; a program with a coupling line is not simulated or emulated yet

module attributes {pulse.coupling_lines} {
  // CHECK-LABEL: func.func @main
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
    %f = arith.constant 1.000000e+06 : f64
    %short = arith.constant 16 : i64
    %long = arith.constant 200 : i64
    %amplitude = arith.constant 1.000000e-01 : f64
    %zero = arith.constant 0.000000e+00 : f64
    %wf = pulse.gaussian %short, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
    %step = pulse.square %long, %amplitude, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %c1 = pulse.drive_baseband %c, %step <{duration_cycles = 200 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
    %d2, %c2 = pulse.sync %d1, %c1 : !pulse.drive_line, !pulse.coupling_line -> !pulse.drive_line, !pulse.coupling_line
    // CHECK: pulse.shift_frequency
    // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 200 : i64
    %t2 = pulse.shift_frequency %t1, %f : !pulse.tone, f64 -> !pulse.tone
    %d3, %t3 = pulse.drive %d2, %wf, %t2 <{duration_cycles = 16 : i64, start_cycles = 200 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
