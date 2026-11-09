// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A drive on a qudit that pulse.modes leaves out has no mode to act on.
// RUN: cudaq-pulse-opt --split-input-file --pulse-to-qop --verify-diagnostics %s

module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.modes = array<i64: 3>} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 13 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q {frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    // expected-error @+1 {{drive on qudit 13, which is not one of pulse.modes}}
    %d1, %t1 = pulse.drive %d0, %x, %t0 {start_cycles = 0 : i64, duration_cycles = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// expected-error @+1 {{pulse.modes must list distinct qudit indices}}
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.modes = array<i64: 3, 3>} {
  func.func @main() {
    return
  }
}
