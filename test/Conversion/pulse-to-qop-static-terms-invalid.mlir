// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-to-qop --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --canonicalize --pulse-to-qop --verify-diagnostics

// The lowering refuses a static term on a qudit the program does not use. A
// drive line nothing uses does not use its qudit: standard passes erase it.

// expected-error@+1 {{static term #pulse.static_term<zz, qudits = [0, 5], rate_hz = 1.000000e+04> names a qudit the program does not use}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.quantum_system = #pulse.quantum_system<static_terms = [#pulse.static_term<zz, qudits = [0, 5], rate_hz = 1.000000e+04>]>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    return
  }
}

// -----

// expected-error@+1 {{static term #pulse.static_term<zz, qudits = [0, 1], rate_hz = 1.000000e+04> names a qudit the program does not use}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.quantum_system = #pulse.quantum_system<static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 1.000000e+04>]>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %d1, %t1 = pulse.get_drive_line %q1 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %zero, %zero : i64, f64, f64 -> !pulse.waveform
    %d2, %t2 = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
