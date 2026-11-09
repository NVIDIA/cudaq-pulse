// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-to-qop --verify-diagnostics

// The lowering refuses pulse.static_terms that is not a list of
// #pulse.static_term, and a term on a qudit the program does not use.

// expected-error@+1 {{pulse.static_terms must be a list of #pulse.static_term}}
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.static_terms = 1 : i64} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    return
  }
}

// -----

// expected-error@+1 {{pulse.static_terms holds 1 : i64, not a #pulse.static_term}}
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.static_terms = [1 : i64]} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    return
  }
}

// -----

// expected-error@+1 {{static term #pulse.static_term<zz, qudits = [0, 5], rate_hz = 1.000000e+04> names a qudit the program does not use}}
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.static_terms = [#pulse.static_term<zz, qudits = [0, 5], rate_hz = 1.000000e+04>]} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    return
  }
}
