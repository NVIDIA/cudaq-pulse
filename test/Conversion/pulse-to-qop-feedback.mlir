// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A program that measures lowers one segment at a time along an outcome
// path: before the measurement, then after it for each outcome.
// RUN: cudaq-pulse-opt --pulse-to-qop --split-input-file --verify-diagnostics %s | FileCheck %s

// Measured 1: the conditional X plays after the readout.
// CHECK-LABEL: module @excited
// CHECK: qop.callback_scalar @drive_envelope_0_x {{.*}}array<f64: 3.000000e+02, 4.000000e+01
module @excited attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.path = [array<i64: 0, 1>], qop.t_start = 3.000000e+02 : f64, qop.t_end = 3.400000e+02 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %r0, %rt0 = pulse.get_readout_line %q {qubit = 0 : i64, frequency_hz = 7.000000e+09 : f64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    %c300 = arith.constant 300 : i64
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d, %t = pulse.drive %d0, %x, %t0 {start_cycles = 300 : i64, duration_cycles = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}

// -----

// Measured 0: nothing plays.
// CHECK-LABEL: module @ground
// CHECK-NOT: qop.callback_scalar
module @ground attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.path = [array<i64: 0, 0>], qop.t_start = 3.000000e+02 : f64, qop.t_end = 3.400000e+02 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %r0, %rt0 = pulse.get_readout_line %q {qubit = 0 : i64, frequency_hz = 7.000000e+09 : f64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    %c300 = arith.constant 300 : i64
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d, %t = pulse.drive %d0, %x, %t0 {start_cycles = 300 : i64, duration_cycles = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}

// -----

// Before the measurement completes, the branch does not matter.
// CHECK-LABEL: module @before
// CHECK-NOT: qop.callback_scalar
module @before attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.path = [], qop.t_start = 0.000000e+00 : f64, qop.t_end = 3.000000e+02 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %r0, %rt0 = pulse.get_readout_line %q {qubit = 0 : i64, frequency_hz = 7.000000e+09 : f64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    %c300 = arith.constant 300 : i64
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d, %t = pulse.drive %d0, %x, %t0 {start_cycles = 300 : i64, duration_cycles = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}

// -----

// After it, an outcome is needed.
module @undecided attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.path = [], qop.t_start = 0.000000e+00 : f64, qop.t_end = 3.400000e+02 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %r0, %rt0 = pulse.get_readout_line %q {qubit = 0 : i64, frequency_hz = 7.000000e+09 : f64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    %c300 = arith.constant 300 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %ro = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    // expected-error @+1 {{no outcome on pulse.path for this measurement, which completes at 3.000000e+02 ns, before the evolution window ends}}
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d, %t = pulse.drive %d0, %ro, %t0 {start_cycles = 300 : i64, duration_cycles = 300 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}

// -----

// Without a path, a program that measures cannot be one evolution.
module @no_path attributes {pulse.clock_ghz = 1.000000e+00 : f64} {
  func.func @main(%x: !pulse.waveform) {
    %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %r0, %rt0 = pulse.get_readout_line %q {qubit = 0 : i64, frequency_hz = 7.000000e+09 : f64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    // expected-error @+1 {{a program that measures evolves in segments between measurements; lower it along a measurement-outcome path (pulse.path), as pulse.simulate() does}}
    %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    return
  }
}
