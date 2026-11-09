// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-verify --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --canonicalize --pulse-schedule-alap %s | cudaq-pulse-opt | FileCheck %s --check-prefix=ROUNDTRIP

// A wait whose length the function computes from a float argument is
// scheduled at its lowest length; the computation stays in the module.

// CHECK-LABEL: func.func @t1(%arg0: f64)
func.func @t1(%tau: f64) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q
      : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c200 = arith.constant 200 : i64
  %amp = arith.constant 0.3 : f64
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %m = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  // CHECK: = pulse.drive %
  // CHECK-SAME: start_cycles = 0
  %d1, %t1 = pulse.drive %d0, %x, %t0
      : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: %[[X:.*]] = arith.addf %arg0
  // CHECK: %[[N:.*]] = arith.fptosi %[[X]] : f64 to i64
  // CHECK: pulse.duration_from_int %[[N]]
  %h = arith.addf %tau, %half : f64
  %n = arith.fptosi %h : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // CHECK: = pulse.wait %
  // CHECK-SAME: duration_cycles = 16
  // CHECK-SAME: pulse.max_cycles = 4000
  // CHECK-SAME: pulse.min_cycles = 16
  // CHECK-SAME: pulse.variable_wait
  // CHECK-SAME: start_cycles = 40
  %d2 = pulse.wait %d1, %len {pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 4000 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %s:2 = pulse.sync %d2, %r0 : !pulse.drive_line, !pulse.readout_line
      -> !pulse.drive_line, !pulse.readout_line
  // CHECK: = pulse.readout %
  // CHECK-SAME: start_cycles = 56
  %r1, %rt1, %res = pulse.readout %s#1, %m, %rt0, "iq"
      : !pulse.readout_line, !pulse.waveform, !pulse.tone
      -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}

// A second line that does not wait ends with the program under ALAP and
// starts at zero under ASAP.

// CHECK-LABEL: func.func @two_lines(%arg0: f64)
func.func @two_lines(%tau: f64) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qref
  %a0, %at = pulse.get_drive_line %q
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %b0, %bt = pulse.get_drive_line %p
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c20 = arith.constant 20 : i64
  %amp = arith.constant 0.3 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c20, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // CHECK: = pulse.wait %
  // CHECK-SAME: duration_cycles = 100
  // CHECK-SAME: start_cycles = 0
  %a1 = pulse.wait %a0, %len {pulse.variable_wait, pulse.min_cycles = 100 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // CHECK: = pulse.drive %
  // CHECK-SAME: start_cycles = 100
  %a2, %at2 = pulse.drive %a1, %x, %at
      : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: = pulse.drive %
  // ALAP-SAME: start_cycles = 100
  // ASAP-SAME: start_cycles = 0
  %b1, %bt1 = pulse.drive %b0, %x, %bt
      : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// Once the argument is a constant, canonicalization folds the length and the
// schedule is the exact one.

// ROUNDTRIP-LABEL: func.func @bound()
func.func @bound() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.3 : f64
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %tau = arith.constant 99.6 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %h = arith.addf %tau, %half : f64
  %n = arith.fptosi %h : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // ROUNDTRIP: = pulse.wait %
  // ROUNDTRIP-SAME: duration_cycles = 100
  %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = 0 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // ROUNDTRIP: = pulse.drive %
  // ROUNDTRIP-SAME: start_cycles = 100
  %d2, %t1 = pulse.drive %d1, %x, %t0
      : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
