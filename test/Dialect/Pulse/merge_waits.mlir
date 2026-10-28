// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-merge-waits %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-merge-waits --pulse-schedule-alap %s | FileCheck %s --check-prefix=TIME

// Waits in a row on one line become one wait of their total length.
// CHECK-LABEL: func.func @in_a_row
// CHECK: arith.constant 350 : i64
// CHECK: pulse.wait
// CHECK-NOT: pulse.wait
// CHECK: pulse.drive
// TIME-LABEL: func.func @in_a_row
// TIME: pulse.drive {{.*}}start_cycles = 350
func.func @in_a_row() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c100 = arith.constant 100 : i64
  %c50 = arith.constant 50 : i64
  %c200 = arith.constant 200 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  %w50 = pulse.duration_from_int %c50 : (i64) -> !pulse.duration
  %w200 = pulse.duration_from_int %c200 : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %w100 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d2 = pulse.wait %d1, %w50 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d3 = pulse.wait %d2, %w200 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %c40 = arith.constant 40 : i64
  %a = arith.constant 1.000000e-01 : f64
  %z = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d4, %t1 = pulse.drive %d3, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A drive between two waits, or a wait whose length is a parameter, keeps
// them apart.
// CHECK-LABEL: func.func @apart
// CHECK-COUNT-2: pulse.wait
// CHECK: pulse.drive
// CHECK-COUNT-2: pulse.wait
func.func @apart(%len: i64) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c100 = arith.constant 100 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  %wlen = pulse.duration_from_int %len : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %w100 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d2 = pulse.wait %d1, %wlen : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %c40 = arith.constant 40 : i64
  %a = arith.constant 1.000000e-01 : f64
  %z = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d3, %t1 = pulse.drive %d2, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d4 = pulse.wait %d3, %w100 {pulse.variable_wait, pulse.min_cycles = 10 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d5 = pulse.wait %d4, %w100 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}
