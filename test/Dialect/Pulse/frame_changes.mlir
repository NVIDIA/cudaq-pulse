// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-frame-changes --split-input-file %s | FileCheck %s

// A change is stamped where its tone's previous drive ends, with no duration;
// one no later drive takes is kept and marked unconsumed.

// CHECK-LABEL: func.func @stamps
func.func @stamps() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %f = arith.constant 1.000000e+06 : f64
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 0 : i64
  %t1 = pulse.shift_frequency %t, %f : !pulse.tone, f64 -> !pulse.tone
  %d1, %t2 = pulse.drive %d, %wf, %t1 {duration_cycles = 16 : i64, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.set_phase
  // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 16 : i64
  %t3 = pulse.set_phase %t2, %ph : !pulse.tone, f64 -> !pulse.tone
  %d2, %t4 = pulse.drive %d1, %wf, %t3 {duration_cycles = 16 : i64, start_cycles = 24 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.shift_phase
  // CHECK-SAME: duration_cycles = 0 : i64, pulse.unconsumed, start_cycles = 40 : i64
  %t5 = pulse.shift_phase %t4, %ph : !pulse.tone, f64 -> !pulse.tone
  return
}

// -----

// Frames that persist leave the change nothing takes unmarked.

// CHECK-LABEL: module attributes {pulse.persistent_frames}
// CHECK-LABEL: func.func @persistent
module attributes {pulse.persistent_frames} {
  func.func @persistent() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %ph = arith.constant 2.500000e-01 : f64
    %duration = arith.constant 16 : i64
    %amplitude = arith.constant 1.000000e-01 : f64
    %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %wf, %t {duration_cycles = 16 : i64, start_cycles = 8 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 24 : i64
    // CHECK-NOT: pulse.unconsumed
    %t2 = pulse.shift_phase %t1, %ph : !pulse.tone, f64 -> !pulse.tone
    return
  }
}

// -----

// The change a loop carries out of its last iteration to nothing is listed
// on the loop.

// CHECK-LABEL: func.func @carried
func.func @carried() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%l = %d, %tone = %t) -> (!pulse.drive_line, !pulse.tone) {
    %l1, %t1 = pulse.drive %l, %wf, %tone {duration_cycles = 16 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: duration_cycles = 0 : i64, period_cycles = array<i64: 16>, pulse.carried = 0 : i64, start_cycles = 16 : i64
    %t2 = pulse.shift_phase %t1, %ph : !pulse.tone, f64 -> !pulse.tone
    scf.yield %l1, %t2 : !pulse.drive_line, !pulse.tone
  }
  // CHECK: {pulse.carried_in = array<i64: 0>, pulse.exit_updates = array<i64: 0>}
  return
}

// -----

// A change made before a branch that only one arm plays with is listed on
// the other arm.

// CHECK-LABEL: func.func @one_arm
func.func @one_arm(%m: i1) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.shift_phase
  // CHECK-SAME: pulse.carried = 0 : i64
  %t1 = pulse.shift_phase %t, %ph : !pulse.tone, f64 -> !pulse.tone
  %r:2 = scf.if %m -> (!pulse.drive_line, !pulse.tone) {
    %l1, %t2 = pulse.drive %d, %wf, %t1 {duration_cycles = 16 : i64, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %l1, %t2 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d, %t1 : !pulse.drive_line, !pulse.tone
  }
  // CHECK: } {pulse.else_updates = array<i64: 0>}
  return
}

// -----

// In a loop whose iterations do not play its tone, a change is at the start
// of each iteration, though the tone is free only later.

// CHECK-LABEL: func.func @idle
func.func @idle() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e, %u = pulse.get_drive_line %r : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t {duration_cycles = 16 : i64, start_cycles = 48 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%tone = %t1, %l = %e, %v = %u) -> (!pulse.tone, !pulse.drive_line, !pulse.tone) {
    %l1, %v1 = pulse.drive %l, %wf, %v {duration_cycles = 16 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: duration_cycles = 0 : i64, period_cycles = array<i64: 16>, pulse.unconsumed, start_cycles = 0 : i64
    %t2 = pulse.shift_phase %tone, %ph : !pulse.tone, f64 -> !pulse.tone
    scf.yield %t2, %l1, %v1 : !pulse.tone, !pulse.drive_line, !pulse.tone
  }
  return
}
