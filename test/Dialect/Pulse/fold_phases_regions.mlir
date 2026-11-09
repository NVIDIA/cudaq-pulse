// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-fold-phases %s | FileCheck %s

// A frame change and its drive inside one arm fold together.
// CHECK-LABEL: func.func @inside_an_arm
// CHECK: scf.if
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.drive {{.*}}frame_phase_offset = 5.000000e-01
func.func @inside_an_arm(%x: !pulse.waveform, %bit: i1) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %phi = arith.constant 0.5 : f64
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %t = pulse.shift_phase %t0, %phi : !pulse.tone, f64 -> !pulse.tone
    %d, %u = pulse.drive %d0, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %u : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// A frame change that both arms pass through folds into the drive after
// the if: it applies on either path.
// CHECK-LABEL: func.func @across_an_if
// CHECK-NOT: pulse.shift_phase
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}frame_phase_offset = 5.000000e-01
func.func @across_an_if(%x: !pulse.waveform, %w: !pulse.duration, %bit: i1) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %phi = arith.constant 0.5 : f64
  %t1 = pulse.shift_phase %t0, %phi : !pulse.tone, f64 -> !pulse.tone
  %d2, %t2 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d = pulse.wait %d0, %w : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d, %t1 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t1 : !pulse.drive_line, !pulse.tone
  }
  %d3, %t3 = pulse.drive %d2, %x, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A frame change in one arm only stays put: the drive after the if plays
// with or without it.
// CHECK-LABEL: func.func @one_arm_only
// CHECK: scf.if
// CHECK: pulse.shift_phase
// CHECK: pulse.drive
// CHECK-NOT: frame_phase_offset
func.func @one_arm_only(%x: !pulse.waveform, %bit: i1) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %phi = arith.constant 0.5 : f64
  %t1 = scf.if %bit -> (!pulse.tone) {
    %t = pulse.shift_phase %t0, %phi : !pulse.tone, f64 -> !pulse.tone
    scf.yield %t : !pulse.tone
  } else {
    scf.yield %t0 : !pulse.tone
  }
  %d1, %t2 = pulse.drive %d0, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// In a loop body the frame change folds into that iteration's drive; one
// before the loop does not fold into the body.
// CHECK-LABEL: func.func @loops
// CHECK: pulse.shift_phase
// CHECK: scf.for
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.drive {{.*}}frame_phase_offset = 2.500000e-01
func.func @loops(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %phi = arith.constant 0.5 : f64
  %quarter = arith.constant 0.25 : f64
  %t1 = pulse.shift_phase %t0, %phi : !pulse.tone, f64 -> !pulse.tone
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t1) -> (!pulse.drive_line, !pulse.tone) {
    %u = pulse.shift_phase %t, %quarter : !pulse.tone, f64 -> !pulse.tone
    %dd, %tt = pulse.drive %d, %x, %u : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
  }
  return
}
