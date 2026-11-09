// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-simplify-control-flow %s | FileCheck %s

// Branching on "outcome != 0" becomes a branch on pulse.is_set, and a
// redundant sync inside an arm goes away while the arms stay.
// CHECK-LABEL: func.func @feedback
// CHECK: %[[BIT:.*]] = pulse.is_set
// CHECK: scf.if %[[BIT]]
// CHECK-COUNT-1: pulse.sync
// CHECK: } else {
func.func @feedback(%x: !pulse.waveform) {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %qa : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %k = pulse.outcome %m : !pulse.measurement -> i64
  %c0 = arith.constant 0 : i64
  %bit = arith.cmpi ne, %k, %c0 : i64
  %a2, %b2 = scf.if %bit -> (!pulse.drive_line, !pulse.drive_line) {
    %a1, %b1 = pulse.sync %a0, %b0 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
    %a3, %b3 = pulse.sync %a1, %b1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
    scf.yield %a3, %b3 : !pulse.drive_line, !pulse.drive_line
  } else {
    scf.yield %a0, %b0 : !pulse.drive_line, !pulse.drive_line
  }
  %a4, %b4 = pulse.sync %a2, %b2 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  return
}

// A loop that runs once is inlined and one that never runs is removed.
// CHECK-LABEL: func.func @trivial_loops
// CHECK-NOT: scf.for
// CHECK-COUNT-1: pulse.drive %
// CHECK: return
func.func @trivial_loops(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d1, %t1 = scf.for %i = %c0 to %c1 step %c1 iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %a, %b = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %a, %b : !pulse.drive_line, !pulse.tone
  }
  %d2, %t2 = scf.for %i = %c0 to %c0 step %c1 iter_args(%d = %d1, %t = %t1) -> (!pulse.drive_line, !pulse.tone) {
    %a, %b = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %a, %b : !pulse.drive_line, !pulse.tone
  }
  return
}
