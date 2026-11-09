// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-fold-phases %s | FileCheck %s

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A frame change on a tone a block takes next moves into the block and folds
// into the block's first drive.
// CHECK-LABEL: func.func @into_a_block
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.schedule_block
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.drive {{.*}}frame_phase_offset = 2.500000e-01
func.func @into_a_block(%x: !w) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %phi = arith.constant 0.25 : f64
  %t1 = pulse.shift_phase %t0, %phi : !t, f64 -> !t
  %d1, %t2 = pulse.schedule_block "asap" (%d0, %t1) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t3 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t3 : !d, !t
  }
  return
}

// A constant frame change that ends a block moves out of it and folds into
// the drive after the block.
// CHECK-LABEL: func.func @out_of_a_block
// CHECK: pulse.schedule_block
// CHECK: pulse.drive
// CHECK-NOT: frame_phase_offset
// CHECK: pulse.yield
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.drive {{.*}}frame_phase_offset = 5.000000e-01
func.func @out_of_a_block(%x: !w) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.schedule_block "alap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    %phi = arith.constant 0.5 : f64
    %t3 = pulse.shift_phase %t2, %phi : !t, f64 -> !t
    pulse.yield %d2, %t3 : !d, !t
  }
  %d3, %t4 = pulse.drive %d1, %x, %t1 : !d, !w, !t -> !d, !t
  return
}

// A frame change on a tone a block only passes through ends up after the
// block, next to the drive that absorbs it.
// CHECK-LABEL: func.func @through_a_block
// CHECK-NOT: pulse.shift_phase
// CHECK: pulse.drive {{.*}}frame_phase_offset = 7.500000e-01
func.func @through_a_block(%x: !w) {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qudit) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qudit) -> (!d, !t)
  %phi = arith.constant 0.75 : f64
  %tb1 = pulse.shift_phase %tb0, %phi : !t, f64 -> !t
  %a1, %ta1, %tb2 = pulse.schedule_block "asap" (%a0, %ta0, %tb1) : (!d, !t, !t) -> (!d, !t, !t) {
  ^bb0(%a: !d, %ta: !t, %tb: !t):
    %a2, %ta2 = pulse.drive %a, %x, %ta : !d, !w, !t -> !d, !t
    pulse.yield %a2, %ta2, %tb : !d, !t, !t
  }
  %b1, %tb3 = pulse.drive %b0, %x, %tb2 : !d, !w, !t -> !d, !t
  return
}
