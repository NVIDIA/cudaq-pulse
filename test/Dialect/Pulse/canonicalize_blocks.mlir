// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-canonicalize %s | FileCheck %s

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A block without drives, readouts or waits takes no time and is inlined.
// CHECK-LABEL: func.func @untimed_block
// CHECK-NOT: pulse.schedule_block
// CHECK: pulse.shift_phase
// CHECK: pulse.drive
func.func @untimed_block(%x: !w) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %phi = arith.constant 0.25 : f64
  %t1 = pulse.schedule_block "asap" (%t0) : (!t) -> (!t) {
  ^bb0(%t: !t):
    %t2 = pulse.shift_phase %t, %phi : !t, f64 -> !t
    pulse.yield %t2 : !t
  }
  %d1, %t3 = pulse.drive %d0, %x, %t1 : !d, !w, !t -> !d, !t
  return
}

// A block with timed ops stays.
// CHECK-LABEL: func.func @timed_block
// CHECK: pulse.schedule_block "alap"
func.func @timed_block(%x: !w) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %d1, %t1 = pulse.schedule_block "alap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t2 : !d, !t
  }
  return
}
