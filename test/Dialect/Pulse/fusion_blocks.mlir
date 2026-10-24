// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-fusion %s | FileCheck %s

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Adjacent square drives fuse inside a block, but not across its boundary:
// the block keeps its own timing.
// CHECK-LABEL: func.func @fusion_and_blocks
// CHECK: pulse.drive {{.*}}duration_vtu = 80 : i64, fused
// CHECK: pulse.schedule_block
// CHECK: pulse.drive {{.*}}duration_vtu = 80 : i64, fused
// CHECK-NEXT: pulse.yield
func.func @fusion_and_blocks() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !d, !w, !t -> !d, !t
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !d, !w, !t -> !d, !t
  %d3, %t3 = pulse.schedule_block "asap" (%d2, %t2) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d4, %t4 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    %d5, %t5 = pulse.drive %d4, %x, %t4 : !d, !w, !t -> !d, !t
    pulse.yield %d5, %t5 : !d, !t
  }
  return
}
