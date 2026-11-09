// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-fuse-pulses %s | FileCheck %s

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Adjacent square drives fuse inside a block, but not across its boundary:
// the block keeps its own timing.
// CHECK-LABEL: func.func @fuse_and_blocks
// CHECK: pulse.drive {{.*}}duration_cycles = 80 : i64, fused
// CHECK: pulse.schedule_block
// CHECK: pulse.drive {{.*}}duration_cycles = 80 : i64, fused
// CHECK-NEXT: pulse.yield
func.func @fuse_and_blocks() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
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
