// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: not cudaq-pulse-opt --pulse-verify --split-input-file %s 2>&1 | FileCheck %s

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A line threaded through a block continues as the block's result.
// CHECK-NOT: error
func.func @threaded(%x: !w) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.schedule_block "asap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t2 : !d, !t
  }
  %d3, %t3 = pulse.drive %d1, %x, %t1 : !d, !w, !t -> !d, !t
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Passing a line to a block consumes it.
// CHECK: error: linear pulse value is consumed more than once on one path
func.func @used_after_block(%x: !w) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.schedule_block "asap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t2 : !d, !t
  }
  %d3, %t3 = pulse.drive %d0, %x, %t1 : !d, !w, !t -> !d, !t
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Inside the block, the block argument is consumed at most once.
// CHECK: error: linear pulse value is consumed more than once on one path
func.func @twice_in_block(%x: !w) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.schedule_block "asap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    %d3, %t3 = pulse.drive %d, %x, %t2 : !d, !w, !t -> !d, !t
    pulse.yield %d3, %t3 : !d, !t
  }
  return
}
