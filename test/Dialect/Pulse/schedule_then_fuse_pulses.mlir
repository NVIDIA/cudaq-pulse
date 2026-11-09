// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-fuse-pulses %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-fuse-pulses %s | cudaq-pulse-opt --pulse-verify > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-fuse-pulses %s | cudaq-pulse-opt --pulse-verify > /dev/null

// Fusing two scheduled drives keeps the schedule of the first, its start and
// its period in each enclosing loop, for the length of both.

// CHECK-LABEL: func.func @fuse_in_loop
// CHECK: scf.for
// CHECK: pulse.drive {{.*}}{duration_cycles = 80 : i64, fused, period_cycles = array<i64: 280>, start_cycles = 0 : i64}
// CHECK-NOT: pulse.drive %
// CHECK: pulse.wait
func.func @fuse_in_loop() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.047 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %c200 = arith.constant 200 : i64
  %w200 = pulse.duration_from_int %c200 : (i64) -> !pulse.duration
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d3 = pulse.wait %d2, %w200 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d3, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}
