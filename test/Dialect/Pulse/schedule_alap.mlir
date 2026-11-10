// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s

// CHECK-LABEL: func.func @simple_schedule
func.func @simple_schedule() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 3.000000e-01 : f64
  %sigma = arith.constant 1.000000e+01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  // CHECK: = pulse.drive
  // CHECK-SAME: duration_cycles = 40
  // CHECK-SAME: start_cycles = 0
  %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
