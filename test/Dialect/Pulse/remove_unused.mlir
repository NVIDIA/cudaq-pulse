// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-remove-unused %s | FileCheck %s

// A waveform and constants nothing uses go; what the drive uses stays, and
// equal constants are not merged.
// CHECK-LABEL: func.func @unused
// CHECK-COUNT-2: arith.constant 2.000000e-01
// CHECK-NOT: arith.constant 5.000000e-01
// CHECK-COUNT-2: pulse.gaussian
// CHECK-NOT: pulse.gaussian
// CHECK: pulse.drive
func.func @unused() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 2.000000e-01 : f64
  %b = arith.constant 2.000000e-01 : f64
  %unused = arith.constant 5.000000e-01 : f64
  %sigma = arith.constant 1.000000e+01 : f64
  %x = pulse.gaussian %c40, %a, %sigma : i64, f64, f64 -> !pulse.waveform
  %y = pulse.gaussian %c40, %b, %sigma : i64, f64, f64 -> !pulse.waveform
  %z = pulse.gaussian %c40, %unused, %sigma : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %y, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
