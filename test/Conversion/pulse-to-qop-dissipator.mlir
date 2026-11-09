// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

module @dissipator_test attributes {
    pulse.t1_times = [50.0e3 : f64],
    pulse.t2_times = [30.0e3 : f64]
} {

// CHECK-LABEL: func @main
func.func @main() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.0e9 : f64}
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0
      {start_cycles = 0 : i64, duration_cycles = 40 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone

  // CHECK: qop.spin{{.*}}spin_lowering
  // CHECK: qop.spin{{.*}}spin_z
  // CHECK: qop.lindblad
  return
}

}
