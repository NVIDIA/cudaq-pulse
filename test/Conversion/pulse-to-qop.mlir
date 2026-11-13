// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-to-qop --qop-to-cudm --cudm-to-llvm \
// RUN:   --canonicalize --convert-arith-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts %s | FileCheck %s --check-prefix=FULL

// FULL-LABEL: llvm.func @main
// FULL: llvm.call @cudm_init
// FULL: llvm.call @cudm_evolve
// FULL: llvm.call @cudm_state_capture
// FULL-NOT: pulse.drive
// FULL-NOT: qop.lindblad
// FULL-NOT: qop.evolve
// FULL-NOT: cudm.evolve

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
      <{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone

  // CHECK: qop.spin
  // CHECK: qop.make_product
  // CHECK: qop.callback_scalar
  // CHECK: qop.make_sum
  // CHECK: %[[L:.*]] = qop.lindblad
  // CHECK-NEXT: qop.evolve %[[L]] : !qop.super_op
  // CHECK-NEXT: return
  return
}
