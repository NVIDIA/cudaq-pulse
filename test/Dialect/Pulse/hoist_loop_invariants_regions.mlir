// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-hoist-loop-invariants %s | FileCheck %s

// Waveform construction leaves the loop; the drive and the frame change it
// consumes stay in the body.
// CHECK-LABEL: func.func @hoist_waveform
// CHECK: pulse.square
// CHECK: scf.for
// CHECK-NOT: pulse.square
// CHECK: pulse.shift_phase
// CHECK: pulse.drive
func.func @hoist_waveform() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c50 = arith.constant 50 : i64
  %amp = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %phi = arith.constant 0.5 : f64
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %x = pulse.square %c50, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %u = pulse.shift_phase %t, %phi : !pulse.tone, f64 -> !pulse.tone
    %a, %b = pulse.drive %d, %x, %u : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %a, %b : !pulse.drive_line, !pulse.tone
  }
  return
}
