// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --loop-invariant-code-motion %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-hoist-loop-invariants %s | FileCheck %s
// RUN: cudaq-pulse-opt --loop-invariant-code-motion --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-hoist-loop-invariants --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --canonicalize %s | FileCheck %s --check-prefix=DEAD

// Hoisting loop invariants out of a loop of pulses: a waveform computed from
// constants is the same every iteration and moves out; a tone made in the
// body is a new frame each iteration and stays.

// CHECK-LABEL: func.func @tone_per_iteration
// CHECK: pulse.square
// CHECK: scf.for
// CHECK: pulse.tone
// CHECK: pulse.drive
// DEAD-LABEL: func.func @tone_per_iteration
// DEAD: pulse.tone
func.func @tone_per_iteration(%c0: index, %c1: index, %c10: index, %d0: !pulse.drive_line) -> !pulse.drive_line {
  %f = arith.constant 5.0e9 : f64
  %p = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %res = scf.for %i = %c0 to %c10 step %c1 iter_args(%d = %d0) -> (!pulse.drive_line) {
    %w = pulse.square %c40, %a, %p : i64, f64, f64 -> !pulse.waveform
    %t = pulse.tone %f, %p : f64, f64 -> !pulse.tone
    %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1 : !pulse.drive_line
  }
  return %res : !pulse.drive_line
}

// A tone nothing uses is still removed as dead.
// DEAD-LABEL: func.func @unused_tone
// DEAD-NOT: pulse.tone
// DEAD: return
func.func @unused_tone() {
  %f = arith.constant 5.0e9 : f64
  %p = arith.constant 0.0 : f64
  %t = pulse.tone %f, %p : f64, f64 -> !pulse.tone
  return
}
