// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp --verify-diagnostics
// RUN: not cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp=max-drives=0 2>&1 | FileCheck %s --check-prefix=LIMITS
// RUN: not cudaq-pulse-opt %s --split-input-file --pulse-unroll-classical-loops 2>&1 | FileCheck %s --check-prefix=UNROLL

// Diagnostics of the resource-constrained schedule and of unrolling loops
// that make classical calls.

// LIMITS: error: resource limits must be positive
// UNROLL-NOT: error
func.func @kernel_with_its_own_schedule(%d0: !pulse.drive_line, %t0: !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{resource-constrained scheduling (rcp, alap_rcp) does not support kernels with their own schedule yet}}
  %d1, %t1 = pulse.schedule_block "asap" (%d0, %t0) : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  ^bb0(%db: !pulse.drive_line, %tb: !pulse.tone):
    %d2, %t2 = pulse.drive %db, %x, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    pulse.yield %d2, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// UNROLL: error: cannot unroll a loop with non-constant bounds
func.func @call_in_a_loop_of_argument_trips(%n: index) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  // expected-error @+1 {{cannot schedule a loop with non-constant bounds}}
  scf.for %i = %lb to %n step %step {
    %l = classical.get_link {device = 0 : i64} : !classical.link
  }
  return
}
