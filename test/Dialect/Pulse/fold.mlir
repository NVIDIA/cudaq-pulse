// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-fold %s | FileCheck %s --check-prefix=FOLD
// RUN: cudaq-pulse-opt --pulse-virtual-z %s | FileCheck %s --check-prefix=VZ
// RUN: cudaq-pulse-opt --pulse-fold --pulse-virtual-z %s | FileCheck %s --check-prefix=BOTH

// pulse-fold drops a frame change by zero and merges equal constants;
// pulse-virtual-z only moves frame changes into the drives they precede.
// FOLD-LABEL: func.func @frames
// FOLD-COUNT-1: arith.constant 2.000000e-01
// FOLD-NOT: arith.constant 2.000000e-01
// FOLD-NOT: pulse.shift_phase
// FOLD-NOT: frame_phase_offset

// VZ-LABEL: func.func @frames
// VZ-COUNT-2: arith.constant 2.000000e-01
// VZ-NOT: pulse.shift_phase
// VZ: pulse.drive {{.*}}frame_phase_offset = 0.000000e+00

// BOTH-LABEL: func.func @frames
// BOTH-COUNT-1: arith.constant 2.000000e-01
// BOTH-NOT: arith.constant 2.000000e-01
// BOTH-NOT: pulse.shift_phase
// BOTH-NOT: frame_phase_offset
func.func @frames() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 2.000000e-01 : f64
  %b = arith.constant 2.000000e-01 : f64
  %none = arith.constant 0.000000e+00 : f64
  %sigma = arith.constant 1.000000e+01 : f64
  %x = pulse.gaussian %c40, %a, %sigma : i64, f64, f64 -> !pulse.waveform
  %y = pulse.gaussian %c40, %b, %sigma : i64, f64, f64 -> !pulse.waveform
  %t1 = pulse.shift_phase %t0, %none : !pulse.tone, f64 -> !pulse.tone
  %d1, %t2 = pulse.drive %d0, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t3 = pulse.drive %d1, %y, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
