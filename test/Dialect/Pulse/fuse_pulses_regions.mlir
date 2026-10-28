// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-fuse-pulses %s | FileCheck %s

// Adjacent pulses inside one arm fuse. Without pulse-fold-constants the new
// duration stays where the fused drive is.
// CHECK-LABEL: func.func @inside_an_arm
// CHECK: scf.if
// CHECK: arith.constant 100
// CHECK: pulse.drive {{.*}}fused
func.func @inside_an_arm(%bit: i1) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c50 = arith.constant 50 : i64
  %amp = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c50, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %e, %u = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %e, %u : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// A pulse that ends one arm does not fuse with the pulse after the if: the
// other arm plays none.
// CHECK-LABEL: func.func @across_an_if
// CHECK-COUNT-2: pulse.drive %
// CHECK-NOT: fused
func.func @across_an_if(%bit: i1) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c50 = arith.constant 50 : i64
  %amp = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c50, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// Adjacent pulses in a loop body fuse once per iteration; the last pulse of
// one iteration does not fuse with the first pulse of the next.
// CHECK-LABEL: func.func @loop_body
// CHECK: scf.for
// CHECK: arith.constant 100
// CHECK: pulse.drive {{.*}}fused
// CHECK-NOT: pulse.drive %
// CHECK: scf.yield
func.func @loop_body() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c50 = arith.constant 50 : i64
  %amp = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c50, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %a, %b = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %c, %e = pulse.drive %a, %x, %b : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %c, %e : !pulse.drive_line, !pulse.tone
  }
  return
}

// CHECK-LABEL: func.func @single_pulse_loop
// CHECK: scf.for
// CHECK: pulse.drive
// CHECK-NOT: fused
func.func @single_pulse_loop() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c50 = arith.constant 50 : i64
  %amp = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c50, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %a, %b = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %a, %b : !pulse.drive_line, !pulse.tone
  }
  return
}
