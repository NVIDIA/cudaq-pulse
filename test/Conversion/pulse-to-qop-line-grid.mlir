// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

// A product is sampled once per cycle of its line's controller. Qudit 0's
// controller runs every 2 program cycles: 0.3 * 0.1 * exp(-(t - 2)^2 / 8) at
// t = 1 and 3 ns, two samples. Qudit 1's runs every cycle: four samples.
// CHECK-LABEL: func @main
// CHECK: qop.callback_scalar @drive_envelope_0_x {cudm.callback_kind = 7 : i32, cudm.callback_params = array<f64: 0.000000e+00, 4.000000e+00, {{.*}}, 0.026474907077537864, 0.026474907077537864>}
// CHECK: qop.callback_scalar @drive_envelope_0_y
// CHECK: qop.callback_scalar @drive_envelope_1_x {cudm.callback_kind = 7 : i32, cudm.callback_params = array<f64: 0.000000e+00, 4.000000e+00, {{.*}}, 0.02264518805967022, 0.029076997034290328, 0.029076997034290328, 0.02264518805967022>}
module attributes {pulse.line_grid = 1 : i64, pulse.drive_line_grid = array<i64: 2, 0>} {
func.func @main() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %q1 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %q1 {qubit = 1 : i64, frequency_hz = 5.2e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 4 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 2.0 : f64
  %real = arith.constant 0.1 : f64
  %imag = arith.constant 0.0 : f64
  %g = pulse.gaussian %duration, %amplitude, %sigma : i64, f64, f64 -> !pulse.waveform
  %s = pulse.square %duration, %real, %imag : i64, f64, f64 -> !pulse.waveform
  %product = pulse.mul %g, %s : !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %product, %t0
      {start_cycles = 0 : i64, duration_cycles = 4 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  %e1, %u1 = pulse.drive %e0, %product, %u0
      {start_cycles = 0 : i64, duration_cycles = 4 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  return
}
}
