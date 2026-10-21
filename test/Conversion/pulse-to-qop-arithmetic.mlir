// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

// A linear combination of waveforms lowers to one callback per term; a
// scale by -2 doubles the square's amplitude and advances its phase by pi.
// CHECK-LABEL: func @main
// CHECK: qop.callback_scalar @drive_envelope_0_0_x {cudm.callback_kind = 2 : i32, cudm.callback_params = array<f64: 0.000000e+00, 4.000000e+00, 3.000000e-01, 0.000000e+00, 2.000000e+00,
// CHECK: qop.callback_scalar @drive_envelope_0_1_x {cudm.callback_kind = 1 : i32, cudm.callback_params = array<f64: 0.000000e+00, 4.000000e+00, 2.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, {{-?}}3.1415926535897931, 0.000000e+00,
// CHECK: qop.callback_scalar @drive_envelope_0_0_y
// CHECK: qop.callback_scalar @drive_envelope_0_1_y
// A product is sampled once per clock cycle: 0.3 * 0.1 * exp(-(t - 2)^2 / 8)
// at t = 0.5, 1.5, 2.5, 3.5 ns. Its imaginary part is zero, so it has one
// callback per quadrature.
// CHECK: qop.callback_scalar @drive_envelope_1_x {cudm.callback_kind = 7 : i32, cudm.callback_params = array<f64: 4.000000e+00, 4.000000e+00, {{.*}}, 0.02264518805967022, 0.029076997034290328, 0.029076997034290328, 0.02264518805967022>}
// CHECK: qop.callback_scalar @drive_envelope_1_y {cudm.callback_kind = 7
// CHECK-NOT: qop.callback_scalar
func.func @main() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 4 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 2.0 : f64
  %real = arith.constant 0.1 : f64
  %imag = arith.constant 0.0 : f64
  %factor = arith.constant -2.0 : f64
  %g = pulse.gaussian %duration, %amplitude, %sigma : i64, f64, f64 -> !pulse.waveform
  %s = pulse.square %duration, %real, %imag : i64, f64, f64 -> !pulse.waveform
  %scaled = pulse.scale %s, %factor : !pulse.waveform, f64 -> !pulse.waveform
  %sum = pulse.add %g, %scaled : !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %sum, %t0
      {start_vtu = 0 : i64, duration_vtu = 4 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  %product = pulse.mul %g, %s : !pulse.waveform
  %d2, %t2 = pulse.drive %d1, %product, %t1
      {start_vtu = 4 : i64, duration_vtu = 4 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  return
}

