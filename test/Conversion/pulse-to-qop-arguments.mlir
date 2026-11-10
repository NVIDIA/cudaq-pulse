// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s
// RUN: not cudaq-pulse-opt --pulse-to-qop --qop-to-cudm %s 2>&1 | FileCheck %s --check-prefix=UNBOUND

// A tone frequency, a frame phase and an amplitude that are float arguments
// stay symbolic: each callback parameter is its constant part plus
// coefficients on the arguments, as (slot, argument) pairs. The drive starts
// at 2 ns, so the envelope phase is -(phase + 2 pi (f - 5 GHz) 2 ns) and it
// turns at 2 pi (5 GHz - f); the amplitude is amp / 2.
// CHECK-LABEL: func @main
// CHECK: qop.callback_scalar @drive_envelope_0_x {cudm.callback_kind = 1 : i32,
// CHECK-SAME: cudm.callback_param_arguments = array<i64: 2, 2, 7, 0, 7, 1, 9, 0>,
// CHECK-SAME: cudm.callback_param_coefficients = array<f64: 5.000000e-01, -1.2566370614359173E-8, -1.000000e+00, -6.2831853071795863E-9>,
// CHECK-SAME: cudm.callback_params = array<f64: 2.000000e+00, 4.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 62.831853071795862, 0.000000e+00, 31.415926535897931>}

// The cuDensityMat runtime takes constant parameters.
// UNBOUND: error: drive callback depends on the program's float arguments; bind them before lowering to cuDensityMat
func.func @main(%f: f64, %phase: f64, %amp: f64) {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 4 : i64
  %zero = arith.constant 0.0 : f64
  %half = arith.constant 0.5 : f64
  %t1 = pulse.set_frequency %t0, %f : !pulse.tone, f64 -> !pulse.tone
  %t2 = pulse.shift_phase %t1, %phase : !pulse.tone, f64 -> !pulse.tone
  %scaled = arith.mulf %amp, %half : f64
  %s = pulse.square %duration, %scaled, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t3 = pulse.drive %d0, %s, %t2
      {start_cycles = 2 : i64, duration_cycles = 4 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone
  return
}
