// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// How frames behave, as the simulator plays them. Each drive's callback has
// its start (ns) first and its phase in the qudit's frame at index 7.
// RUN: cudaq-pulse-opt --split-input-file --pulse-schedule-asap --pulse-to-qop %s | FileCheck %s

// A frame change takes effect at the next pulse on its tone: the first drive
// plays at phase 0, the second at the shift (as -pi/2: the pulse turns about
// an axis at -theta).
// CHECK: @drive_envelope_0_x {{.*}}array<f64: 0.000000e+00, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}0.000000e+00,
// CHECK: @drive_envelope_1_x {{.*}}array<f64: 4.000000e+01, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}-1.5707963267948966,
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.1 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %half_pi = arith.constant 1.5707963267948966 : f64
    %t2 = pulse.shift_phase %t1, %half_pi : !pulse.tone, f64 -> !pulse.tone
    %d2, %t3 = pulse.drive %d1, %x, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// A wait does not advance a tone's frame phase (unlike OpenPulse, where a
// delay advances the frame): the drive after a 100-cycle wait plays at the
// same phase as the one before. The carrier runs on through the wait, so a
// tone 1 MHz above its qudit is 2 pi 1e6 t round in the qudit's frame at each
// start t: -1.13 rad at 180 ns and -2.01 rad at 320 ns, after a wait as
// without one.
// CHECK: {{^// -{5}$}}
// CHECK: @drive_envelope_0_x {{.*}}array<f64: 0.000000e+00, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}0.000000e+00,
// CHECK: @drive_envelope_1_x {{.*}}array<f64: 1.400000e+02, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}0.000000e+00,
// CHECK: @drive_envelope_2_x {{.*}}array<f64: 1.800000e+02, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}-1.1309733552923256,
// CHECK: @drive_envelope_3_x {{.*}}array<f64: 3.200000e+02, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}-2.0106192982974678,
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c100 = arith.constant 100 : i64
    %amp = arith.constant 0.1 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %idle = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d2 = pulse.wait %d1, %idle : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d3, %t2 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %f = arith.constant 5.001000e+09 : f64
    %off = pulse.tone %f, %zero : f64, f64 -> !pulse.tone
    %d4, %t3 = pulse.drive %d3, %x, %off : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d5 = pulse.wait %d4, %idle : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d6, %t4 = pulse.drive %d5, %x, %t3 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// A frame rotates through a sync: the shift made before the sync applies to
// the drive after it, which the sync holds back to cycle 140, where the other
// line's wait ends.
// CHECK: {{^// -{5}$}}
// CHECK: @drive_envelope_0_x {{.*}}array<f64: 0.000000e+00, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}0.000000e+00,
// CHECK: @drive_envelope_1_x {{.*}}array<f64: 1.400000e+02, 4.000000e+01, {{([^,]*, )([^,]*, )([^,]*, )([^,]*, )([^,]*, )}}-1.5707963267948966,
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %e0, %u0 = pulse.get_drive_line %q1 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c140 = arith.constant 140 : i64
    %amp = arith.constant 0.1 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %idle = pulse.duration_from_int %c140 : (i64) -> !pulse.duration
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %e1 = pulse.wait %e0, %idle : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %half_pi = arith.constant 1.5707963267948966 : f64
    %t2 = pulse.shift_phase %t1, %half_pi : !pulse.tone, f64 -> !pulse.tone
    %s:2 = pulse.sync %d1, %e1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
    %d2, %t3 = pulse.drive %s#0, %x, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
