// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

// A drive and a readout print as they parse, with every operand and result.

// CHECK-LABEL: func @test_qudit_and_drive
// CHECK: %[[Q:.*]] = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
// CHECK: %[[D:.*]], %[[T:.*]] = pulse.get_drive_line %[[Q]] : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
// CHECK: %[[WF:.*]] = pulse.gaussian
// CHECK: %[[D1:.*]], %[[T1:.*]] = pulse.drive %[[D]], %[[WF]], %[[T]] : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
// CHECK: return %[[D1]], %[[T1]]
func.func @test_qudit_and_drive() -> (!pulse.drive_line, !pulse.tone) {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}

// CHECK-LABEL: func @test_readout
// CHECK: %[[Q:.*]] = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
// CHECK: %[[R:.*]], %[[T:.*]] = pulse.get_readout_line %[[Q]] : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
// CHECK: %[[WF:.*]] = pulse.square
// CHECK: %[[R1:.*]], %[[T1:.*]], %[[M:.*]] = pulse.readout %[[R]], %[[WF]], %[[T]], iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
// CHECK: return %[[R1]], %[[T1]], %[[M]]
func.func @test_readout() -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %duration = arith.constant 1000 : i64
  %amplitude = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %wf, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return %r1, %rt1, %m : !pulse.readout_line, !pulse.tone, !pulse.measurement
}

// Each readout mode and schedule policy prints as its keyword.

// CHECK-LABEL: func @test_readout_modes
// CHECK: pulse.readout %{{.*}}, iq :
// CHECK: pulse.readout %{{.*}}, discriminated :
// CHECK: pulse.readout %{{.*}}, trace :
// CHECK: pulse.readout %{{.*}}, counts :
// CHECK: pulse.schedule_block alap (
func.func @test_readout_modes(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> (!pulse.readout_line, !pulse.tone) {
  %r1, %t1, %m1 = pulse.readout %r0, %wf, %t0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r2, %t2, %m2 = pulse.readout %r1, %wf, %t1, discriminated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r3, %t3, %m3 = pulse.readout %r2, %wf, %t2, trace : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r4, %t4, %m4 = pulse.readout %r3, %wf, %t3, counts : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r5, %t5 = pulse.schedule_block alap (%r4, %t4) : (!pulse.readout_line, !pulse.tone) -> (!pulse.readout_line, !pulse.tone) {
  ^bb0(%r: !pulse.readout_line, %t: !pulse.tone):
    pulse.yield %r, %t : !pulse.readout_line, !pulse.tone
  }
  return %r5, %t5 : !pulse.readout_line, !pulse.tone
}
