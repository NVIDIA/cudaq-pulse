// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

// A drive and a readout print as they parse, with every operand and result.

// CHECK-LABEL: func @test_qudit_and_drive
// CHECK: %[[Q:.*]] = pulse.qudit_alloc : !pulse.qref
// CHECK: %[[D:.*]], %[[T:.*]] = pulse.get_drive_line %[[Q]] : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
// CHECK: %[[WF:.*]] = pulse.gaussian
// CHECK: %[[D1:.*]], %[[T1:.*]] = pulse.drive %[[D]], %[[WF]], %[[T]] : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
// CHECK: return %[[D1]], %[[T1]]
func.func @test_qudit_and_drive() -> (!pulse.drive_line, !pulse.tone) {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}

// CHECK-LABEL: func @test_readout
// CHECK: %[[Q:.*]] = pulse.qudit_alloc : !pulse.qref
// CHECK: %[[R:.*]], %[[T:.*]] = pulse.get_readout_line %[[Q]] : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
// CHECK: %[[WF:.*]] = pulse.square
// CHECK: %[[R1:.*]], %[[T1:.*]], %[[M:.*]] = pulse.readout %[[R]], %[[WF]], %[[T]], "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
// CHECK: return %[[R1]], %[[T1]], %[[M]]
func.func @test_readout() -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %duration = arith.constant 1000 : i64
  %amplitude = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r0, %wf, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return %r1, %rt1, %m : !pulse.readout_line, !pulse.tone, !pulse.measurement
}
