// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --canonicalize | FileCheck %s

// Each built-in waveform prints as it parses, with its operands, in the
// custom and the generic form. The function returns them, so no fold or
// dead-code removal can drop one.

// CHECK-LABEL: func @test_waveforms
// CHECK-DAG: %[[F0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[F025:.*]] = arith.constant 2.500000e-01 : f64
// CHECK-DAG: %[[F03:.*]] = arith.constant 3.000000e-01 : f64
// CHECK-DAG: %[[F04:.*]] = arith.constant 4.000000e-01 : f64
// CHECK-DAG: %[[F05:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-DAG: %[[F5:.*]] = arith.constant 5.000000e+00 : f64
// CHECK-DAG: %[[F8:.*]] = arith.constant 8.000000e+00 : f64
// CHECK-DAG: %[[F10:.*]] = arith.constant 1.000000e+01 : f64
// CHECK-DAG: %[[I20:.*]] = arith.constant 20 : i64
// CHECK-DAG: %[[I40:.*]] = arith.constant 40 : i64
// CHECK-DAG: %[[I80:.*]] = arith.constant 80 : i64
// CHECK-DAG: %[[I100:.*]] = arith.constant 100 : i64
// CHECK-DAG: %[[I200:.*]] = arith.constant 200 : i64
// CHECK: %[[SQ:.*]] = pulse.square %[[I100]], %[[F05]], %[[F0]] : i64, f64, f64 -> !pulse.waveform
// CHECK: %[[G:.*]] = pulse.gaussian %[[I40]], %[[F03]], %[[F10]] : i64, f64, f64 -> !pulse.waveform
// CHECK: %[[GS:.*]] = pulse.gaussian_square %[[I200]], %[[F04]], %[[F8]], %[[I20]] : i64, f64, f64, i64 -> !pulse.waveform
// CHECK: %[[DR:.*]] = pulse.drag %[[I40]], %[[F025]], %[[F10]], %[[F05]] : i64, f64, f64, f64 -> !pulse.waveform
// CHECK: %[[COS:.*]] = pulse.cosine %[[I100]], %[[F03]] : i64, f64 -> !pulse.waveform
// CHECK: %[[RAMP:.*]] = pulse.tanh_ramp %[[I80]], %[[F05]], %[[F5]] : i64, f64, f64 -> !pulse.waveform
// CHECK: return %[[SQ]], %[[G]], %[[GS]], %[[DR]], %[[COS]], %[[RAMP]]
func.func @test_waveforms() -> (!pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform) {
  %c0 = arith.constant 0.0 : f64
  %c025 = arith.constant 0.25 : f64
  %c03 = arith.constant 0.3 : f64
  %c04 = arith.constant 0.4 : f64
  %c05 = arith.constant 0.5 : f64
  %c5 = arith.constant 5.0 : f64
  %c8 = arith.constant 8.0 : f64
  %c10 = arith.constant 10.0 : f64
  %c20 = arith.constant 20 : i64
  %c40 = arith.constant 40 : i64
  %c80 = arith.constant 80 : i64
  %c100 = arith.constant 100 : i64
  %c200 = arith.constant 200 : i64
  %sq = pulse.square %c100, %c05, %c0 : i64, f64, f64 -> !pulse.waveform
  %g = pulse.gaussian %c40, %c03, %c10 : i64, f64, f64 -> !pulse.waveform
  %gs = pulse.gaussian_square %c200, %c04, %c8, %c20
      : i64, f64, f64, i64 -> !pulse.waveform
  %dr = pulse.drag %c40, %c025, %c10, %c05
      : i64, f64, f64, f64 -> !pulse.waveform
  %cos = pulse.cosine %c100, %c03 : i64, f64 -> !pulse.waveform
  %ramp = pulse.tanh_ramp %c80, %c05, %c5
      : i64, f64, f64 -> !pulse.waveform
  return %sq, %g, %gs, %dr, %cos, %ramp : !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform
}
