// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

// Waveform algebra prints as it parses, with its operands. The function
// returns every result, so none is dropped as unused.

// CHECK-LABEL: func @test_waveform_algebra
// CHECK: %[[A:.*]] = pulse.square
// CHECK: %[[B:.*]] = pulse.square
// CHECK: %[[SUM:.*]] = pulse.add %[[A]], %[[B]] : !pulse.waveform
// CHECK: %[[DIFF:.*]] = pulse.sub %[[A]], %[[B]] : !pulse.waveform
// CHECK: %[[PROD:.*]] = pulse.mul %[[A]], %[[B]] : !pulse.waveform
// CHECK: %[[C:.*]] = arith.constant 2.000000e+00 : f64
// CHECK: %[[SCALED:.*]] = pulse.scale %[[A]], %[[C]] : !pulse.waveform, f64 -> !pulse.waveform
// CHECK: %[[NEG:.*]] = pulse.neg %[[A]] : !pulse.waveform
// CHECK: return %[[SUM]], %[[DIFF]], %[[PROD]], %[[SCALED]], %[[NEG]]
func.func @test_waveform_algebra() -> (!pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform) {
  %duration = arith.constant 40 : i64
  %amp_a = arith.constant 0.3 : f64
  %amp_b = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %a = pulse.square %duration, %amp_a, %zero
      : i64, f64, f64 -> !pulse.waveform
  %b = pulse.square %duration, %amp_b, %zero
      : i64, f64, f64 -> !pulse.waveform
  %sum = pulse.add %a, %b : !pulse.waveform
  %diff = pulse.sub %a, %b : !pulse.waveform
  %prod = pulse.mul %a, %b : !pulse.waveform
  %c = arith.constant 2.0 : f64
  %scaled = pulse.scale %a, %c : !pulse.waveform, f64 -> !pulse.waveform
  %negated = pulse.neg %a : !pulse.waveform
  return %sum, %diff, %prod, %scaled, %negated : !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform, !pulse.waveform
}
