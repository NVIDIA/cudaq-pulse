// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// Waveform arithmetic is element-wise: operands of different durations are
// rejected instead of truncated.
func.func @add_of_different_durations() {
  %short = arith.constant 4 : i64
  %long = arith.constant 8 : i64
  %real = arith.constant 0.1 : f64
  %imag = arith.constant 0.0 : f64
  %a = pulse.square %short, %real, %imag : i64, f64, f64 -> !pulse.waveform
  %b = pulse.square %long, %real, %imag : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{operands have durations 4 and 8; waveform arithmetic needs equal durations}}
  %sum = pulse.add %a, %b : !pulse.waveform
  return
}

// -----

func.func @mul_of_different_durations() {
  %short = arith.constant 4 : i64
  %long = arith.constant 8 : i64
  %amplitude = arith.constant 0.1 : f64
  %sigma = arith.constant 2.0 : f64
  %a = pulse.gaussian %short, %amplitude, %sigma : i64, f64, f64 -> !pulse.waveform
  %b = pulse.custom_samples [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] : !pulse.waveform
  // expected-error @+1 {{operands have durations 4 and 8}}
  %product = pulse.mul %a, %b : !pulse.waveform
  return
}

// -----

func.func @equal_durations_through_scale_and_neg() {
  %duration = arith.constant 4 : i64
  %real = arith.constant 0.1 : f64
  %imag = arith.constant 0.0 : f64
  %factor = arith.constant 0.5 : f64
  %a = pulse.square %duration, %real, %imag : i64, f64, f64 -> !pulse.waveform
  %b = pulse.scale %a, %factor : !pulse.waveform, f64 -> !pulse.waveform
  %c = pulse.neg %b : !pulse.waveform
  %sum = pulse.sub %a, %c : !pulse.waveform
  return
}
