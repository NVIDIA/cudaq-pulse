// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file > /dev/null
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// The values a waveform, a variable wait or a measurement takes from the ops
// that compute them, one case per diagnostic. Op verifiers check an op on its
// own and accept every case here (the first RUN); pulse-verify rejects them.

func.func @square_of_zero_duration() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  // expected-error @+1 {{'pulse.square' op duration must be positive, got 0}}
  %w = pulse.square %d, %a, %a : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_of_negative_duration() {
  %d = arith.constant -4 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  // expected-error @+1 {{'pulse.gaussian' op duration must be positive, got -4}}
  %w = pulse.gaussian %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_of_zero_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 0.0 : f64
  // expected-error @+1 {{'pulse.gaussian' op sigma must be positive}}
  %w = pulse.gaussian %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

// A duration that is a function argument is checked once it is a constant.
func.func @gaussian_of_symbolic_duration(%d: i64) {
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  %w = pulse.gaussian %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_square_of_zero_duration() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  %r = arith.constant 4 : i64
  // expected-error @+1 {{'pulse.gaussian_square' op duration must be positive, got 0}}
  %w = pulse.gaussian_square %d, %a, %s, %r : i64, f64, f64, i64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_square_of_negative_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant -1.0 : f64
  %r = arith.constant 4 : i64
  // expected-error @+1 {{'pulse.gaussian_square' op sigma must be positive}}
  %w = pulse.gaussian_square %d, %a, %s, %r : i64, f64, f64, i64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_square_of_zero_risefall() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  %r = arith.constant 0 : i64
  // expected-error @+1 {{'pulse.gaussian_square' op risefall must be positive, got 0}}
  %w = pulse.gaussian_square %d, %a, %s, %r : i64, f64, f64, i64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_square_edges_longer_than_the_pulse() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  %r = arith.constant 21 : i64
  // expected-error @+1 {{'pulse.gaussian_square' op risefall (21) exceeds half the duration (40)}}
  %w = pulse.gaussian_square %d, %a, %s, %r : i64, f64, f64, i64 -> !pulse.waveform
  return
}

// -----

// 2 * risefall would overflow an i64.
func.func @gaussian_square_of_huge_risefall() {
  %d = arith.constant 100 : i64
  %a = arith.constant 0.5 : f64
  %s = arith.constant 2.0 : f64
  %r = arith.constant 4611686018427387904 : i64
  // expected-error @+1 {{'pulse.gaussian_square' op risefall (4611686018427387904) exceeds half the duration (100)}}
  %w = pulse.gaussian_square %d, %a, %s, %r : i64, f64, f64, i64 -> !pulse.waveform
  return
}

// -----

func.func @gaussian_of_nan_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 0x7FF8000000000000 : f64
  // expected-error @+1 {{'pulse.gaussian' op sigma must be positive and finite, got nan}}
  %w = pulse.gaussian %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @drag_of_infinite_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 0x7FF0000000000000 : f64
  %b = arith.constant 0.5 : f64
  // expected-error @+1 {{'pulse.drag' op sigma must be positive and finite}}
  %w = pulse.drag %d, %a, %s, %b : i64, f64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @drag_of_zero_duration() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  %b = arith.constant 0.5 : f64
  // expected-error @+1 {{'pulse.drag' op duration must be positive, got 0}}
  %w = pulse.drag %d, %a, %s, %b : i64, f64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @drag_of_zero_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 0.0 : f64
  %b = arith.constant 0.5 : f64
  // expected-error @+1 {{'pulse.drag' op sigma must be positive}}
  %w = pulse.drag %d, %a, %s, %b : i64, f64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @cosine_of_zero_duration() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  // expected-error @+1 {{'pulse.cosine' op duration must be positive, got 0}}
  %w = pulse.cosine %d, %a : i64, f64 -> !pulse.waveform
  return
}

// -----

func.func @tanh_ramp_of_zero_duration() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 2.0 : f64
  // expected-error @+1 {{'pulse.tanh_ramp' op duration must be positive, got 0}}
  %w = pulse.tanh_ramp %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @tanh_ramp_of_zero_sigma() {
  %d = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 0.0 : f64
  // expected-error @+1 {{'pulse.tanh_ramp' op sigma must be positive}}
  %w = pulse.tanh_ramp %d, %a, %s : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

func.func @custom_of_zero_duration() {
  %d = arith.constant 0 : i64
  // expected-error @+1 {{'pulse.custom' op duration must be positive, got 0}}
  %w = pulse.custom @envelope, %d : i64 -> !pulse.waveform
  return
}

// -----

func.func @outcome_not_from_a_readout(%b: i1) -> i64 {
  %m = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
  // expected-error @+1 {{'pulse.outcome' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast}}
  %k = pulse.outcome %m : !pulse.measurement -> i64
  return %k : i64
}

// -----

// A duration computed from constants is a constant once folded.
func.func @duration_folding_to_a_negative() -> !pulse.waveform {
  %a = arith.constant 20 : i64
  %b = arith.constant -40 : i64
  %d = arith.addi %a, %b : i64
  %amp = arith.constant 0.5 : f64
  %z = arith.constant 0.0 : f64
  // expected-error @+1 {{'pulse.square' op duration must be positive, got -20}}
  %w = pulse.square %d, %amp, %z : i64, f64, f64 -> !pulse.waveform
  return %w : !pulse.waveform
}
