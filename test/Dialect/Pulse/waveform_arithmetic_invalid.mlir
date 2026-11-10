// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

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

// -----

func.func @sub_of_different_durations() {
  %short = arith.constant 4 : i64
  %long = arith.constant 8 : i64
  %real = arith.constant 0.1 : f64
  %imag = arith.constant 0.0 : f64
  %a = pulse.square %short, %real, %imag : i64, f64, f64 -> !pulse.waveform
  %b = pulse.square %long, %real, %imag : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{'pulse.sub' op operands have durations 4 and 8}}
  %diff = pulse.sub %a, %b : !pulse.waveform
  return
}

// -----

// The length of a sum comes from either operand: 10 cycles here, though its
// first operand's length is not known.
func.func @known_through_the_second_operand(%w: !pulse.waveform) -> !pulse.waveform {
  %d10 = arith.constant 10 : i64
  %d20 = arith.constant 20 : i64
  %a = arith.constant 0.5 : f64
  %z = arith.constant 0.0 : f64
  %w10 = pulse.square %d10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %w20 = pulse.square %d20, %a, %z : i64, f64, f64 -> !pulse.waveform
  %sum1 = pulse.add %w, %w10 : !pulse.waveform
  // expected-error @+1 {{'pulse.add' op operands have durations 10 and 20}}
  %sum2 = pulse.add %sum1, %w20 : !pulse.waveform
  return %sum2 : !pulse.waveform
}
