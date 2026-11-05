// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of the pulse op verifiers. Waveform arithmetic,
// measurement sources (pulse.is_set) and schedule blocks have their own files.

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

func.func @custom_samples_without_samples() {
  // expected-error @+1 {{'pulse.custom_samples' op samples array must not be empty}}
  %w = pulse.custom_samples [] : !pulse.waveform
  return
}

// -----

func.func @custom_samples_bad_rate() {
  // expected-error @+1 {{'pulse.custom_samples' op rate must be two positive integers}}
  %w = pulse.custom_samples [0.1, 0.2] {rate = array<i64: 0, 1>} : !pulse.waveform
  return
}

// -----

func.func @custom_samples_partial_cycle() {
  // expected-error @+1 {{'pulse.custom_samples' op 3 samples at 2 every 1 cycles do not last a whole number of cycles}}
  %w = pulse.custom_samples [0.1, 0.2, 0.3] {rate = array<i64: 2, 1>} : !pulse.waveform
  return
}

// -----

func.func @custom_samples_longer_than_an_i64() {
  // expected-error @+1 {{'pulse.custom_samples' op 3 samples at 1 every 4611686018427387904 cycles last more cycles than an i64 holds}}
  %w = pulse.custom_samples [0.1, 0.2, 0.3] {rate = array<i64: 1, 4611686018427387904>} : !pulse.waveform
  return
}

// -----

func.func @qvec_of_fewer_results_than_qudits() {
  // expected-error @+1 {{'pulse.qvec_alloc' op allocates 3 qudits, but returns 1 references}}
  %q = pulse.qvec_alloc 3 : !pulse.qref
  return
}

// -----

func.func @qvec_of_three_qudits() {
  %q:3 = pulse.qvec_alloc 3 : !pulse.qref, !pulse.qref, !pulse.qref
  return
}

// -----

func.func @wait_that_changes_the_line_kind(%line: !pulse.drive_line, %dur: !pulse.duration) {
  // expected-error @+1 {{'pulse.wait' op failed to verify that all of {line, updated_line} have same type}}
  %r = pulse.wait %line, %dur : (!pulse.drive_line, !pulse.duration) -> !pulse.readout_line
  return
}

// -----

func.func @sync_of_one_line(%line: !pulse.drive_line) {
  // expected-error @+1 {{'pulse.sync' op sync requires at least 2 lines, got 1}}
  %s = pulse.sync %line : !pulse.drive_line -> !pulse.drive_line
  return
}

// -----

func.func @sync_that_drops_a_line(%a: !pulse.drive_line, %b: !pulse.readout_line) {
  // expected-error @+1 {{'pulse.sync' op number of input lines (2) must match number of output lines (1)}}
  %s = pulse.sync %a, %b : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line
  return
}

// -----

// Each result is its line, of the same kind.
func.func @sync_that_swaps_its_lines(%a: !pulse.drive_line, %b: !pulse.readout_line) {
  // expected-error @+1 {{'pulse.sync' op line 0 is a '!pulse.drive_line', but its result is a '!pulse.readout_line'}}
  %s:2 = pulse.sync %a, %b : !pulse.drive_line, !pulse.readout_line -> !pulse.readout_line, !pulse.drive_line
  return
}

// -----

// A sync may join lines of both kinds.
func.func @sync_of_a_drive_and_a_readout_line(%a: !pulse.drive_line, %b: !pulse.readout_line) {
  %s:2 = pulse.sync %a, %b : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
  return
}

// -----

func.func @outcome_not_from_a_readout(%b: i1) -> i64 {
  %m = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
  // expected-error @+1 {{'pulse.outcome' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast}}
  %k = pulse.outcome %m : !pulse.measurement -> i64
  return %k : i64
}
