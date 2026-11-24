// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// What a coupling line refuses: a tone, a complex waveform, a frame op, a
// readout, a qudit named twice, and a second acquisition of the line.

module attributes {pulse.coupling_lines} {
func.func @tone_on_a_coupling_line(%w : !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{'pulse.drive' op operand #0 must be Physical drive channel (linear resource), but got '!pulse.coupling_line'}}
  %c1, %t1 = pulse.drive %c, %w, %t : !pulse.coupling_line, !pulse.waveform, !pulse.tone -> !pulse.coupling_line, !pulse.tone
  return
}
}

// -----

// A drive line plays at a tone: the op of a coupling line does not take it.
module attributes {pulse.coupling_lines} {
func.func @drive_line_at_baseband(%w : !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{'pulse.drive_baseband' op operand #0 must be Physical coupling channel (linear resource), but got '!pulse.drive_line'}}
  %d1 = pulse.drive_baseband %d, %w : !pulse.drive_line, !pulse.waveform -> !pulse.drive_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @another_line_returned(%w : !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  // expected-error @+1 {{'pulse.drive_baseband' op result #0 must be Physical coupling channel (linear resource), but got '!pulse.drive_line'}}
  %d = pulse.drive_baseband %c, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.drive_line
  return
}
}

// -----

// A pulse on a coupling line, too, is in a module that says so.
func.func @pulse_in_a_module_that_does_not_say(%c : !pulse.coupling_line, %w : !pulse.waveform) {
  // expected-error @+1 {{'pulse.drive_baseband' op is in a module that does not say it holds a coupling line; give the module the unit attribute pulse.coupling_lines}}
  %c1 = pulse.drive_baseband %c, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}

// -----

// The passes look for the ops of a coupling line only in a module that
// says it holds one.
func.func @module_that_does_not_say() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{'pulse.get_coupling_line' op is in a module that does not say it holds a coupling line; give the module the unit attribute pulse.coupling_lines}}
  %c = "pulse.get_coupling_line"(%q) : (!pulse.qudit) -> !pulse.coupling_line
  return
}

// -----

module attributes {pulse.coupling_lines} {
func.func @frame_op_on_a_coupling_line() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %phase = arith.constant 0.5 : f64
  // expected-error @+1 {{'pulse.shift_phase' op operand #0 must be Carrier signal tone (frequency and phase, linear resource), but got '!pulse.coupling_line'}}
  %c1 = "pulse.shift_phase"(%c, %phase) : (!pulse.coupling_line, f64) -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @readout_on_a_coupling_line(%w : !pulse.waveform, %t : !pulse.tone) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  // expected-error @+1 {{'pulse.readout' op operand #0 must be Physical readout channel (linear resource), but got '!pulse.coupling_line'}}
  %c1, %t1, %m = "pulse.readout"(%c, %w, %t) <{mode = 0 : i32}> : (!pulse.coupling_line, !pulse.waveform, !pulse.tone) -> (!pulse.coupling_line, !pulse.tone, !pulse.measurement)
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @no_qudit() {
  // expected-error @+1 {{'pulse.get_coupling_line' op needs at least one qudit: the qudits whose control the line reaches}}
  %c = pulse.get_coupling_line : () -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @qudit_twice() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %again = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{'pulse.get_coupling_line' op names a qudit twice: a coupling line reaches each of its qudits once}}
  %c = pulse.get_coupling_line %q, %again : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @acquired_twice() {
  %w = pulse.custom_samples [0.1, 0.2] : !pulse.waveform
  %cycles = arith.constant 4 : i64
  %len = pulse.duration_from_int %cycles : (i64) -> !pulse.duration
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  // expected-note @+1 {{acquired first here}}
  %p = pulse.get_coupling_line %q0, %q1 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %p1 = pulse.drive_baseband %p, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  // The line of one qudit of the pair is another line.
  %c = pulse.get_coupling_line %q0 : (!pulse.qudit) -> !pulse.coupling_line
  %c1 = pulse.wait %c, %len : (!pulse.coupling_line, !pulse.duration) -> !pulse.coupling_line
  // expected-error @+1 {{this coupling line is already acquired; use the line acquired first}}
  %again = pulse.get_coupling_line %q1, %q0 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %again1 = pulse.drive_baseband %again, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @complex_square() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %duration = arith.constant 40 : i64
  %re = arith.constant 0.3 : f64
  %im = arith.constant 0.1 : f64
  // expected-note @+1 {{this pulse.square may have an imaginary part}}
  %w = pulse.square %duration, %re, %im : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{'pulse.drive_baseband' op plays a real waveform; the imaginary part of this one is not known to be zero}}
  %c1 = pulse.drive_baseband %c, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @drag_in_a_sum() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %beta = arith.constant 0.5 : f64
  %g = pulse.gaussian %duration, %amplitude, %sigma : i64, f64, f64 -> !pulse.waveform
  // expected-note @+1 {{this pulse.drag may have an imaginary part}}
  %d = pulse.drag %duration, %amplitude, %sigma, %beta : i64, f64, f64, f64 -> !pulse.waveform
  %sum = pulse.add %g, %d : !pulse.waveform
  %neg = pulse.neg %sum : !pulse.waveform
  // expected-error @+1 {{'pulse.drive_baseband' op plays a real waveform; the imaginary part of this one is not known to be zero}}
  %c1 = pulse.drive_baseband %c, %neg : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func @waveform_from_outside(%w : !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  // expected-note @-3 {{this func.func may have an imaginary part}}
  // expected-error @+1 {{'pulse.drive_baseband' op plays a real waveform; the imaginary part of this one is not known to be zero}}
  %c1 = pulse.drive_baseband %c, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}

// -----

// Real waveforms pass: a square with no imaginary part, a DRAG pulse with
// beta 0, and sums, real multiples and negations of real waveforms.
module attributes {pulse.coupling_lines} {
func.func @real_waveforms() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %zero = arith.constant 0.0 : f64
  %s = pulse.square %duration, %amplitude, %zero : i64, f64, f64 -> !pulse.waveform
  %d = pulse.drag %duration, %amplitude, %sigma, %zero : i64, f64, f64, f64 -> !pulse.waveform
  %sum = pulse.add %s, %d : !pulse.waveform
  %half = arith.constant -0.5 : f64
  %scaled = pulse.scale %sum, %half : !pulse.waveform, f64 -> !pulse.waveform
  %neg = pulse.neg %scaled : !pulse.waveform
  %c1 = pulse.drive_baseband %c, %neg : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}
