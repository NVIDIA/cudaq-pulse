// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of the pulse op verifiers, which check an op on its
// own. The values a waveform takes from other ops are checked by
// pulse-verify (verify_waveforms_invalid.mlir); schedule blocks have their
// own file.

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

func.func @qudit_without_an_index() {
  // expected-error @+1 {{'pulse.qudit_alloc' op requires attribute 'qudit'}}
  %q = pulse.qudit_alloc : !pulse.qref
  return
}

// -----

func.func @qudit_of_a_negative_index() {
  // expected-error @+1 {{'pulse.qudit_alloc' op attribute 'qudit' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %q = pulse.qudit_alloc {qudit = -1 : i64} : !pulse.qref
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
