// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// One case per diagnostic of pulse-verify, and the programs it must accept.

// A qudit has one drive line: acquiring it twice on one path would give two
// versions of the same line.
func.func @drive_line_acquired_twice() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired; use the line acquired first}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b, %x, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

func.func @readout_line_acquired_twice(%q: !pulse.qref) {
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  // expected-error @+1 {{this qudit's readout line is already acquired}}
  %b, %tb = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %s:2 = pulse.sync %a, %b : !pulse.readout_line, !pulse.readout_line -> !pulse.readout_line, !pulse.readout_line
  return
}

// -----

// Its drive and readout lines are two lines, and so are two qudits' drive
// lines.
func.func @one_line_of_each_kind(%q: !pulse.qref, %p: !pulse.qref) {
  %d, %td = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r, %tr = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %e, %te = pulse.get_drive_line %p : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %s:3 = pulse.sync %d, %r, %e : !pulse.drive_line, !pulse.readout_line, !pulse.drive_line -> !pulse.drive_line, !pulse.readout_line, !pulse.drive_line
  return
}

// -----

// The timing check keys a line on its qudit and kind, so pulses on two
// versions of one line overlap.
func.func @overlap_across_acquisitions() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta {duration_cycles = 40 : i64, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // expected-error @+1 {{operation overlaps or precedes its physical-line predecessor: start 0 < prior end 40}}
  %b1, %tb1 = pulse.drive %b, %x, %tb {duration_cycles = 40 : i64, start_cycles = 0 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
