// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// Calibrated line delays: qudit 0's drive line starts everything 3 cycles
// later than scheduled, qudit 1's readout line 5; the other lines keep their
// times, and the sync still meets where the schedule put it.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP

module attributes {pulse.line_delays = {drive = array<i64: 3, 0>, readout = array<i64: 0, 5>}} {

// CHECK-LABEL: func.func @delayed_lines
// ASAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 3 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 4 : i64, start_cycles = 0 : i64}
// ASAP: pulse.wait {{.*}}{duration_cycles = 2 : i64, start_cycles = 11 : i64}
// ASAP: pulse.readout {{.*}}{duration_cycles = 6 : i64, start_cycles = 15 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 3 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 4 : i64, start_cycles = 12 : i64}
// ALAP: pulse.wait {{.*}}{duration_cycles = 2 : i64, start_cycles = 11 : i64}
// ALAP: pulse.readout {{.*}}{duration_cycles = 6 : i64, start_cycles = 15 : i64}
func.func @delayed_lines() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %a0, %ta = pulse.get_drive_line %qa {qubit = 0 : i64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb {qubit = 1 : i64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %tr = pulse.get_readout_line %qb {qubit = 1 : i64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %c6 = arith.constant 6 : i64
  %c4 = arith.constant 4 : i64
  %c2 = arith.constant 2 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w8 = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w6 = pulse.square %c6, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w4 = pulse.square %c4, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %a1, %ta1 = pulse.drive %a0, %w8, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b0, %w4, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %dur = pulse.duration_from_int %c2 : (i64) -> !pulse.duration
  %a2 = pulse.wait %a1, %dur : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %a3, %r1 = pulse.sync %a2, %r0 : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
  %r2, %tr1, %m = pulse.readout %r1, %w6, %tr, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}

}
