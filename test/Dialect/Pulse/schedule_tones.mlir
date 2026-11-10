// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A tone is busy while a pulse plays on it.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null

// A cross-resonance pulse on line a at qubit b's tone runs between the two
// pulses on line b that the tone is threaded through, though the lines differ.
// CHECK-LABEL: func.func @shared_tone
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 200 : i64, start_cycles = 40 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 240 : i64}
func.func @shared_tone() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c200 = arith.constant 200 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %sx = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %cr = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %b1, %tb1 = pulse.drive %b0, %sx, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %a1, %tb2 = pulse.drive %a0, %cr, %tb1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b2, %tb3 = pulse.drive %b1, %sx, %tb2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// Tone b only changes phase in the loop, so it keeps its time and the loop
// is still scheduled as one pattern.
// CHECK-LABEL: func.func @phase_only_tone
// CHECK: scf.for
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 280 : i64}
func.func @phase_only_tone() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 0.25 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %a1, %ta1, %tb1 = scf.for %i = %lb to %ub step %step iter_args(%a = %a0, %t = %ta, %u = %tb) -> (!pulse.drive_line, !pulse.tone, !pulse.tone) {
    %aa, %tt = pulse.drive %a, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %uu = pulse.shift_phase %u, %phi : !pulse.tone, f64 -> !pulse.tone
    scf.yield %aa, %tt, %uu : !pulse.drive_line, !pulse.tone, !pulse.tone
  }
  %b1, %tb2 = pulse.drive %b0, %x, %tb1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
