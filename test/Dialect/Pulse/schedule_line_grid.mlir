// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Lines whose controllers start operations every few program cycles: qudit
// 0's drive line every 4, qudit 1's every 3. Every operation starts on its
// line's grid, and a sync meets on the grid of every line it syncs.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null

module attributes {pulse.line_grid = 1 : i64, pulse.drive_line_grid = array<i64: 4, 3>} {

// The sync after 8 and 6 cycles meets at 12, the first time on both grids.
// Under ALAP, the 3-cycle drive on qudit 1 ends early, at 18, the latest
// start on its grid.
// CHECK-LABEL: func.func @sync_on_common_grid
// ASAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 6 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 12 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 3 : i64, start_cycles = 12 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 4 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 6 : i64, start_cycles = 6 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 12 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 3 : i64, start_cycles = 15 : i64}
func.func @sync_on_common_grid() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %qb = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %b0, %tb = pulse.get_drive_line %qb : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %c6 = arith.constant 6 : i64
  %c3 = arith.constant 3 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w8 = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w6 = pulse.square %c6, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %w3 = pulse.square %c3, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %a1, %ta1 = pulse.drive %a0, %w8, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b0, %w6, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %a2, %b2 = pulse.sync %a1, %b1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  %a3, %ta2 = pulse.drive %a2, %w8, %ta1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b3, %tb2 = pulse.drive %b2, %w3, %tb1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A 6-cycle drive on a 4-cycle grid repeats every 8 cycles.
// CHECK-LABEL: func.func @loop_on_grid
// CHECK: scf.for
// CHECK: pulse.drive {{.*}}{duration_cycles = 6 : i64, period_cycles = array<i64: 8>, start_cycles = 0 : i64}
func.func @loop_on_grid() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c6 = arith.constant 6 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w6 = pulse.square %c6, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 1 : index
  %a1, %ta1 = scf.for %i = %lb to %ub step %step iter_args(%a = %a0, %t = %ta) -> (!pulse.drive_line, !pulse.tone) {
    %aa, %tt = pulse.drive %a, %w6, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %aa, %tt : !pulse.drive_line, !pulse.tone
  }
  return
}

// Samples made for qudit 0's controller, one every 4 cycles: 3 samples last
// 12 cycles.
// CHECK-LABEL: func.func @samples_at_the_lines_rate
// CHECK: pulse.drive {{.*}}{duration_cycles = 12 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 8 : i64, start_cycles = 12 : i64}
func.func @samples_at_the_lines_rate() {
  %qa = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %a0, %ta = pulse.get_drive_line %qa : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %w = pulse.custom_samples [0.1, 0.2, 0.3] {rate = array<i64: 1, 4>} : !pulse.waveform
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w8 = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %a1, %ta1 = pulse.drive %a0, %w, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %a2, %ta2 = pulse.drive %a1, %w8, %ta1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
