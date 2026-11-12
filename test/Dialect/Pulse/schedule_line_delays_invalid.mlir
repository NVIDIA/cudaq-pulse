// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A line delay counts towards the program's length: delayed, the program
// lasts at most 2**60 cycles, as building the module checks it undelayed.
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics

// An 8-cycle drive on a line 2**60 - 8 cycles late ends at 2**60: it fits.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0], delay_cycles = 1152921504606846968>]>} {
func.func @ends_at_2_60() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
}

// -----

// One cycle later it ends past 2**60.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0], delay_cycles = 1152921504606846969>]>} {
// expected-error @+1 {{the program lasts more than 2**60 cycles with its lines delayed (their delay_ns), more than a schedule counts}}
func.func @ends_past_2_60() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
}
