// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: not cudaq-pulse-opt --split-input-file --pulse-verify %s 2>&1 | FileCheck %s --check-prefix=UNMARKED
// RUN: not cudaq-pulse-opt --split-input-file --pulse-schedule-alap --pulse-to-qop %s 2>&1 | FileCheck %s --check-prefix=UNBOUND

// A wait whose length is not a constant must be a variable wait, and a
// variable wait is lowered only once its arguments are bound: unbound, it
// stands at its lowest length.

// UNMARKED: error: a wait whose length is not a constant must be marked pulse.variable_wait
// UNBOUND: error: the length of this wait depends on the program's float arguments; bind them (specialize the module) before lowering it

func.func @unmarked(%tau: f64) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %len : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @main(%tau: f64) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.3 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = 0 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d2, %t1 = pulse.drive %d1, %x, %t0
      : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
