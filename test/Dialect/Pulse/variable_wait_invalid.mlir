// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// The bounds of a variable wait (pulse.variable_wait, pulse.min_cycles, pulse.max_cycles).

func.func @negative_lowest(%tau: f64, %d0: !pulse.drive_line) {
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{'pulse.wait' op pulse.min_cycles must not be negative, got -1}}
  %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = -1 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @highest_below_lowest(%tau: f64, %d0: !pulse.drive_line) {
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{'pulse.wait' op pulse.max_cycles (4) is below pulse.min_cycles (8)}}
  %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = 8 : i64, pulse.max_cycles = 4 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @no_lowest(%tau: f64, %d0: !pulse.drive_line) {
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{'pulse.wait' op a pulse.variable_wait wait needs pulse.min_cycles}}
  %d1 = pulse.wait %d0, %len {pulse.variable_wait}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @bounds_without_variable(%d0: !pulse.drive_line) {
  %n = arith.constant 8 : i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{'pulse.wait' op pulse.min_cycles and pulse.max_cycles apply to a pulse.variable_wait wait}}
  %d1 = pulse.wait %d0, %len {pulse.min_cycles = 0 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @bound_length_outside_bounds(%d0: !pulse.drive_line) {
  %n = arith.constant 5000 : i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{'pulse.wait' op a pulse.variable_wait wait of 5000 cycles is outside its bounds [0, 4000]}}
  %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = 0 : i64, pulse.max_cycles = 4000 : i64}
      : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}
