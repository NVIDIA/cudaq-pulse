// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

// A timed op holds its schedule in its properties: the custom form prints
// them with its other attributes, the generic form as properties.

// CHECK-LABEL: func.func @scheduled
// CHECK: pulse.drive {{.*}} {duration_cycles = 40 : i64, frame_phase_offset = 5.000000e-01 : f64, period_cycles = array<i64: 60>, start_cycles = 0 : i64}
// CHECK: pulse.wait {{.*}} {duration_cycles = 20 : i64, period_cycles = array<i64: 60>, start_cycles = 40 : i64}
// GENERIC-LABEL: sym_name = "scheduled"
// GENERIC: "pulse.drive"({{.*}}) <{duration_cycles = 40 : i64, period_cycles = array<i64: 60>, start_cycles = 0 : i64}> {frame_phase_offset = 5.000000e-01 : f64}
// GENERIC: "pulse.wait"({{.*}}) <{duration_cycles = 20 : i64, period_cycles = array<i64: 60>, start_cycles = 40 : i64}>
func.func @scheduled(%d0: !pulse.drive_line, %t0: !pulse.tone, %w: !pulse.waveform, %idle: !pulse.duration) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %d, %t = scf.for %i = %c0 to %c3 step %c1 iter_args(%d1 = %d0, %t1 = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %d2, %t2 = pulse.drive %d1, %w, %t1 {start_cycles = 0 : i64, duration_cycles = 40 : i64, period_cycles = array<i64: 60>, frame_phase_offset = 0.5 : f64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d3 = pulse.wait %d2, %idle {start_cycles = 40 : i64, duration_cycles = 20 : i64, period_cycles = array<i64: 60>} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d3, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}
