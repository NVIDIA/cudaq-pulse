// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s | FileCheck %s --check-prefix=EXACT --strict-whitespace
// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s --check-prefix=EXACT --strict-whitespace

// A timed op holds its schedule as integers in its properties, printed in
// a <{...}> dictionary before its other attributes in both forms. An op not
// scheduled prints none in the custom form, and no space for it; the generic
// form lists every property, a missing start or length as [] and no periods
// as array<i64>.

// CHECK-LABEL: func.func @scheduled
// CHECK: pulse.drive {{.*}} <{duration_cycles = 40 : i64, period_cycles = array<i64: 60>, start_cycles = 0 : i64}> {frame_phase_offset = 5.000000e-01 : f64}
// CHECK: pulse.wait {{.*}} <{duration_cycles = 20 : i64, period_cycles = array<i64: 60>, start_cycles = 40 : i64}>
// GENERIC-LABEL: sym_name = "scheduled"
// GENERIC: "pulse.drive"({{.*}}) <{duration_cycles = 40 : i64, period_cycles = array<i64: 60>, start_cycles = 0 : i64}> {frame_phase_offset = 5.000000e-01 : f64}
// GENERIC: "pulse.wait"({{.*}}) <{duration_cycles = 20 : i64, period_cycles = array<i64: 60>, start_cycles = 40 : i64}>
func.func @scheduled(%d0: !pulse.drive_line, %t0: !pulse.tone, %w: !pulse.waveform, %idle: !pulse.duration) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %d, %t = scf.for %i = %c0 to %c3 step %c1 iter_args(%d1 = %d0, %t1 = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %d2, %t2 = pulse.drive %d1, %w, %t1 <{duration_cycles = 40 : i64, period_cycles = array<i64: 60>, start_cycles = 0 : i64}> {frame_phase_offset = 0.5 : f64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d3 = pulse.wait %d2, %idle <{duration_cycles = 20 : i64, period_cycles = array<i64: 60>, start_cycles = 40 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d3, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}

// CHECK-LABEL: func.func @unscheduled
// CHECK: pulse.drive %{{[^ ]*}}, %{{[^ ]*}}, %{{[^ ]*}} : !pulse.drive_line
// EXACT-LABEL: func.func @unscheduled
// EXACT: {{^}}    %updated_line, %updated_tone = pulse.drive %arg0, %arg2, %arg1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone{{$}}
// EXACT: {{^}}    %updated_line_0, %updated_tone_1 = pulse.drive %updated_line, %arg2, %updated_tone {frame_phase_offset = 5.000000e-01 : f64} : !pulse.drive_line
// GENERIC-LABEL: sym_name = "unscheduled"
// GENERIC: "pulse.drive"({{.*}}) <{duration_cycles = [], period_cycles = array<i64>, start_cycles = []}> :
func.func @unscheduled(%d0: !pulse.drive_line, %t0: !pulse.tone, %w: !pulse.waveform) {
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %w, %t1 {frame_phase_offset = 0.5 : f64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// The one other inherent attribute of schedule_block (callee) and of
// classical.wait (timeout_ns) shares the dictionary, in its sorted place.

// EXACT-LABEL: func.func @shared_dictionary
// EXACT: {{^}}    %0:2 = pulse.schedule_block alap (%arg0, %arg1, %arg2) <{callee = "measure", duration_cycles = 8 : i64, start_cycles = 2 : i64}> : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {{[{]$}}
// EXACT: {{^}}    %1:2 = pulse.schedule_block alap (%arg0, %0#0, %0#1) <{callee = "measure"}> : (!pulse.measurement
// EXACT: {{^}}    %updated_link, %future = classical.submit %arg3, @est(%arg4) <{start_cycles = 4 : i64}> : (f64) -> (!classical.link, !classical.future){{$}}
// EXACT: {{^}}    %2:2 = classical.wait %future, @est <{start_cycles = 12 : i64, timeout_ns = 2.500000e+02 : f64}> : i1, f64{{$}}
// EXACT: {{^}}    %3:2 = classical.wait %future, @est <{timeout_ns = 2.500000e+02 : f64}> {tag = 1 : i64} : i1, f64{{$}}
// EXACT: {{^}}    %4:2 = classical.wait %future, @est : i1, f64{{$}}
// EXACT: {{^}}    %5 = pulse.shift_phase %0#1, %arg4 <{period_cycles = array<i64: 10, 20>}> : !pulse.tone, f64 -> !pulse.tone{{$}}
classical.callable @est {function_id = 2023343271 : i64, device = 0 : i64,
    args = ["f64"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 1000.0, latency_max_ns = 1500.0, mode = "timeout",
    timeout_ns = 2000.0, submit = "lazy"}
func.func @shared_dictionary(%m0: !pulse.measurement, %d0: !pulse.drive_line, %t0: !pulse.tone, %l0: !classical.link, %x: f64) {
  %d1, %t1 = pulse.schedule_block alap (%m0, %d0, %t0) <{start_cycles = 2 : i64, callee = "measure", duration_cycles = 8 : i64}> : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  ^bb0(%m: !pulse.measurement, %d: !pulse.drive_line, %t: !pulse.tone):
    pulse.yield %d, %t : !pulse.drive_line, !pulse.tone
  }
  %d2, %t2 = pulse.schedule_block alap (%m0, %d1, %t1) <{callee = "measure"}> : (!pulse.measurement, !pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
  ^bb0(%m: !pulse.measurement, %d: !pulse.drive_line, %t: !pulse.tone):
    pulse.yield %d, %t : !pulse.drive_line, !pulse.tone
  }
  %l1, %f = classical.submit %l0, @est(%x) <{start_cycles = 4 : i64}> : (f64) -> (!classical.link, !classical.future)
  %ok, %v = classical.wait %f, @est <{timeout_ns = 250.0 : f64, start_cycles = 12 : i64}> : i1, f64
  %ok2, %v2 = classical.wait %f, @est <{timeout_ns = 250.0 : f64}> {tag = 1 : i64} : i1, f64
  %ok3, %v3 = classical.wait %f, @est : i1, f64
  %t3 = pulse.shift_phase %t1, %x <{period_cycles = array<i64: 10, 20>}> : !pulse.tone, f64 -> !pulse.tone
  return
}
