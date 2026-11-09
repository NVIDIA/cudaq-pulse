// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Two square drives on one line are not fused across a barrier on another
// line: the second plays after it.
// RUN: cudaq-pulse-opt --pulse-fuse-pulses %s | FileCheck %s

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

// CHECK-LABEL: func.func @across_a_variable_wait
// CHECK-COUNT-2: pulse.drive
// CHECK-NOT: fused
func.func @across_a_variable_wait() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %wf = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %e1, %u1 = pulse.drive %e0, %wf, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 0 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %e2, %u2 = pulse.drive %e1, %wf, %u1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
