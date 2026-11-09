// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A controller that blocks on calls (pulse.blocks_on_calls): each call holds
// every line from its submission until its answer is due. At 2 GHz a
// 1500 ns latency is 3000 cycles.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check --pulse-verify %s > /dev/null

module attributes {pulse.clock_ghz = 2.0 : f64, pulse.blocks_on_calls} {

classical.callable @decode {function_id = 3007678287 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 500.0,
    latency_nominal_ns = 900.0, latency_max_ns = 1500.0, mode = "bounded",
    submit = "eager"}

// The drive on qudit 1 ends before the call goes out at 300 (under ALAP, just
// before it); the one after it in program order waits until the answer is due
// at 3301, though nothing it plays depends on the answer.
// CHECK-LABEL: func.func @held
// ASAP: pulse.drive {{.*}}start_cycles = 0 : i64
// ALAP: pulse.drive {{.*}}start_cycles = 260 : i64
// CHECK: pulse.readout {{.*}}start_cycles = 0
// CHECK: classical.submit {{.*}}start_cycles = 300 : i64}
// CHECK: pulse.drive {{.*}}start_cycles = 3301 : i64
// CHECK: classical.wait {{.*}}start_cycles = 3301 : i64}
func.func @held() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %f = classical.submit %l0, @decode(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %fix = classical.wait %f, @decode : i1
  return
}

}
