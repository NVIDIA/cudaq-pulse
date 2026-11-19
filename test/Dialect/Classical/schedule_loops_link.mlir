// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loops one after another on a link: a loop whose last iterations are not
// simulated still holds the slots they hold, so the next loop's first
// requests wait for them, under every policy.
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-classical-check %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-classical-check %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-rcp --pulse-classical-check %s > /dev/null
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' --pulse-classical-check %s > /dev/null

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000>,
                   classical.links = {"0" = {fixed_ns = 4.0 : f64, slots = 1 : i64},
                                      "1" = {fixed_ns = 4.0 : f64, slots = 6 : i64}}} {

classical.callable @ask {dispatch_key = 1 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}
classical.callable @ask_wide {dispatch_key = 2 : i64, device = 1 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}
classical.callable @ask_lazy {dispatch_key = 3 : i64, device = 1 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "lazy", dispatch_key_override}

// Rounds branching on their answer, then readouts whose requests nobody
// awaits: on one slot the second loop's first request waits for the first
// loop's last answer to be due.
func.func @answers_then_unawaited() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c16 = arith.constant 16 : i64
  %c4 = arith.constant 4 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c16, %a, %z : i64, f64, f64 -> !pulse.waveform
  %y = pulse.square %c4, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %lb = arith.constant 0 : index
  %ub = arith.constant 6 : index
  %step = arith.constant 1 : index
  %d1, %t1, %r1, %rt1, %l1 = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %r = %r0, %rt = %rt0, %l = %l0)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %ds, %rs = pulse.sync %d, %r : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
    %rn, %rtn, %m = pulse.readout %rs, %ro, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %ln, %f = classical.submit %l, @ask(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %fix = classical.wait %f, @ask : i1
    %dn, %tn = scf.if %fix -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %ds, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %ds, %t : !pulse.drive_line, !pulse.tone
    }
    scf.yield %dn, %tn, %rn, %rtn, %ln : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  %d2, %t2, %r2, %rt2, %l2 = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d1, %t = %t1, %r = %r1, %rt = %rt1, %l = %l1)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %rn, %rtn, %m = pulse.readout %r, %ro, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %ln, %f = classical.submit %l, @ask(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %dn, %tn = pulse.drive %d, %y, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %dn, %tn, %rn, %rtn, %ln : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  return
}

// Requests nobody awaits, then lazy ones on six slots: under alap the lazy
// loop is placed first, backward, and the requests before it need the
// slots its first iterations leave them.
func.func @unawaited_then_lazy() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c30 = arith.constant 30 : i64
  %c6 = arith.constant 6 : i64
  %c4 = arith.constant 4 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %ro2 = pulse.square %c30, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c6, %a, %z : i64, f64, f64 -> !pulse.waveform
  %y = pulse.square %c4, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l0 = classical.get_link {device = 1 : i64} : !classical.link
  %lb = arith.constant 0 : index
  %ub = arith.constant 6 : index
  %step = arith.constant 1 : index
  %d1, %t1, %r1, %rt1, %l1 = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d0, %t = %t0, %r = %r0, %rt = %rt0, %l = %l0)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %rn, %rtn, %m = pulse.readout %r, %ro, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %ln, %f = classical.submit %l, @ask_wide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %dn, %tn = pulse.drive %d, %y, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %dn, %tn, %rn, %rtn, %ln : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  %d2, %t2, %r2, %rt2, %l2 = scf.for %i = %lb to %ub step %step
      iter_args(%d = %d1, %t = %t1, %r = %r1, %rt = %rt1, %l = %l1)
      -> (!pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link) {
    %rn, %rtn, %m = pulse.readout %r, %ro2, %rt, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %ln, %f = classical.submit %l, @ask_lazy(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
    %fix = classical.wait %f, @ask_lazy : i1
    %dn, %tn = scf.if %fix -> (!pulse.drive_line, !pulse.tone) {
      %dd, %tt = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %dd, %tt : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    }
    scf.yield %dn, %tn, %rn, %rtn, %ln : !pulse.drive_line, !pulse.tone, !pulse.readout_line, !pulse.tone, !classical.link
  }
  return
}

}
