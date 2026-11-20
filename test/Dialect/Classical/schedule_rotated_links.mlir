// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Links handed from one iteration argument to another each trip, as tones
// are in Pulse/schedule_rotated_tones.mlir: a call after the loop on a link
// it hands out waits for the loop's last call on that link, as in the
// unrolled loop. Three links of one device, so pulse-classical-check
// rejects both forms (the calls after the loop start before the loop's
// last); this checks only the schedule.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefix=SEQ
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefix=SEQ
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefix=RCP
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' %s | FileCheck %s --check-prefix=RCP

!l = !classical.link
!r = !pulse.readout_line
!t = !pulse.tone
!m = !pulse.measurement
!w = !pulse.waveform

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000>,
                   classical.links = {"0" = {fixed_ns = 20.0 : f64, slots = 8 : i64}}} {

classical.callable @ask {dispatch_key = 1 : i64, device = 0 : i64,
    args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// Each trip sends the last trip's measurement on the link of the first
// argument (in an inner loop of one trip), then reads out; the links
// rotate. 7 trips, a call each 40 from 40: link 2 is last used in trip 4
// (ending at 240), link 3 in trip 5 (280), link 1 in trip 6 (320).
// SEQ-LABEL: func.func @three_links_rotated
// SEQ: scf.yield
// SEQ-NEXT: }
// SEQ: classical.submit {{.*}}start_cycles = 240 : i64
// SEQ-NEXT: classical.submit {{.*}}start_cycles = 280 : i64
// SEQ-NEXT: classical.submit {{.*}}start_cycles = 320 : i64
// RCP-LABEL: func.func @three_links_rotated
// RCP: scf.yield
// RCP-NEXT: }
// RCP: classical.submit {{.*}}start_cycles = 320 : i64
// RCP-NEXT: classical.submit {{.*}}start_cycles = 320 : i64
// RCP-NEXT: classical.submit {{.*}}start_cycles = 320 : i64
func.func @three_links_rotated() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!r, !t)
  %q9 = pulse.qudit_alloc {qudit = 9 : i64} : !pulse.qudit
  %r9, %rt9 = pulse.get_readout_line %q9 : (!pulse.qudit) -> (!r, !t)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %k0 = classical.get_link {device = 0 : i64} : !l
  %k1 = classical.get_link {device = 0 : i64} : !l
  %k2 = classical.get_link {device = 0 : i64} : !l
  %r1, %rt1, %m0 = pulse.readout %r0, %ro, %rt0, iq : !r, !w, !t -> !r, !t, !m
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %st = arith.constant 1 : index
  %res:6 = scf.for %i = %lb to %ub step %st
      iter_args(%r = %r1, %rt = %rt1, %mp = %m0, %a0 = %k0, %a1 = %k1, %a2 = %k2)
      -> (!r, !t, !m, !l, !l, !l) {
    %y = scf.for %j = %lb to %st step %st iter_args(%x = %a0) -> (!l) {
      %s, %f = classical.submit %x, @ask(%mp) : (!m) -> (!l, !classical.future)
      scf.yield %s : !l
    }
    %rn, %rtn, %m = pulse.readout %r, %ro, %rt, iq : !r, !w, !t -> !r, !t, !m
    scf.yield %rn, %rtn, %m, %a1, %a2, %y : !r, !t, !m, !l, !l, !l
  }
  %r9n, %rt9n, %m9 = pulse.readout %r9, %ro, %rt9, iq : !r, !w, !t -> !r, !t, !m
  %e0, %f0 = classical.submit %res#3, @ask(%m9) : (!m) -> (!l, !classical.future)
  %e1, %f1 = classical.submit %res#4, @ask(%m9) : (!m) -> (!l, !classical.future)
  %e2, %f2 = classical.submit %res#5, @ask(%m9) : (!m) -> (!l, !classical.future)
  return
}

}
