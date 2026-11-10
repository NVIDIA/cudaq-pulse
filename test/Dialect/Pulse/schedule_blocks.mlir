// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A pulse.schedule_block is scheduled on its own with its policy, then placed
// whole by the enclosing schedule and inlined.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=CHECK,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefixes=CHECK,ALAP
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null
// Without the blocks, the same ops are scheduled as one flat program.
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-asap %s | FileCheck %s --check-prefix=FLAT-ASAP
// RUN: cudaq-pulse-opt --pulse-inline-schedule-blocks --pulse-schedule-alap %s | FileCheck %s --check-prefix=FLAT-ALAP
// Scheduling inlines every block.
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefix=INLINED
// INLINED-NOT: pulse.schedule_block
// INLINED-NOT: pulse.yield

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

// An ASAP block on lines b and c: inside it both start together. The ALAP
// parent places the block as late as it can, so it ends with line a; line
// c keeps its place at the start of the block instead of moving to the end.
// CHECK-LABEL: func.func @asap_in_alap
// CHECK: pulse.drive {{.*}}{duration_cycles = 200 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 100 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 100 : i64}
// FLAT-ALAP-LABEL: func.func @asap_in_alap
// FLAT-ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 160 : i64}
func.func @asap_in_alap() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %qc = pulse.qudit_alloc : !pulse.qref
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qref) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qref) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c200 = arith.constant 200 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %x200 = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %a1, %ta1 = pulse.drive %a0, %x200, %ta0 : !d, !w, !t -> !d, !t
  %b1, %tb1, %c1, %tc1 = pulse.schedule_block "asap" (%b0, %tb0, %c0, %tc0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%b: !d, %tb: !t, %c: !d, %tc: !t):
    %b2, %tb2 = pulse.drive %b, %x100, %tb : !d, !w, !t -> !d, !t
    %c2, %tc2 = pulse.drive %c, %x40, %tc : !d, !w, !t -> !d, !t
    pulse.yield %b2, %tb2, %c2, %tc2 : !d, !t, !d, !t
  }
  return
}

// An ALAP block in an ASAP parent: the short line c is right-aligned inside
// the block, and the drive after the block waits for the block's end on c.
// CHECK-LABEL: func.func @alap_in_asap
// CHECK: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 60 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 100 : i64}
// FLAT-ASAP-LABEL: func.func @alap_in_asap
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// FLAT-ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 40 : i64}
func.func @alap_in_asap() {
  %qb = pulse.qudit_alloc : !pulse.qref
  %qc = pulse.qudit_alloc : !pulse.qref
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qref) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %b1, %tb1, %c1, %tc1 = pulse.schedule_block "alap" (%b0, %tb0, %c0, %tc0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%b: !d, %tb: !t, %c: !d, %tc: !t):
    %b2, %tb2 = pulse.drive %b, %x100, %tb : !d, !w, !t -> !d, !t
    %c2, %tc2 = pulse.drive %c, %x40, %tc : !d, !w, !t -> !d, !t
    pulse.yield %b2, %tb2, %c2, %tc2 : !d, !t, !d, !t
  }
  %c3, %tc3 = pulse.drive %c1, %x40, %tc1 : !d, !w, !t -> !d, !t
  return
}

// Three levels: an ASAP block holding an ALAP block holding an ASAP block.
// Each level is placed whole in the one above, so start times add up.
// Level 3 runs x for 40 and y for 20 from its start. Level 2 drives x for
// 100 first and places level 3 at 100. Level 1 drives y for 60, which level
// 2 does not wait for: its first op on y is at 100. The ALAP root places
// level 1 (length 140) to end with the 300-long drive on z.
// CHECK-LABEL: func.func @three_levels
// CHECK: pulse.drive {{.*}}{duration_cycles = 300 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 60 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 100 : i64}
// ASAP: pulse.drive {{.*}}{duration_cycles = 20 : i64, start_cycles = 100 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 60 : i64, start_cycles = 160 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 160 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 260 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 20 : i64, start_cycles = 260 : i64}
func.func @three_levels() {
  %qx = pulse.qudit_alloc : !pulse.qref
  %qy = pulse.qudit_alloc : !pulse.qref
  %qz = pulse.qudit_alloc : !pulse.qref
  %x0, %tx0 = pulse.get_drive_line %qx : (!pulse.qref) -> (!d, !t)
  %y0, %ty0 = pulse.get_drive_line %qy : (!pulse.qref) -> (!d, !t)
  %z0, %tz0 = pulse.get_drive_line %qz : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c300 = arith.constant 300 : i64
  %c100 = arith.constant 100 : i64
  %c60 = arith.constant 60 : i64
  %c40 = arith.constant 40 : i64
  %c20 = arith.constant 20 : i64
  %w300 = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !w
  %w100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %w60 = pulse.square %c60, %amp, %zero : i64, f64, f64 -> !w
  %w40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %w20 = pulse.square %c20, %amp, %zero : i64, f64, f64 -> !w
  %z1, %tz1 = pulse.drive %z0, %w300, %tz0 : !d, !w, !t -> !d, !t
  %x1, %tx1, %y1, %ty1 = pulse.schedule_block "asap" (%x0, %tx0, %y0, %ty0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
  ^bb0(%x: !d, %tx: !t, %y: !d, %ty: !t):
    %y2, %ty2 = pulse.drive %y, %w60, %ty : !d, !w, !t -> !d, !t
    %x3, %tx3, %y3, %ty3 = pulse.schedule_block "alap" (%x, %tx, %y2, %ty2) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
    ^bb0(%xx: !d, %txx: !t, %yy: !d, %tyy: !t):
      %x4, %tx4 = pulse.drive %xx, %w100, %txx : !d, !w, !t -> !d, !t
      %x5, %tx5, %y5, %ty5 = pulse.schedule_block "asap" (%x4, %tx4, %yy, %tyy) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
      ^bb0(%xxx: !d, %txxx: !t, %yyy: !d, %tyyy: !t):
        %x6, %tx6 = pulse.drive %xxx, %w40, %txxx : !d, !w, !t -> !d, !t
        %y6, %ty6 = pulse.drive %yyy, %w20, %tyyy : !d, !w, !t -> !d, !t
        pulse.yield %x6, %tx6, %y6, %ty6 : !d, !t, !d, !t
      }
      pulse.yield %x5, %tx5, %y5, %ty5 : !d, !t, !d, !t
    }
    pulse.yield %x3, %tx3, %y3, %ty3 : !d, !t, !d, !t
  }
  return
}

// A block in a loop has the same shape in every iteration, so the loop stays
// affine: the block's ops take the loop's period, followed by the period of
// the loop inside the block.
// CHECK-LABEL: func.func @block_in_loop
// CHECK: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 0 : i64}
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: pulse.drive {{.*}}{duration_cycles = 40 : i64, period_cycles = array<i64: 100, 40>, start_cycles = 100 : i64}
// CHECK: pulse.wait {{.*}}{duration_cycles = 20 : i64, period_cycles = array<i64: 100>, start_cycles = 180 : i64}
func.func @block_in_loop() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %c20 = arith.constant 20 : i64
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %w20 = pulse.duration_from_int %c20 : (i64) -> !pulse.duration
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %two = arith.constant 2 : index
  %step = arith.constant 1 : index
  %d1, %t1 = pulse.drive %d0, %x100, %t0 : !d, !w, !t -> !d, !t
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d1, %t = %t1) -> (!d, !t) {
    %d3, %t3 = pulse.schedule_block "asap" (%d, %t) : (!d, !t) -> (!d, !t) {
    ^bb0(%bd: !d, %bt: !t):
      %d4, %t4 = scf.for %j = %lb to %two step %step iter_args(%e = %bd, %u = %bt) -> (!d, !t) {
        %e1, %u1 = pulse.drive %e, %x40, %u : !d, !w, !t -> !d, !t
        scf.yield %e1, %u1 : !d, !t
      }
      %d5 = pulse.wait %d4, %w20 : (!d, !pulse.duration) -> !d
      pulse.yield %d5, %t4 : !d, !t
    }
    scf.yield %d3, %t3 : !d, !t
  }
  return
}

// A block in a measurement-conditioned arm starts once the measurement is
// available; inside it, the ALAP policy right-aligns the short line.
// CHECK-LABEL: func.func @block_in_if
// CHECK: pulse.readout {{.*}}{duration_cycles = 400 : i64, start_cycles = 0 : i64}
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 400 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 460 : i64}
func.func @block_in_if() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %r0, %tr0 = pulse.get_readout_line %qa : (!pulse.qref) -> (!r, !t)
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qref) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, "discriminate" : !r, !w, !t -> !r, !t, !m
  %bit = pulse.is_set %m : !m -> i1
  %a1, %ta1, %b1, %tb1 = scf.if %bit -> (!d, !t, !d, !t) {
    %a2, %ta2, %b2, %tb2 = pulse.schedule_block "alap" (%a0, %ta0, %b0, %tb0) : (!d, !t, !d, !t) -> (!d, !t, !d, !t) {
    ^bb0(%a: !d, %ta: !t, %b: !d, %tb: !t):
      %a3, %ta3 = pulse.drive %a, %x100, %ta : !d, !w, !t -> !d, !t
      %b3, %tb3 = pulse.drive %b, %x40, %tb : !d, !w, !t -> !d, !t
      pulse.yield %a3, %ta3, %b3, %tb3 : !d, !t, !d, !t
    }
    scf.yield %a2, %ta2, %b2, %tb2 : !d, !t, !d, !t
  } else {
    scf.yield %a0, %ta0, %b0, %tb0 : !d, !t, !d, !t
  }
  return
}

// A measurement-conditioned branch inside a block: the measurement is an
// operand that only the branch, 40 into the block, waits for. The block
// starts so that the branch begins when the readout is done.
// CHECK-LABEL: func.func @if_in_block
// CHECK: pulse.readout {{.*}}{duration_cycles = 400 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 360 : i64}
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}{duration_cycles = 100 : i64, start_cycles = 400 : i64}
func.func @if_in_block() {
  %q = pulse.qudit_alloc : !pulse.qref
  %r0, %tr0 = pulse.get_readout_line %q : (!pulse.qref) -> (!r, !t)
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c400 = arith.constant 400 : i64
  %c100 = arith.constant 100 : i64
  %c40 = arith.constant 40 : i64
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %r1, %tr1, %m = pulse.readout %r0, %ro, %tr0, "discriminate" : !r, !w, !t -> !r, !t, !m
  %d1, %t1 = pulse.schedule_block "asap" (%m, %d0, %t0) : (!m, !d, !t) -> (!d, !t) {
  ^bb0(%bm: !m, %d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x40, %t : !d, !w, !t -> !d, !t
    %bit = pulse.is_set %bm : !m -> i1
    %d3, %t3 = scf.if %bit -> (!d, !t) {
      %d4, %t4 = pulse.drive %d2, %x100, %t2 : !d, !w, !t -> !d, !t
      scf.yield %d4, %t4 : !d, !t
    } else {
      scf.yield %d2, %t2 : !d, !t
    }
    pulse.yield %d3, %t3 : !d, !t
  }
  return
}

// Tones crossing the boundary. Tone b is busy with the cross-resonance pulse
// on line a until 300, so the block that plays on it starts then. Tone c
// only changes phase in the block, so it passes through: the drive on line c
// after the block does not wait for the block.
// CHECK-LABEL: func.func @tones_across_blocks
// CHECK: pulse.drive {{.*}}{duration_cycles = 300 : i64, start_cycles = 0 : i64}
// CHECK: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 300 : i64}
// CHECK: pulse.shift_phase
// ASAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 0 : i64}
// ALAP: pulse.drive {{.*}}{duration_cycles = 40 : i64, start_cycles = 300 : i64}
func.func @tones_across_blocks() {
  %qa = pulse.qudit_alloc : !pulse.qref
  %qb = pulse.qudit_alloc : !pulse.qref
  %qc = pulse.qudit_alloc : !pulse.qref
  %a0, %ta0 = pulse.get_drive_line %qa : (!pulse.qref) -> (!d, !t)
  %b0, %tb0 = pulse.get_drive_line %qb : (!pulse.qref) -> (!d, !t)
  %c0, %tc0 = pulse.get_drive_line %qc : (!pulse.qref) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %phi = arith.constant 0.5 : f64
  %c300 = arith.constant 300 : i64
  %c40 = arith.constant 40 : i64
  %cr = pulse.square %c300, %amp, %zero : i64, f64, f64 -> !w
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %a1, %tb1 = pulse.drive %a0, %cr, %tb0 : !d, !w, !t -> !d, !t
  %b1, %tb2, %tc1 = pulse.schedule_block "asap" (%b0, %tb1, %tc0) : (!d, !t, !t) -> (!d, !t, !t) {
  ^bb0(%b: !d, %tb: !t, %tc: !t):
    %b2, %tb3 = pulse.drive %b, %x40, %tb : !d, !w, !t -> !d, !t
    %tc2 = pulse.shift_phase %tc, %phi : !t, f64 -> !t
    pulse.yield %b2, %tb3, %tc2 : !d, !t, !t
  }
  %c1, %tc3 = pulse.drive %c0, %x40, %tc1 : !d, !w, !t -> !d, !t
  return
}

// A sync inside a block aligns the lines it names; the block waits for both.
// Line a is busy until 200, so the block, whose readout follows the sync,
// starts at 200 even though line r is free.
// CHECK-LABEL: func.func @sync_in_block
// CHECK: pulse.drive {{.*}}{duration_cycles = 200 : i64, start_cycles = 0 : i64}
// CHECK: pulse.readout {{.*}}{duration_cycles = 400 : i64, start_cycles = 200 : i64}
func.func @sync_in_block() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!d, !t)
  %r0, %tr0 = pulse.get_readout_line %q : (!pulse.qref) -> (!r, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c200 = arith.constant 200 : i64
  %c400 = arith.constant 400 : i64
  %x200 = pulse.square %c200, %amp, %zero : i64, f64, f64 -> !w
  %ro = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d0, %x200, %t0 : !d, !w, !t -> !d, !t
  %d2, %r2, %tr2, %m = pulse.schedule_block "alap" (%d1, %r0, %tr0) : (!d, !r, !t) -> (!d, !r, !t, !m) {
  ^bb0(%d: !d, %r: !r, %tr: !t):
    %ds, %rs = pulse.sync %d, %r : !d, !r -> !d, !r
    %r3, %tr3, %mm = pulse.readout %rs, %ro, %tr, "discriminate" : !r, !w, !t -> !r, !t, !m
    pulse.yield %ds, %r3, %tr3, %mm : !d, !r, !t, !m
  }
  return
}
