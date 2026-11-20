// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A loop that hands tones from one iteration argument to another (tones
// rotated or swapped each trip, played in the loop or in an inner loop)
// holds a different tone in each argument every trip: under every policy, a drive on a tone the loop hands out waits
// for the loop's last use of that tone, and one before the loop ends before
// the loop's first.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s --check-prefixes=EARLY,ASAP
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s --check-prefixes=EARLY,RCP
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s --check-prefix=LATE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' %s | FileCheck %s --check-prefix=LATE
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap max-drives=3' %s | FileCheck %s --check-prefix=CAPPED
// RUN: cudaq-pulse-opt --pulse-schedule-asap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-rcp --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt --pulse-schedule-alap --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap' --pulse-verify %s > /dev/null
// RUN: cudaq-pulse-opt '--pulse-schedule-rcp=alap max-drives=3' --pulse-verify %s > /dev/null

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Three tones rotated through the loop's arguments, 7 trips, the first
// played each trip: tone 2 is last played in trip 4 (ending at 200), tone 3
// in trip 5 (240), tone 1 in trip 6 (280). Before, asap started the drive
// on tone 2 at 0, while the loop still played it.
// EARLY-LABEL: func.func @three_tones_rotated
// EARLY: scf.yield
// EARLY-NEXT: }
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 200 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 240 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-LABEL: func.func @three_tones_rotated
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
func.func @three_tones_rotated() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %q3 = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %d3, %t3 = pulse.get_drive_line %q3 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %st = arith.constant 1 : index
  %l, %a, %b, %c = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t1, %b1 = %t2, %c1 = %t3) -> (!d, !t, !t, !t) {
    %l2, %a2 = pulse.drive %l1, %x40, %a1 : !d, !w, !t -> !d, !t
    scf.yield %l2, %b1, %c1, %a2 : !d, !t, !t, !t
  }
  %e1, %f1 = pulse.drive %d1, %x40, %a : !d, !w, !t -> !d, !t
  %e2, %f2 = pulse.drive %d2, %x40, %b : !d, !w, !t -> !d, !t
  %e3, %f3 = pulse.drive %d3, %x40, %c : !d, !w, !t -> !d, !t
  return
}

// Two tones swapped each trip, 7 trips, the second busy until 40: tone 2
// is last played in trip 5 (ending at 240). Before, asap started the drive
// on it at 40.
// EARLY-LABEL: func.func @two_tones_swapped
// EARLY: scf.yield
// EARLY-NEXT: }
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 240 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-LABEL: func.func @two_tones_swapped
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
func.func @two_tones_swapped() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %p2, %u2 = pulse.drive %d2, %x40, %t2 : !d, !w, !t -> !d, !t
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %st = arith.constant 1 : index
  %l, %a, %b = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t1, %b1 = %u2) -> (!d, !t, !t) {
    %l2, %a2 = pulse.drive %l1, %x40, %a1 : !d, !w, !t -> !d, !t
    scf.yield %l2, %b1, %a2 : !d, !t, !t
  }
  %e1, %f1 = pulse.drive %d1, %x40, %a : !d, !w, !t -> !d, !t
  %e2, %f2 = pulse.drive %p2, %x40, %b : !d, !w, !t -> !d, !t
  return
}

// Four tones rotated, 5 trips, the fourth busy until 140: under asap and
// rcp the last two trips do not repeat and are peeled off after the loop.
// The loop's results are the tones it still holds (4, 1, 2, 3), each passed
// to the peeled trips and the drives after as its own. Before, every drive
// after the loop took the last trip's tone (4 drives on one tone at 220),
// and alap started the drive before the loop on tone 4 at 80, overlapping
// the loop's trip 3 on it (140..180). alap_rcp at three drives per line
// peels them too; before, it also gave every drive after the loop the last
// trip's tone.
// CAPPED-LABEL: func.func @four_tones_rotated_peeled
// CAPPED: %[[R:.*]]:5 = scf.for {{.*}} to %c3
// CAPPED: }
// CAPPED: %[[L3:.*]], %[[T3:.*]] = pulse.drive %[[R]]#0, %{{.*}}, %[[R]]#1 <{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// CAPPED: %[[L4:.*]], %[[T4:.*]] = pulse.drive %[[L3]], %{{.*}}, %[[R]]#2 <{duration_cycles = 40 : i64, start_cycles = 180 : i64}>
// CAPPED: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#3 <{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// CAPPED: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#4 <{duration_cycles = 40 : i64, start_cycles = 180 : i64}>
// CAPPED: pulse.drive %{{.*}}, %{{.*}}, %[[T3]] <{duration_cycles = 40 : i64, start_cycles = 180 : i64}>
// CAPPED: pulse.drive %{{.*}}, %{{.*}}, %[[T4]] <{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
// EARLY-LABEL: func.func @four_tones_rotated_peeled
// EARLY: %[[R:.*]]:5 = scf.for {{.*}} to %c3
// EARLY: }
// EARLY: %[[L3:.*]], %[[T3:.*]] = pulse.drive %[[R]]#0, %{{.*}}, %[[R]]#1 <{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// EARLY: %[[L4:.*]], %[[T4:.*]] = pulse.drive %[[L3]], %{{.*}}, %[[R]]#2 <{duration_cycles = 40 : i64, start_cycles = 180 : i64}>
// ASAP: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#3 <{duration_cycles = 40 : i64, start_cycles = 80 : i64}>
// ASAP: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#4 <{duration_cycles = 40 : i64, start_cycles = 120 : i64}>
// RCP: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#3 <{duration_cycles = 40 : i64, start_cycles = 120 : i64}>
// RCP: pulse.drive %{{.*}}, %{{.*}}, %[[R]]#4 <{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// EARLY: pulse.drive %{{.*}}, %{{.*}}, %[[T3]] <{duration_cycles = 40 : i64, start_cycles = 180 : i64}>
// EARLY: pulse.drive %{{.*}}, %{{.*}}, %[[T4]] <{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
// LATE-LABEL: func.func @four_tones_rotated_peeled
// LATE: pulse.drive {{.*}}<{duration_cycles = 140 : i64, start_cycles = 0 : i64}>
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 220 : i64}>
func.func @four_tones_rotated_peeled() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %q3 = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
  %q4 = pulse.qudit_alloc {qudit = 4 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %d3, %t3 = pulse.get_drive_line %q3 : (!pulse.qudit) -> (!d, !t)
  %d4, %t4 = pulse.get_drive_line %q4 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %c140 = arith.constant 140 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %x140 = pulse.square %c140, %amp, %zero : i64, f64, f64 -> !w
  %p4, %u4 = pulse.drive %d4, %x140, %t4 : !d, !w, !t -> !d, !t
  %lb = arith.constant 0 : index
  %ub = arith.constant 5 : index
  %st = arith.constant 1 : index
  %l, %a, %b, %c, %e = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t1, %b1 = %t2, %c1 = %t3, %e1 = %u4) -> (!d, !t, !t, !t, !t) {
    %l2, %a2 = pulse.drive %l1, %x40, %a1 : !d, !w, !t -> !d, !t
    scf.yield %l2, %b1, %c1, %e1, %a2 : !d, !t, !t, !t, !t
  }
  %g1, %f1 = pulse.drive %d1, %x40, %a : !d, !w, !t -> !d, !t
  %g2, %f2 = pulse.drive %d2, %x40, %b : !d, !w, !t -> !d, !t
  %g3, %f3 = pulse.drive %d3, %x40, %c : !d, !w, !t -> !d, !t
  %g4, %f4 = pulse.drive %p4, %x40, %e : !d, !w, !t -> !d, !t
  return
}

// Three tones rotated, 3 trips, the third driven for 100 before the loop,
// which plays it in trip 2. Before, alap started that drive at 40, over the
// loop's trip 2 on the tone (100..140).
// EARLY-LABEL: func.func @tone_rotated_after_a_drive
// EARLY: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 0 : i64}>
// EARLY: scf.yield
// EARLY-NEXT: }
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 80 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// LATE-LABEL: func.func @tone_rotated_after_a_drive
// LATE: pulse.drive {{.*}}<{duration_cycles = 100 : i64, start_cycles = 0 : i64}>
// LATE: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 20 : i64}>
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 140 : i64}>
func.func @tone_rotated_after_a_drive() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c20 = arith.constant 20 : i64
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %x100 = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !w
  %w20 = pulse.duration_from_int %c20 : (i64) -> !pulse.duration
  %p2, %u2 = pulse.drive %d2, %x100, %t2 : !d, !w, !t -> !d, !t
  %lb = arith.constant 0 : index
  %ub = arith.constant 3 : index
  %st = arith.constant 1 : index
  %l, %a, %b, %c = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t0, %b1 = %t1, %c1 = %u2) -> (!d, !t, !t, !t) {
    %l2, %a2 = pulse.drive %l1, %x40, %a1 : !d, !w, !t -> !d, !t
    scf.yield %l2, %b1, %c1, %a2 : !d, !t, !t, !t
  }
  %g0, %f0 = pulse.drive %l, %x40, %a : !d, !w, !t -> !d, !t
  %g1, %f1 = pulse.drive %d1, %x40, %b : !d, !w, !t -> !d, !t
  %g2, %f2 = pulse.drive %p2, %x40, %c : !d, !w, !t -> !d, !t
  return
}

// As @three_tones_rotated, the drive in an inner loop of one trip: the
// outer loop hands on the inner loop's result, which is the tone the inner
// loop is given. Before, asap started the drive on tone 2 at 0.
// EARLY-LABEL: func.func @three_tones_rotated_through_an_inner_loop
// EARLY: scf.yield
// EARLY: scf.yield
// EARLY-NEXT: }
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 200 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 240 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-LABEL: func.func @three_tones_rotated_through_an_inner_loop
// LATE: scf.yield
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 280 : i64}>
func.func @three_tones_rotated_through_an_inner_loop() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %q3 = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %d3, %t3 = pulse.get_drive_line %q3 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %one = arith.constant 1 : index
  %st = arith.constant 1 : index
  %l, %a, %b, %c = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t1, %b1 = %t2, %c1 = %t3) -> (!d, !t, !t, !t) {
    %l2, %a2 = scf.for %j = %lb to %one step %st iter_args(%l3 = %l1, %a3 = %a1) -> (!d, !t) {
      %l4, %a4 = pulse.drive %l3, %x40, %a3 : !d, !w, !t -> !d, !t
      scf.yield %l4, %a4 : !d, !t
    }
    scf.yield %l2, %b1, %c1, %a2 : !d, !t, !t, !t
  }
  %e1, %f1 = pulse.drive %d1, %x40, %a : !d, !w, !t -> !d, !t
  %e2, %f2 = pulse.drive %d2, %x40, %b : !d, !w, !t -> !d, !t
  %e3, %f3 = pulse.drive %d3, %x40, %c : !d, !w, !t -> !d, !t
  return
}

// The same with two trips of the inner loop: each outer trip plays its tone
// twice (80 cycles), so tone 2 is last played in trip 4 (ending at 400),
// tone 3 in trip 5 (480), tone 1 in trip 6 (560); rcp starts the drive on
// tone 2 at 440, as it does the unrolled loop's. Before, asap started it at 0.
// EARLY-LABEL: func.func @three_tones_rotated_through_an_inner_loop_of_two
// EARLY: scf.yield
// EARLY: scf.yield
// EARLY-NEXT: }
// ASAP-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 400 : i64}>
// RCP-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 440 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 480 : i64}>
// EARLY-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 560 : i64}>
// LATE-LABEL: func.func @three_tones_rotated_through_an_inner_loop_of_two
// LATE: scf.yield
// LATE: scf.yield
// LATE-NEXT: }
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 560 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 560 : i64}>
// LATE-NEXT: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 560 : i64}>
func.func @three_tones_rotated_through_an_inner_loop_of_two() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %q3 = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!d, !t)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!d, !t)
  %d2, %t2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!d, !t)
  %d3, %t3 = pulse.get_drive_line %q3 : (!pulse.qudit) -> (!d, !t)
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %x40 = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 7 : index
  %two = arith.constant 2 : index
  %st = arith.constant 1 : index
  %l, %a, %b, %c = scf.for %i = %lb to %ub step %st iter_args(%l1 = %d0, %a1 = %t1, %b1 = %t2, %c1 = %t3) -> (!d, !t, !t, !t) {
    %l2, %a2 = scf.for %j = %lb to %two step %st iter_args(%l3 = %l1, %a3 = %a1) -> (!d, !t) {
      %l4, %a4 = pulse.drive %l3, %x40, %a3 : !d, !w, !t -> !d, !t
      scf.yield %l4, %a4 : !d, !t
    }
    scf.yield %l2, %b1, %c1, %a2 : !d, !t, !t, !t
  }
  %e1, %f1 = pulse.drive %d1, %x40, %a : !d, !w, !t -> !d, !t
  %e2, %f2 = pulse.drive %d2, %x40, %b : !d, !w, !t -> !d, !t
  %e3, %f3 = pulse.drive %d3, %x40, %c : !d, !w, !t -> !d, !t
  return
}
