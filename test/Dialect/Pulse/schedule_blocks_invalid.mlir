// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func @unknown_policy(%d0: !d) {
  // expected-error @+1 {{policy must be "asap" or "alap", got "rcp"}}
  %d1 = pulse.schedule_block "rcp" (%d0) : (!d) -> (!d) {
  ^bb0(%d: !d):
    pulse.yield %d : !d
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A line from outside the block must come in as an operand.
func.func @captured_line(%d0: !d, %t0: !t, %x: !w) {
  %t1 = pulse.schedule_block "asap" (%t0) : (!t) -> (!t) {
  ^bb0(%t: !t):
    // expected-error @+1 {{uses a line, tone, measurement or link from outside its pulse.schedule_block}}
    %d1, %t2 = pulse.drive %d0, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %t2 : !t
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func @argument_types(%d0: !d) {
  // expected-error @+2 {{successor operand type #0 '!pulse.drive_line' should match successor input type #0 '!pulse.tone'}}
  // expected-note @+1 {{region branch point}}
  %d1 = pulse.schedule_block "asap" (%d0) : (!d) -> (!d) {
  ^bb0(%t: !t):
    pulse.yield %t : !t
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func @result_types(%d0: !d, %t0: !t) {
  // expected-error @+1 {{region branch point has 2 operands, but region successor needs 1 inputs}}
  %d1 = pulse.schedule_block "alap" (%d0, %t0) : (!d, !t) -> (!d) {
  ^bb0(%d: !d, %t: !t):
    // expected-note @+1 {{region branch point}}
    pulse.yield %d, %t : !d, !t
  }
  return
}

// -----

// Waveforms are not timed: they may be used from outside the block.
!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func @captured_waveform(%d0: !d, %t0: !t, %x: !w) {
  %d1, %t1 = pulse.schedule_block "alap" (%d0, %t0) : (!d, !t) -> (!d, !t) {
  ^bb0(%d: !d, %t: !t):
    %d2, %t2 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
    pulse.yield %d2, %t2 : !d, !t
  }
  return
}

// -----

// A classical link is timed like a line, so it crosses the boundary only as
// an operand too.
func.func @captured_link(%l: !classical.link, %d: !pulse.drive_line) -> (!pulse.drive_line, !classical.link) {
  %d1, %l1 = pulse.schedule_block "asap" (%d) : (!pulse.drive_line) -> (!pulse.drive_line, !classical.link) {
  ^bb0(%d_in: !pulse.drive_line):
    // expected-error @+1 {{uses a line, tone, measurement or link from outside its pulse.schedule_block; pass it as an operand}}
    pulse.yield %d_in, %l : !pulse.drive_line, !classical.link
  }
  return %d1, %l1 : !pulse.drive_line, !classical.link
}

// -----

func.func @block_without_a_yield(%d: !pulse.drive_line) {
  // expected-error @+1 {{'pulse.schedule_block' op body must end in pulse.yield}}
  %d1 = pulse.schedule_block "asap" (%d) : (!pulse.drive_line) -> (!pulse.drive_line) {
  ^bb0(%d_in: !pulse.drive_line):
    llvm.unreachable
  }
  return
}
