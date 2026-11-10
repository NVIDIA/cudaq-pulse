// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | FileCheck %s --check-prefix=PARSED
// RUN: cudaq-pulse-opt --canonicalize %s | FileCheck %s --check-prefix=FOLDED
// RUN: cudaq-pulse-opt --sccp %s | FileCheck %s --check-prefix=FOLDED
// RUN: not cudaq-pulse-opt --pulse-verify %s 2>&1 | FileCheck %s --check-prefix=VERIFY
// RUN: not cudaq-pulse-opt --canonicalize --pulse-verify %s 2>&1 | FileCheck %s --check-prefix=VERIFY

// Folding must not make valid IR invalid: op verifiers check an op on its
// own, so a duration that folds to a constant verifies before and after;
// pulse-verify rejects it either way.

// PARSED: arith.addi
// FOLDED: arith.constant -20 : i64
// FOLDED: pulse.square
// VERIFY: error: 'pulse.square' op duration must be positive, got -20
func.func @duration_that_folds(%amp: f64) -> !pulse.waveform {
  %a = arith.constant 20 : i64
  %b = arith.constant -40 : i64
  %d = arith.addi %a, %b : i64
  %z = arith.constant 0.0 : f64
  %w = pulse.square %d, %amp, %z : i64, f64, f64 -> !pulse.waveform
  return %w : !pulse.waveform
}
