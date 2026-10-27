/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The cudaq-pulse passes and conversions, and their registration under the
// names cudaq-pulse-opt and the Python extension run them by.

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <cstdint>
#include <memory>

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();

std::unique_ptr<mlir::Pass> createPulseVerifyPass();

void populateCanonicalizePatterns(mlir::RewritePatternSet &patterns);
void populateVirtualZPatterns(mlir::RewritePatternSet &patterns);
void populateFusionPatterns(mlir::RewritePatternSet &patterns);

/// A pass that is one greedy rewrite of a function with its patterns.
struct RewriteStage {
  llvm::StringLiteral argument;
  llvm::StringLiteral description;
  /// Adds the patterns; null for pulse-fold, which has none and only folds.
  void (*populate)(mlir::RewritePatternSet &);
};

/// pulse-fold, pulse-canonicalize, pulse-virtual-z and pulse-fusion.
llvm::ArrayRef<RewriteStage> rewriteStages();
/// The stage registered as `argument`, or null.
const RewriteStage *findRewriteStage(llvm::StringRef argument);
/// One greedy rewrite of a function with the patterns of all `stages`, which
/// also folds ops and merges equal constants when `fold`. The stages share a
/// walk of the function, so a pipeline that runs several back to back pays
/// for one.
std::unique_ptr<mlir::Pass>
createPulseRewritePass(llvm::ArrayRef<const RewriteStage *> stages, bool fold);

std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap);
std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass();
std::unique_ptr<mlir::Pass> createPulseClassicalCheckPass();
std::unique_ptr<mlir::Pass> createPulseUnrollClassicalLoopsPass();
} // namespace pulse

namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
} // namespace qop

namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
} // namespace cudm

namespace pulse {
/// Register the conversions and the pulse passes. Registration is global and
/// not thread-safe: call it once, before any pipeline is parsed.
inline void registerPulsePasses() {
  mlir::registerPass(createPulseToQOpPass);
  mlir::registerPass(qop::createQOpToCuDensityMatPass);
  mlir::registerPass(cudm::createCuDensityMatToLLVMPass);

  mlir::registerPass(createPulseVerifyPass);
  for (const RewriteStage &stage : rewriteStages())
    mlir::registerPass([&stage] {
      return createPulseRewritePass(&stage, stage.populate == nullptr);
    });
  mlir::registerPass(createPulseScheduleAsapPass);
  mlir::registerPass(createPulseScheduleAlapPass);
  mlir::registerPass(createPulseInlineScheduleBlocksPass);
  mlir::registerPass(createPulseClassicalCheckPass);
  // Resource-constrained scheduling with the default machine model (4
  // drives, 2 readouts, no readout latency or switch penalty).
  mlir::registerPass(
      [] { return createPulseScheduleRcpPass(4, 2, 0, 0, false); });
}
} // namespace pulse
