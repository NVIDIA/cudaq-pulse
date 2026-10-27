/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The pattern passes: pulse-fold, pulse-canonicalize, pulse-virtual-z and
// pulse-fusion, each one greedy rewrite of a function, and the pass that
// applies several of them in a single rewrite.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cudaq-pulse/Transforms/Passes.h"

#include <string>

namespace {

constexpr pulse::RewriteStage stages[] = {
    {"pulse-fold",
     "Fold ops, merge equal constants and remove dead ops and blocks", nullptr},
    {"pulse-canonicalize",
     "Remove redundant syncs, merge waits in a row, inline untimed schedule "
     "blocks, and branch on is_set and drop loops that run once or never",
     pulse::populateCanonicalizePatterns},
    {"pulse-virtual-z",
     "Fold shift_phase ops into adjacent drive ops as phase attributes",
     pulse::populateVirtualZPatterns},
    {"pulse-fusion",
     "Fuse adjacent same-amplitude square-pulse drives into one",
     pulse::populateFusionPatterns},
};

struct PulseRewritePass
    : public mlir::PassWrapper<PulseRewritePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseRewritePass)

  PulseRewritePass(llvm::ArrayRef<const pulse::RewriteStage *> stages,
                   bool fold)
      : stages(stages), fold(fold) {
    if (stages.size() == 1) {
      argument = stages.front()->argument.str();
      description = stages.front()->description.str();
      return;
    }
    argument = "pulse-rewrite";
    description = "One greedy rewrite with the patterns of";
    for (const pulse::RewriteStage *stage : stages)
      description += " " + stage->argument.str();
  }

  llvm::StringRef getArgument() const override { return argument; }
  llvm::StringRef getDescription() const override { return description; }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    mlir::RewritePatternSet set(ctx);
    for (const pulse::RewriteStage *stage : stages)
      if (stage->populate)
        stage->populate(set);
    patterns = mlir::FrozenRewritePatternSet(std::move(set));
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::GreedyRewriteConfig config;
    config.enableFolding(fold).enableConstantCSE(fold);
    if (!fold)
      config.setRegionSimplificationLevel(
          mlir::GreedySimplifyRegionLevel::Disabled);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), patterns, config)))
      signalPassFailure();
  }

  llvm::SmallVector<const pulse::RewriteStage *, 4> stages;
  bool fold;
  std::string argument;
  std::string description;
  mlir::FrozenRewritePatternSet patterns;
};

} // namespace

namespace pulse {
llvm::ArrayRef<RewriteStage> rewriteStages() { return stages; }

const RewriteStage *findRewriteStage(llvm::StringRef argument) {
  for (const RewriteStage &stage : stages)
    if (stage.argument == argument)
      return &stage;
  return nullptr;
}

std::unique_ptr<mlir::Pass>
createPulseRewritePass(llvm::ArrayRef<const RewriteStage *> stages, bool fold) {
  return std::make_unique<PulseRewritePass>(stages, fold);
}
} // namespace pulse
