// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The pattern passes, each one greedy rewrite of a function, the pass that
// applies several of them in a single rewrite, and
// pulse-hoist-loop-invariants.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Transforms/Passes.h"

#include <string>

namespace {

// Every greedy rewrite also erases the operations whose results nothing
// uses; pulse-remove-unused is that alone, with region simplification.
constexpr pulse::RewriteStage stages[] = {
    {"pulse-fold-constants",
     "Compute constant arithmetic at compile time and merge equal constants",
     nullptr, /*foldConstants=*/true},
    {"pulse-remove-unused",
     "Remove operations whose results nothing uses, unreachable blocks and "
     "unused block arguments",
     nullptr, /*foldConstants=*/false, /*removeUnused=*/true},
    {"pulse-merge-waits",
     "Join constant waits in a row on one line into one wait",
     pulse::populateMergeWaitsPatterns, false, false, false,
     /*unscheduledOnly=*/true},
    {"pulse-simplify-control-flow",
     "Remove syncs that order nothing, inline schedule blocks that take no "
     "time, branch on is_set, and inline or remove loops that run once or "
     "never",
     pulse::populateSimplifyControlFlowPatterns},
    {"pulse-fold-phases",
     "Fold constant frame changes into the next drive as phase attributes",
     pulse::populateVirtualZPatterns, false, false, /*onBinding=*/true},
    {"pulse-fuse-pulses",
     "Merge back-to-back square-pulse drives of the same amplitude into one",
     pulse::populateFusionPatterns, false, false, /*onBinding=*/true},
};

struct PulseRewritePass
    : public mlir::PassWrapper<PulseRewritePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseRewritePass)

  explicit PulseRewritePass(llvm::ArrayRef<const pulse::RewriteStage *> stages)
      : stages(stages) {
    for (const pulse::RewriteStage *stage : stages) {
      foldConstants |= stage->foldConstants;
      removeUnused |= stage->removeUnused;
      if (stage->unscheduledOnly && !unscheduledOnly)
        unscheduledOnly = stage;
    }
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
    if (unscheduledOnly) {
      pulse::ScheduleState state = pulse::scheduleStateOf(getOperation());
      if (state != pulse::ScheduleState::Unscheduled) {
        getOperation().emitError()
            << unscheduledOnly->argument
            << " takes an unscheduled function, and this one is "
            << pulse::stringifyScheduleState(state)
            << "; run it before pulse-schedule";
        return signalPassFailure();
      }
    }
    mlir::GreedyRewriteConfig config;
    config.enableFolding(foldConstants).enableConstantCSE(foldConstants);
    if (!removeUnused)
      config.setRegionSimplificationLevel(
          mlir::GreedySimplifyRegionLevel::Disabled);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), patterns, config)))
      signalPassFailure();
  }

  llvm::SmallVector<const pulse::RewriteStage *, 4> stages;
  bool foldConstants = false;
  bool removeUnused = false;
  /// The first of `stages` that takes only an unscheduled function.
  const pulse::RewriteStage *unscheduledOnly = nullptr;
  std::string argument;
  std::string description;
  mlir::FrozenRewritePatternSet patterns;
};

struct PulseHoistLoopInvariantsPass
    : public mlir::PassWrapper<PulseHoistLoopInvariantsPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseHoistLoopInvariantsPass)

  llvm::StringRef getArgument() const override {
    return "pulse-hoist-loop-invariants";
  }
  llvm::StringRef getDescription() const override {
    return "Move operations that compute the same value in every iteration "
           "out of loops";
  }

  void runOnOperation() override {
    getOperation()->walk([](mlir::LoopLikeOpInterface loop) {
      mlir::moveLoopInvariantCode(loop);
    });
  }
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
createPulseRewritePass(llvm::ArrayRef<const RewriteStage *> stages) {
  return std::make_unique<PulseRewritePass>(stages);
}

std::unique_ptr<mlir::Pass> createPulseHoistLoopInvariantsPass() {
  return std::make_unique<PulseHoistLoopInvariantsPass>();
}
} // namespace pulse
