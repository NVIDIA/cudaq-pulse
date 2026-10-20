/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Virtual-Z pass: fold shift_phase ops into the next drive's waveform phase.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

struct FoldShiftPhaseIntoDrive
    : public mlir::OpRewritePattern<pulse::ShiftPhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::ShiftPhaseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    if (!result.hasOneUse())
      return mlir::failure();

    auto *user = *result.getUsers().begin();
    auto driveOp = mlir::dyn_cast<pulse::DriveOp>(user);
    if (!driveOp || driveOp.getTone() != result)
      return mlir::failure();

    double existing = 0.0;
    if (auto attr =
            driveOp->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
      existing = attr.getValueAsDouble();

    auto phaseVal = op.getPhaseRad();
    if (auto cst = phaseVal.getDefiningOp<mlir::arith::ConstantFloatOp>()) {
      double delta = cst.value().convertToDouble();
      driveOp->setAttr("frame_phase_offset",
                       rewriter.getF64FloatAttr(existing + delta));
    } else {
      return mlir::failure();
    }

    // Drive now consumes the original tone (input to shift_phase)
    driveOp.getToneMutable().assign(op.getTone());
    rewriter.replaceOp(op, op.getTone());
    return mlir::success();
  }
};

/// A frame change that both arms of an `scf.if` pass through unchanged
/// applies after the `if` on either path: move it there, so that it can fold
/// into the drive that follows.
struct SinkShiftPhaseThroughIf
    : public mlir::OpRewritePattern<pulse::ShiftPhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::ShiftPhaseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value tone = op.getResult();
    mlir::scf::IfOp branch;
    std::optional<unsigned> index;
    for (mlir::OpOperand &use : tone.getUses()) {
      auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(use.getOwner());
      auto parent = yield
                        ? mlir::dyn_cast<mlir::scf::IfOp>(yield->getParentOp())
                        : mlir::scf::IfOp();
      if (!parent || (branch && parent != branch) ||
          (index && *index != use.getOperandNumber()))
        return mlir::failure();
      branch = parent;
      index = use.getOperandNumber();
    }
    if (!branch || branch.getElseRegion().empty() ||
        std::distance(tone.use_begin(), tone.use_end()) != 2)
      return mlir::failure();

    mlir::Value result = branch.getResult(*index);
    rewriter.modifyOpInPlace(branch.thenYield(), [&] {
      branch.thenYield()->setOperand(*index, op.getTone());
    });
    rewriter.modifyOpInPlace(branch.elseYield(), [&] {
      branch.elseYield()->setOperand(*index, op.getTone());
    });
    rewriter.setInsertionPointAfter(branch);
    auto moved = pulse::ShiftPhaseOp::create(
        rewriter, op.getLoc(), op.getType(), result, op.getPhaseRad());
    rewriter.replaceAllUsesExcept(result, moved.getResult(), moved);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct VirtualZPass
    : public mlir::PassWrapper<VirtualZPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VirtualZPass)

  llvm::StringRef getArgument() const override { return "pulse-virtual-z"; }
  llvm::StringRef getDescription() const override {
    return "Fold shift_phase ops into adjacent drive ops as phase attributes";
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldShiftPhaseIntoDrive, SinkShiftPhaseThroughIf>(
        &getContext());
    mlir::GreedyRewriteConfig config;
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createVirtualZPass() {
  return std::make_unique<VirtualZPass>();
}
} // namespace pulse
