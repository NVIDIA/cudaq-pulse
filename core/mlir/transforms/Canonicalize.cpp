/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Pulse canonicalization pass: redundant sync elimination, dead line
// elimination, waveform CSE.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

/// Remove sync ops with only one operand (no-op synchronization).
struct RemoveSingleOperandSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumOperands() <= 1) {
      if (op.getNumResults() == 1 && op.getNumOperands() == 1) {
        rewriter.replaceOp(op, op.getOperand(0));
      } else if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else {
        return mlir::failure();
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Remove consecutive duplicate sync ops on the same set of lines.
struct RemoveRedundantSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op->getPrevNode();
    if (!prevOp)
      return mlir::failure();

    auto prevSync = mlir::dyn_cast<pulse::SyncOp>(prevOp);
    if (!prevSync)
      return mlir::failure();

    if (prevSync.getNumResults() != op.getNumOperands())
      return mlir::failure();

    bool allMatch = true;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (op.getOperand(i) != prevSync.getResult(i)) {
        allMatch = false;
        break;
      }
    }

    if (!allMatch)
      return mlir::failure();

    // This sync immediately follows the previous one on the same lines
    rewriter.replaceOp(op, prevSync.getResults());
    return mlir::success();
  }
};

/// A schedule block without drives, readouts or waits has no duration and
/// schedules the same inlined.
struct InlineUntimedScheduleBlock
    : public mlir::OpRewritePattern<pulse::ScheduleBlockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::ScheduleBlockOp block,
                  mlir::PatternRewriter &rewriter) const override {
    bool timed =
        block.getBody()
            .walk([](mlir::Operation *op) {
              return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(
                         op)
                         ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
            })
            .wasInterrupted();
    if (timed)
      return mlir::failure();
    mlir::Block &body = block.getBody().front();
    mlir::Operation *yield = body.getTerminator();
    llvm::SmallVector<mlir::Value> results(yield->getOperands());
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&body, block, block.getInputs());
    rewriter.replaceOp(block, results);
    return mlir::success();
  }
};

} // namespace

namespace pulse {
void populateCanonicalizePatterns(mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *ctx = patterns.getContext();
  patterns.add<RemoveSingleOperandSync, RemoveRedundantSync,
               InlineUntimedScheduleBlock>(ctx);
  // Branch on pulse.is_set, and inline or drop loops that run once or
  // never. scf.if patterns are left out: they may turn a branch over
  // lines and tones into arith.select.
  pulse::IsSetOp::getCanonicalizationPatterns(patterns, ctx);
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
}
} // namespace pulse
