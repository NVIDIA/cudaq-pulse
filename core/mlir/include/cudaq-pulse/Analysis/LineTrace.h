// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Follow a linear pulse value (line or tone) back to the allocation it is a
// version of, through pulse ops and structured control flow.
//
// Include after the Pulse dialect op and type headers.

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace pulse {

/// A `classical` link or call handle: linear and timed, like a line.
inline bool isClassicalType(mlir::Type type) {
  return type.getDialect().getNamespace() == "classical";
}

inline bool isLinearPulseType(mlir::Type type) {
  return mlir::isa<DriveLineType, ReadoutLineType, ToneType>(type) ||
         isClassicalType(type);
}

/// Whether `user` reads a line without consuming it, so the version it reads
/// stays usable: pulse.get_current_time observes where a line's time stands.
inline bool readsWithoutConsuming(mlir::Operation *user) {
  return mlir::isa<GetCurrentTimeOp>(user);
}

/// The value `value` is the next version of, or null when `value` comes
/// straight from an allocation (or from something this analysis does not
/// model).
///
/// - An `scf.for` iteration argument and result continue the matching init
///   operand.
/// - An `scf.if` result continues the matching `then` yield operand; the
///   `else` arm yields a version of the same allocation.
/// - A `pulse.schedule_block` argument continues its operand, and a result
///   continues the value the body yields for it.
/// - Result `k` of type `T` of any other op continues its `k`-th operand of
///   type `T` (a drive's line result continues its line operand, a sync's
///   third line continues its third input line).
inline mlir::Value linearPredecessor(mlir::Value value) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (auto block = mlir::dyn_cast_or_null<ScheduleBlockOp>(parent))
      return block.getInputs()[argument.getArgNumber()];
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
    if (!loop || argument.getArgNumber() == 0)
      return {};
    return loop.getInitArgs()[argument.getArgNumber() - 1];
  }
  auto result = mlir::cast<mlir::OpResult>(value);
  mlir::Operation *op = result.getOwner();
  unsigned index = result.getResultNumber();
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return loop.getInitArgs()[index];
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return branch.thenYield().getOperand(index);
  // A schedule block result continues the value its body yields.
  if (auto block = mlir::dyn_cast<ScheduleBlockOp>(op))
    return block.getBody().front().getTerminator()->getOperand(index);
  if (mlir::isa<GetDriveLineOp, GetReadoutLineOp>(op) ||
      op->getName().getStringRef() == "classical.get_link")
    return {};
  mlir::Type type = value.getType();
  unsigned rank = 0;
  for (mlir::OpResult earlier : op->getResults().take_front(index))
    rank += earlier.getType() == type;
  for (mlir::Value operand : op->getOperands())
    if (operand.getType() == type && rank-- == 0)
      return operand;
  return {};
}

/// Whether `a` and `b` lie in different arms of some `scf.if`, so at most
/// one of them runs.
inline bool mutuallyExclusive(mlir::Operation *a, mlir::Operation *b) {
  llvm::DenseMap<mlir::Operation *, mlir::Region *> armsOfA;
  for (mlir::Region *region = a->getParentRegion(); region;
       region = region->getParentOp()->getParentRegion())
    if (mlir::isa<mlir::scf::IfOp>(region->getParentOp()))
      armsOfA[region->getParentOp()] = region;
  for (mlir::Region *region = b->getParentRegion(); region;
       region = region->getParentOp()->getParentRegion()) {
    auto it = armsOfA.find(region->getParentOp());
    if (it != armsOfA.end() && it->second != region)
      return true;
  }
  return false;
}

/// Whether `a` runs before `b` in program order, both in one function: `a`
/// or the op holding it comes before `b` or the op holding it in a block they
/// share (an op runs before the ops of its regions).
inline bool runsBefore(mlir::Operation *a, mlir::Operation *b) {
  for (mlir::Operation *outer = a; outer; outer = outer->getParentOp())
    if (mlir::Operation *other = outer->getBlock()->findAncestorOpInBlock(*b))
      return outer == other || outer->isBeforeInBlock(other);
  return false;
}

/// A qudit has one drive and one readout line, so acquiring one where an
/// earlier acquisition of it may also have run would make two versions of
/// the same line, with two timelines.
inline mlir::LogicalResult checkOneAcquisition(mlir::Operation *acquisition) {
  for (mlir::Operation *first : acquisition->getOperand(0).getUsers()) {
    if (first == acquisition || first->getName() != acquisition->getName() ||
        !runsBefore(first, acquisition) ||
        mutuallyExclusive(first, acquisition))
      continue;
    mlir::InFlightDiagnostic diagnostic = acquisition->emitError();
    diagnostic << "this qudit's "
               << (mlir::isa<GetDriveLineOp>(acquisition) ? "drive" : "readout")
               << " line is already acquired; use the line acquired first";
    diagnostic.attachNote(first->getLoc()) << "acquired first here";
    return mlir::failure();
  }
  return mlir::success();
}

/// The physical line (or link) a line, tone or link value is a version of:
/// the `pulse.get_drive_line` / `pulse.get_readout_line` op it comes from (or
/// the `classical.get_link` of a link), or null. A qudit has one drive and
/// one readout line, so every acquisition of the same kind on one qref stands
/// for the first made. Each value is traced once, so a line of n
/// ops costs O(n) rather than O(n^2). The IR must not change while a tracer
/// is in use.
class AllocationTracer {
public:
  mlir::Operation *operator()(mlir::Value value) {
    llvm::SmallVector<mlir::Value, 16> chain;
    mlir::Operation *found = nullptr;
    while (value) {
      auto known = memo.find(value);
      if (known != memo.end()) {
        found = known->second;
        break;
      }
      chain.push_back(value);
      if (mlir::Operation *def = value.getDefiningOp()) {
        if (mlir::isa<GetDriveLineOp, GetReadoutLineOp>(def)) {
          found = firstAcquisition(def);
          break;
        }
        if (def->getName().getStringRef() == "classical.get_link") {
          found = def;
          break;
        }
      }
      value = linearPredecessor(value);
    }
    for (mlir::Value traced : chain)
      memo[traced] = found;
    return found;
  }

private:
  /// Of the acquisitions like `acquisition` on its qref, the first made (a
  /// use list runs from its newest use).
  static mlir::Operation *firstAcquisition(mlir::Operation *acquisition) {
    mlir::Operation *first = acquisition;
    for (mlir::Operation *user : acquisition->getOperand(0).getUsers())
      if (user->getName() == acquisition->getName())
        first = user;
    return first;
  }

  llvm::DenseMap<mlir::Value, mlir::Operation *> memo;
};

} // namespace pulse
