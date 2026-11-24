// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Follow a linear pulse value (line or tone) back to the allocation it is a
// version of, through pulse ops and structured control flow.
//
// Include after the Pulse dialect op and type headers.

#include "cudaq-pulse/Analysis/Variant.h"

// Included once per variant (see Variant.h).
#if PULSE_COUPLED == 0
#ifdef CUDAQ_PULSE_ANALYSIS_LINETRACE_H
#define CUDAQ_PULSE_LINETRACE_SEEN
#else
#define CUDAQ_PULSE_ANALYSIS_LINETRACE_H
#endif
#else
#ifdef CUDAQ_PULSE_ANALYSIS_LINETRACE_COUPLED_H
#define CUDAQ_PULSE_LINETRACE_SEEN
#else
#define CUDAQ_PULSE_ANALYSIS_LINETRACE_COUPLED_H
#endif
#endif
#ifndef CUDAQ_PULSE_LINETRACE_SEEN

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace pulse {
PULSE_VARIANT_OPEN

/// A `classical` link or call handle: linear and timed, like a line.
inline bool isClassicalType(mlir::Type type) {
  return type.getDialect().getNamespace() == "classical";
}

/// The index of the qudit whose line `acquisition` (a `pulse.get_drive_line`
/// or `pulse.get_readout_line`) acquires, or -1 for a runtime qudit not bound
/// yet (a function argument). A coupling line is not one qudit's: -1, so
/// that nothing a control system says of a qudit's drive or readout line is
/// read for it (`couplingQudits` gives its qudits).
inline int64_t quditOf(mlir::Operation *acquisition) {
  if (auto alloc = acquisition->getOperand(0).getDefiningOp<QuditAllocOp>())
    return kCoupled && mlir::isa<GetCouplingLineOp>(acquisition)
               ? -1
               : alloc.getQuditAttr().getInt();
  return -1;
}

/// The indices of the qudits a `pulse.get_coupling_line` reaches, in operand
/// order; -1 for a runtime qudit not bound yet.
inline llvm::SmallVector<int64_t, 2>
couplingQudits(mlir::Operation *acquisition) {
  llvm::SmallVector<int64_t, 2> qudits;
  for (mlir::Value qudit : acquisition->getOperands()) {
    auto alloc = qudit.getDefiningOp<QuditAllocOp>();
    qudits.push_back(alloc ? alloc.getQuditAttr().getInt() : -1);
  }
  return qudits;
}

/// Whether two `pulse.get_coupling_line` ops name one line: the same qudits,
/// in any order.
inline bool sameCouplingLine(mlir::Operation *a, mlir::Operation *b) {
  if (a->getNumOperands() != b->getNumOperands())
    return false;
  for (mlir::Value qudit : a->getOperands())
    if (!llvm::is_contained(b->getOperands(), qudit))
      return false;
  return true;
}

inline bool isLinearPulseType(mlir::Type type) {
  return mlir::isa<DriveLineType, ReadoutLineType, ToneType>(type) ||
         isClassicalType(type) ||
         (kCoupled && mlir::isa<CouplingLineType>(type));
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
  // A sync's result continues the operand in its place (SyncOp::verify), the
  // one the scan below finds: looked up directly, tracing every line of a
  // sync of n lines takes n steps instead of n^2.
  if (mlir::isa<SyncOp>(op))
    return op->getOperand(index);
  if (mlir::isa<GetDriveLineOp, GetReadoutLineOp>(op) ||
      (kCoupled && mlir::isa<GetCouplingLineOp>(op)) ||
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

/// Whether every two of `ops` are mutuallyExclusive, in time linear in the
/// ops and the regions around them rather than in their pairs.
///
/// Two ops are exclusive exactly when the chains of regions around them,
/// from the outermost, first differ at two regions of one scf.if. So all of
/// them are, when at each region on their chains: at most one op lies in it
/// directly, and then no other op lies deeper in it; the deeper ones all lie
/// in the regions of one op; and if they lie in two or more of its regions,
/// that op is an scf.if. A region's chain is its own, so the region itself
/// keys what is seen there.
inline bool allMutuallyExclusive(llvm::ArrayRef<mlir::Operation *> ops) {
  if (ops.size() < 2)
    return true;
  struct Seen {
    bool direct = false;           // an op lies in this region directly
    mlir::Region *child = nullptr; // a region of an op in this one
  };
  llvm::DenseMap<mlir::Region *, Seen> seen;
  for (mlir::Operation *op : ops) {
    mlir::Region *region = op->getParentRegion();
    if (!region)
      return false;
    Seen &here = seen[region];
    if (here.direct || here.child)
      return false;
    here.direct = true;
    mlir::Region *inner = region;
    while (mlir::Operation *holder = inner->getParentOp()) {
      mlir::Region *outer = holder->getParentRegion();
      if (!outer)
        break;
      Seen &around = seen[outer];
      if (around.direct)
        return false;
      if (around.child == inner)
        break; // seen from here outwards by an earlier op
      if (around.child) {
        // Another region of the op an earlier op came through, which
        // walked on outwards from here.
        if (around.child->getParentOp() != holder ||
            !mlir::isa<mlir::scf::IfOp>(holder))
          return false;
        break;
      }
      around.child = inner;
      inner = outer;
    }
  }
  return true;
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
  bool coupling = kCoupled && mlir::isa<GetCouplingLineOp>(acquisition);
  for (mlir::Operation *first : acquisition->getOperand(0).getUsers()) {
    if (first == acquisition || first->getName() != acquisition->getName() ||
        (coupling && !sameCouplingLine(first, acquisition)) ||
        !runsBefore(first, acquisition) ||
        mutuallyExclusive(first, acquisition))
      continue;
    mlir::InFlightDiagnostic diagnostic = acquisition->emitError();
    if (coupling)
      diagnostic << "this coupling line is already acquired; use the line "
                    "acquired first";
    else
      diagnostic << "this qudit's "
                 << (mlir::isa<GetDriveLineOp>(acquisition) ? "drive"
                                                            : "readout")
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
        if (kCoupled && mlir::isa<GetCouplingLineOp>(def)) {
          found = firstCouplingAcquisition(def);
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

  /// Of the acquisitions of the coupling line `acquisition` names, the
  /// first made.
  static mlir::Operation *
  firstCouplingAcquisition(mlir::Operation *acquisition) {
    mlir::Operation *first = acquisition;
    for (mlir::Operation *user : acquisition->getOperand(0).getUsers())
      if (user->getName() == acquisition->getName() &&
          sameCouplingLine(user, acquisition))
        first = user;
    return first;
  }

  llvm::DenseMap<mlir::Value, mlir::Operation *> memo;
};

PULSE_VARIANT_CLOSE
} // namespace pulse

#endif // CUDAQ_PULSE_LINETRACE_SEEN
#undef CUDAQ_PULSE_LINETRACE_SEEN
