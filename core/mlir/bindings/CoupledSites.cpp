//===- CoupledSites.cpp - The site tree of a module with coupling lines ---===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// The walk of PulseModule::site_tree with the line trace and the timing of a
// module that holds a coupling line (Variant.h), written with Python's own
// calls (the limited API). As in Lines.cpp, the unit's own text is in a
// section of its own and not in string literals, so that it adds none to
// the library's and no literal of the library moves.
//
//===----------------------------------------------------------------------===//

// The lengths given with a string are Py_ssize_t.
#define PY_SSIZE_T_CLEAN
#include "CoupledSites.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

// The whole unit is the variant that knows the ops of a coupling line.
#define PULSE_COUPLED 1
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include <algorithm>
#include <optional>

namespace cudaq_pulse {
namespace {

#if defined(__ELF__)
#define CUDAQ_PULSE_SITES_TEXT __attribute__((section("cudaq_pulse_lines")))
#else
#define CUDAQ_PULSE_SITES_TEXT
#endif
CUDAQ_PULSE_SITES_TEXT const char kDrive[] = "drive";
CUDAQ_PULSE_SITES_TEXT const char kReadout[] = "readout";
CUDAQ_PULSE_SITES_TEXT const char kCoupling[] = "coupling";
CUDAQ_PULSE_SITES_TEXT const char kSync[] = "sync";
CUDAQ_PULSE_SITES_TEXT const char kAnswer[] = "answer";
CUDAQ_PULSE_SITES_TEXT const char kFor[] = "for";
CUDAQ_PULSE_SITES_TEXT const char kIf[] = "if";
CUDAQ_PULSE_SITES_TEXT const char kSite[] = "pulse.site";
CUDAQ_PULSE_SITES_TEXT const char kVariable[] = "classical.variable";
CUDAQ_PULSE_SITES_TEXT const char kBlocking[] = "classical.blocking";
// (role, qudit); a node; ("sync", site, lines); ("answer",); ("for", trips,
// nodes); ("if", nodes, nodes); (lines, nodes).
CUDAQ_PULSE_SITES_TEXT const char kLine[] = "(sL)";
CUDAQ_PULSE_SITES_TEXT const char kNode[] = "(s#OLLOOOO)";
CUDAQ_PULSE_SITES_TEXT const char kJoined[] = "(sOO)";
CUDAQ_PULSE_SITES_TEXT const char kOne[] = "(s)";
CUDAQ_PULSE_SITES_TEXT const char kLoop[] = "(sLO)";
CUDAQ_PULSE_SITES_TEXT const char kTree[] = "(OO)";
CUDAQ_PULSE_SITES_TEXT const char kUntraced[] =
    "a pulse on a coupling line does not trace back to the line's fetch";

/// A new reference: `value`, or None.
PyObject *number(std::optional<int64_t> value) {
  if (value)
    return PyLong_FromLongLong(*value);
  Py_INCREF(Py_None);
  return Py_None;
}

/// Appends `item`, which it takes, to `list`; false when either failed.
bool append(PyObject *list, PyObject *item) {
  if (!item)
    return false;
  bool appended = PyList_Append(list, item) == 0;
  Py_DECREF(item);
  return appended;
}

/// The walk: the lines by their acquisition, and where it stands in the
/// loops around it.
struct Walk {
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  llvm::SmallVector<int64_t, 2> iteration;
  pulse::AllocationTracer trace;

  int64_t indexOf(mlir::Value value) {
    mlir::Operation *allocation = trace(value);
    if (!allocation)
      return -1;
    auto found = lineIndex.find(allocation);
    return found == lineIndex.end() ? -1 : found->second;
  }

  /// A new reference: the site number a packed site record gave `op`, or
  /// None (PulseModule::siteOf).
  static PyObject *site(mlir::Operation &op) {
    auto loc = mlir::dyn_cast<mlir::FileLineColLoc>(op.getLoc());
    if (!loc || loc.getFilename().getValue() != llvm::StringRef(kSite))
      return number(std::nullopt);
    int64_t line = loc.getLine();
    return number(loc.getColumn() ? -line - 1 : line);
  }

  /// Appends the nodes of `block` to the list `out`; false when Python
  /// failed.
  bool build(mlir::Block &block, PyObject *out) {
    for (mlir::Operation &op : block) {
      bool played = mlir::isa<pulse::DriveOp, pulse::ReadoutOp>(op);
      bool wait = mlir::isa<pulse::WaitOp>(op);
      bool coupling = mlir::isa<pulse::DriveBasebandOp>(op);
      if (played || wait || coupling ||
          mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                    pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
        if (!timed(op, played, wait, coupling, out))
          return false;
      } else if (mlir::isa<pulse::SyncOp>(op)) {
        PyObject *joined = PyList_New(0);
        bool listed = joined != nullptr;
        for (mlir::Value line : op.getOperands())
          listed = listed && append(joined, PyLong_FromLongLong(indexOf(line)));
        PyObject *number = listed ? site(op) : nullptr;
        listed = number &&
                 append(out, Py_BuildValue(kJoined, kSync, number, joined));
        Py_XDECREF(number);
        Py_XDECREF(joined);
        if (!listed)
          return false;
      } else if (mlir::isa<classical::WaitOp>(op)) {
        if (op.getDiscardableAttr(kBlocking) &&
            !append(out, Py_BuildValue(kOne, kAnswer)))
          return false;
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        PyObject *body = PyList_New(0);
        iteration.push_back(0);
        bool listed = body && build(*loop.getBody(), body);
        iteration.pop_back();
        listed = listed &&
                 append(out, Py_BuildValue(kLoop, kFor,
                                           static_cast<long long>(
                                               std::max<int64_t>(trips, 0)),
                                           body));
        Py_XDECREF(body);
        if (!listed)
          return false;
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        PyObject *then = PyList_New(0);
        PyObject *other = PyList_New(0);
        bool listed = then && other &&
                      build(branch.getThenRegion().front(), then) &&
                      (branch.getElseRegion().empty() ||
                       build(branch.getElseRegion().front(), other)) &&
                      append(out, Py_BuildValue(kJoined, kIf, then, other));
        Py_XDECREF(then);
        Py_XDECREF(other);
        if (!listed)
          return false;
      } else {
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            if (!build(nested, out))
              return false;
      }
    }
    return true;
  }

  /// Appends the node of the timed op `op`: one for a drive, readout, wait
  /// or frame change, and for a pulse on a coupling line one per qudit the
  /// line reaches.
  bool timed(mlir::Operation &op, bool played, bool wait, bool coupling,
             PyObject *out) {
    std::optional<int64_t> start = pulse::scheduledStart(&op, iteration);
    PyObject *shifts = PyList_New(0);
    bool listed = shifts != nullptr;
    if (start)
      for (int64_t &k : iteration) {
        k = 1;
        std::optional<int64_t> next = pulse::scheduledStart(&op, iteration);
        k = 0;
        listed =
            listed &&
            append(shifts, number(next ? std::optional<int64_t>(*next - *start)
                                       : std::nullopt));
      }
    PyObject *begins = listed ? number(start) : nullptr;
    PyObject *lasts = begins ? number(pulse::scheduledDuration(&op)) : nullptr;
    listed = lasts != nullptr;
    if (listed && coupling) {
      mlir::Operation *line = trace(op.getOperand(0));
      int64_t first = indexOf(op.getOperand(0));
      // Listed on no qudit, the pulse would leave its qudits idle.
      if (!line || first < 0) {
        PyErr_SetString(PyExc_RuntimeError, kUntraced);
        listed = false;
      }
      int64_t reached = listed ? line->getNumOperands() : 0;
      PyObject *none = number(std::nullopt);
      PyObject *variable =
          op.getDiscardableAttr(kVariable) ? Py_True : Py_False;
      for (int64_t k = 0; listed && k < reached; ++k)
        listed = append(
            out, Py_BuildValue(kNode, kReadout,
                               static_cast<Py_ssize_t>(sizeof(kReadout) - 1),
                               none, static_cast<long long>(first + k), -1LL,
                               begins, lasts, shifts, variable));
      Py_DECREF(none);
    } else if (listed) {
      llvm::StringRef kind = op.getName().stripDialect();
      PyObject *number = site(op);
      listed =
          number &&
          append(out,
                 Py_BuildValue(
                     kNode, kind.data(), static_cast<Py_ssize_t>(kind.size()),
                     number,
                     static_cast<long long>(
                         played || wait ? indexOf(op.getOperand(0)) : -1),
                     static_cast<long long>(
                         wait ? -1 : indexOf(op.getOperand(played ? 2 : 0))),
                     begins, lasts, shifts,
                     op.getDiscardableAttr(kVariable) ? Py_True : Py_False));
      Py_XDECREF(number);
    }
    Py_XDECREF(shifts);
    Py_XDECREF(begins);
    Py_XDECREF(lasts);
    return listed;
  }
};

} // namespace

PyObject *coupledSiteTree(mlir::Operation *module) {
  Walk walk;
  PyObject *lines = PyList_New(0);
  PyObject *nodes = PyList_New(0);
  bool listed = lines && nodes;
  int64_t count = 0;
  module->walk([&](mlir::Operation *op) {
    if (!listed)
      return;
    if (mlir::isa<pulse::GetCouplingLineOp>(op)) {
      walk.lineIndex.try_emplace(op, count);
      for (int64_t qudit : pulse::couplingQudits(op)) {
        listed = listed &&
                 append(lines, Py_BuildValue(kLine, kCoupling,
                                             static_cast<long long>(qudit)));
        ++count;
      }
      return;
    }
    if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
      return;
    walk.lineIndex.try_emplace(op, count++);
    listed = append(
        lines,
        Py_BuildValue(kLine,
                      mlir::isa<pulse::GetDriveLineOp>(op) ? kDrive : kReadout,
                      static_cast<long long>(pulse::quditOf(op))));
  });
  PyObject *tree = nullptr;
  if (listed) {
    auto top = mlir::cast<mlir::ModuleOp>(module);
    if (walk.build(*top.getBody(), nodes))
      tree = Py_BuildValue(kTree, lines, nodes);
  }
  Py_XDECREF(lines);
  Py_XDECREF(nodes);
  return tree;
}

} // namespace cudaq_pulse
