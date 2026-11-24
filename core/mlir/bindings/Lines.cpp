//===- Lines.cpp - The lines of a module, in allocation order -------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// The method is bound here with Python's own calls (the limited API, as in
// LengthInference.cpp), not as a method the bindings define, and the
// bindings give only how a module is read: they keep every function they
// had as it was compiled before. Of the bindings' library it asks whether
// the instance was constructed, and for its pointer.
//
//===----------------------------------------------------------------------===//

#include "Lines.h"

#include <nanobind/nanobind.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace cudaq_pulse {
namespace {

// The unit's text is in a section of its own, and none of it is a string
// literal: a literal would go among the library's other literals and move
// those after it, some across a page end, where comparing them costs more.
#if defined(__ELF__)
#define CUDAQ_PULSE_LINES_TEXT __attribute__((section("cudaq_pulse_lines")))
#else
#define CUDAQ_PULSE_LINES_TEXT
#endif
CUDAQ_PULSE_LINES_TEXT const char kDrive[] = "drive";
CUDAQ_PULSE_LINES_TEXT const char kReadout[] = "readout";
CUDAQ_PULSE_LINES_TEXT const char kCoupling[] = "coupling";
CUDAQ_PULSE_LINES_TEXT const char kRow[] = "(sO)";
CUDAQ_PULSE_LINES_TEXT const char kClass[] = "PulseModule";
CUDAQ_PULSE_LINES_TEXT const char kName[] = "lines";
CUDAQ_PULSE_LINES_TEXT const char kNotReady[] =
    "lines() of a PulseModule that was not constructed";
CUDAQ_PULSE_LINES_TEXT const char kFailed[] =
    "listing the lines of the module failed";
CUDAQ_PULSE_LINES_TEXT const char kDoc[] =
    "lines($self, /)\n--\n\n(role, qudits) of each drive, readout and "
    "coupling line, in the order events() numbers them";

/// The role of the line `op` acquires, or null when it acquires none.
const char *roleOf(mlir::Operation *op) {
  if (mlir::isa<pulse::GetDriveLineOp>(op))
    return kDrive;
  if (mlir::isa<pulse::GetReadoutLineOp>(op))
    return kReadout;
  if (mlir::isa<pulse::GetCouplingLineOp>(op))
    return kCoupling;
  return nullptr;
}

/// A new reference: the indices of the qudits of the acquisition `op`.
PyObject *quditsOf(mlir::Operation *op) {
  // A drive or readout line is its first operand's; a coupling line reaches
  // every qudit it is given.
  unsigned count =
      mlir::isa<pulse::GetCouplingLineOp>(op) ? op->getNumOperands() : 1;
  PyObject *qudits = PyTuple_New(count);
  if (!qudits)
    return nullptr;
  for (unsigned place = 0; place < count; ++place) {
    auto alloc = op->getOperand(place).getDefiningOp<pulse::QuditAllocOp>();
    PyObject *index =
        PyLong_FromLongLong(alloc ? alloc.getQuditAttr().getInt() : -1);
    // The tuple takes the reference, also when it fails.
    if (!index || PyTuple_SetItem(qudits, place, index) < 0) {
      Py_DECREF(qudits);
      return nullptr;
    }
  }
  return qudits;
}

/// What lists the lines of one module: the table, null once Python failed.
struct Listing {
  PyObject *table;
};

/// Adds the lines of `module` to the Listing `state`.
void listLines(mlir::Operation *module, void *state) {
  auto *listing = static_cast<Listing *>(state);
  // The walk of PulseModule::events, which numbers the lines.
  module->walk([&](mlir::Operation *op) {
    const char *role = listing->table ? roleOf(op) : nullptr;
    if (!role)
      return;
    PyObject *qudits = quditsOf(op);
    PyObject *row = qudits ? Py_BuildValue(kRow, role, qudits) : nullptr;
    Py_XDECREF(qudits);
    if (!row || PyList_Append(listing->table, row) < 0)
      Py_CLEAR(listing->table);
    Py_XDECREF(row);
  });
}

/// How a PulseModule is read (bindLineTable).
ModuleAccess moduleAccess = nullptr;

/// PulseModule.lines(): `self` is a PulseModule, which the method's
/// descriptor checks.
PyObject *lines(PyObject *self, PyObject *) {
  nanobind::handle held(self);
  if (!nanobind::inst_ready(held)) {
    PyErr_SetString(PyExc_TypeError, kNotReady);
    return nullptr;
  }
  Listing listing{PyList_New(0)};
  // Nothing is thrown through Python's frame, which called this.
  try {
    if (listing.table)
      moduleAccess(nanobind::inst_ptr<void>(held), listLines, &listing);
  } catch (...) {
    Py_CLEAR(listing.table);
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError, kFailed);
  }
  return listing.table;
}

PyMethodDef linesMethod = {kName, lines, METH_NOARGS, kDoc};

} // namespace

bool bindLineTable(PyObject *module, ModuleAccess access) {
  moduleAccess = access;
  PyObject *cls = PyObject_GetAttrString(module, kClass);
  if (!cls)
    return false;
  PyObject *method =
      PyDescr_NewMethod(reinterpret_cast<PyTypeObject *>(cls), &linesMethod);
  bool bound = method && PyObject_SetAttrString(cls, kName, method) == 0;
  Py_XDECREF(method);
  Py_DECREF(cls);
  return bound;
}

} // namespace cudaq_pulse
