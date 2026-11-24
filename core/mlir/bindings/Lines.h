//===- Lines.h - The lines of a module, in allocation order ---------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What a compiled kernel's table of lines reads from its module: each line's
// role and qudits. A unit of its own that binds its own method, so that the
// bindings gain one call and one accessor, and no other code or text.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_LINES_H
#define CUDAQ_PULSE_BINDINGS_LINES_H

#include <Python.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace cudaq_pulse {

/// What reads a module: called with its top-level operation and `state`.
using ModuleReader = void (*)(mlir::Operation *module, void *state);

/// Runs a reader on the module that `self`, a PulseModule, holds, under the
/// module's lock.
using ModuleAccess = void (*)(void *self, ModuleReader read, void *state);

/// Gives the class `PulseModule` of the extension module `module` its
/// method `lines()`: a list with one `(role, qudits)` for each line the
/// module acquires (a drive, readout or coupling line), in the order its
/// events number them (PulseModule::events). `role` is "drive", "readout"
/// or "coupling"; `qudits` is a tuple of the qudits' indices, in operand
/// order, -1 for a runtime qudit not bound yet. `access` reads a module.
/// False with an exception set when Python fails.
bool bindLineTable(PyObject *module, ModuleAccess access);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_LINES_H
