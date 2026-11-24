//===- BindingsCoupled.h - The bindings' code for coupling lines ----------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What the bindings do for a program that holds a coupling line: decode its
// packed stream and list its events. It is bindings.cpp compiled a second
// time (bindingsCoupled.cpp), as the passes that know a coupling line are
// their sources compiled a second time (Variant.h), so that what
// bindings.cpp compiles to for a program without one is what it compiled to
// before coupling lines existed.
//
// Each unit has its own PulseModule and its own exception types (they are
// in an anonymous namespace), so nothing of those crosses: a module crosses
// as its MLIR module and plain fields, a failure as a CoupledFailure, which
// the unit that asked throws as its own type.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_BINDINGSCOUPLED_H
#define CUDAQ_PULSE_BINDINGS_BINDINGSCOUPLED_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <nanobind/ndarray.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq_pulse {

/// What the coupled unit refused or could not read, for the unit that asked
/// to throw as its own exception type.
struct CoupledFailure {
  enum Kind { None, PackedValue, NotSupportedYet, Unsupported, KernelRefused };
  Kind kind = None;
  std::string what;
  /// Of a PackedValue: the offset of the record that holds the value.
  size_t word = 0;
};

/// A decoded program: its module and what the builder says of it.
struct DecodedModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  bool runTime = true, loops = true;
  /// Of a module with waits whose length floats set (PulseModule::Length).
  bool hasLength = false;
  int64_t cycles = 0, most = 0, controller = 0;
  std::vector<int64_t> repeats, excess;
};

/// PulseModule::Event.
using CoupledEvent = std::tuple<std::string, int64_t, int64_t,
                                std::optional<int64_t>, std::optional<int64_t>>;

/// PulseModuleBuilder::build_from_packed for a stream that holds the record
/// of a coupling line, in the builder's context `context`; `lost` is what
/// the builder was told to expect lost (expectLost).
DecodedModule decodeCoupled(
    const std::shared_ptr<mlir::MLIRContext> &context,
    const std::vector<int64_t> &lost,
    const nanobind::ndarray<int64_t, nanobind::ndim<1>, nanobind::c_contig>
        &stream,
    double clockGHz, int64_t quditCount,
    const nanobind::ndarray<double, nanobind::ndim<1>, nanobind::c_contig>
        &qubitFreqs,
    const std::vector<std::string> &paramNames,
    const std::vector<std::string> &paramTypes, const std::string &name,
    int64_t slotCount,
    const std::optional<nanobind::ndarray<const double, nanobind::ndim<1>,
                                          nanobind::c_contig>> &readoutFreqs,
    CoupledFailure &failure);

/// PulseModule::events() of `module`, which holds a coupling line and is
/// owned by the caller, who holds its lock.
std::vector<CoupledEvent>
coupledEvents(const std::shared_ptr<mlir::MLIRContext> &context,
              mlir::ModuleOp module, CoupledFailure &failure);

/// PulseModule::event_tree(limit) of such a module: a new reference, or
/// null with `failure` set. A Python error, like any exception other than
/// the four kinds of CoupledFailure, propagates out of this function.
PyObject *coupledEventTree(const std::shared_ptr<mlir::MLIRContext> &context,
                           mlir::ModuleOp module, int64_t limit,
                           CoupledFailure &failure);

/// Sets the Python error for what a program asks that is not available yet
/// (cudaq_pulse.UnsupportedError) and for what the compiler refuses of a
/// kernel (cudaq_pulse.CompilationError): where the bindings' exception
/// translator sends its two types for them, so that it names no module or
/// class of its own.
void raiseUnsupported(const char *what);
void raiseKernelRefused(const char *what);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_BINDINGSCOUPLED_H
