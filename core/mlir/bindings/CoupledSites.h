//===- CoupledSites.h - The site tree of a module with coupling lines -----===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What idle replacement reads of a scheduled module that holds a coupling
// line (PulseModule::site_tree gives it for a module without one). A unit of
// its own, so that the walk of a module without a coupling line compiles as
// it did before this one existed.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_COUPLEDSITES_H
#define CUDAQ_PULSE_BINDINGS_COUPLEDSITES_H

#include <Python.h>

namespace mlir {
class Operation;
} // namespace mlir

namespace cudaq_pulse {

/// A new reference: what PulseModule::site_tree gives, `(lines, nodes)`, for
/// `module`, which holds a coupling line. Beside the drive and readout lines
/// and their ops it lists each coupling line, as one `("coupling", qudit)`
/// per qudit the line reaches (the first is the line's own index, which its
/// waits and syncs name), and each pulse on one as a `"readout"` node on each
/// of those entries, with no site and no tone: time on each qudit the line
/// reaches, which the program orders with nothing on the qudit's drive line,
/// as it orders the qudit's own readout. Null with an exception set only
/// when Python fails.
PyObject *coupledSiteTree(mlir::Operation *module);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_COUPLEDSITES_H
