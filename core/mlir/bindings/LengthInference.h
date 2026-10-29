//===- LengthInference.h - List result lengths of a classical body --------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// The lengths of a classical function's list results for one version, read
// from its bytecode as classical._infer reads them, for the bodies most
// functions have. A body outside that part of Python gets None, and
// classical._infer gives its lengths or the reason it has none, so a length
// this gives is the length classical._infer gives.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_LENGTHINFERENCE_H
#define CUDAQ_PULSE_BINDINGS_LENGTHINFERENCE_H

#include <Python.h>

namespace cudaq_pulse {

/// Reads the running interpreter's opcodes. Call once, when the module is
/// imported; until then inferListLengths gives None.
void loadLengthInference();

/// A new reference: a tuple of the lengths of the results at `positions` (a
/// sequence of ints) of `function`, whose parameters are `names` and whose
/// list parameters have the lengths of the dict `lengths`; or None when the
/// body is not one this follows. `resultCount` is how many results the
/// function declares. Null with an exception set only when Python fails.
PyObject *inferListLengths(PyObject *function, PyObject *names,
                           PyObject *lengths, PyObject *positions,
                           Py_ssize_t resultCount);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_LENGTHINFERENCE_H
