// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_CLASSICAL_TYPES
#define CUDAQ_PULSE_DIALECT_CLASSICAL_TYPES

include "cudaq-pulse/Dialect/Classical/ClassicalDialect.td"
include "mlir/IR/AttrTypeBase.td"

class ClassicalType<string name, string typeMnemonic>
    : TypeDef<ClassicalDialect, name> {
  let mnemonic = typeMnemonic;
}

def Classical_LinkType : ClassicalType<"Link", "link"> {
  let summary = "The request channel to one classical device";
  let description = [{
    A linear value, like a line: `classical.submit` consumes it and returns
    the next version, so the requests on one device are ordered as the
    program submits them. A link starts at time 0.
  }];
}

def Classical_FutureType : ClassicalType<"Future", "future"> {
  let summary = "A request whose answer is still on its way";
  let description = [{
    What `classical.submit` returns. It becomes available when the answer
    is due, and `classical.wait` consumes it at most once. A future that is
    never waited on is a call whose answer the program does not use.
  }];
}

#endif // CUDAQ_PULSE_DIALECT_CLASSICAL_TYPES
