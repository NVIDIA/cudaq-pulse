// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_CLASSICAL
#define CUDAQ_PULSE_DIALECT_CLASSICAL

include "mlir/IR/OpBase.td"

def ClassicalDialect : Dialect {
  let name = "classical";
  let summary = "A classical real-time dialect for co-processing at QPU runtime";
  let description = [{
    The `classical` dialect describes the classical services a pulse program
    calls while the QPU runs: functions on a GPU, a host or an FPGA that
    decode syndromes, update calibrations or receive bulk readout data, and
    whose answers later pulses can use.

    A call is two timed events. `classical.submit` sends the request on the
    device's link, a linear value like a line, so the requests on one link
    go out in program order. `classical.wait` is the point where the answer
    is needed; it is ready a bounded latency after the request is sent.
    Pulses that do not depend on the answer are not delayed by it.

    `classical.callable` declares a function: its name (whose 32-bit FNV-1a
    hash identifies it on the wire), the link it is reached on, the types of
    its arguments and results, and its latency.

    ```mlir
    classical.callable @decode {dispatch_key = 3007678287 : i64, device = 0 : i64,
        args = ["bits:2"], results = ["bits:1"],
        latency_min_ns = 500.0, latency_nominal_ns = 900.0,
        latency_max_ns = 1500.0, mode = "bounded", submit = "eager"}
    %link = classical.get_link {device = 0 : i64} : !classical.link
    %link1, %answer = classical.submit %link, @decode(%m0, %m1)
        : (!pulse.measurement, !pulse.measurement)
        -> (!classical.link, !classical.future)
    %fix = classical.wait %answer, @decode : i1
    ```
  }];
  let cppNamespace = "::classical";
  let useDefaultTypePrinterParser = 1;
}

#endif // CUDAQ_PULSE_DIALECT_CLASSICAL
