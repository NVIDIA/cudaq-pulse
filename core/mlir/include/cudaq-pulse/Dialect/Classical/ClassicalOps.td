// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_CLASSICAL_OPS
#define CUDAQ_PULSE_DIALECT_CLASSICAL_OPS

include "cudaq-pulse/Dialect/Classical/ClassicalDialect.td"
include "cudaq-pulse/Dialect/Classical/ClassicalTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ClassicalOp<string mnemonic, list<Trait> traits = []> :
    Op<ClassicalDialect, mnemonic, traits>;

def Classical_CallableOp : ClassicalOp<"callable", [Symbol]> {
  let summary = "Declare a classical function the program calls at runtime";
  let description = [{
    The schema and timing of one classical function.

    - `function_id`: the 32-bit FNV-1a hash of `sym_name`, which identifies
      the function on the wire. With `function_id_override` it may be any
      32-bit value instead.
    - `device`: the id of the link the requests travel on.
    - `args`, `results`: one type descriptor per argument and result, in
      wire order: `u8`, `i32`, `i64`, `f32`, `f64`; `bits:N` (`N`
      measurements, sent as `outcome != 0`, bit-packed LSB-first);
      `levels:N` (`N` measurement outcomes, one byte each); `iq:N` (the IQ
      points of `N` readouts, as `2N` float32 values, I then Q); and arrays
      `u8[N]`, `i32[N]`, `f32[N]`, `f64[N]`. At most 8 arguments and 4
      results. A result `bits:N` gives `N` `i1` values, `levels:N` gives
      `N` `i64`s, `iq:N` gives `2N` `f64`s, an array gives its elements, and
      a scalar one value.
    - `latency_min_ns`, `latency_nominal_ns`, `latency_max_ns`: how long the
      answer takes, from the end of the submission until it is back.
      Scheduling budgets `latency_max_ns`.
    - `mode`: `"bounded"` (the answer is assumed back within the maximum),
      `"timeout"` (wait gives a leading `i1` that is false when no answer
      came within `timeout_ns`, which is what scheduling then budgets), or
      `"blocking"` (the program stops until the answer arrives; the wait
      starts a new timing segment).
    - `submit`: `"eager"` (sent as soon as its arguments are known, under
      every scheduling policy) or `"lazy"` (placed by the policy like any
      timed op).
    - `deadline_ns` (optional): the answer must be back within this long of
      the start of the submission; `pulse-classical-check` rejects a
      schedule that cannot meet it.
    - `reaction_deadline_ns` (optional): in every execution of a call, the answer
      must be due within this long of the end of the latest readout whose
      outcome the request carries, or of the start of the submission when
      it carries none; `pulse-classical-check` rejects a schedule where it
      is not.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$function_id,
    I64Attr:$device,
    StrArrayAttr:$args,
    StrArrayAttr:$results,
    F64Attr:$latency_min_ns,
    F64Attr:$latency_nominal_ns,
    F64Attr:$latency_max_ns,
    StrAttr:$mode,
    StrAttr:$submit,
    OptionalAttr<F64Attr>:$timeout_ns,
    OptionalAttr<F64Attr>:$deadline_ns,
    UnitAttr:$function_id_override,
    OptionalAttr<F64Attr>:$reaction_deadline_ns
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Classical_GetLinkOp : ClassicalOp<"get_link"> {
  let summary = "The link to one classical device";
  let description = [{
    The link of device `device`, at time 0. Like a line, it is a linear
    value that each `classical.submit` consumes and returns.
  }];
  let arguments = (ins I64Attr:$device);
  let results = (outs Classical_LinkType:$link);
  let assemblyFormat = "attr-dict `:` qualified(type($link))";
  let hasVerifier = 1;
}

def Classical_SubmitOp : ClassicalOp<"submit",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Send a request to a classical function";
  let description = [{
    Sends the request for `callee` on `link`, with `args` in the order the
    function's argument descriptors take them: one SSA value per scalar or
    array element, and one `!pulse.measurement` per measurement of a
    `bits`, `levels` or `iq` argument. The submission cannot start before
    its arguments are available, and it holds the link for the time the
    request takes to send. It returns the link and a future that becomes
    available when the answer is due.
  }];
  let arguments = (ins
    Classical_LinkType:$link,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Classical_LinkType:$updated_link,
                      Classical_FutureType:$future);
  let assemblyFormat = [{
    $link `,` $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}

def Classical_WaitOp : ClassicalOp<"wait",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Take the answer to a request";
  let description = [{
    The values `callee` returned for the request `future` stands for. They
    are available when the future is: the end of the submission plus the
    budgeted latency (and the link's feedback latency). Pulses whose
    parameters, tones or branch conditions are computed from them start no
    earlier. Scheduling records that time as `start_cycles`.

    With `timeout_ns` (or when the function's mode is `"timeout"`) the wait
    gives a leading `i1`, false when no answer came within the timeout;
    scheduling then budgets the timeout instead of the maximum latency.
  }];
  let arguments = (ins Classical_FutureType:$future, FlatSymbolRefAttr:$callee,
                       OptionalAttr<F64Attr>:$timeout_ns);
  let results = (outs Variadic<AnyType>:$values);
  let assemblyFormat = "$future `,` $callee attr-dict `:` type($values)";
}

#endif // CUDAQ_PULSE_DIALECT_CLASSICAL_OPS
