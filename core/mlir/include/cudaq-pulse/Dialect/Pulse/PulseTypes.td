// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_PULSE_TYPES
#define CUDAQ_PULSE_DIALECT_PULSE_TYPES

include "cudaq-pulse/Dialect/Pulse/PulseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class PulseType<string name, string typeMnemonic, list<Trait> traits = [],
        string baseCppClass = "mlir::Type">
    : TypeDef<PulseDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Qudit
//===----------------------------------------------------------------------===//

def QuditType : PulseType<"Qudit", "qudit"> {
  let summary = "A qudit of the quantum system";
  let description = [{
    A reference to one qudit, as `pulse.qudit_alloc` gives it or as a
    function takes a runtime qudit. Its number of levels is the quantum
    system's, not part of the type.
  }];
}

//===----------------------------------------------------------------------===//
// Measurement result
//===----------------------------------------------------------------------===//

def MeasurementType : PulseType<"Measurement", "measurement"> {
  let summary = "The level a readout observed";
}

//===----------------------------------------------------------------------===//
// Drive and readout lines (linear resources)
//===----------------------------------------------------------------------===//

def DriveLineType : PulseType<"DriveLine", "drive_line"> {
  let summary = "Physical drive channel (linear resource)";
  let description = [{
    A `drive_line` represents a physical drive channel. It is a linear resource:
    each value is produced exactly once and consumed exactly once. Operations
    that consume a drive line produce a new value with advanced time state.
  }];
}

def ReadoutLineType : PulseType<"ReadoutLine", "readout_line"> {
  let summary = "Physical readout channel (linear resource)";
  let description = [{
    A `readout_line` represents a physical readout channel. Like drive lines,
    readout lines are linear resources with time state.
  }];
}

//===----------------------------------------------------------------------===//
// Tone and timing types
//===----------------------------------------------------------------------===//

def ToneType : PulseType<"Tone", "tone"> {
  let summary = "Carrier signal tone (frequency and phase, linear resource)";
  let description = [{
    A `tone` represents a carrier tone tracking frequency and phase. Tones are
    separate from physical lines, enabling cross-resonance and
    position-independent calibrations.
  }];
}

def TimepointType : PulseType<"Timepoint", "timepoint"> {
  let summary = "Abstract point in time on a line";
}

def DurationType : PulseType<"Duration", "duration"> {
  let summary = "Time interval in clock cycles";
}

//===----------------------------------------------------------------------===//
// Waveform (value type, not linear)
//===----------------------------------------------------------------------===//

def WaveformType : PulseType<"Waveform", "waveform"> {
  let summary = "Pulse envelope shape (value type, not linear)";
  let description = [{
    A `waveform` represents a time-domain pulse envelope. Waveforms are value
    types and can be freely passed around and applied to multiple lines.
  }];
}

//===----------------------------------------------------------------------===//
// IQ data
//===----------------------------------------------------------------------===//

def IQDataType : PulseType<"IQData", "iq_data"> {
  let summary = "Raw IQ signal data from readout (value type)";
  let description = [{
    An IQ point: what a `pulse.readout` of mode `integrated` gives, which a
    classical call takes as an `iq` argument, and what `pulse.iq_acquire`
    returns. `pulse.extract_iq`, `pulse.iq_magnitude` and `pulse.iq_phase`
    read its components. It holds no level: `pulse.is_set` and
    `pulse.outcome` read only a `!pulse.measurement`.
  }];
}

//===----------------------------------------------------------------------===//
// Readout modes and scheduling policies
//===----------------------------------------------------------------------===//

def ReadoutMode : I32Enum<"ReadoutMode", "readout mode", [
    I32EnumCase<"Level", 0, "level">,
    I32EnumCase<"Discriminated", 1, "discriminated">,
    I32EnumCase<"Trace", 2, "trace">,
    I32EnumCase<"Counts", 3, "counts">,
    I32EnumCase<"Integrated", 4, "integrated">
  ]> {
  let cppNamespace = "::pulse";
}

def SchedulePolicy : I32Enum<"SchedulePolicy", "schedule policy", [
    I32EnumCase<"ASAP", 0, "asap">,
    I32EnumCase<"ALAP", 1, "alap">
  ]> {
  let cppNamespace = "::pulse";
}

//===----------------------------------------------------------------------===//
// Composite type predicates
//===----------------------------------------------------------------------===//

def AnyLineType : Type<
  Or<[DriveLineType.predicate, ReadoutLineType.predicate]>,
  "any pulse line type"
>;

// What a pulse.readout gives: the level it observed, or the IQ point of a
// readout of mode integrated.
def ReadoutResultType : AnyTypeOf<[MeasurementType, IQDataType],
  "the level or IQ point a readout acquires">;

// Values that become available at a time: lines, tones, measurements and IQ
// points, and the links and call handles of the classical dialect.
def AnyTimedType : Type<
  Or<[DriveLineType.predicate, ReadoutLineType.predicate,
      ToneType.predicate, MeasurementType.predicate, IQDataType.predicate,
      CPred<"$_self.getDialect().getNamespace() == \"classical\"">]>,
  "a pulse line, tone, measurement, IQ point, link or call handle"
>;

#endif // CUDAQ_PULSE_DIALECT_PULSE_TYPES
