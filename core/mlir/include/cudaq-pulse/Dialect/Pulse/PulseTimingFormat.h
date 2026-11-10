// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The custom form of a timed op's schedule (`custom<Timing>` in the ops'
// assembly formats): `<{duration_cycles = 20 : i64, start_cycles = 40 : i64}>`
// after the operands, the text `prop-dict` gives, written from the integers
// themselves. prop-dict makes an attribute of every property and a dictionary
// of them each time an op prints; this makes none. An op not scheduled prints
// nothing there, not even a space. Parsing takes the same text, with
// prop-dict's diagnostics.

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>
#include <optional>

namespace pulse {
namespace timing_format {

/// The one other inherent attribute an op keeps in the same dictionary
/// (schedule_block's callee, classical.wait's timeout_ns), printed in its
/// sorted place among the schedule's names.
struct Other {
  llvm::StringRef name;
  mlir::Attribute value;
};

inline void print(mlir::OpAsmPrinter &p, const std::optional<int64_t> &start,
                  const std::optional<int64_t> &duration,
                  llvm::ArrayRef<int64_t> periods, Other other = {}) {
  if (!start && !duration && periods.empty() && !other.value)
    return;
  llvm::raw_ostream &os = p.getStream();
  bool first = true;
  auto entry = [&](llvm::StringRef name) -> llvm::raw_ostream & {
    os << (first ? " <{" : ", ") << name << " = ";
    first = false;
    return os;
  };
  auto otherBefore = [&](llvm::StringRef name) {
    if (other.value && other.name < name) {
      entry(other.name);
      p.printAttribute(other.value);
      other.value = {};
    }
  };
  otherBefore("duration_cycles");
  if (duration)
    entry("duration_cycles") << *duration << " : i64";
  otherBefore("period_cycles");
  if (!periods.empty()) {
    entry("period_cycles") << "array<i64: ";
    llvm::interleaveComma(periods, os);
    os << '>';
  }
  otherBefore("start_cycles");
  if (start)
    entry("start_cycles") << *start << " : i64";
  if (other.value) {
    entry(other.name);
    p.printAttribute(other.value);
  }
  os << "}>";
}

/// Parses an optional `<{...}>`. `takeOther` receives the entry named
/// `otherName`, if the op has one, and says whether it has the right kind.
inline mlir::ParseResult
parse(mlir::OpAsmParser &parser, llvm::StringRef opName,
      std::optional<int64_t> &start, std::optional<int64_t> &duration,
      llvm::SmallVector<int64_t> &periods, llvm::StringRef otherName = {},
      llvm::function_ref<bool(mlir::Attribute)> takeOther = nullptr) {
  if (mlir::failed(parser.parseOptionalLess()))
    return mlir::success();
  mlir::Attribute attr;
  if (parser.parseAttribute(attr) || parser.parseGreater())
    return mlir::failure();
  // Where and how the generic parser reports properties it cannot take: at
  // the op's name, without the "custom op" prefix of parser.emitError.
  auto emitError = [&]() {
    return mlir::emitError(parser.getEncodedSourceLoc(parser.getNameLoc()))
           << "invalid properties " << attr << " for op " << opName << ": ";
  };
  auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
  if (!dict)
    return emitError() << "expected DictionaryAttr to set properties";
  if (takeOther)
    if (mlir::Attribute value = dict.get(otherName); value && !takeOther(value))
      return emitError() << "Invalid attribute `" << otherName
                         << "` in property conversion: " << value;
  auto cycles = [&](llvm::StringRef name,
                    std::optional<int64_t> &to) -> mlir::LogicalResult {
    mlir::Attribute value = dict.get(name);
    if (!value)
      return mlir::success();
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
      to = integer.getInt();
      return mlir::success();
    }
    auto none = llvm::dyn_cast<mlir::ArrayAttr>(value);
    if (!none || !none.empty())
      return emitError() << "expected a number of cycles, an i64";
    to = std::nullopt;
    return mlir::success();
  };
  if (mlir::failed(cycles("start_cycles", start)) ||
      mlir::failed(cycles("duration_cycles", duration)))
    return mlir::failure();
  if (mlir::Attribute value = dict.get("period_cycles")) {
    auto array = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(value);
    if (!array)
      return emitError() << "expected periods as array<i64: ...>";
    periods.assign(array.asArrayRef().begin(), array.asArrayRef().end());
  }
  return mlir::success();
}

} // namespace timing_format

inline void printTiming(mlir::OpAsmPrinter &p, mlir::Operation *,
                        llvm::StringRef, const std::optional<int64_t> &start,
                        const std::optional<int64_t> &duration,
                        llvm::ArrayRef<int64_t> periods) {
  timing_format::print(p, start, duration, periods);
}

inline mlir::ParseResult parseTiming(mlir::OpAsmParser &parser,
                                     llvm::StringRef opName,
                                     std::optional<int64_t> &start,
                                     std::optional<int64_t> &duration,
                                     llvm::SmallVector<int64_t> &periods) {
  return timing_format::parse(parser, opName, start, duration, periods);
}

/// schedule_block: its callee shares the dictionary.
inline void printTiming(mlir::OpAsmPrinter &p, mlir::Operation *,
                        llvm::StringRef, const std::optional<int64_t> &start,
                        const std::optional<int64_t> &duration,
                        llvm::ArrayRef<int64_t> periods,
                        mlir::StringAttr callee) {
  timing_format::print(p, start, duration, periods, {"callee", callee});
}

inline mlir::ParseResult
parseTiming(mlir::OpAsmParser &parser, llvm::StringRef opName,
            std::optional<int64_t> &start, std::optional<int64_t> &duration,
            llvm::SmallVector<int64_t> &periods, mlir::StringAttr &callee) {
  return timing_format::parse(parser, opName, start, duration, periods,
                              "callee", [&](mlir::Attribute value) {
                                callee =
                                    llvm::dyn_cast<mlir::StringAttr>(value);
                                return bool(callee);
                              });
}

/// classical.wait: its timeout_ns shares the dictionary.
inline void printTiming(mlir::OpAsmPrinter &p, mlir::Operation *,
                        llvm::StringRef, const std::optional<int64_t> &start,
                        const std::optional<int64_t> &duration,
                        llvm::ArrayRef<int64_t> periods,
                        mlir::FloatAttr timeout) {
  timing_format::print(p, start, duration, periods, {"timeout_ns", timeout});
}

inline mlir::ParseResult
parseTiming(mlir::OpAsmParser &parser, llvm::StringRef opName,
            std::optional<int64_t> &start, std::optional<int64_t> &duration,
            llvm::SmallVector<int64_t> &periods, mlir::FloatAttr &timeout) {
  return timing_format::parse(parser, opName, start, duration, periods,
                              "timeout_ns", [&](mlir::Attribute value) {
                                timeout =
                                    llvm::dyn_cast<mlir::FloatAttr>(value);
                                return bool(timeout);
                              });
}

} // namespace pulse
