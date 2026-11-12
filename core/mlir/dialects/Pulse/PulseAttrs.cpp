// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"

using namespace mlir;

#include "cudaq-pulse/Dialect/Pulse/PulseAttrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.cpp.inc"

namespace pulse {

unsigned StaticTermAttr::arity(StaticTermKind kind) {
  switch (kind) {
  case StaticTermKind::ZZ:
    return 2;
  }
  llvm_unreachable("unknown static term kind");
}

LogicalResult
StaticTermAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       StaticTermKind kind, ArrayRef<int64_t> qudits,
                       llvm::APFloat rateHz) {
  const unsigned expected = arity(kind);
  if (qudits.size() != expected)
    return emitError() << "a " << stringifyStaticTermKind(kind)
                       << " term acts on " << expected << " qudits, got "
                       << qudits.size();
  llvm::SmallSet<int64_t, 4> seen;
  for (int64_t qudit : qudits) {
    if (qudit < 0)
      return emitError() << "qudit " << qudit << " is not a device index";
    if (!seen.insert(qudit).second)
      return emitError() << "names qudit " << qudit << " twice";
  }
  if (&rateHz.getSemantics() != &llvm::APFloat::IEEEdouble())
    return emitError() << "rate_hz must be an f64";
  if (!rateHz.isFinite())
    return emitError() << "rate_hz must be finite, got "
                       << (rateHz.isNaN()        ? "nan"
                           : rateHz.isNegative() ? "-inf"
                                                 : "inf");
  return success();
}

// #pulse.static_term<zz, qudits = [3, 4], rate_hz = 4.5e+04>
Attribute StaticTermAttr::parse(AsmParser &parser, Type) {
  StringRef keyword;
  SMLoc kindLoc = parser.getCurrentLocation();
  if (parser.parseLess() || parser.parseKeyword(&keyword))
    return {};
  std::optional<StaticTermKind> kind = symbolizeStaticTermKind(keyword);
  if (!kind) {
    parser.emitError(kindLoc) << "unknown static term kind '" << keyword << "'";
    return {};
  }
  SmallVector<int64_t> qudits;
  llvm::APFloat rateHz(0.0);
  if (parser.parseComma() || parser.parseKeyword("qudits") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       return parser.parseInteger(
                                           qudits.emplace_back());
                                     }) ||
      parser.parseComma() || parser.parseKeyword("rate_hz") ||
      parser.parseEqual() ||
      parser.parseFloat(llvm::APFloat::IEEEdouble(), rateHz) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<StaticTermAttr>(parser.getContext(), *kind, qudits,
                                           rateHz);
}

void StaticTermAttr::print(AsmPrinter &printer) const {
  printer << '<' << stringifyStaticTermKind(getKind()) << ", qudits = [";
  llvm::interleaveComma(getQudits(), printer);
  printer << "], rate_hz = ";
  printer.printFloat(getRateHz());
  printer << '>';
}

namespace {

/// The fields of a `#pulse.<name><key = value, ...>` attribute, each parsed
/// by its own function, in any order and at most once.
struct FieldParser {
  AsmParser &parser;
  StringRef attribute;

  ParseResult
  parse(ArrayRef<std::pair<StringRef, function_ref<ParseResult()>>> fields) {
    if (parser.parseLess())
      return failure();
    if (succeeded(parser.parseOptionalGreater()))
      return success();
    llvm::SmallSet<StringRef, 8> seen;
    do {
      StringRef key;
      SMLoc loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&key))
        return failure();
      const auto *field = llvm::find_if(
          fields, [&](const auto &entry) { return entry.first == key; });
      if (field == fields.end())
        return parser.emitError(loc)
               << "unknown field '" << key << "' of #pulse." << attribute;
      if (!seen.insert(key).second)
        return parser.emitError(loc)
               << "#pulse." << attribute << " gives '" << key << "' twice";
      if (parser.parseEqual() || field->second())
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseGreater();
  }

  ParseResult floats(DenseF64ArrayAttr &out) {
    SmallVector<double> values;
    if (parser.parseCommaSeparatedList(
            AsmParser::Delimiter::Square, [&]() -> ParseResult {
              llvm::APFloat value(0.0);
              if (parser.parseFloat(llvm::APFloat::IEEEdouble(), value))
                return failure();
              values.push_back(value.convertToDouble());
              return success();
            }))
      return failure();
    if (!values.empty())
      out = DenseF64ArrayAttr::get(parser.getContext(), values);
    return success();
  }

  ParseResult integers(SmallVectorImpl<int64_t> &out) {
    return parser.parseCommaSeparatedList(
        AsmParser::Delimiter::Square, [&]() -> ParseResult {
          return parser.parseInteger(out.emplace_back());
        });
  }

  ParseResult integers(DenseI64ArrayAttr &out) {
    SmallVector<int64_t> values;
    if (integers(values))
      return failure();
    if (!values.empty())
      out = DenseI64ArrayAttr::get(parser.getContext(), values);
    return success();
  }

  template <typename Attr>
  ParseResult attributes(ArrayAttr &out, StringRef what) {
    SmallVector<Attribute> values;
    if (parser.parseCommaSeparatedList(
            AsmParser::Delimiter::Square, [&]() -> ParseResult {
              SMLoc loc = parser.getCurrentLocation();
              Attribute value;
              if (parser.parseAttribute(value))
                return failure();
              if (!isa<Attr>(value))
                return parser.emitError(loc) << "expected a " << what;
              values.push_back(value);
              return success();
            }))
      return failure();
    if (!values.empty())
      out = ArrayAttr::get(parser.getContext(), values);
    return success();
  }
};

/// Prints `key = value` fields, a comma between two.
struct FieldPrinter {
  AsmPrinter &printer;
  bool first = true;

  raw_ostream &key(StringRef name) {
    if (!first)
      printer << ", ";
    first = false;
    return printer.getStream() << name << " = ";
  }

  void floats(StringRef name, DenseF64ArrayAttr values) {
    if (!values || values.empty())
      return;
    key(name) << '[';
    llvm::interleaveComma(values.asArrayRef(), printer, [&](double value) {
      printer.printFloat(llvm::APFloat(value));
    });
    printer << ']';
  }

  void integers(StringRef name, ArrayRef<int64_t> values) {
    key(name) << '[';
    llvm::interleaveComma(values, printer);
    printer << ']';
  }

  void integers(StringRef name, DenseI64ArrayAttr values) {
    if (values && !values.empty())
      integers(name, values.asArrayRef());
  }

  void attributes(StringRef name, ArrayAttr values) {
    if (!values || values.empty())
      return;
    key(name) << '[';
    llvm::interleaveComma(values, printer,
                          [&](Attribute value) { printer << value; });
    printer << ']';
  }
};

} // namespace

LogicalResult QuantumSystemAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseF64ArrayAttr quditFrequencyHz, DenseI64ArrayAttr levelCounts,
    DenseF64ArrayAttr anharmonicityHz, DenseF64ArrayAttr t1Ns,
    DenseF64ArrayAttr t2Ns, DenseF64ArrayAttr driveScaleRadPerNs,
    ArrayAttr staticTerms) {
  std::optional<int64_t> count;
  auto sameCount = [&](StringRef name, int64_t size) -> LogicalResult {
    if (size == 0)
      return success();
    if (!count)
      count = size;
    if (*count != size)
      return emitError() << name << " lists " << size
                         << " qudits, and the other lists " << *count;
    return success();
  };
  const std::pair<StringRef, DenseF64ArrayAttr> floats[] = {
      {"qudit_frequency_hz", quditFrequencyHz},
      {"anharmonicity_hz", anharmonicityHz},
      {"t1_ns", t1Ns},
      {"t2_ns", t2Ns},
      {"drive_scale_rad_per_ns", driveScaleRadPerNs}};
  for (auto [name, values] : floats)
    if (values && failed(sameCount(name, values.size())))
      return failure();
  if (levelCounts) {
    if (failed(sameCount("level_counts", levelCounts.size())))
      return failure();
    for (int64_t levels : levelCounts.asArrayRef())
      if (levels < 2)
        return emitError() << "a qudit has at least 2 levels, got " << levels;
  }
  for (auto [name, values] : {floats[2], floats[3]})
    if (values)
      for (double time : values.asArrayRef())
        if (!(time >= 0.0))
          return emitError() << name << " entries are 0 or more, got " << time;
  if (staticTerms)
    for (Attribute term : staticTerms)
      if (!isa<StaticTermAttr>(term))
        return emitError() << "static_terms holds " << term
                           << ", not a #pulse.static_term";
  return success();
}

// #pulse.quantum_system<qudit_frequency_hz = [5.0e+09], level_counts = [3]>
Attribute QuantumSystemAttr::parse(AsmParser &parser, Type) {
  DenseF64ArrayAttr frequencies, anharmonicities, t1, t2, scales;
  DenseI64ArrayAttr levels;
  ArrayAttr terms;
  FieldParser fields{parser, "quantum_system"};
  if (fields.parse({
          {"qudit_frequency_hz", [&] { return fields.floats(frequencies); }},
          {"level_counts", [&] { return fields.integers(levels); }},
          {"anharmonicity_hz", [&] { return fields.floats(anharmonicities); }},
          {"t1_ns", [&] { return fields.floats(t1); }},
          {"t2_ns", [&] { return fields.floats(t2); }},
          {"drive_scale_rad_per_ns", [&] { return fields.floats(scales); }},
          {"static_terms",
           [&] {
             return fields.attributes<StaticTermAttr>(terms,
                                                      "#pulse.static_term");
           }},
      }))
    return {};
  return parser.getChecked<QuantumSystemAttr>(parser.getContext(), frequencies,
                                              levels, anharmonicities, t1, t2,
                                              scales, terms);
}

void QuantumSystemAttr::print(AsmPrinter &printer) const {
  printer << '<';
  FieldPrinter fields{printer};
  fields.floats("qudit_frequency_hz", getQuditFrequencyHz());
  fields.integers("level_counts", getLevelCounts());
  fields.floats("anharmonicity_hz", getAnharmonicityHz());
  fields.floats("t1_ns", getT1Ns());
  fields.floats("t2_ns", getT2Ns());
  fields.floats("drive_scale_rad_per_ns", getDriveScaleRadPerNs());
  fields.attributes("static_terms", getStaticTerms());
  printer << '>';
}

void PulseDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.cpp.inc"
      >();
}

} // namespace pulse
