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

void PulseDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.cpp.inc"
      >();
}

} // namespace pulse
