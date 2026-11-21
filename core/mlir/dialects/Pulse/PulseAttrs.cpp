// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"

#include <cmath>
#include <numeric>

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

/// The fields of a `#pulse.<name><key = value, flag, ...>` attribute, each
/// value parsed by its own function and each flag a bare keyword, in any
/// order and at most once.
struct FieldParser {
  using Fields = ArrayRef<std::pair<StringRef, function_ref<ParseResult()>>>;
  using Flags = ArrayRef<std::pair<StringRef, bool *>>;

  AsmParser &parser;
  StringRef attribute;

  /// `<` fields `>`, none allowed.
  ParseResult parse(Fields fields, Flags flags = {}) {
    if (parser.parseLess())
      return failure();
    if (succeeded(parser.parseOptionalGreater()))
      return success();
    return parseRest(fields, flags);
  }

  /// One or more fields, then `>`.
  ParseResult parseRest(Fields fields, Flags flags = {}) {
    llvm::SmallSet<StringRef, 8> seen;
    do {
      StringRef key;
      SMLoc loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&key))
        return failure();
      const auto *field = llvm::find_if(
          fields, [&](const auto &entry) { return entry.first == key; });
      const auto *flag = llvm::find_if(
          flags, [&](const auto &entry) { return entry.first == key; });
      if (field == fields.end() && flag == flags.end())
        return parser.emitError(loc)
               << "unknown field '" << key << "' of #pulse." << attribute;
      if (!seen.insert(key).second)
        return parser.emitError(loc)
               << "#pulse." << attribute << " gives '" << key << "' twice";
      if (flag != flags.end())
        *flag->second = true;
      else if (parser.parseEqual() || field->second())
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseGreater();
  }

  ParseResult integer(int64_t &out) { return parser.parseInteger(out); }

  ParseResult real(llvm::APFloat &out) {
    return parser.parseFloat(llvm::APFloat::IEEEdouble(), out);
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

  void integer(StringRef name, int64_t value) { key(name) << value; }

  void flag(StringRef name, bool set) {
    if (!set)
      return;
    if (!first)
      printer << ", ";
    first = false;
    printer << name;
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

// #pulse.line<drive, qudits = [0, 1], max_concurrent_tones = 2>
LogicalResult LineAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               LineKind kind, ArrayRef<int64_t> qudits,
                               int64_t maxConcurrentTones, int64_t delayCycles,
                               int64_t controller, int64_t fineCycles) {
  if (qudits.empty())
    return emitError() << "a line reaches at least one qudit";
  llvm::SmallSet<int64_t, 4> seen;
  for (int64_t qudit : qudits) {
    if (qudit < 0)
      return emitError() << "qudit " << qudit << " is not a device index";
    if (!seen.insert(qudit).second)
      return emitError() << "a line reaches qudit " << qudit << " twice";
  }
  if (maxConcurrentTones < 1)
    return emitError() << "max_concurrent_tones is 1 or more, got "
                       << maxConcurrentTones;
  if (controller < 0)
    return emitError() << "controller " << controller
                       << " is not a controller index";
  if (fineCycles < 0)
    return emitError() << "fine_cycles is 0 or more, got " << fineCycles;
  return success();
}

Attribute LineAttr::parse(AsmParser &parser, Type) {
  StringRef keyword;
  SMLoc kindLoc = parser.getCurrentLocation();
  if (parser.parseLess() || parser.parseKeyword(&keyword))
    return {};
  std::optional<LineKind> kind = symbolizeLineKind(keyword);
  if (!kind) {
    parser.emitError(kindLoc)
        << "unknown line kind '" << keyword << "'; a line is drive or readout";
    return {};
  }
  SmallVector<int64_t> qudits;
  int64_t tones = 1, delay = 0, controller = 0, fine = 0;
  FieldParser fields{parser, "line"};
  if (succeeded(parser.parseOptionalComma())) {
    if (fields.parseRest({
            {"qudits", [&] { return fields.integers(qudits); }},
            {"max_concurrent_tones", [&] { return fields.integer(tones); }},
            {"delay_cycles", [&] { return fields.integer(delay); }},
            {"controller", [&] { return fields.integer(controller); }},
            {"fine_cycles", [&] { return fields.integer(fine); }},
        }))
      return {};
  } else if (parser.parseGreater()) {
    return {};
  }
  return parser.getChecked<LineAttr>(parser.getContext(), *kind, qudits, tones,
                                     delay, controller, fine);
}

void LineAttr::print(AsmPrinter &printer) const {
  printer << '<' << stringifyLineKind(getKind());
  FieldPrinter fields{printer, /*first=*/false};
  fields.integers("qudits", getQudits());
  if (getMaxConcurrentTones() != 1)
    fields.integer("max_concurrent_tones", getMaxConcurrentTones());
  if (getDelayCycles() != 0)
    fields.integer("delay_cycles", getDelayCycles());
  if (getController() != 0)
    fields.integer("controller", getController());
  if (getFineCycles() != 0)
    fields.integer("fine_cycles", getFineCycles());
  printer << '>';
}

LogicalResult ControllerAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, int64_t gridCycles,
    int64_t clockRatioNumerator, int64_t clockRatioDenominator,
    int64_t feedbackLatencyCycles, int64_t peerLatencyCycles,
    ArrayRef<std::pair<int64_t, int64_t>> peerLatencyByController,
    int64_t startUncertaintyCycles, bool persistentFrames, bool blocksOnCalls,
    int64_t maxDurationCycles) {
  if (gridCycles < 1)
    return emitError() << "grid_cycles is 1 or more, got " << gridCycles;
  if (clockRatioNumerator < 1 || clockRatioDenominator < 1)
    return emitError() << "clock_ratio is positive, got " << clockRatioNumerator
                       << "/" << clockRatioDenominator;
  if (std::gcd(clockRatioNumerator, clockRatioDenominator) != 1)
    return emitError() << "clock_ratio " << clockRatioNumerator << "/"
                       << clockRatioDenominator << " is not in lowest terms";
  for (auto [name, cycles] :
       {std::pair<StringRef, int64_t>{"feedback_latency_cycles",
                                      feedbackLatencyCycles},
        {"peer_latency_cycles", peerLatencyCycles},
        {"start_uncertainty_cycles", startUncertaintyCycles},
        {"max_duration_cycles", maxDurationCycles}})
    if (cycles < 0)
      return emitError() << name << " is 0 or more, got " << cycles;
  if (!peerLatencyByController.empty() && peerLatencyCycles != 0)
    return emitError()
           << "peer_latency_cycles is one value or one by controller, not both";
  int64_t previous = -1;
  for (auto [controller, cycles] : peerLatencyByController) {
    if (controller <= previous)
      return emitError() << "peer_latency_cycles names controllers by "
                            "increasing index, got "
                         << controller << " after " << previous;
    if (cycles < 0)
      return emitError() << "peer_latency_cycles is 0 or more, got " << cycles;
    previous = controller;
  }
  return success();
}

// #pulse.controller<clock_ratio = 3/5, peer_latency_cycles = {1 = 10}>
Attribute ControllerAttr::parse(AsmParser &parser, Type) {
  int64_t grid = 1, numerator = 1, denominator = 1, feedback = 0, peer = 0,
          uncertainty = 0, budget = 0;
  SmallVector<std::pair<int64_t, int64_t>> peers;
  bool persistent = false, blocks = false;
  FieldParser fields{parser, "controller"};
  // `N`, or `{index = N, ...}`. An unreachable peer (`index = none`) is kept
  // for later.
  auto peerLatency = [&]() -> ParseResult {
    if (failed(parser.parseOptionalLBrace()))
      return parser.parseInteger(peer);
    if (succeeded(parser.parseOptionalRBrace()))
      return success();
    do {
      auto &[controller, cycles] = peers.emplace_back();
      if (parser.parseInteger(controller) || parser.parseEqual())
        return failure();
      SMLoc loc = parser.getCurrentLocation();
      if (succeeded(parser.parseOptionalKeyword("none")))
        return parser.emitError(loc)
               << "unreachable peers are not supported yet";
      if (parser.parseInteger(cycles))
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    return parser.parseRBrace();
  };
  if (fields.parse(
          {
              {"grid_cycles", [&] { return fields.integer(grid); }},
              {"clock_ratio",
               [&]() -> ParseResult {
                 if (parser.parseInteger(numerator))
                   return failure();
                 if (succeeded(parser.parseOptionalSlash()))
                   return parser.parseInteger(denominator);
                 return success();
               }},
              {"feedback_latency_cycles",
               [&] { return fields.integer(feedback); }},
              {"peer_latency_cycles", peerLatency},
              {"start_uncertainty_cycles",
               [&] { return fields.integer(uncertainty); }},
              {"max_duration_cycles", [&] { return fields.integer(budget); }},
          },
          {{"persistent_frames", &persistent}, {"blocks_on_calls", &blocks}}))
    return {};
  return parser.getChecked<ControllerAttr>(
      parser.getContext(), grid, numerator, denominator, feedback, peer, peers,
      uncertainty, persistent, blocks, budget);
}

void ControllerAttr::print(AsmPrinter &printer) const {
  printer << '<';
  FieldPrinter fields{printer};
  if (getGridCycles() != 1)
    fields.integer("grid_cycles", getGridCycles());
  if (getClockRatioNumerator() != 1 || getClockRatioDenominator() != 1) {
    fields.key("clock_ratio") << getClockRatioNumerator();
    if (getClockRatioDenominator() != 1)
      printer << '/' << getClockRatioDenominator();
  }
  if (getFeedbackLatencyCycles() != 0)
    fields.integer("feedback_latency_cycles", getFeedbackLatencyCycles());
  if (!getPeerLatencyByController().empty()) {
    fields.key("peer_latency_cycles") << '{';
    llvm::interleaveComma(getPeerLatencyByController(), printer,
                          [&](const std::pair<int64_t, int64_t> &entry) {
                            printer << entry.first << " = " << entry.second;
                          });
    printer << '}';
  } else if (getPeerLatencyCycles() != 0) {
    fields.integer("peer_latency_cycles", getPeerLatencyCycles());
  }
  if (getStartUncertaintyCycles() != 0)
    fields.integer("start_uncertainty_cycles", getStartUncertaintyCycles());
  fields.flag("persistent_frames", getPersistentFrames());
  fields.flag("blocks_on_calls", getBlocksOnCalls());
  if (getMaxDurationCycles() != 0)
    fields.integer("max_duration_cycles", getMaxDurationCycles());
  printer << '>';
}

// Periods and their clocks convert exactly: both parts of the fraction, and
// 1e6 times its denominator, are whole doubles.
static constexpr int64_t kExactLimit = int64_t(1) << 53;
static constexpr int64_t kFsPerGHz = 1000000;

double ControlSystemAttr::clockGHz() const {
  return static_cast<double>(getClockPeriodFsDenominator() * kFsPerGHz) /
         static_cast<double>(getClockPeriodFsNumerator());
}

std::optional<std::pair<int64_t, int64_t>>
ControlSystemAttr::periodFs(double clockGHz) {
  if (!(clockGHz > 0.0) || !std::isfinite(clockGHz))
    return std::nullopt;
  // clockGHz = a / b exactly, b a power of two (or a whole clock, b = 1).
  int exponent = 0;
  double mantissa = std::frexp(clockGHz, &exponent);
  auto a = static_cast<uint64_t>(std::ldexp(mantissa, 53));
  int shift = 53 - exponent;
  while (shift > 0 && !(a & 1)) {
    a >>= 1;
    --shift;
  }
  if (shift < 0 || shift > 62)
    return std::nullopt;
  uint64_t b = uint64_t(1) << shift;
  // The first convergent p / q of a / b's continued fraction whose double
  // quotient is clockGHz: the last one is a / b itself.
  uint64_t p0 = 0, p1 = 1, q0 = 1, q1 = 0;
  for (uint64_t x = a, y = b; y != 0;) {
    uint64_t term = x / y;
    uint64_t p = term * p1 + p0, q = term * q1 + q0;
    if (p >= uint64_t(kExactLimit) || q >= uint64_t(kExactLimit))
      return std::nullopt;
    if (static_cast<double>(p) / static_cast<double>(q) == clockGHz) {
      // The period in fs is 1e6 q / p.
      uint64_t numerator = q * kFsPerGHz, denominator = p;
      if (q > uint64_t(kExactLimit / kFsPerGHz))
        return std::nullopt;
      uint64_t common = std::gcd(numerator, denominator);
      numerator /= common;
      denominator /= common;
      if (denominator > uint64_t(kExactLimit / kFsPerGHz))
        return std::nullopt;
      return std::pair<int64_t, int64_t>(numerator, denominator);
    }
    p0 = p1;
    p1 = p;
    q0 = q1;
    q1 = q;
    uint64_t rest = x - term * y;
    x = y;
    y = rest;
  }
  return std::nullopt;
}

// The latencies between the controllers of a system of several, in program
// cycles, fit 64 bits: ceil((feedback_r + peer_r(u) + uncertainty_r) *
// den_r / num_r + uncertainty_u * den_u / num_u), which bounds the latency on
// a controller itself and the spreads too. Every product below fits 128 bits.
static LogicalResult latenciesFit(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<ControllerAttr> controllers) {
  using Unsigned = unsigned __int128;
  if (controllers.size() < 2)
    return success();
  constexpr auto kMost = Unsigned(INT64_MAX);
  auto late = [](ControllerAttr c, int64_t peer) {
    return (Unsigned(c.getFeedbackLatencyCycles()) + peer +
            c.getStartUncertaintyCycles()) *
           c.getClockRatioDenominator();
  };
  auto start = [](ControllerAttr c) {
    return Unsigned(c.getStartUncertaintyCycles()) *
           c.getClockRatioDenominator();
  };
  auto ceilOf = [](Unsigned value, int64_t by) {
    return value / Unsigned(by) + Unsigned(value % Unsigned(by) != 0);
  };
  // A bound first: the latest reader plus the least known start.
  Unsigned latest = 0, least = 0;
  for (ControllerAttr c : controllers) {
    int64_t peer = c.getPeerLatencyCycles();
    for (auto [index, cycles] : c.getPeerLatencyByController())
      peer = std::max(peer, cycles);
    latest =
        std::max(latest, ceilOf(late(c, peer), c.getClockRatioNumerator()));
    least = std::max(least, ceilOf(start(c), c.getClockRatioNumerator()));
  }
  if (latest + least <= kMost)
    return success();
  // Then every pair, exactly: whole parts and the carry of the remainders.
  for (auto [r, reader] : llvm::enumerate(controllers)) {
    Unsigned num = Unsigned(reader.getClockRatioNumerator());
    for (auto [u, user] : llvm::enumerate(controllers)) {
      if (r == u)
        continue;
      int64_t peer = reader.getPeerLatencyCycles();
      for (auto [index, cycles] : reader.getPeerLatencyByController())
        if (index == static_cast<int64_t>(u))
          peer = cycles;
      Unsigned a = late(reader, peer), x = start(user);
      Unsigned userNum = Unsigned(user.getClockRatioNumerator());
      Unsigned aRest = a % num, xRest = x % userNum;
      Unsigned total = a / num + x / userNum;
      if (aRest || xRest)
        total += 1 + Unsigned(aRest * userNum + xRest * num > num * userNum);
      if (total > kMost)
        return emitError()
               << "the outcome of a readout on controller " << r
               << " reaches controller " << u
               << " more than 2^63 - 1 program cycles after the readout ends";
    }
  }
  return success();
}

LogicalResult
ControlSystemAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          int64_t numerator, int64_t denominator,
                          ArrayRef<LineAttr> lines,
                          ArrayRef<ControllerAttr> controllers) {
  if (numerator < 1 || denominator < 1)
    return emitError() << "clock_period_fs is positive, got " << numerator
                       << "/" << denominator;
  if (std::gcd(numerator, denominator) != 1)
    return emitError() << "clock_period_fs " << numerator << "/" << denominator
                       << " is not in lowest terms";
  if (numerator >= kExactLimit || denominator > kExactLimit / kFsPerGHz)
    return emitError() << "clock_period_fs " << numerator << "/" << denominator
                       << " has no exact clock";
  auto count = static_cast<int64_t>(controllers.size());
  for (auto [index, controller] : llvm::enumerate(controllers)) {
    for (auto [peer, cycles] : controller.getPeerLatencyByController()) {
      if (peer == static_cast<int64_t>(index))
        return emitError() << "controller " << index
                           << "'s peer_latency_cycles names itself";
      if (peer >= count)
        return emitError() << "controller " << index
                           << "'s peer_latency_cycles names controller " << peer
                           << ", of " << count;
    }
  }
  if (failed(latenciesFit(emitError, controllers)))
    return failure();
  llvm::SmallSet<std::pair<int64_t, int64_t>, 16> reached;
  for (LineAttr line : lines) {
    if (line.getController() >= std::max<int64_t>(count, 1))
      return emitError() << "a line is on controller " << line.getController()
                         << ", of " << std::max<int64_t>(count, 1);
    for (int64_t qudit : line.getQudits())
      if (!reached.insert({static_cast<int64_t>(line.getKind()), qudit}).second)
        return emitError() << "two " << stringifyLineKind(line.getKind())
                           << " lines reach qudit " << qudit;
  }
  return success();
}

// #pulse.control_system<clock_period_fs = 1250000/3, lines = [..]>
Attribute ControlSystemAttr::parse(AsmParser &parser, Type) {
  int64_t numerator = 0, denominator = 1;
  bool clock = false;
  SmallVector<LineAttr> lines;
  SmallVector<ControllerAttr> controllers;
  FieldParser fields{parser, "control_system"};
  auto list = [&](auto &out, StringRef what) {
    return parser.parseCommaSeparatedList(
        AsmParser::Delimiter::Square, [&]() -> ParseResult {
          SMLoc loc = parser.getCurrentLocation();
          Attribute value;
          if (parser.parseAttribute(value))
            return failure();
          auto entry = dyn_cast<std::decay_t<decltype(out[0])>>(value);
          if (!entry)
            return parser.emitError(loc) << "expected a " << what;
          out.push_back(entry);
          return success();
        });
  };
  SMLoc loc = parser.getCurrentLocation();
  if (fields.parse({
          {"clock_period_fs",
           [&]() -> ParseResult {
             clock = true;
             if (parser.parseInteger(numerator))
               return failure();
             if (succeeded(parser.parseOptionalSlash()))
               return parser.parseInteger(denominator);
             return success();
           }},
          {"lines", [&] { return list(lines, "#pulse.line"); }},
          {"controllers",
           [&] { return list(controllers, "#pulse.controller"); }},
      }))
    return {};
  if (!clock) {
    parser.emitError(loc) << "#pulse.control_system needs its clock_period_fs";
    return {};
  }
  return parser.getChecked<ControlSystemAttr>(parser.getContext(), numerator,
                                              denominator, lines, controllers);
}

void ControlSystemAttr::print(AsmPrinter &printer) const {
  printer << "<clock_period_fs = " << getClockPeriodFsNumerator();
  if (getClockPeriodFsDenominator() != 1)
    printer << '/' << getClockPeriodFsDenominator();
  auto list = [&](StringRef name, auto values) {
    if (values.empty())
      return;
    printer << ", " << name << " = [";
    llvm::interleaveComma(values, printer,
                          [&](Attribute value) { printer << value; });
    printer << ']';
  };
  list("lines", getLines());
  list("controllers", getControllers());
  printer << '>';
}

void PulseDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.cpp.inc"
      >();
}

} // namespace pulse
