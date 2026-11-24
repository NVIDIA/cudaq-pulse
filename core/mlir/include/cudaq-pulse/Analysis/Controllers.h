// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The controllers of a program whose lines several controllers play: which
// one plays each line, and what crossing between two of them costs.
//
// A module's control system (`pulse.control_system`) of several controllers
// gives, on each line, the controller that plays it, and on each controller
// the facts its costs follow from, in its own cycles: `clock_ratio` (its
// cycles per program cycle), `feedback_latency_cycles` (from the end of one
// of its readouts until the outcome is usable on it), `peer_latency_cycles`
// (from there until it reaches another controller, one value for every peer
// or by controller index) and `start_uncertainty_cycles` (how well its start
// is known). In program cycles, an outcome read on `r` is usable on `r`
// ceil(feedback_r / ratio_r) after the readout ends, and on another
// controller `u` ceil((feedback_r + peer_r(u)) / ratio_r + uncertainty_r /
// ratio_r + uncertainty_u / ratio_u) after; the starts of two controllers
// may be ceil(uncertainty_a / ratio_a + uncertainty_b / ratio_b) apart.
//
// Each controller's values in program cycles are kept as a whole part and a
// remainder over its clock ratio's numerator, exactly for any 64-bit facts
// (the control system's verifier rejects facts whose latencies do not fit
// 64 bits). Up to `kTabulated` controllers, every latency and spread is
// computed once into a table each query reads. Above, each query adds the
// whole parts and the carry of the remainders, in O(1) for two given
// controllers and in a few steps besides one per peer value given by index
// for one of "any". A line of no known controller, such as a runtime
// qudit's, may be on any. A module with one controller pays one attribute
// lookup.
//
// Include after the Pulse dialect op and type headers and LineTrace.h.

#pragma once

#include "cudaq-pulse/Analysis/Variant.h"

#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Compiler.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

namespace pulse {
PULSE_VARIANT_OPEN

class Controllers {
public:
  /// Up to this many controllers, the latencies and spreads between every
  /// two are tabulated; above, each query computes its answer.
  static constexpr int64_t kTabulated = 64;

  Controllers() = default;

  explicit Controllers(mlir::ModuleOp module)
      : Controllers(ClockAttributes(module)) {}

  /// Of the control system `attrs` read, if it has several controllers.
  explicit Controllers(const ClockAttributes &attrs) {
    if (!attrs.system)
      return;
    llvm::ArrayRef<ControllerAttr> controllers = attrs.system.getControllers();
    auto size = static_cast<int64_t>(controllers.size());
    if (size < 2)
      return;
    n = size;
    drive = attrs.driveOwner;
    readout = attrs.readoutOwner;
    tabulated = n <= kTabulated;
    llvm::SmallVector<Facts, 8> local;
    llvm::SmallVectorImpl<Facts> &each =
        tabulated ? static_cast<llvm::SmallVectorImpl<Facts> &>(local) : facts;
    each.resize_for_overwrite(controllers.size());
    // The clock ratios' common numerator, if small enough that remainders
    // over it add without overflow; 0 if not.
    whole = 1;
    for (auto [i, controller] : llvm::enumerate(controllers)) {
      Facts &f = each[i];
      f.num = controller.getClockRatioNumerator();
      f.den = controller.getClockRatioDenominator();
      f.feedback = controller.getFeedbackLatencyCycles();
      f.peer = controller.getPeerLatencyCycles();
      f.peers = controller.getPeerLatencyByController();
      f.uncertainty = controller.getStartUncertaintyCycles();
      if (whole > 1 || (whole == 1 && f.num != 1)) {
        Unsigned common = Unsigned(whole / std::gcd(whole, f.num)) * f.num;
        whole =
            common <= Unsigned(kCommonLimit) ? static_cast<int64_t>(common) : 0;
      }
    }
    for (Facts &f : each) {
      f.own = ceilOf(Unsigned(f.feedback) * f.den, f.num);
      divide(Unsigned(f.uncertainty) * f.den, f.num, f.xWhole, f.xRest);
      divide((Unsigned(f.feedback) + f.peer + f.uncertainty) * f.den, f.num,
             f.aWhole, f.aRest);
    }
    double clockGHz = attrs.system.clockGHz();
    uncertaintyNs.resize_for_overwrite(n);
    for (int64_t c = 0; c < n; ++c)
      uncertaintyNs[c] = nanoseconds(each[c], clockGHz);
    if (tabulated)
      tabulate(each);
    else
      index();
  }

  /// Whether the program's lines are on several controllers.
  bool enabled() const { return n > 0; }

  int64_t count() const { return n; }

  /// The controller that plays the line `line` is a version of, or -1.
  int64_t of(mlir::Value line, AllocationTracer &trace) const {
    mlir::Operation *allocation = trace(line);
    if (!allocation)
      return -1;
#if PULSE_COUPLED
    if (mlir::isa<GetCouplingLineOp>(allocation)) {
      int64_t named = couplingFact(allocation, "controller", -1);
      return named >= 0 && named < n ? named : -1;
    }
#endif
    int64_t qudit = quditOf(allocation);
    const llvm::SmallVector<int64_t> &owners =
        mlir::isa<GetDriveLineOp>(allocation) ? drive : readout;
    if (qudit < 0 || qudit >= static_cast<int64_t>(owners.size()))
      return -1;
    int64_t owner = owners[qudit];
    return owner >= 0 && owner < n ? owner : -1;
  }

  /// After a readout on `reader` (-1: any) ends, when its outcome is usable
  /// on the reader itself.
  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t own(int64_t reader) const {
    if (tabulated) {
      if (reader >= 0)
        return at(latencyOf, reader, reader);
      int64_t most = 0;
      for (int64_t r = 0; r < n; ++r)
        most = std::max(most, at(latencyOf, r, r));
      return most;
    }
    return reader >= 0 ? ownOf[reader] : mostOwn;
  }

  /// How much later than on `reader` itself the outcome of its readout is
  /// usable on `user`; either may be -1, for any. The queries are inlined,
  /// as reading a table is, whatever their callers' size.
  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t extra(int64_t reader,
                                             int64_t user) const {
    if (tabulated) {
      int64_t most = 0;
      for (int64_t r = reader < 0 ? 0 : reader;
           r < (reader < 0 ? n : reader + 1); ++r)
        for (int64_t c = user < 0 ? 0 : user; c < (user < 0 ? n : user + 1);
             ++c)
          most = std::max(most, at(latencyOf, r, c) - at(latencyOf, r, r));
      return most;
    }
    if (reader >= 0 && user >= 0 && plain[reader]) {
      if (reader == user)
        return 0;
      int64_t base = aExtra[reader] + xWhole[user];
      return whole ? base + carry(aRest[reader] + xRest[user])
                   : base + carry(aRest[reader], numOf[reader], xRest[user],
                                  numOf[user]);
    }
    return extraOtherwise(reader, user);
  }

  /// How far apart the starts of `a` and `b` may be; either may be -1, for
  /// any other controller.
  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t spread(int64_t a, int64_t b) const {
    if (tabulated) {
      if (a >= 0 && b >= 0)
        return at(spreadOf, a, b);
      int64_t most = 0;
      for (int64_t x = a < 0 ? 0 : a; x < (a < 0 ? n : a + 1); ++x)
        for (int64_t y = b < 0 ? 0 : b; y < (b < 0 ? n : b + 1); ++y)
          most = std::max(most, at(spreadOf, x, y));
      return most;
    }
    if (a >= 0 && b >= 0) {
      if (a == b)
        return 0;
      int64_t base = xWhole[a] + xWhole[b];
      return whole ? base + carry(xRest[a] + xRest[b])
                   : base + carry(xRest[a], numOf[a], xRest[b], numOf[b]);
    }
    return spreadOtherwise(a, b);
  }

  /// How well the start of `controller` is known, in ns: its
  /// start_uncertainty_cycles in program cycles over the program clock, the
  /// double nearest the exact quotient.
  double uncertainty(int64_t controller) const {
    return uncertaintyNs[controller];
  }

private:
  using Unsigned = unsigned __int128;

  /// The largest common numerator of the clock ratios kept as one.
  static constexpr int64_t kCommonLimit = int64_t(1) << 31;

  struct Facts {
    // Every field is set by the constructor (`level` by index()).
    int64_t num, den;
    int64_t feedback, peer, uncertainty;
    llvm::ArrayRef<std::pair<int64_t, int64_t>> peers;
    /// ceil(feedback / ratio): the latency on itself.
    int64_t own;
    /// uncertainty / ratio and (feedback + peer + uncertainty) / ratio, in
    /// program cycles: each a whole part and a remainder over `num`.
    int64_t xWhole, xRest, aWhole, aRest;
    /// aWhole less `own`.
    int64_t level;
  };

  /// `value / by` as a whole part and a remainder; the whole part fits 64
  /// bits for every value the verifier lets through.
  static void divide(Unsigned value, int64_t by, int64_t &quotient,
                     int64_t &rest) {
    if (by == 1 && value <= Unsigned(INT64_MAX)) {
      quotient = static_cast<int64_t>(value);
      rest = 0;
    } else if (value <= Unsigned(INT64_MAX)) {
      auto narrow = static_cast<int64_t>(value);
      quotient = narrow / by;
      rest = narrow % by;
    } else {
      quotient = static_cast<int64_t>(value / static_cast<uint64_t>(by));
      rest = static_cast<int64_t>(value % static_cast<uint64_t>(by));
    }
  }

  static int64_t ceilOf(Unsigned value, int64_t by) {
    int64_t quotient = 0, rest = 0;
    divide(value, by, quotient, rest);
    return quotient + static_cast<int64_t>(rest != 0);
  }

  /// What the ceiling adds to two whole parts whose remainders, over the
  /// common numerator `whole`, sum to `rests`.
  int64_t carry(int64_t rests) const {
    return static_cast<int64_t>(rests > 0) +
           static_cast<int64_t>(rests > whole);
  }

  /// The same for remainders `a` over `aOver` and `b` over `bOver`.
  static int64_t carry(int64_t a, int64_t aOver, int64_t b, int64_t bOver) {
    if ((a | b) == 0)
      return 0;
    return 1 + static_cast<int64_t>(Unsigned(a) * bOver + Unsigned(b) * aOver >
                                    Unsigned(aOver) * bOver);
  }

  /// a + b + c in 64 bits, wrapping: a table's entry for a controller and
  /// itself, overwritten after, may pass 2^63 (the verifier bounds those of
  /// two controllers).
  static int64_t add(int64_t a, int64_t b, int64_t c = 0) {
    return static_cast<int64_t>(static_cast<uint64_t>(a) +
                                static_cast<uint64_t>(b) +
                                static_cast<uint64_t>(c));
  }

  /// The value `reader` gives `user` by index, if it does.
  static const std::pair<int64_t, int64_t> *find(const Facts &reader,
                                                 int64_t user) {
    const auto *it = llvm::lower_bound(
        reader.peers, user,
        [](const auto &entry, int64_t key) { return entry.first < key; });
    return it != reader.peers.end() && it->first == user ? it : nullptr;
  }

  bool given(int64_t reader, int64_t user) const {
    return !facts[reader].peers.empty() && find(facts[reader], user);
  }

  /// After a readout on `r` ends, when its outcome is usable on another
  /// controller `u` at a peer latency of `peer`.
  static int64_t reach(const Facts &r, int64_t peer, const Facts &u) {
    int64_t quotient = 0, rest = 0;
    divide((Unsigned(r.feedback) + peer + r.uncertainty) * r.den, r.num,
           quotient, rest);
    return quotient + u.xWhole + carry(rest, r.num, u.xRest, u.num);
  }

  /// How far apart the starts of two controllers may be.
  static int64_t apart(const Facts &x, const Facts &y) {
    return add(x.xWhole, y.xWhole, carry(x.xRest, x.num, y.xRest, y.num));
  }

  /// The same, at the peer latency `reader` gives `user`, untabulated.
  int64_t latency(int64_t reader, int64_t user) const {
    const Facts &r = facts[reader], &u = facts[user];
    if (!r.peers.empty())
      if (const auto *entry = find(r, user))
        return peerLatency[peerStart[reader] + (entry - r.peers.begin())];
    return r.aWhole + u.xWhole + carry(r.aRest, r.num, u.xRest, u.num);
  }

  /// `extra` of an untabulated system, but for two controllers the first
  /// of which gives no peer value by index.
  LLVM_ATTRIBUTE_NOINLINE int64_t extraOtherwise(int64_t reader,
                                                 int64_t user) const {
    if (n == 0)
      return 0;
    if (reader >= 0 && user >= 0)
      return reader == user ? 0 : latency(reader, user) - facts[reader].own;
    if (reader < 0 && user < 0)
      return mostExtra;
    int64_t most = 0;
    if (reader >= 0) {
      // Its peers by index, and the one of the others whose start is known
      // least well.
      const Facts &from = facts[reader];
      for (auto [peer, cycles] : from.peers)
        most = std::max(most, latency(reader, peer) - from.own);
      for (int64_t peer : byUncertainty)
        if (peer != reader && !given(reader, peer)) {
          most = std::max(most, latency(reader, peer) - from.own);
          break;
        }
      return most;
    }
    // The readers that give `user` a value by index; of the others, the
    // latest of the highest level and of the level below it (one two or
    // more below adds no more).
    for (int64_t r : llvm::ArrayRef<int64_t>(byIndex).slice(
             readersOf[user], readersOf[user + 1] - readersOf[user]))
      most = std::max(most, latency(r, user) - facts[r].own);
    auto other = [&](int64_t r) { return r != user && !given(r, user); };
    int64_t i = 0;
    while (i < n && !other(byLevel[i]))
      ++i;
    if (i == n)
      return most;
    int64_t first = byLevel[i];
    most = std::max(most, latency(first, user) - facts[first].own);
    i = nextLevel[i];
    if (i == n || facts[byLevel[i]].level != facts[first].level - 1)
      return most;
    for (int64_t end = nextLevel[i]; i < end; ++i)
      if (other(byLevel[i])) {
        int64_t r = byLevel[i];
        return std::max(most, latency(r, user) - facts[r].own);
      }
    return most;
  }

  /// `spread` of an untabulated system with -1 for one or both.
  LLVM_ATTRIBUTE_NOINLINE int64_t spreadOtherwise(int64_t a, int64_t b) const {
    if (n == 0)
      return 0;
    int64_t first = byUncertainty[0], second = byUncertainty[1];
    if (a < 0 && b < 0)
      return apart(facts[first], facts[second]);
    int64_t one = a >= 0 ? a : b;
    return apart(facts[one], facts[one == first ? second : first]);
  }

  /// Every latency and spread, read by the queries as tables.
  void tabulate(llvm::ArrayRef<Facts> each) {
    latencyOf.resize_for_overwrite(n * n);
    spreadOf.resize_for_overwrite(n * n);
    if (whole && n <= 8) {
      // A few controllers with a common numerator: sums and carries only.
      for (int64_t r = 0; r < n; ++r) {
        const Facts &from = each[r];
        const int64_t ar = from.aRest * (whole / from.num),
                      sr = from.xRest * (whole / from.num);
        for (int64_t u = 0; u < n; ++u) {
          const Facts &to = each[u];
          int64_t xr = to.xRest * (whole / to.num);
          latencyOf[r * n + u] = add(from.aWhole, to.xWhole, carry(ar + xr));
          spreadOf[r * n + u] = add(from.xWhole, to.xWhole, carry(sr + xr));
        }
        for (auto [user, cycles] : from.peers)
          latencyOf[r * n + user] = reach(from, cycles, each[user]);
        latencyOf[r * n + r] = from.own;
        spreadOf[r * n + r] = 0;
      }
      return;
    }
    // Each row from the reader's values and every controller's, in
    // contiguous arrays the loops vectorize over.
    llvm::SmallVector<int64_t, 16> wholes(n), rests(n);
    for (int64_t c = 0; c < n; ++c) {
      wholes[c] = each[c].xWhole;
      rests[c] = whole ? each[c].xRest * (whole / each[c].num) : 0;
    }
    const int64_t w = whole;
    const int64_t *xw = wholes.data(), *xr = rests.data();
    for (int64_t r = 0; r < n; ++r) {
      const Facts &from = each[r];
      int64_t *row = &latencyOf[r * n];
      int64_t *apartRow = &spreadOf[r * n];
      const int64_t aw = from.aWhole, sw = from.xWhole;
      if (w == 1) {
        for (int64_t u = 0; u < n; ++u)
          row[u] = add(aw, xw[u]);
        for (int64_t u = 0; u < n; ++u)
          apartRow[u] = add(sw, xw[u]);
      } else if (w) {
        const int64_t ar = from.aRest * (w / from.num),
                      sr = from.xRest * (w / from.num);
        for (int64_t u = 0; u < n; ++u) {
          int64_t rest = ar + xr[u];
          row[u] = add(aw, xw[u],
                       static_cast<int64_t>(rest > 0) +
                           static_cast<int64_t>(rest > w));
        }
        for (int64_t u = 0; u < n; ++u) {
          int64_t rest = sr + xr[u];
          apartRow[u] = add(sw, xw[u],
                            static_cast<int64_t>(rest > 0) +
                                static_cast<int64_t>(rest > w));
        }
      } else {
        for (int64_t u = 0; u < n; ++u) {
          row[u] = add(aw, each[u].xWhole,
                       carry(from.aRest, from.num, each[u].xRest, each[u].num));
          apartRow[u] = apart(from, each[u]);
        }
      }
      for (auto [user, cycles] : from.peers)
        row[user] = reach(from, cycles, each[user]);
      row[r] = from.own;
      apartRow[r] = 0;
    }
  }

  /// The arrays and orders the queries of an untabulated system read.
  void index() {
    // The latency to each peer given by index, in the order given.
    for (int64_t r = 0; r < n; ++r) {
      const Facts &f = facts[r];
      peerStart.push_back(static_cast<int64_t>(peerLatency.size()));
      for (auto [user, cycles] : f.peers)
        peerLatency.push_back(reach(f, cycles, facts[user]));
    }
    for (Facts &f : facts) {
      mostOwn = std::max(mostOwn, f.own);
      ownOf.push_back(f.own);
      plain.push_back(f.peers.empty());
      aExtra.push_back(f.aWhole - f.own);
      xWhole.push_back(f.xWhole);
      int64_t scale = whole ? whole / f.num : 1;
      aRest.push_back(f.aRest * scale);
      xRest.push_back(f.xRest * scale);
      numOf.push_back(f.num);
      f.level = f.aWhole - f.own;
    }

    // By start uncertainty in program cycles, most first: the other
    // controller a reader's outcome reaches last, of those it gives no peer
    // value by index.
    byUncertainty.resize(n);
    std::iota(byUncertainty.begin(), byUncertainty.end(), 0);
    llvm::stable_sort(byUncertainty, [&](int64_t a, int64_t b) {
      const Facts &x = facts[a], &y = facts[b];
      if (x.xWhole != y.xWhole)
        return x.xWhole > y.xWhole;
      return Unsigned(x.xRest) * y.num > Unsigned(y.xRest) * x.num;
    });

    // The readers that give a peer value by index for each user.
    readersOf.assign(n + 1, 0);
    for (const Facts &f : facts)
      for (auto [user, cycles] : f.peers)
        ++readersOf[user + 1];
    for (int64_t c = 0; c < n; ++c)
      readersOf[c + 1] += readersOf[c];
    byIndex.resize(readersOf[n]);
    llvm::SmallVector<int64_t> next(readersOf.begin(), readersOf.end() - 1);
    for (int64_t r = 0; r < n; ++r)
      for (auto [user, cycles] : facts[r].peers)
        byIndex[next[user]++] = r;

    // Readers by how much later than on themselves their outcome reaches
    // a controller they give no value by index, before that controller's
    // own start uncertainty: a whole part `level` and a fraction under one,
    // aRest / num. To a user's start uncertainty, of whole part w and
    // fraction h, a reader adds level + w + ceil(fraction + h), and
    // ceil(fraction + h) is 0, 1 or 2, rising with the fraction.
    byLevel.resize(n);
    std::iota(byLevel.begin(), byLevel.end(), 0);
    llvm::stable_sort(byLevel, [&](int64_t a, int64_t b) {
      const Facts &x = facts[a], &y = facts[b];
      if (x.level != y.level)
        return x.level > y.level;
      return Unsigned(x.aRest) * y.num > Unsigned(y.aRest) * x.num;
    });
    nextLevel.assign(n, n);
    for (int64_t i = n - 2; i >= 0; --i)
      nextLevel[i] = facts[byLevel[i]].level == facts[byLevel[i + 1]].level
                         ? nextLevel[i + 1]
                         : i + 1;

    for (int64_t r = 0; r < n; ++r)
      mostExtra = std::max(mostExtra, extraOtherwise(r, -1));
  }

  /// The start uncertainty of `f` in ns at a `clockGHz` program clock: the
  /// double nearest (uncertainty * den) / (num * clockGHz).
  static double nanoseconds(const Facts &f, double clockGHz) {
    if (f.uncertainty == 0)
      return 0.0;
    Unsigned top = Unsigned(f.uncertainty) * f.den;
    constexpr int64_t kExact = int64_t(1) << 53;
    if (top < Unsigned(kExact) && f.num < kExact) {
      auto t = static_cast<double>(static_cast<int64_t>(top));
      auto num = static_cast<double>(f.num);
      // num * clockGHz is p + e exactly; with e = 0, one division rounds
      // the exact quotient.
      double p = num * clockGHz;
      double e = std::fma(num, clockGHz, -p);
      if (e == 0.0)
        return t / p;
      // Else q = t / p is within an ulp of the quotient t / (p + e), which
      // is q + (t - q (p + e)) / (p + e); t - q p is exact. Unless that
      // remainder is within 1% of half an ulp's worth (or q is a power of
      // two, where the ulp changes), q or its neighbour toward the quotient
      // is the nearest double.
      double q = t / p;
      auto bits = llvm::bit_cast<uint64_t>(q);
      constexpr uint64_t kMantissa = (uint64_t(1) << 52) - 1;
      uint64_t biased = bits >> 52;
      if ((bits & kMantissa) != 0 && biased > 54) {
        // Half of q's ulp: 2^(exponent - 53), q in [2^exponent, ...).
        double half = p * llvm::bit_cast<double>((biased - 53) << 52);
        double rest = std::fma(-q, p, t) - q * e;
        double size = std::fabs(rest);
        if (size < 0.99 * half)
          return q;
        if (size > 1.01 * half && size < 2.99 * half)
          return llvm::bit_cast<double>(rest > 0 ? bits + 1 : bits - 1);
      }
    }
    // clockGHz = mantissa * 2^(exponent - 53), the mantissa whole.
    int exponent = 0;
    double fraction = std::frexp(clockGHz, &exponent);
    auto mantissa = static_cast<uint64_t>(std::ldexp(fraction, 53));
    llvm::APInt dividend = llvm::APInt(192, f.uncertainty) *
                           llvm::APInt(192, static_cast<uint64_t>(f.den));
    llvm::APInt divisor = llvm::APInt(192, static_cast<uint64_t>(f.num)) *
                          llvm::APInt(192, mantissa);
    return std::ldexp(nearest(dividend, divisor), 53 - exponent);
  }

  /// The double nearest `top / bottom`, ties to even.
  static double nearest(const llvm::APInt &top, const llvm::APInt &bottom) {
    // A quotient of 66 or 67 bits, with a set last bit if it is not exact,
    // rounds as the exact quotient does.
    int shift = 66 - static_cast<int>(top.getActiveBits()) +
                static_cast<int>(bottom.getActiveBits());
    unsigned width = top.getActiveBits() + bottom.getActiveBits() + 140;
    llvm::APInt dividend = top.zextOrTrunc(width),
                divisor = bottom.zextOrTrunc(width);
    if (shift > 0)
      dividend <<= shift;
    else
      divisor <<= -shift;
    llvm::APInt quotient, remainder;
    llvm::APInt::udivrem(dividend, divisor, quotient, remainder);
    if (!remainder.isZero())
      quotient.setBit(0);
    llvm::APFloat value(llvm::APFloat::IEEEdouble());
    value.convertFromAPInt(quotient, /*IsSigned=*/false,
                           llvm::APFloat::rmNearestTiesToEven);
    return std::ldexp(value.convertToDouble(), -shift);
  }

  int64_t at(const llvm::SmallVector<int64_t> &table, int64_t a,
             int64_t b) const {
    return table[a * n + b];
  }

  int64_t n = 0;
  /// With no controllers too: the queries read the empty tables.
  bool tabulated = true;
  /// The clock ratios' common numerator; 0 if too large to keep.
  int64_t whole = 1;
  llvm::SmallVector<double> uncertaintyNs;
  /// Tabulated: by reader, then user, the latency; by controller pair, the
  /// spread.
  llvm::SmallVector<int64_t> latencyOf, spreadOf;
  /// Untabulated: each controller's facts, and the orders "any" follows.
  llvm::SmallVector<Facts, 0> facts;
  int64_t mostOwn = 0, mostExtra = 0;
  /// Untabulated, by controller: its own latency, whether it gives no peer
  /// value by index, its latency to another less its own and its start
  /// uncertainty as whole parts and remainders (over `whole`, or with none
  /// over its numerator, `numOf`).
  llvm::SmallVector<int64_t> ownOf, aExtra, aRest, xWhole, xRest, numOf;
  llvm::SmallVector<bool> plain;
  /// Untabulated: the latencies to the peers given by index, reader by
  /// reader (from peerStart[reader]) in the order given.
  llvm::SmallVector<int64_t> peerStart, peerLatency;
  /// Controllers by start uncertainty in program cycles, most first.
  llvm::SmallVector<int64_t> byUncertainty;
  /// Readers by `level`, highest first, then by the rest of their latency;
  /// and for each place, where the next level starts.
  llvm::SmallVector<int64_t> byLevel, nextLevel;
  /// For each user c, byIndex[readersOf[c]..readersOf[c + 1]) are the
  /// readers that give it a peer value by index.
  llvm::SmallVector<int64_t> readersOf, byIndex;
  llvm::SmallVector<int64_t> drive, readout;
};

PULSE_VARIANT_CLOSE
} // namespace pulse
