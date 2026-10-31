/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Numbers that are affine in the f64 arguments of a pulse program's entry
// function: c + sum_i a_i x_i. A kernel's float arguments stay function
// arguments through compilation; frame phases, detunings and amplitudes
// built from them by addition, subtraction, negation and scaling by
// constants stay affine, so a lowering can carry them symbolically and bind
// the arguments per execution.

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace pulse {

class Affine {
public:
  using Term = std::pair<unsigned, double>;

  Affine(double constant = 0.0) : constant(constant) {}

  static Affine argument(unsigned index) {
    Affine result;
    result.terms.push_back({index, 1.0});
    return result;
  }

  bool isConstant() const { return terms.empty(); }
  /// The (argument index, coefficient) terms, by argument index.
  llvm::ArrayRef<Term> coefficients() const { return terms; }

  double evaluate(llvm::ArrayRef<double> arguments) const {
    double value = constant;
    for (auto [index, coefficient] : terms)
      value += coefficient * arguments[index];
    return value;
  }

  Affine operator+(const Affine &other) const { return combine(other, 1.0); }
  Affine operator-(const Affine &other) const { return combine(other, -1.0); }
  Affine operator-() const { return *this * -1.0; }
  Affine operator*(double factor) const {
    Affine result(constant * factor);
    for (auto [index, coefficient] : terms)
      if (coefficient * factor != 0.0)
        result.terms.push_back({index, coefficient * factor});
    return result;
  }
  Affine operator/(double divisor) const {
    Affine result(constant / divisor);
    for (auto [index, coefficient] : terms)
      result.terms.push_back({index, coefficient / divisor});
    return result;
  }
  Affine &operator+=(const Affine &other) { return *this = *this + other; }
  Affine &operator-=(const Affine &other) { return *this = *this - other; }

  /// The product, if at most one factor depends on the arguments.
  std::optional<Affine> times(const Affine &other) const {
    if (isConstant())
      return other * constant;
    if (other.isConstant())
      return *this * other.constant;
    return std::nullopt;
  }

  bool operator==(const Affine &other) const {
    return constant == other.constant && terms == other.terms;
  }
  bool operator!=(const Affine &other) const { return !(*this == other); }

  /// Equal up to `tolerance` in the constant and a relative 1e-12 in each
  /// coefficient.
  bool near(const Affine &other, double tolerance) const {
    if (std::abs(constant - other.constant) > tolerance)
      return false;
    Affine difference = *this - other;
    for (auto [index, coefficient] : difference.terms) {
      double scale = 0.0;
      for (const Affine *side : {this, &other})
        for (auto [i, c] : side->terms)
          if (i == index)
            scale = std::max(scale, std::abs(c));
      if (std::abs(coefficient) > 1e-12 * scale)
        return false;
    }
    return true;
  }

  double constant;

private:
  Affine combine(const Affine &other, double sign) const {
    Affine result(constant + sign * other.constant);
    auto a = terms.begin(), b = other.terms.begin();
    while (a != terms.end() || b != other.terms.end()) {
      if (b == other.terms.end() || (a != terms.end() && a->first < b->first)) {
        result.terms.push_back(*a++);
      } else if (a == terms.end() || b->first < a->first) {
        result.terms.push_back({b->first, sign * b->second});
        ++b;
      } else {
        double sum = a->second + sign * b->second;
        if (sum != 0.0)
          result.terms.push_back({a->first, sum});
        ++a;
        ++b;
      }
    }
    return result;
  }

  llvm::SmallVector<Term, 1> terms;
};

/// The answer simulation bound to `value`, a result of a `classical.wait`
/// that carries `classical.answers`: the wait stays in the program, so what
/// it feeds keeps its timing, and its results read as these constants.
inline std::optional<double> boundAnswer(mlir::Value value) {
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return std::nullopt;
  mlir::Operation *op = result.getOwner();
  if (op->getName().getStringRef() != "classical.wait")
    return std::nullopt;
  auto answers =
      op->getAttrOfType<mlir::DenseF64ArrayAttr>("classical.answers");
  if (!answers || result.getResultNumber() >= answers.size())
    return std::nullopt;
  return answers[result.getResultNumber()];
}

template <typename Leaf>
std::optional<double> computedTerm(mlir::Value value, Leaf &leaf);

/// A number computed from constants and the values `leaf` gives (else
/// none), by the terms of the controller export's expression vocabulary
/// and the bool logic it expresses with them. A bool is 1 or 0.
template <typename Leaf>
std::optional<double> computedNumber(mlir::Value value, Leaf &leaf) {
  std::optional<double> out = computedTerm(value, leaf);
  if (out && value.getType().isInteger(1))
    out = *out != 0.0 ? 1.0 : 0.0;
  return out;
}

template <typename Leaf>
std::optional<double> computedTerm(mlir::Value value, Leaf &leaf) {
  mlir::Attribute attribute;
  if (mlir::matchPattern(value, mlir::m_Constant(&attribute))) {
    if (auto number = mlir::dyn_cast<mlir::FloatAttr>(attribute))
      return number.getValueAsDouble();
    if (auto number = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
      return static_cast<double>(number.getInt());
    return std::nullopt;
  }
  if (auto known = leaf(value))
    return known;
  mlir::Operation *def = value.getDefiningOp();
  if (!def || def->getNumRegions() != 0)
    return std::nullopt;
  llvm::StringRef name = def->getName().getStringRef();
  if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    auto condition = computedNumber(select.getCondition(), leaf);
    if (!condition)
      return std::nullopt;
    return computedNumber(*condition != 0.0 ? select.getTrueValue()
                                            : select.getFalseValue(),
                          leaf);
  }
  if (def->getNumOperands() == 1) {
    auto x = computedNumber(def->getOperand(0), leaf);
    if (!x)
      return std::nullopt;
    if (name == "arith.negf")
      return -*x;
    if (name == "arith.fptosi")
      return std::trunc(*x);
    if (name == "arith.extsi" && def->getOperand(0).getType().isInteger(1))
      return -*x;
    if (name == "arith.sitofp" || name == "arith.extui" ||
        name == "arith.extsi" || name == "arith.trunci" ||
        name == "arith.index_cast")
      return *x;
    return std::nullopt;
  }
  if (def->getNumOperands() != 2)
    return std::nullopt;
  auto a = computedNumber(def->getOperand(0), leaf),
       b = computedNumber(def->getOperand(1), leaf);
  if (!a || !b)
    return std::nullopt;
  if (name == "arith.xori")
    return static_cast<double>(static_cast<int64_t>(*a) ^
                               static_cast<int64_t>(*b));
  if (name == "arith.andi")
    return static_cast<double>(static_cast<int64_t>(*a) &
                               static_cast<int64_t>(*b));
  if (name == "arith.ori")
    return static_cast<double>(static_cast<int64_t>(*a) |
                               static_cast<int64_t>(*b));
  if (name == "arith.addf" || name == "arith.addi")
    return *a + *b;
  if (name == "arith.subf" || name == "arith.subi")
    return *a - *b;
  if (name == "arith.mulf" || name == "arith.muli")
    return *a * *b;
  if (name == "arith.divf")
    return *a / *b;
  if (name == "arith.divsi" && *b != 0)
    return std::trunc(*a / *b);
  if (name == "arith.floordivsi" && *b != 0)
    return std::floor(*a / *b);
  if (name == "arith.remsi" && *b != 0)
    return std::fmod(*a, *b);
  if (name == "arith.maxsi")
    return std::max(*a, *b);
  if (name == "arith.minsi")
    return std::min(*a, *b);
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
    using P = mlir::arith::CmpIPredicate;
    switch (cmp.getPredicate()) {
    case P::eq:
      return *a == *b;
    case P::ne:
      return *a != *b;
    case P::slt:
      return *a < *b;
    case P::sle:
      return *a <= *b;
    case P::sgt:
      return *a > *b;
    case P::sge:
      return *a >= *b;
    case P::ult:
      return *a < *b;
    case P::ule:
      return *a <= *b;
    case P::ugt:
      return *a > *b;
    case P::uge:
      return *a >= *b;
    }
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
    using P = mlir::arith::CmpFPredicate;
    switch (cmp.getPredicate()) {
    case P::OEQ:
      return *a == *b;
    case P::ONE:
      return *a < *b || *a > *b;
    case P::OLT:
      return *a < *b;
    case P::OLE:
      return *a <= *b;
    case P::OGT:
      return *a > *b;
    case P::OGE:
      return *a >= *b;
    case P::UEQ:
      return !(*a < *b || *a > *b);
    case P::UNE:
      return *a != *b;
    case P::ULT:
      return !(*a >= *b);
    case P::ULE:
      return !(*a > *b);
    case P::UGT:
      return !(*a <= *b);
    case P::UGE:
      return !(*a < *b);
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

/// A number computed from constants and bound answers alone.
inline std::optional<double> boundNumber(mlir::Value value) {
  auto bound = [](mlir::Value v) { return boundAnswer(v); };
  return computedNumber(value, bound);
}

/// `value` as an affine function of the entry function's arguments, if it
/// is a constant, an argument, or arithmetic on them that stays affine.
inline std::optional<Affine> affineNumber(mlir::Value value) {
  mlir::Attribute attribute;
  if (mlir::matchPattern(value, mlir::m_Constant(&attribute))) {
    if (auto number = mlir::dyn_cast<mlir::FloatAttr>(attribute))
      return Affine(number.getValueAsDouble());
    if (auto number = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
      return Affine(static_cast<double>(number.getInt()));
    return std::nullopt;
  }
  if (auto answer = boundAnswer(value))
    return Affine(*answer);
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = argument.getOwner();
    if (owner->isEntryBlock() &&
        mlir::isa_and_nonnull<mlir::func::FuncOp>(owner->getParentOp()) &&
        argument.getType().isF64())
      return Affine::argument(argument.getArgNumber());
    return std::nullopt;
  }
  mlir::Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto operand = [&](unsigned index) {
    return affineNumber(op->getOperand(index));
  };
  if (mlir::isa<mlir::arith::NegFOp>(op)) {
    auto x = operand(0);
    return x ? std::optional(-*x) : std::nullopt;
  }
  if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                 mlir::arith::DivFOp>(op))
    return std::nullopt;
  auto lhs = operand(0), rhs = operand(1);
  if (!lhs || !rhs)
    return std::nullopt;
  if (mlir::isa<mlir::arith::AddFOp>(op))
    return *lhs + *rhs;
  if (mlir::isa<mlir::arith::SubFOp>(op))
    return *lhs - *rhs;
  if (mlir::isa<mlir::arith::MulFOp>(op))
    return lhs->times(*rhs);
  if (!rhs->isConstant())
    return std::nullopt;
  return *lhs / rhs->constant;
}

} // namespace pulse
