// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The expressions of what a controller program computes at run time, in
// the export's closed vocabulary: the walks of the controller export and
// of the check (bindings.cpp) record their run-time values with them.
// Included by bindings.cpp inside its anonymous namespace.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

struct ControllerExpressions {
  static bool same(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
  }

  static constexpr double kUnknown = std::numeric_limits<double>::quiet_NaN();

  /// A number computed from constants alone, else NaN.
  static double numberOf(mlir::Value value) {
    if (auto constant = pulse::constantNumber(value))
      return *constant;
    if (auto affine = pulse::affineNumber(value);
        affine && affine->isConstant())
      return affine->evaluate({});
    return kUnknown;
  }

  /// numberOf(), once per value: the walk asks again at every execution
  /// and at every node of an expression.
  double number(mlir::Value value) {
    auto [found, inserted] = numbers.try_emplace(value);
    if (inserted)
      found->second = numberOf(value);
    return found->second;
  }
  llvm::DenseMap<mlir::Value, double> numbers;

  // The record of the latest execution of each readout and classical wait,
  // which an expression of its result refers to.
  llvm::DenseMap<mlir::Operation *, int64_t> latest;

  /// How a value is computed at run time, children before their parent: a
  /// number (a frame change's value), a constant, the result of a readout
  /// or answer record, an op of the export's vocabulary (`name`) over the
  /// expressions `of` of one `type`, or what the vocabulary has no term
  /// for (`name`: an op's MLIR name, else what it is), which no export
  /// gives.
  struct Expression {
    enum Kind : uint8_t { Number, Constant, Unknown, Readout, Answer, Op };
    enum Type : uint8_t { Int, Float, Bool, Types };
    // The terms of the export's expression vocabulary, in `names` order.
    enum Term : uint8_t {
      Add,
      Sub,
      Mul,
      Div,
      FloorDiv,
      Rem,
      Neg,
      Max,
      Min,
      Eq,
      Ne,
      Lt,
      Le,
      Gt,
      Ge,
      Select,
      ToFloat,
      ToInt,
      Terms
    };
    static constexpr const char *names[Terms] = {
        "add", "sub", "mul", "div",    "floor_div", "rem",
        "neg", "max", "min", "eq",     "ne",        "lt",
        "le",  "gt",  "ge",  "select", "to_float",  "to_int"};
    static constexpr const char *typeNames[Types] = {"int", "float", "bool"};
    Kind kind = Number;
    Type type = Float;
    Term term = Add;
    double number = 0.0;
    int64_t event = 0, result = 0;
    llvm::StringRef name; // what an Unknown is computed from
    std::vector<int32_t> of;
  };
  std::vector<Expression> expressions;

  int32_t add(Expression expression) {
    expressions.push_back(std::move(expression));
    return static_cast<int32_t>(expressions.size()) - 1;
  }

  int32_t add(Expression::Kind kind, double number, int64_t event = 0,
              int64_t result = 0, llvm::StringRef name = {}) {
    Expression expression;
    expression.kind = kind;
    expression.number = number;
    expression.event = event;
    expression.result = result;
    expression.name = name;
    return add(std::move(expression));
  }

  int32_t constant(double number, Expression::Type type) {
    int32_t index = add(Expression::Constant, number);
    expressions[index].type = type;
    return index;
  }

  int32_t op(Expression::Term term, Expression::Type type,
             std::vector<int32_t> of) {
    Expression expression;
    expression.kind = Expression::Op;
    expression.type = type;
    expression.term = term;
    expression.of = std::move(of);
    return add(std::move(expression));
  }

  static std::optional<Expression::Type> valueType(mlir::Type type) {
    if (type.isInteger(1))
      return Expression::Bool;
    if (type.isInteger(64) || type.isIndex())
      return Expression::Int;
    if (type.isF64())
      return Expression::Float;
    return std::nullopt;
  }

  static std::optional<Expression::Term>
  comparison(mlir::arith::CmpIPredicate predicate) {
    using P = mlir::arith::CmpIPredicate;
    switch (predicate) {
    case P::eq:
      return Expression::Eq;
    case P::ne:
      return Expression::Ne;
    case P::slt:
      return Expression::Lt;
    case P::sle:
      return Expression::Le;
    case P::sgt:
      return Expression::Gt;
    case P::sge:
      return Expression::Ge;
    default:
      return std::nullopt;
    }
  }

  // Ordered, but for ne: a NaN compares unequal and in no order.
  static std::optional<Expression::Term>
  comparison(mlir::arith::CmpFPredicate predicate) {
    using P = mlir::arith::CmpFPredicate;
    switch (predicate) {
    case P::OEQ:
      return Expression::Eq;
    case P::UNE:
      return Expression::Ne;
    case P::OLT:
      return Expression::Lt;
    case P::OLE:
      return Expression::Le;
    case P::OGT:
      return Expression::Gt;
    case P::OGE:
      return Expression::Ge;
    default:
      return std::nullopt;
    }
  }

  /// How `value` is computed at run time: a constant, a result of a readout
  /// or answer record, or a term of the export's vocabulary over such
  /// expressions (see term()).
  int32_t expression(mlir::Value value, int depth = 0) {
    double number = this->number(value);
    if (!std::isnan(number)) {
      if (value.getType().isInteger(1))
        return constant(number != 0.0, Expression::Bool);
      return constant(number, Expression::Float);
    }
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return add(Expression::Unknown, 0.0, 0, 0, "a value the kernel is given");
    if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
      auto found = latest.find(def);
      if (found == latest.end())
        return add(Expression::Unknown, 0.0, 0, 0,
                   "a readout or answer the program has not reached");
      if (mlir::isa<pulse::ReadoutOp>(def))
        return add(Expression::Readout, 0.0, found->second);
      return add(Expression::Answer, 0.0, found->second,
                 static_cast<int64_t>(
                     mlir::cast<mlir::OpResult>(value).getResultNumber()));
    }
    if (depth > 64)
      return add(Expression::Unknown, 0.0, 0, 0,
                 "an expression more than 64 operations deep");
    if (def->getNumRegions() != 0 || def->getNumResults() != 1)
      return add(Expression::Unknown, 0.0, 0, 0, def->getName().getStringRef());
    return term(def, depth);
  }

  /// The term of the export's vocabulary `def` computes, over its operands'
  /// expressions; forms that compute the same value give the same term.
  int32_t term(mlir::Operation *def, int depth) {
    namespace arith = mlir::arith;
    auto of = [&](mlir::Value value) { return expression(value, depth + 1); };
    auto unknown = [&] {
      return add(Expression::Unknown, 0.0, 0, 0, def->getName().getStringRef());
    };
    // The export counts times in cycles, which a duration holds.
    if (mlir::isa<pulse::DurationFromIntOp>(def))
      return of(def->getOperand(0));
    // A readout's leaf is the level it observed.
    if (mlir::isa<pulse::OutcomeOp>(def))
      return of(def->getOperand(0));
    // A readout's outcome is set when it observed a level other than 0.
    if (mlir::isa<pulse::IsSetOp>(def)) {
      int32_t outcome = of(def->getOperand(0));
      return op(Expression::Ne, Expression::Int,
                {outcome, constant(0.0, Expression::Int)});
    }
    std::optional<Expression::Type> result =
        valueType(def->getResult(0).getType());
    if (!result || def->getNumOperands() == 0)
      return unknown();
    std::optional<Expression::Type> type =
        valueType(def->getOperand(0).getType());
    if (!type)
      return unknown();
    using T = Expression::Type;
    struct Simple {
      mlir::TypeID op;
      Expression::Term term;
      T type, result;
    };
    static const Simple simple[] = {
        {mlir::TypeID::get<arith::AddFOp>(), Expression::Add, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::MulFOp>(), Expression::Mul, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::FPToSIOp>(), Expression::ToInt, T::Float,
         T::Int},
        {mlir::TypeID::get<arith::SIToFPOp>(), Expression::ToFloat, T::Int,
         T::Float},
        {mlir::TypeID::get<arith::SubFOp>(), Expression::Sub, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::DivFOp>(), Expression::Div, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::NegFOp>(), Expression::Neg, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::AddIOp>(), Expression::Add, T::Int, T::Int},
        {mlir::TypeID::get<arith::MulIOp>(), Expression::Mul, T::Int, T::Int},
        {mlir::TypeID::get<arith::DivSIOp>(), Expression::Div, T::Int, T::Int},
        {mlir::TypeID::get<arith::FloorDivSIOp>(), Expression::FloorDiv, T::Int,
         T::Int},
        {mlir::TypeID::get<arith::RemSIOp>(), Expression::Rem, T::Int, T::Int},
        {mlir::TypeID::get<arith::MaxSIOp>(), Expression::Max, T::Int, T::Int},
        {mlir::TypeID::get<arith::MinSIOp>(), Expression::Min, T::Int, T::Int},
    };
    mlir::TypeID id = def->getName().getTypeID();
    for (const Simple &s : simple) {
      if (s.op != id)
        continue;
      if (*type != s.type || *result != s.result)
        return unknown();
      std::vector<int32_t> operands;
      for (mlir::Value operand : def->getOperands()) {
        if (valueType(operand.getType()) != s.type)
          return unknown();
        operands.push_back(of(operand));
      }
      return op(s.term, s.type, std::move(operands));
    }
    // Integer subtraction from 0 is negation.
    if (mlir::isa<arith::SubIOp>(def)) {
      if (*type != T::Int || *result != T::Int)
        return unknown();
      if (this->number(def->getOperand(0)) == 0.0)
        return op(Expression::Neg, T::Int, {of(def->getOperand(1))});
      int32_t lhs = of(def->getOperand(0));
      return op(Expression::Sub, T::Int, {lhs, of(def->getOperand(1))});
    }
    if (auto select = mlir::dyn_cast<arith::SelectOp>(def)) {
      int32_t extremum = minMax(select, *result, depth);
      if (extremum >= 0)
        return extremum;
      // A choice of bool constants is the condition or its negation.
      double then = number(select.getTrueValue()),
             other = number(select.getFalseValue());
      if (*result == T::Bool && !std::isnan(then) && !std::isnan(other) &&
          (then != 0.0) != (other != 0.0))
        return equality(true, select.getCondition(), select.getTrueValue(),
                        depth);
      int32_t condition = of(select.getCondition());
      int32_t chosen = of(select.getTrueValue());
      return op(Expression::Select, *result,
                {condition, chosen, of(select.getFalseValue())});
    }
    if (mlir::isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(def)) {
      if (*type != T::Bool || *result != T::Bool)
        return unknown();
      return logic(def, depth);
    }
    if (auto cmp = mlir::dyn_cast<arith::CmpIOp>(def)) {
      using P = arith::CmpIPredicate;
      if (*type == T::Bool &&
          (cmp.getPredicate() == P::eq || cmp.getPredicate() == P::ne))
        return equality(cmp.getPredicate() == P::eq, cmp.getLhs(), cmp.getRhs(),
                        depth);
      std::optional<Expression::Term> term = comparison(cmp.getPredicate());
      if (!term || *type == T::Bool)
        return unknown();
      return compared(*term, *type, cmp.getLhs(), cmp.getRhs(), depth);
    }
    if (auto cmp = mlir::dyn_cast<arith::CmpFOp>(def)) {
      std::optional<Expression::Term> term = comparison(cmp.getPredicate());
      if (!term)
        return unknown();
      return compared(*term, *type, cmp.getLhs(), cmp.getRhs(), depth);
    }
    // A bool as an integer: 1, or -1 sign-extended, when true, else 0.
    if (mlir::isa<arith::ExtUIOp, arith::ExtSIOp>(def)) {
      if (*type != T::Bool || *result != T::Int)
        return unknown();
      int32_t condition = of(def->getOperand(0));
      double set = mlir::isa<arith::ExtUIOp>(def) ? 1.0 : -1.0;
      int32_t then = constant(set, T::Int);
      return op(Expression::Select, T::Int,
                {condition, then, constant(0.0, T::Int)});
    }
    return unknown();
  }

  /// Bool logic `def` computes: a and b is select(a, b, false), a or b is
  /// select(a, true, b), and exclusive or is ne.
  LLVM_ATTRIBUTE_NOINLINE int32_t logic(mlir::Operation *def, int depth) {
    using T = Expression::Type;
    mlir::Value x = def->getOperand(0), y = def->getOperand(1);
    if (mlir::isa<mlir::arith::XOrIOp>(def))
      return equality(false, x, y, depth);
    bool both = mlir::isa<mlir::arith::AndIOp>(def);
    if (!std::isnan(number(x)))
      std::swap(x, y);
    // With a constant: a and true is a, a or false is a, and the others
    // the constant.
    if (double c = number(y); !std::isnan(c))
      return (c != 0.0) == both ? expression(x, depth + 1)
                                : constant(c != 0.0, T::Bool);
    int32_t a = expression(x, depth + 1), b = expression(y, depth + 1);
    if (both)
      return op(Expression::Select, T::Bool, {a, b, constant(0.0, T::Bool)});
    return op(Expression::Select, T::Bool, {a, constant(1.0, T::Bool), b});
  }

  /// Comparison `term` of `x` with `y` in `type`, a constant second: c < x
  /// is x > c, for integers and, as neither holds of NaN, for floats.
  LLVM_ATTRIBUTE_NOINLINE int32_t compared(Expression::Term term,
                                           Expression::Type type, mlir::Value x,
                                           mlir::Value y, int depth) {
    if (!std::isnan(number(x)) && std::isnan(number(y))) {
      std::swap(x, y);
      switch (term) {
      case Expression::Lt:
        term = Expression::Gt;
        break;
      case Expression::Le:
        term = Expression::Ge;
        break;
      case Expression::Gt:
        term = Expression::Lt;
        break;
      case Expression::Ge:
        term = Expression::Le;
        break;
      default:
        break;
      }
    }
    int32_t lhs = expression(x, depth + 1);
    return op(term, type, {lhs, expression(y, depth + 1)});
  }

  /// Makes comparison `e` its negation where that is a comparison too, and
  /// says whether it did: not (a < b) is a >= b for integers, and for
  /// floats, where NaN is in no order, only eq and ne negate each other.
  static bool negate(Expression &e) {
    if (e.kind != Expression::Op)
      return false;
    using X = Expression;
    switch (e.term) {
    case X::Eq:
      e.term = X::Ne;
      return true;
    case X::Ne:
      e.term = X::Eq;
      return true;
    case X::Lt:
    case X::Le:
    case X::Gt:
    case X::Ge:
      if (e.type != X::Int)
        return false;
      e.term = e.term == X::Lt   ? X::Ge
               : e.term == X::Le ? X::Gt
               : e.term == X::Gt ? X::Le
                                 : X::Lt;
      return true;
    default:
      return false;
    }
  }

  /// Whether bools `x` and `y` are equal (`equal`), else whether they
  /// differ. Against a constant that is the other bool itself or its
  /// negation, ne(it, true).
  LLVM_ATTRIBUTE_NOINLINE int32_t equality(bool equal, mlir::Value x,
                                           mlir::Value y, int depth) {
    double first = number(x), second = number(y);
    if (!std::isnan(first)) {
      std::swap(x, y);
      std::swap(first, second);
    }
    int32_t lhs = expression(x, depth + 1);
    if (!std::isnan(second)) {
      if ((second != 0.0) == equal)
        return lhs;
      if (negate(expressions[lhs]))
        return lhs;
      return op(Expression::Ne, Expression::Bool,
                {lhs, constant(1.0, Expression::Bool)});
    }
    return op(equal ? Expression::Eq : Expression::Ne, Expression::Bool,
              {lhs, expression(y, depth + 1)});
  }

  /// `select` as the max or min of the values its condition compares,
  /// when it is that for every value they take, else -1: max(a, b) is a
  /// when a > b, else b, so select(a > b, a, b) is max(a, b) and
  /// select(a < b, b, a) is max(b, a), and integers also for >= and <=.
  /// The max and min of integers do not depend on the order of their
  /// operands, which keep the comparison's.
  int32_t minMax(mlir::arith::SelectOp select, Expression::Type type,
                 int depth) {
    namespace arith = mlir::arith;
    mlir::Value a, b;
    bool greater;
    if (auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>()) {
      using P = arith::CmpIPredicate;
      P predicate = cmp.getPredicate();
      if (type != Expression::Int ||
          (predicate != P::sgt && predicate != P::sge && predicate != P::slt &&
           predicate != P::sle))
        return -1;
      greater = predicate == P::sgt || predicate == P::sge;
      a = cmp.getLhs();
      b = cmp.getRhs();
    } else if (auto cmp =
                   select.getCondition().getDefiningOp<arith::CmpFOp>()) {
      using P = arith::CmpFPredicate;
      P predicate = cmp.getPredicate();
      if (type != Expression::Float ||
          (predicate != P::OGT && predicate != P::OLT))
        return -1;
      greater = predicate == P::OGT;
      a = cmp.getLhs();
      b = cmp.getRhs();
    } else {
      return -1;
    }
    mlir::Value then = select.getTrueValue(), other = select.getFalseValue();
    if (then == b && other == a) {
      greater = !greater;
      if (type != Expression::Int)
        std::swap(a, b);
    } else if (then != a || other != b)
      return -1;
    int32_t first = expression(a, depth + 1);
    return op(greater ? Expression::Max : Expression::Min, type,
              {first, expression(b, depth + 1)});
  }

  /// Whether expression `y` is `x` one iteration on: records [a, b) moved
  /// to [b, 2b - a), the rest the same.
  bool sameExpression(int32_t x, int32_t y, int64_t a, int64_t b) const {
    return sameExpression(x, y, a, b, b - a);
  }

  /// Whether expression `y` is `x` with the records it refers to in
  /// [low, high) `shift` later, the rest the same.
  bool sameExpression(int32_t x, int32_t y, int64_t low, int64_t high,
                      int64_t shift) const {
    const Expression &p = expressions[x], &q = expressions[y];
    if (p.kind != q.kind || p.term != q.term || p.name != q.name ||
        p.type != q.type || p.result != q.result ||
        p.of.size() != q.of.size() || !same(p.number, q.number))
      return false;
    if (p.kind == Expression::Readout || p.kind == Expression::Answer) {
      int64_t moved =
          p.event >= low && p.event < high ? p.event + shift : p.event;
      if (q.event != moved)
        return false;
    }
    for (size_t i = 0; i < p.of.size(); ++i)
      if (!sameExpression(p.of[i], q.of[i], low, high, shift))
        return false;
    return true;
  }

  /// Whether `e` is whether a readout is set: ne(readout, 0).
  bool isSet(const Expression &e) const {
    if (e.kind != Expression::Op || e.term != Expression::Ne ||
        e.type != Expression::Int || e.of.size() != 2)
      return false;
    const Expression &level = expressions[e.of[0]],
                     &zero = expressions[e.of[1]];
    return level.kind == Expression::Readout &&
           zero.kind == Expression::Constant && zero.number == 0.0;
  }
};
