/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// The waveform-callback table of the cuDensityMat runtime, run against the
// host fake in fake_cudensitymat/: concurrent runs each see their own
// envelopes, the capacity is per run, and a callback evaluated outside its
// run's evolution fails instead of reading another run's parameters.

#include "cudm_runtime.h"

#include <cudensitymat.h>

#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<int> failures{0};

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

class Barrier {
public:
  explicit Barrier(size_t count) : remaining_(count) {}
  void wait() {
    std::unique_lock lock(mutex_);
    if (--remaining_ == 0) {
      released_.notify_all();
      return;
    }
    released_.wait(lock, [this] { return remaining_ == 0; });
  }

private:
  std::mutex mutex_;
  std::condition_variable released_;
  size_t remaining_;
};

constexpr double durationNs = 10.0;
constexpr int32_t squareKind = 1;

// A square envelope of `amplitude` over [0, durationNs): the callback value
// is amplitude / 2 (WaveformCallback.h).
std::array<double, 10> squareEnvelope(double amplitude) {
  return {0.0, durationNs, amplitude, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

// A one-qubit program whose operator has one product per envelope.
struct Run {
  CudmHandle handle = nullptr;
  CudmState in = nullptr;
  CudmState out = nullptr;
  CudmWorkspace workspace = nullptr;
  CudmOperator op = nullptr;
  CudmElementaryOp elementary = nullptr;
  CudmOpTerm term = nullptr;

  explicit Run(const std::string &name) {
    constexpr std::array<int64_t, 1> extents = {2};
    constexpr std::array<double, 8> identity = {1, 0, 0, 0, 0, 0, 1, 0};
    const bool built =
        cudm_init(&handle) == CUDM_SUCCESS &&
        cudm_state_alloc(handle, &in, extents.data(), 1, 0, 16) ==
            CUDM_SUCCESS &&
        cudm_state_alloc(handle, &out, extents.data(), 1, 0, 16) ==
            CUDM_SUCCESS &&
        cudm_workspace_create(handle, &workspace) == CUDM_SUCCESS &&
        cudm_operator_create(handle, &op, extents.data(), 1) == CUDM_SUCCESS &&
        cudm_elementary_op_create(handle, &elementary, identity.data(),
                                  identity.size(), extents.data(), 1,
                                  16) == CUDM_SUCCESS &&
        cudm_op_term_create(handle, &term, extents.data(), 1) == CUDM_SUCCESS;
    check(built,
          name + ": building the program failed: " + cudm_last_error_message());
  }

  CudmStatus appendEnvelope(double amplitude) {
    constexpr int32_t modes[] = {0};
    constexpr int32_t duality[] = {0};
    const auto parameters = squareEnvelope(amplitude);
    return cudm_op_term_append(handle, term, &elementary, modes, duality, 1,
                               1.0, 0.0, squareKind, parameters.data(),
                               static_cast<int32_t>(parameters.size()));
  }

  // The amplitude of |0> after evolving from |0> over the envelopes.
  double evolve(const std::string &name) {
    if (cudm_operator_append(handle, op, term, 0, 1.0, 0.0) != CUDM_SUCCESS ||
        cudm_evolve(handle, op, in, out, workspace, 0.0, durationNs, 200, 4) !=
            CUDM_SUCCESS ||
        cudm_state_capture(out) != CUDM_SUCCESS) {
      check(false, name + ": evolution failed: " + cudm_last_error_message());
      return NAN;
    }
    std::array<double, 4> state{};
    check(cudm_last_result_size() == sizeof(state), name + ": result size");
    cudm_last_result_copy(state.data(), sizeof(state));
    return state[0];
  }

  ~Run() {
    cudm_op_term_destroy(term);
    cudm_elementary_op_destroy(elementary);
    cudm_operator_destroy(op);
    cudm_workspace_destroy(workspace);
    cudm_state_destroy(out);
    cudm_state_destroy(in);
    cudm_destroy(handle);
  }
};

// Under the fake, d|psi>/dt = s |psi> with s the sum of the callback values.
double expected(const std::vector<double> &amplitudes) {
  double rate = 0.0;
  for (double amplitude : amplitudes)
    rate += 0.5 * amplitude;
  return std::exp(rate * durationNs);
}

// Runs whose envelopes together far exceed one run's 1024 slots are all
// alive before any evolves, and each evolves under its own envelopes only.
void concurrentRunsKeepTheirOwnEnvelopes() {
  constexpr size_t threads = 8;
  constexpr size_t envelopesPerRun = 300;
  Barrier built(threads);
  std::vector<std::thread> workers;
  for (size_t t = 0; t < threads; ++t) {
    workers.emplace_back([t, &built] {
      const std::string name = "run " + std::to_string(t);
      Run run(name);
      std::vector<double> amplitudes;
      for (size_t k = 0; k < envelopesPerRun; ++k) {
        const double amplitude = 1.0e-4 * static_cast<double>(t + 1) +
                                 1.0e-7 * static_cast<double>(k);
        amplitudes.push_back(amplitude);
        if (run.appendEnvelope(amplitude) != CUDM_SUCCESS) {
          check(false, name + ": append failed: " + cudm_last_error_message());
          break;
        }
      }
      built.wait();
      const double amplitude = run.evolve(name);
      const double want = expected(amplitudes);
      check(std::abs(amplitude - want) < 1.0e-9 * want,
            name + ": |0> amplitude " + std::to_string(amplitude) +
                ", expected " + std::to_string(want));
    });
  }
  for (auto &worker : workers)
    worker.join();
}

// The 1024-envelope limit applies to one run, whatever else is in flight.
void capacityIsPerRun() {
  Run first("first");
  Run second("second");
  for (int k = 0; k < 1024; ++k) {
    check(first.appendEnvelope(1.0e-6) == CUDM_SUCCESS, "first: append");
    check(second.appendEnvelope(1.0e-6) == CUDM_SUCCESS, "second: append");
  }
  check(first.appendEnvelope(1.0e-6) == CUDM_ERROR_INTERNAL,
        "the 1025th envelope of a run is refused");
  check(std::strstr(cudm_last_error_message(), "1024") != nullptr,
        "the refusal names the per-run limit");
}

// cuDensityMat evaluates the callbacks only inside cudm_evolve; a callback
// called outside it has no run and reports an error.
void callbacksOutsideAnEvolutionFail() {
  Run run("outside");
  check(run.appendEnvelope(0.25) == CUDM_SUCCESS, "outside: append");
  check(cudm_operator_append(run.handle, run.op, run.term, 0, 1.0, 0.0) ==
            CUDM_SUCCESS,
        "outside: operator append");
  cuDoubleComplex value{};
  check(fakeCudmLastOperatorScalar(1.0, &value) != CUDENSITYMAT_STATUS_SUCCESS,
        "a callback evaluated outside cudm_evolve fails");
}

// Invalid envelope parameters are reported as such, not as a full table.
void invalidParametersAreRejected() {
  Run run("invalid");
  constexpr int32_t modes[] = {0};
  constexpr int32_t duality[] = {0};
  constexpr std::array<double, 3> tooShort = {0.0, 1.0, 1.0};
  check(cudm_op_term_append(run.handle, run.term, &run.elementary, modes,
                            duality, 1, 1.0, 0.0, squareKind, tooShort.data(),
                            tooShort.size()) == CUDM_ERROR_INTERNAL,
        "a short parameter list is refused");
  check(std::strstr(cudm_last_error_message(), "invalid waveform callback") !=
            nullptr,
        "the refusal names the invalid parameters");
}

} // namespace

int main() {
  concurrentRunsKeepTheirOwnEnvelopes();
  capacityIsPerRun();
  callbacksOutsideAnEvolutionFail();
  invalidParametersAreRejected();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures.load());
    return 1;
  }
  std::puts("cuDensityMat callback table: all checks passed");
  return 0;
}
