// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The waveform-callback table of the cuDensityMat runtime, run against the
// host fake in fake_cudensitymat/: concurrent runs each see their own
// envelopes, the capacity is per run, a callback evaluated outside its
// run's evolution fails instead of reading another run's parameters, the
// members of a batched run evolve as their own runs do, a workspace serves
// more than one evolution, and a sum callback adds its envelopes. Also the
// traffic of an evolution: RK4 uploads its factors once, and an evolution
// that fails frees what it allocated.

#include "cudm_runtime.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"

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
    auto states = evolveAll(name, 1);
    return states.empty() ? NAN : states[0];
  }

  // The states (interleaved real and imaginary parts) after evolving
  // `count` members over the envelopes.
  std::vector<double> evolveAll(const std::string &name, size_t count) {
    if (cudm_operator_append(handle, op, term, 0, 1.0, 0.0) != CUDM_SUCCESS ||
        cudm_evolve(handle, op, in, out, workspace, 0.0, durationNs, 200, 4) !=
            CUDM_SUCCESS ||
        cudm_state_capture(out) != CUDM_SUCCESS) {
      check(false, name + ": evolution failed: " + cudm_last_error_message());
      return {};
    }
    std::vector<double> states(4 * count);
    check(cudm_last_result_size() ==
              static_cast<int64_t>(states.size() * sizeof(double)),
          name + ": result size");
    cudm_last_result_copy(states.data(), states.size() * sizeof(double));
    return states;
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

// A batch of three initial states evolves under one operator, and each
// member ends where a run of its own from that state ends. The next run is
// not batched.
void batchedMembersEvolveAsTheirOwnRuns() {
  const std::vector<double> initial = {1.0, 0.0, 0.0, 0.0,  0.6, 0.0,
                                       0.0, 0.8, 0.0, -1.0, 0.0, 0.0};
  std::vector<double> batched;
  {
    check(cudm_set_initial_state(initial.data(),
                                 initial.size() * sizeof(double),
                                 3) == CUDM_SUCCESS,
          "batch: set initial states");
    Run run("batch");
    check(run.appendEnvelope(0.25) == CUDM_SUCCESS, "batch: append");
    batched = run.evolveAll("batch", 3);
  }
  for (size_t member = 0; member < 3 && batched.size() == 12; ++member) {
    const std::string name = "member " + std::to_string(member);
    check(cudm_set_initial_state(initial.data() + 4 * member,
                                 4 * sizeof(double), 1) == CUDM_SUCCESS,
          name + ": set initial state");
    Run run(name);
    check(run.appendEnvelope(0.25) == CUDM_SUCCESS, name + ": append");
    const auto single = run.evolveAll(name, 1);
    check(single.size() == 4 &&
              std::memcmp(single.data(), batched.data() + 4 * member,
                          4 * sizeof(double)) == 0,
          name + ": batched and single runs differ");
  }
  Run after("after a batch");
  check(after.appendEnvelope(0.25) == CUDM_SUCCESS, "after: append");
  check(after.evolveAll("after a batch", 1).size() == 4,
        "the run after a batch evolves one state");
  check(cudm_set_initial_state(initial.data(), 3 * sizeof(double), 5) ==
            CUDM_ERROR_INVALID_STATE,
        "a buffer that does not split into the batch is refused");
}

// A multi-diagonal operator takes one vector per ascending offset on one mode.
void multidiagonalOperatorsAreChecked() {
  Run run("multidiagonal");
  constexpr std::array<int64_t, 1> extents = {4};
  constexpr std::array<int32_t, 2> offsets = {-1, 1};
  constexpr std::array<int32_t, 2> descending = {1, -1};
  constexpr std::array<int32_t, 1> outside = {4};
  const std::array<double, 16> data{};
  CudmElementaryOp elementary = nullptr;
  check(cudm_elementary_op_create_multidiagonal(
            run.handle, &elementary, data.data(), data.size(), extents.data(),
            1, 16, 2, offsets.data()) == CUDM_SUCCESS &&
            elementary,
        "an operator on the two off-diagonals is made");
  cudm_elementary_op_destroy(elementary);
  check(cudm_elementary_op_create_multidiagonal(
            run.handle, &elementary, data.data(), data.size(), extents.data(),
            1, 16, 2, descending.data()) == CUDM_ERROR_INVALID_HANDLE &&
            !elementary,
        "descending offsets are refused");
  check(cudm_elementary_op_create_multidiagonal(
            run.handle, &elementary, data.data(), 8, extents.data(), 1, 16, 1,
            outside.data()) == CUDM_ERROR_INVALID_HANDLE,
        "an offset outside the mode is refused");
  check(cudm_elementary_op_create_multidiagonal(
            run.handle, &elementary, data.data(), 8, extents.data(), 1, 16, 2,
            offsets.data()) == CUDM_ERROR_INVALID_HANDLE,
        "data of the wrong size is refused");
}

// The parameters of a sum callback over square envelopes of `amplitudes`,
// with `count` as the number of envelopes it claims.
std::vector<double> squareSum(const std::vector<double> &amplitudes,
                              double count) {
  std::vector<double> parameters = {count};
  for (double amplitude : amplitudes) {
    const auto envelope = squareEnvelope(amplitude);
    parameters.push_back(squareKind);
    parameters.push_back(static_cast<double>(envelope.size()));
    parameters.insert(parameters.end(), envelope.begin(), envelope.end());
  }
  return parameters;
}

// One product whose callback sums envelopes evolves as one product per
// envelope does; a sum whose parameters do not add up fails the evolution.
void aSumCallbackAddsItsEnvelopes() {
  constexpr int32_t modes[] = {0};
  constexpr int32_t duality[] = {0};
  const std::vector<double> amplitudes = {0.25, 0.5, -0.125};
  {
    Run run("sum");
    const auto parameters = squareSum(amplitudes, 3.0);
    check(cudm_op_term_append(
              run.handle, run.term, &run.elementary, modes, duality, 1, 1.0,
              0.0, cudaq_pulse::kSumKind, parameters.data(),
              static_cast<int32_t>(parameters.size())) == CUDM_SUCCESS,
          "sum: append");
    double separate = NAN;
    {
      Run single("one product per envelope");
      for (double amplitude : amplitudes)
        check(single.appendEnvelope(amplitude) == CUDM_SUCCESS,
              "one product per envelope: append");
      separate = single.evolve("one product per envelope");
    }
    const double amplitude = run.evolve("sum");
    check(amplitude == separate && std::abs(amplitude - expected(amplitudes)) <
                                       1.0e-8 * expected(amplitudes),
          "sum: |0> amplitude " + std::to_string(amplitude) +
              ", one product per envelope " + std::to_string(separate));
  }
  for (double count : {2.0, 4.0, -1.0}) {
    Run run("malformed sum");
    const auto parameters = squareSum(amplitudes, count);
    check(cudm_op_term_append(
              run.handle, run.term, &run.elementary, modes, duality, 1, 1.0,
              0.0, cudaq_pulse::kSumKind, parameters.data(),
              static_cast<int32_t>(parameters.size())) == CUDM_SUCCESS,
          "malformed sum: append");
    check(cudm_operator_append(run.handle, run.op, run.term, 0, 1.0, 0.0) ==
                  CUDM_SUCCESS &&
              cudm_evolve(run.handle, run.op, run.in, run.out, run.workspace,
                          0.0, durationNs, 200, 4) != CUDM_SUCCESS,
          "a sum claiming " + std::to_string(count) +
              " of 3 envelopes fails the evolution");
  }
}

// A second evolution with the same workspace re-prepares it, attaches the
// buffer the first one allocated, and ends where the first one did.
void aWorkspaceServesASecondEvolution() {
  Run run("second evolution");
  check(run.appendEnvelope(0.25) == CUDM_SUCCESS, "second evolution: append");
  const auto first = run.evolveAll("first evolution", 1);
  const bool evolved =
      cudm_evolve(run.handle, run.op, run.in, run.out, run.workspace, 0.0,
                  durationNs, 200, 4) == CUDM_SUCCESS &&
      cudm_state_capture(run.out) == CUDM_SUCCESS;
  check(evolved,
        std::string("second evolution failed: ") + cudm_last_error_message());
  std::vector<double> second(4);
  if (evolved)
    cudm_last_result_copy(second.data(), second.size() * sizeof(double));
  check(evolved && first.size() == 4 &&
            std::memcmp(first.data(), second.data(), 4 * sizeof(double)) == 0,
        "the second evolution ends where the first one did");
}

// The host-to-device copies of an RK4 evolution of `batch` states in `steps`
// steps.
int64_t rk4Copies(int64_t batch, int64_t steps) {
  const std::string name = "rk4 copies, batch " + std::to_string(batch) + ", " +
                           std::to_string(steps) + " steps";
  const std::vector<double> initial(4 * static_cast<size_t>(batch), 0.5);
  check(cudm_set_initial_state(initial.data(), initial.size() * sizeof(double),
                               batch) == CUDM_SUCCESS,
        name + ": set initial states");
  Run run(name);
  check(run.appendEnvelope(0.25) == CUDM_SUCCESS &&
            cudm_operator_append(run.handle, run.op, run.term, 0, 1.0, 0.0) ==
                CUDM_SUCCESS,
        name + ": building the operator");
  const int64_t before = fakeCudaHostToDeviceCopies();
  check(cudm_evolve(run.handle, run.op, run.in, run.out, run.workspace, 0.0,
                    durationNs, steps, 4) == CUDM_SUCCESS,
        name + ": evolution failed: " + cudm_last_error_message());
  return fakeCudaHostToDeviceCopies() - before;
}

// The factors of an RK4 step are the same for every step of an evolution,
// so the evolution uploads them once: its host-to-device copies do not grow
// with its steps, for one state and for a batch.
void rk4UploadsItsFactorsOnce() {
  for (int64_t batch : {1, 3}) {
    const int64_t few = rk4Copies(batch, 5);
    const int64_t many = rk4Copies(batch, 400);
    check(few == many, "an RK4 evolution of batch " + std::to_string(batch) +
                           " made " + std::to_string(few) +
                           " host-to-device copies in 5 steps and " +
                           std::to_string(many) + " in 400");
  }
}

// Whichever host-to-device copy of an evolution fails, the evolution reports
// a CUDA error and frees every allocation it made, for each integrator.
void aFailedCopyFreesTheEvolution() {
  for (int32_t integrator = 2; integrator <= 6; ++integrator) {
    const std::string name =
        "failed copy, integrator " + std::to_string(integrator);
    Run run(name);
    check(run.appendEnvelope(0.25) == CUDM_SUCCESS &&
              cudm_operator_append(run.handle, run.op, run.term, 0, 1.0, 0.0) ==
                  CUDM_SUCCESS,
          name + ": building the operator");
    const auto evolve = [&run, integrator] {
      return cudm_evolve(run.handle, run.op, run.in, run.out, run.workspace,
                         0.0, durationNs, 3, integrator);
    };
    // The first evolution allocates the workspace's buffer, which stays.
    check(evolve() == CUDM_SUCCESS, name + ": evolution failed");
    const int64_t live = fakeCudaLiveAllocations();
    bool factorFailed = false;
    int64_t failures = 0;
    for (;; ++failures) {
      fakeCudaFailHostToDeviceCopy(failures);
      const CudmStatus status = evolve();
      fakeCudaFailHostToDeviceCopy(-1);
      check(fakeCudaLiveAllocations() == live,
            name + ": allocations left after copy " + std::to_string(failures));
      if (status == CUDM_SUCCESS)
        break;
      check(status == CUDM_ERROR_CUDA,
            name + ": status of a failed copy " + std::to_string(failures));
      factorFailed |= std::strstr(cudm_last_error_message(),
                                  "cudaMemcpy(state factor)") != nullptr;
      if (failures > 10000) {
        check(false, name + ": every evolution fails");
        break;
      }
    }
    check(failures > 0 && factorFailed,
          name + ": no copy of a factor was made to fail");
  }
}

} // namespace

int main() {
  concurrentRunsKeepTheirOwnEnvelopes();
  capacityIsPerRun();
  callbacksOutsideAnEvolutionFail();
  invalidParametersAreRejected();
  batchedMembersEvolveAsTheirOwnRuns();
  multidiagonalOperatorsAreChecked();
  aWorkspaceServesASecondEvolution();
  aSumCallbackAddsItsEnvelopes();
  rk4UploadsItsFactorsOnce();
  aFailedCopyFreesTheEvolution();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures.load());
    return 1;
  }
  std::puts("cuDensityMat callback table: all checks passed");
  return 0;
}
