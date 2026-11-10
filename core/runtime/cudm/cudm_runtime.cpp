// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cudm_runtime.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"
#include "cudaq-pulse/Runtime/WireFormat.h"

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cudensitymat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

struct WaveformCallback {
  bool active = false;
  int32_t kind = 0;
  std::vector<double> parameters;
};

// One handle is one run: the JIT-compiled program creates it, builds its
// operators on it, evolves and destroys it, all on the calling thread. The
// run owns its drive envelopes, so concurrent runs share no callback state.
struct HandleData {
  cudensitymatHandle_t native = nullptr;
  std::vector<WaveformCallback> callbacks;
  std::vector<size_t> freeCallbackSlots;
  // The run's classical handlers, by function id; a program calls a few.
  std::vector<std::pair<uint32_t, CudmClassicalHandler>> classical;
};

struct StateData {
  HandleData *owner = nullptr;
  cudensitymatState_t native = nullptr;
  std::vector<int64_t> modeExtents;
  std::vector<void *> buffers;
  std::vector<size_t> bufferSizes;
  int32_t purity = 0;
};

// The CuDensityMat dialect ComputeType::F64; the runtime computes in
// complex128 only.
constexpr int32_t complex128DataType = 16;

struct WorkspaceData {
  HandleData *owner = nullptr;
  cudensitymatWorkspaceDescriptor_t native = nullptr;
  void *deviceBuffer = nullptr;
  size_t deviceBufferSize = 0;
};

struct OperatorData {
  HandleData *owner = nullptr;
  cudensitymatOperator_t native = nullptr;
};

struct ElementaryOpData {
  HandleData *owner = nullptr;
  cudensitymatElementaryOperator_t native = nullptr;
  void *tensorData = nullptr;
};

struct OpTermData {
  HandleData *owner = nullptr;
  cudensitymatOperatorTerm_t native = nullptr;
  std::vector<size_t> callbackSlots;
};

// cuDensityMat scalar callbacks are plain function pointers with no user
// data, so each envelope of a run gets one of a fixed set of trampolines, and
// a trampoline finds its run through `activeRun`. cuDensityMat invokes CPU
// callbacks synchronously on the thread computing the operator action, which
// is the thread inside cudm_evolve for that run.
constexpr size_t callbackCapacity = 1024;
thread_local const HandleData *activeRun = nullptr;

class ActiveRunScope {
public:
  explicit ActiveRunScope(const HandleData *run) : previous_(activeRun) {
    activeRun = run;
  }
  ~ActiveRunScope() { activeRun = previous_; }
  ActiveRunScope(const ActiveRunScope &) = delete;
  ActiveRunScope &operator=(const ActiveRunScope &) = delete;

private:
  const HandleData *previous_;
};

int32_t evaluateWaveformCallback(size_t slot, double time,
                                 cudaDataType_t dataType, void *scalarStorage) {
  const HandleData *run = activeRun;
  if (!run || slot >= run->callbacks.size() || !run->callbacks[slot].active)
    return 1;
  const WaveformCallback &callback = run->callbacks[slot];
  if (!scalarStorage)
    return 2;
  double value = 0.0;
  if (int32_t status = cudaq_pulse::evaluateWaveform(
          callback.kind, callback.parameters.data(), callback.parameters.size(),
          time, value))
    return status;
  if (dataType != CUDA_C_64F)
    return 4;
  *static_cast<cuDoubleComplex *>(scalarStorage) =
      make_cuDoubleComplex(value, 0.0);
  return 0;
}

template <size_t Slot>
int32_t waveformCallback(double time, int64_t, int32_t, const double *,
                         cudaDataType_t dataType, void *scalarStorage,
                         cudaStream_t) {
  return evaluateWaveformCallback(Slot, time, dataType, scalarStorage);
}

template <size_t... Slots>
constexpr auto makeWaveformCallbackTable(std::index_sequence<Slots...>) {
  return std::array<cudensitymatScalarCallback_t, sizeof...(Slots)>{
      &waveformCallback<Slots>...};
}

constexpr auto waveformCallbackTable =
    makeWaveformCallbackTable(std::make_index_sequence<callbackCapacity>{});

std::optional<size_t> allocateCallbackSlot(HandleData *run, int32_t kind,
                                           const double *parameters,
                                           int32_t parameterCount) {
  size_t slot = run->callbacks.size();
  if (!run->freeCallbackSlots.empty()) {
    slot = run->freeCallbackSlots.back();
    run->freeCallbackSlots.pop_back();
  } else if (slot < callbackCapacity) {
    run->callbacks.emplace_back();
  } else {
    return std::nullopt;
  }
  WaveformCallback &callback = run->callbacks[slot];
  callback.active = true;
  callback.kind = kind;
  callback.parameters.assign(parameters, parameters + parameterCount);
  return slot;
}

void releaseCallbackSlot(HandleData *run, size_t slot) {
  run->callbacks[slot] = {};
  run->freeCallbackSlots.push_back(slot);
}

thread_local std::vector<std::byte> lastResult;
thread_local std::vector<std::byte> pendingInitialState;
thread_local std::string lastError;

CudmStatus fail(CudmStatus status, const char *message) {
  lastError = message;
  lastResult.clear();
  return status;
}

CudmStatus checkCudm(cudensitymatStatus_t status, const char *operation) {
  if (status == CUDENSITYMAT_STATUS_SUCCESS)
    return CUDM_SUCCESS;
  lastError = std::string(operation) + " failed with cuDensityMat status " +
              std::to_string(static_cast<int>(status));
  lastResult.clear();
  return CUDM_ERROR_CUDA;
}

CudmStatus checkCuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return CUDM_SUCCESS;
  lastError = std::string(operation) + " failed: " + cudaGetErrorString(status);
  lastResult.clear();
  return status == cudaErrorNoDevice ? CUDM_ERROR_NO_GPU : CUDM_ERROR_CUDA;
}

CudmStatus initializeBasisZero(StateData *state) {
  auto status = checkCudm(cudensitymatStateInitializeZero(
                              state->owner->native, state->native, nullptr),
                          "cudensitymatStateInitializeZero");
  if (status != CUDM_SUCCESS)
    return status;
  if (state->buffers.empty() || state->bufferSizes.front() == 0)
    return fail(CUDM_ERROR_INVALID_STATE,
                "cuDensityMat state has no component storage");

  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  return checkCuda(cudaMemcpy(state->buffers.front(), &one, sizeof(one),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(initial state)");
}

CudmStatus setZero(StateData *state) {
  return checkCudm(cudensitymatStateInitializeZero(state->owner->native,
                                                   state->native, nullptr),
                   "cudensitymatStateInitializeZero");
}

CudmStatus setFactor(void *deviceFactor, cuDoubleComplex value) {
  return checkCuda(
      cudaMemcpy(deviceFactor, &value, sizeof(value), cudaMemcpyHostToDevice),
      "cudaMemcpy(state factor)");
}

CudmStatus accumulate(StateData *source, StateData *destination,
                      void *deviceFactor, cuDoubleComplex factor) {
  auto status = setFactor(deviceFactor, factor);
  if (status != CUDM_SUCCESS)
    return status;
  return checkCudm(cudensitymatStateComputeAccumulation(
                       source->owner->native, source->native,
                       destination->native, deviceFactor, nullptr),
                   "cudensitymatStateComputeAccumulation");
}

CudmStatus copyState(StateData *source, StateData *destination,
                     void *deviceFactor) {
  auto status = setZero(destination);
  if (status != CUDM_SUCCESS)
    return status;
  return accumulate(source, destination, deviceFactor,
                    make_cuDoubleComplex(1.0, 0.0));
}

CudmStatus prepareWorkspace(HandleData *handle, OperatorData *op,
                            StateData *stateIn, StateData *stateOut,
                            WorkspaceData *workspace) {
  auto status = checkCudm(cudensitymatOperatorPrepareAction(
                              handle->native, op->native, stateIn->native,
                              stateOut->native, CUDENSITYMAT_COMPUTE_64F,
                              std::numeric_limits<size_t>::max(),
                              workspace->native, nullptr),
                          "cudensitymatOperatorPrepareAction");
  if (status != CUDM_SUCCESS)
    return status;

  size_t required = 0;
  status = checkCudm(cudensitymatWorkspaceGetMemorySize(
                         handle->native, workspace->native,
                         CUDENSITYMAT_MEMSPACE_DEVICE,
                         CUDENSITYMAT_WORKSPACE_SCRATCH, &required),
                     "cudensitymatWorkspaceGetMemorySize");
  if (status != CUDM_SUCCESS || required <= workspace->deviceBufferSize)
    return status;

  if (workspace->deviceBuffer)
    cudaFree(workspace->deviceBuffer);
  workspace->deviceBuffer = nullptr;
  workspace->deviceBufferSize = 0;
  status = checkCuda(cudaMalloc(&workspace->deviceBuffer, required),
                     "cudaMalloc(workspace)");
  if (status != CUDM_SUCCESS)
    return status;
  workspace->deviceBufferSize = required;
  return checkCudm(
      cudensitymatWorkspaceSetMemory(
          handle->native, workspace->native, CUDENSITYMAT_MEMSPACE_DEVICE,
          CUDENSITYMAT_WORKSPACE_SCRATCH, workspace->deviceBuffer, required),
      "cudensitymatWorkspaceSetMemory");
}

CudmStatus rhs(HandleData *handle, OperatorData *op, StateData *stateIn,
               StateData *stateOut, WorkspaceData *workspace, double time) {
  auto status = setZero(stateOut);
  if (status != CUDM_SUCCESS)
    return status;
  return checkCudm(cudensitymatOperatorComputeAction(
                       handle->native, op->native, time, 1, 0, nullptr,
                       stateIn->native, stateOut->native, workspace->native,
                       nullptr),
                   "cudensitymatOperatorComputeAction");
}

// Pulse callbacks are right-continuous and use half-open intervals. When a
// stage must sample the right endpoint of the current step, pull the sample
// just inside the interval so a discontinuity at the next pulse boundary does
// not bleed the following pulse's coefficients into this step. The next step's
// first stage still samples the new pulse at the exact boundary.
double boundarySafeSampleTime(double time, double dt) {
  const double endpoint = time + dt;
  const double boundaryEpsilon = std::max(
      std::abs(dt) * 1.0e-9, std::numeric_limits<double>::epsilon() *
                                 std::max(1.0, std::abs(endpoint)) * 8.0);
  return std::max(time, endpoint - boundaryEpsilon);
}

StateData *allocateLike(HandleData *handle, StateData *prototype) {
  CudmState state = nullptr;
  if (cudm_state_alloc(handle, &state, prototype->modeExtents.data(),
                       static_cast<int32_t>(prototype->modeExtents.size()),
                       prototype->purity, complex128DataType) != CUDM_SUCCESS)
    return nullptr;
  auto *result = static_cast<StateData *>(state);
  if (setZero(result) != CUDM_SUCCESS) {
    cudm_state_destroy(result);
    return nullptr;
  }
  return result;
}

CudmStatus integrateStep(HandleData *handle, OperatorData *op,
                         StateData *current, StateData *next,
                         StateData *temporary, StateData *k1, StateData *k2,
                         StateData *k3, StateData *k4, WorkspaceData *workspace,
                         void *deviceFactor, double time, double dt,
                         int32_t integrator) {
  auto status = rhs(handle, op, current, k1, workspace, time);
  if (status != CUDM_SUCCESS)
    return status;

  if (integrator == 2) {
    status = copyState(current, next, deviceFactor);
    if (status == CUDM_SUCCESS)
      status =
          accumulate(k1, next, deviceFactor, make_cuDoubleComplex(dt, 0.0));
    return status;
  }

  // RK2 midpoint, or the first half of RK4.
  status = copyState(current, temporary, deviceFactor);
  if (status == CUDM_SUCCESS)
    status = accumulate(k1, temporary, deviceFactor,
                        make_cuDoubleComplex(0.5 * dt, 0.0));
  if (status == CUDM_SUCCESS)
    status = rhs(handle, op, temporary, k2, workspace, time + 0.5 * dt);
  if (status != CUDM_SUCCESS)
    return status;

  if (integrator == 3) {
    status = copyState(current, next, deviceFactor);
    if (status == CUDM_SUCCESS)
      status =
          accumulate(k2, next, deviceFactor, make_cuDoubleComplex(dt, 0.0));
    return status;
  }

  status = copyState(current, temporary, deviceFactor);
  if (status == CUDM_SUCCESS)
    status = accumulate(k2, temporary, deviceFactor,
                        make_cuDoubleComplex(0.5 * dt, 0.0));
  if (status == CUDM_SUCCESS)
    status = rhs(handle, op, temporary, k3, workspace, time + 0.5 * dt);
  if (status == CUDM_SUCCESS)
    status = copyState(current, temporary, deviceFactor);
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k3, temporary, deviceFactor, make_cuDoubleComplex(dt, 0.0));
  // Sample the final RK stage from inside the current interval so a
  // discontinuity at the next pulse boundary does not shorten this interval by
  // dt / 6 (see boundarySafeSampleTime).
  if (status == CUDM_SUCCESS)
    status = rhs(handle, op, temporary, k4, workspace,
                 boundarySafeSampleTime(time, dt));
  if (status == CUDM_SUCCESS)
    status = copyState(current, next, deviceFactor);
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k1, next, deviceFactor, make_cuDoubleComplex(dt / 6.0, 0.0));
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k2, next, deviceFactor, make_cuDoubleComplex(dt / 3.0, 0.0));
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k3, next, deviceFactor, make_cuDoubleComplex(dt / 3.0, 0.0));
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k4, next, deviceFactor, make_cuDoubleComplex(dt / 6.0, 0.0));
  return status;
}

// Magnus expansion (first-order / midpoint) with a Taylor series for the
// matrix exponential action. Mirrors cudaq::integrators::magnus_expansion:
// with the Liouvillian L frozen at the interval midpoint, advance
//   next = sum_{k=0}^{N} (dt L)^k / k! * current.
// Each Taylor term reuses the previous one via w_k = L * w_{k-1}, so a single
// Liouvillian action per term suffices. `result` receives the new state;
// `w` and `Lw` are scratch buffers.
CudmStatus integrateStepMagnus(HandleData *handle, OperatorData *op,
                               StateData *current, StateData *result,
                               StateData *w, StateData *Lw,
                               WorkspaceData *workspace, void *deviceFactor,
                               double time, double dt, int numTaylorTerms) {
  const double tMid = time + 0.5 * dt;
  // k = 0 term: result = current; running vector w_0 = current.
  auto status = copyState(current, result, deviceFactor);
  if (status == CUDM_SUCCESS)
    status = copyState(current, w, deviceFactor);

  double coeff = 1.0;
  for (int k = 1; status == CUDM_SUCCESS && k <= numTaylorTerms; ++k) {
    status = rhs(handle, op, w, Lw, workspace, tMid);
    if (status != CUDM_SUCCESS)
      break;
    coeff *= dt / static_cast<double>(k);
    status =
        accumulate(Lw, result, deviceFactor, make_cuDoubleComplex(coeff, 0.0));
    // Advance the running vector: w_k <- L * w_{k-1}. Swapping reuses buffers;
    // the next rhs() zero-initializes its output before accumulating.
    std::swap(w, Lw);
  }
  return status;
}

// Crank-Nicolson predictor-corrector. Mirrors
// cudaq::integrators::crank_nicolson:
//   k1 = L(t) * current
//   rho_iter = current + dt * k1                       (explicit predictor)
//   repeat: k2 = L(t + dt) * rho_iter
//           rho_iter = current + (dt/2) (k1 + k2)       (trapezoidal corrector)
// The endpoint sample uses boundarySafeSampleTime so pulse discontinuities at
// the next boundary do not leak into this step. `next` receives the new state.
CudmStatus integrateStepCrankNicolson(HandleData *handle, OperatorData *op,
                                      StateData *current, StateData *next,
                                      StateData *k1, StateData *k2,
                                      StateData *rhoIter, StateData *rhoNext,
                                      WorkspaceData *workspace,
                                      void *deviceFactor, double time,
                                      double dt, int numCorrectorSteps) {
  auto status = rhs(handle, op, current, k1, workspace, time);
  if (status != CUDM_SUCCESS)
    return status;

  status = copyState(current, rhoIter, deviceFactor);
  if (status == CUDM_SUCCESS)
    status =
        accumulate(k1, rhoIter, deviceFactor, make_cuDoubleComplex(dt, 0.0));

  const double tNext = boundarySafeSampleTime(time, dt);
  for (int iter = 0; status == CUDM_SUCCESS && iter < numCorrectorSteps;
       ++iter) {
    status = rhs(handle, op, rhoIter, k2, workspace, tNext);
    if (status == CUDM_SUCCESS)
      status = copyState(current, rhoNext, deviceFactor);
    if (status == CUDM_SUCCESS)
      status = accumulate(k1, rhoNext, deviceFactor,
                          make_cuDoubleComplex(0.5 * dt, 0.0));
    if (status == CUDM_SUCCESS)
      status = accumulate(k2, rhoNext, deviceFactor,
                          make_cuDoubleComplex(0.5 * dt, 0.0));
    std::swap(rhoIter, rhoNext);
  }

  if (status == CUDM_SUCCESS)
    status = copyState(rhoIter, next, deviceFactor);
  return status;
}

} // namespace

extern "C" {

int64_t cudm_runtime_version(void) {
  return static_cast<int64_t>(cudensitymatGetVersion());
}

const char *cudm_last_error_message(void) { return lastError.c_str(); }

CudmStatus cudm_init(CudmHandle *handle) {
  if (!handle)
    return fail(CUDM_ERROR_INVALID_HANDLE, "cudm_init received a null output");
  *handle = nullptr;
  auto *data = new (std::nothrow) HandleData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL, "failed to allocate cuDensityMat handle");
  auto status =
      checkCudm(cudensitymatCreate(&data->native), "cudensitymatCreate");
  if (status != CUDM_SUCCESS) {
    delete data;
    return status;
  }
  lastError.clear();
  lastResult.clear();
  *handle = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_destroy(CudmHandle handle) {
  if (!handle)
    return CUDM_SUCCESS;
  auto *data = static_cast<HandleData *>(handle);
  auto status =
      checkCudm(cudensitymatDestroy(data->native), "cudensitymatDestroy");
  delete data;
  return status;
}

CudmStatus cudm_state_alloc(CudmHandle handle, CudmState *state,
                            const int64_t *modeExtents, int32_t numModes,
                            int32_t purity, int32_t dataType) {
  if (state)
    *state = nullptr;
  if (!handle || !state || !modeExtents || numModes <= 0)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "cudm_state_alloc received invalid arguments");
  if (dataType != complex128DataType)
    return fail(CUDM_ERROR_INTERNAL,
                "cudm-runtime supports complex128 (f64) states only");
  auto *owner = static_cast<HandleData *>(handle);
  auto *data = new (std::nothrow) StateData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL, "failed to allocate state wrapper");
  data->owner = owner;
  data->modeExtents.assign(modeExtents, modeExtents + numModes);
  data->purity = purity;
  auto nativePurity = purity == 0 ? CUDENSITYMAT_STATE_PURITY_PURE
                                  : CUDENSITYMAT_STATE_PURITY_MIXED;
  auto status = checkCudm(cudensitymatCreateState(owner->native, nativePurity,
                                                  numModes, modeExtents, 0,
                                                  CUDA_C_64F, &data->native),
                          "cudensitymatCreateState");
  int32_t numComponents = 0;
  if (status == CUDM_SUCCESS)
    status = checkCudm(cudensitymatStateGetNumComponents(
                           owner->native, data->native, &numComponents),
                       "cudensitymatStateGetNumComponents");
  if (status == CUDM_SUCCESS && numComponents <= 0)
    status = fail(CUDM_ERROR_INVALID_STATE,
                  "cuDensityMat created a state with no components");
  if (status == CUDM_SUCCESS) {
    data->buffers.resize(numComponents, nullptr);
    data->bufferSizes.resize(numComponents, 0);
    status = checkCudm(cudensitymatStateGetComponentStorageSize(
                           owner->native, data->native, numComponents,
                           data->bufferSizes.data()),
                       "cudensitymatStateGetComponentStorageSize");
  }
  for (int32_t i = 0; status == CUDM_SUCCESS && i < numComponents; ++i)
    status = checkCuda(cudaMalloc(&data->buffers[i], data->bufferSizes[i]),
                       "cudaMalloc(state component)");
  if (status == CUDM_SUCCESS)
    status = checkCudm(cudensitymatStateAttachComponentStorage(
                           owner->native, data->native, numComponents,
                           data->buffers.data(), data->bufferSizes.data()),
                       "cudensitymatStateAttachComponentStorage");
  if (status == CUDM_SUCCESS)
    status = initializeBasisZero(data);
  if (status == CUDM_SUCCESS && !pendingInitialState.empty()) {
    size_t total = 0;
    for (size_t componentSize : data->bufferSizes)
      total += componentSize;
    if (total != pendingInitialState.size()) {
      status = fail(CUDM_ERROR_INVALID_STATE,
                    "initial state size does not match the allocated state");
    } else {
      size_t offset = 0;
      for (size_t i = 0; status == CUDM_SUCCESS && i < data->buffers.size();
           ++i) {
        status = checkCuda(
            cudaMemcpy(data->buffers[i], pendingInitialState.data() + offset,
                       data->bufferSizes[i], cudaMemcpyHostToDevice),
            "cudaMemcpy(initial state)");
        offset += data->bufferSizes[i];
      }
    }
    pendingInitialState.clear();
  }
  if (status != CUDM_SUCCESS) {
    for (void *buffer : data->buffers)
      if (buffer)
        cudaFree(buffer);
    if (data->native)
      cudensitymatDestroyState(data->native);
    delete data;
    return status;
  }
  *state = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_state_destroy(CudmState state) {
  if (!state)
    return CUDM_SUCCESS;
  auto *data = static_cast<StateData *>(state);
  auto status = checkCudm(cudensitymatDestroyState(data->native),
                          "cudensitymatDestroyState");
  for (void *buffer : data->buffers)
    if (buffer)
      cudaFree(buffer);
  delete data;
  return status;
}

CudmStatus cudm_state_capture(CudmState state) {
  // Generated LLVM currently continues into cleanup after a runtime call
  // fails. Preserve the first diagnostic and, critically, do not turn an
  // unsuccessful evolution into an apparently valid |0> result.
  if (!lastError.empty()) {
    lastResult.clear();
    return CUDM_ERROR_INTERNAL;
  }
  if (!state)
    return fail(CUDM_ERROR_INVALID_STATE, "cannot capture a null state");
  auto *data = static_cast<StateData *>(state);
  const size_t total = [&] {
    size_t size = 0;
    for (auto componentSize : data->bufferSizes)
      size += componentSize;
    return size;
  }();
  lastResult.resize(total);
  size_t offset = 0;
  for (size_t i = 0; i < data->buffers.size(); ++i) {
    auto status =
        checkCuda(cudaMemcpy(lastResult.data() + offset, data->buffers[i],
                             data->bufferSizes[i], cudaMemcpyDeviceToHost),
                  "cudaMemcpy(capture state)");
    if (status != CUDM_SUCCESS)
      return status;
    offset += data->bufferSizes[i];
  }
  lastError.clear();
  return CUDM_SUCCESS;
}

CudmStatus cudm_set_initial_state(const void *data, int64_t size) {
  if (size < 0 || (size > 0 && !data))
    return fail(CUDM_ERROR_INVALID_STATE, "invalid initial state buffer");
  const auto *bytes = static_cast<const std::byte *>(data);
  pendingInitialState.assign(bytes, bytes + size);
  return CUDM_SUCCESS;
}

int64_t cudm_last_result_size(void) {
  return static_cast<int64_t>(lastResult.size());
}

CudmStatus cudm_last_result_copy(void *destination, int64_t destinationSize) {
  if (!destination || destinationSize < 0 ||
      static_cast<size_t>(destinationSize) < lastResult.size())
    return fail(CUDM_ERROR_INTERNAL, "result destination is too small");
  std::memcpy(destination, lastResult.data(), lastResult.size());
  return CUDM_SUCCESS;
}

CudmStatus cudm_workspace_create(CudmHandle handle, CudmWorkspace *workspace) {
  if (workspace)
    *workspace = nullptr;
  if (!handle || !workspace)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "workspace create received invalid arguments");
  auto *data = new (std::nothrow) WorkspaceData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL, "failed to allocate workspace wrapper");
  data->owner = static_cast<HandleData *>(handle);
  auto status =
      checkCudm(cudensitymatCreateWorkspace(data->owner->native, &data->native),
                "cudensitymatCreateWorkspace");
  if (status != CUDM_SUCCESS) {
    delete data;
    return status;
  }
  *workspace = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_workspace_destroy(CudmWorkspace workspace) {
  if (!workspace)
    return CUDM_SUCCESS;
  auto *data = static_cast<WorkspaceData *>(workspace);
  auto status = checkCudm(cudensitymatDestroyWorkspace(data->native),
                          "cudensitymatDestroyWorkspace");
  if (data->deviceBuffer)
    cudaFree(data->deviceBuffer);
  delete data;
  return status;
}

CudmStatus cudm_operator_create(CudmHandle handle, CudmOperator *op,
                                const int64_t *modeExtents, int32_t numModes) {
  if (op)
    *op = nullptr;
  if (!handle || !op || !modeExtents || numModes <= 0)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "operator create received invalid arguments");
  auto *data = new (std::nothrow) OperatorData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL, "failed to allocate operator wrapper");
  data->owner = static_cast<HandleData *>(handle);
  auto status =
      checkCudm(cudensitymatCreateOperator(data->owner->native, numModes,
                                           modeExtents, &data->native),
                "cudensitymatCreateOperator");
  if (status != CUDM_SUCCESS) {
    delete data;
    return status;
  }
  *op = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_operator_destroy(CudmOperator op) {
  if (!op)
    return CUDM_SUCCESS;
  auto *data = static_cast<OperatorData *>(op);
  auto status = checkCudm(cudensitymatDestroyOperator(data->native),
                          "cudensitymatDestroyOperator");
  delete data;
  return status;
}

CudmStatus cudm_elementary_op_create(CudmHandle handle,
                                     CudmElementaryOp *elementaryOp,
                                     const void *tensorData,
                                     int64_t tensorValueCount,
                                     const int64_t *modeExtents,
                                     int32_t numModes, int32_t dataType) {
  if (elementaryOp)
    *elementaryOp = nullptr;
  if (!handle || !elementaryOp || !tensorData || tensorValueCount <= 0 ||
      !modeExtents || numModes <= 0)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "elementary operator create received invalid arguments");
  if (dataType != complex128DataType)
    return fail(CUDM_ERROR_INTERNAL,
                "cudm-runtime supports complex128 (f64) operators only");
  auto *data = new (std::nothrow) ElementaryOpData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL,
                "failed to allocate elementary operator wrapper");
  data->owner = static_cast<HandleData *>(handle);
  const size_t bytes = static_cast<size_t>(tensorValueCount) * sizeof(double);
  auto status = checkCuda(cudaMalloc(&data->tensorData, bytes),
                          "cudaMalloc(elementary tensor)");
  if (status == CUDM_SUCCESS)
    status = checkCuda(
        cudaMemcpy(data->tensorData, tensorData, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(elementary tensor)");
  if (status == CUDM_SUCCESS)
    status = checkCudm(
        cudensitymatCreateElementaryOperator(
            data->owner->native, numModes, modeExtents,
            CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F,
            data->tensorData, cudensitymatTensorCallbackNone,
            cudensitymatTensorGradientCallbackNone, &data->native),
        "cudensitymatCreateElementaryOperator");
  if (status != CUDM_SUCCESS) {
    if (data->tensorData)
      cudaFree(data->tensorData);
    delete data;
    return status;
  }
  *elementaryOp = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_elementary_op_destroy(CudmElementaryOp elementaryOp) {
  if (!elementaryOp)
    return CUDM_SUCCESS;
  auto *data = static_cast<ElementaryOpData *>(elementaryOp);
  auto status = checkCudm(cudensitymatDestroyElementaryOperator(data->native),
                          "cudensitymatDestroyElementaryOperator");
  if (data->tensorData)
    cudaFree(data->tensorData);
  delete data;
  return status;
}

CudmStatus cudm_op_term_create(CudmHandle handle, CudmOpTerm *term,
                               const int64_t *modeExtents, int32_t numModes) {
  if (term)
    *term = nullptr;
  if (!handle || !term || !modeExtents || numModes <= 0)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "operator term create received invalid arguments");
  auto *data = new (std::nothrow) OpTermData{};
  if (!data)
    return fail(CUDM_ERROR_INTERNAL,
                "failed to allocate operator term wrapper");
  data->owner = static_cast<HandleData *>(handle);
  auto status =
      checkCudm(cudensitymatCreateOperatorTerm(data->owner->native, numModes,
                                               modeExtents, &data->native),
                "cudensitymatCreateOperatorTerm");
  if (status != CUDM_SUCCESS) {
    delete data;
    return status;
  }
  *term = data;
  return CUDM_SUCCESS;
}

CudmStatus cudm_op_term_append(CudmHandle handle, CudmOpTerm term,
                               const CudmElementaryOp *elementaryOps,
                               const int32_t *modesActedOn,
                               const int32_t *duality, int32_t numElementaryOps,
                               double coefficientReal, double coefficientImag,
                               int32_t callbackKind,
                               const double *callbackParameters,
                               int32_t callbackParameterCount) {
  if (!handle || !term || !elementaryOps || !modesActedOn || !duality ||
      numElementaryOps <= 0)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "operator term append received invalid arguments");
  std::vector<cudensitymatElementaryOperator_t> nativeOps;
  nativeOps.reserve(numElementaryOps);
  for (int32_t i = 0; i < numElementaryOps; ++i) {
    if (!elementaryOps[i])
      return fail(CUDM_ERROR_INTERNAL,
                  "operator term contains a null elementary operator");
    nativeOps.push_back(
        static_cast<ElementaryOpData *>(elementaryOps[i])->native);
  }
  auto *owner = static_cast<HandleData *>(handle);
  auto *termData = static_cast<OpTermData *>(term);
  cudensitymatWrappedScalarCallback_t wrappedCallback =
      cudensitymatScalarCallbackNone;
  std::optional<size_t> callbackSlot;
  if (callbackKind != 0) {
    if (!callbackParameters ||
        callbackParameterCount <
            static_cast<int32_t>(cudaq_pulse::kWaveformHeader))
      return fail(CUDM_ERROR_INTERNAL,
                  "operator term append received invalid waveform callback "
                  "parameters");
    if (termData->owner != owner)
      return fail(CUDM_ERROR_INVALID_HANDLE,
                  "operator term belongs to a different cuDensityMat handle");
    callbackSlot = allocateCallbackSlot(owner, callbackKind, callbackParameters,
                                        callbackParameterCount);
    if (!callbackSlot)
      return fail(CUDM_ERROR_INTERNAL,
                  "the drives of this program need more than 1024 envelope "
                  "callbacks");
    wrappedCallback.callback = waveformCallbackTable[*callbackSlot];
    wrappedCallback.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU;
    wrappedCallback.wrapper = nullptr;
  }
  auto status =
      checkCudm(cudensitymatOperatorTermAppendElementaryProduct(
                    owner->native, termData->native, numElementaryOps,
                    nativeOps.data(), modesActedOn, duality,
                    make_cuDoubleComplex(coefficientReal, coefficientImag),
                    wrappedCallback, cudensitymatScalarGradientCallbackNone),
                "cudensitymatOperatorTermAppendElementaryProduct");
  if (status == CUDM_SUCCESS && callbackSlot)
    termData->callbackSlots.push_back(*callbackSlot);
  else if (callbackSlot)
    releaseCallbackSlot(owner, *callbackSlot);
  return status;
}

CudmStatus cudm_op_term_destroy(CudmOpTerm term) {
  if (!term)
    return CUDM_SUCCESS;
  auto *data = static_cast<OpTermData *>(term);
  auto status = checkCudm(cudensitymatDestroyOperatorTerm(data->native),
                          "cudensitymatDestroyOperatorTerm");
  for (size_t slot : data->callbackSlots)
    releaseCallbackSlot(data->owner, slot);
  delete data;
  return status;
}

CudmStatus cudm_operator_append(CudmHandle handle, CudmOperator op,
                                CudmOpTerm term, int32_t duality,
                                double coefficientReal,
                                double coefficientImag) {
  if (!handle || !op || !term)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "operator append received invalid arguments");
  auto *owner = static_cast<HandleData *>(handle);
  auto *opData = static_cast<OperatorData *>(op);
  auto *termData = static_cast<OpTermData *>(term);
  return checkCudm(cudensitymatOperatorAppendTerm(
                       owner->native, opData->native, termData->native, duality,
                       make_cuDoubleComplex(coefficientReal, coefficientImag),
                       cudensitymatScalarCallbackNone,
                       cudensitymatScalarGradientCallbackNone),
                   "cudensitymatOperatorAppendTerm");
}

CudmStatus cudm_evolve(CudmHandle handle, CudmOperator op, CudmState stateIn,
                       CudmState stateOut, CudmWorkspace workspace,
                       double timeStart, double timeEnd, int64_t numSteps,
                       int32_t integrator) {
  if (!handle || !op || !stateIn || !stateOut || !workspace)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "cudm_evolve received a null runtime object");
  if (numSteps <= 0 || !(timeEnd > timeStart))
    return fail(CUDM_ERROR_INTERNAL, "invalid evolution interval");
  // The dialect IntegratorKind values map as: 2=rk1, 3=rk2, 4=rk4,
  // 5=magnus (Taylor-series midpoint), 6=crank_nicolson (predictor-corrector).
  // These mirror the mainlined cudaq::integrators algorithms of the same name,
  // driven here through the cuDensityMat Liouvillian action.
  if (integrator < 2 || integrator > 6)
    return fail(CUDM_ERROR_INTERNAL,
                "cudm-runtime supports rk1, rk2, rk4, magnus, and "
                "crank_nicolson integrators");

  // Match the mainlined integrator defaults for parity.
  constexpr int kMagnusTaylorTerms = 10;
  constexpr int kCrankNicolsonCorrectorSteps = 2;

  auto *owner = static_cast<HandleData *>(handle);
  auto *operatorData = static_cast<OperatorData *>(op);
  auto *input = static_cast<StateData *>(stateIn);
  auto *output = static_cast<StateData *>(stateOut);
  auto *workspaceData = static_cast<WorkspaceData *>(workspace);
  const ActiveRunScope runScope(owner);

  std::vector<StateData *> scratch;
  scratch.reserve(7);
  for (int i = 0; i < 7; ++i) {
    auto *state = allocateLike(owner, input);
    if (!state) {
      for (auto *allocated : scratch)
        cudm_state_destroy(allocated);
      return CUDM_ERROR_CUDA;
    }
    scratch.push_back(state);
  }
  auto *current = scratch[0];
  auto *next = scratch[1];
  auto *temporary = scratch[2];
  auto *k1 = scratch[3];
  auto *k2 = scratch[4];
  auto *k3 = scratch[5];
  auto *k4 = scratch[6];

  void *deviceFactor = nullptr;
  auto status = checkCuda(cudaMalloc(&deviceFactor, sizeof(cuDoubleComplex)),
                          "cudaMalloc(integration factor)");
  if (status == CUDM_SUCCESS)
    status = copyState(input, current, deviceFactor);
  if (status == CUDM_SUCCESS)
    status = prepareWorkspace(owner, operatorData, current, k1, workspaceData);

  const double dt = (timeEnd - timeStart) / static_cast<double>(numSteps);
  for (int64_t step = 0; status == CUDM_SUCCESS && step < numSteps; ++step) {
    const double stepTime = timeStart + static_cast<double>(step) * dt;
    if (integrator == 5) {
      // Magnus: next=result, temporary=w, k1=Lw.
      status = integrateStepMagnus(owner, operatorData, current, next,
                                   temporary, k1, workspaceData, deviceFactor,
                                   stepTime, dt, kMagnusTaylorTerms);
    } else if (integrator == 6) {
      // Crank-Nicolson: k1/k2 = Liouvillian actions, k3=rho_iter, k4=rho_next.
      status = integrateStepCrankNicolson(
          owner, operatorData, current, next, k1, k2, k3, k4, workspaceData,
          deviceFactor, stepTime, dt, kCrankNicolsonCorrectorSteps);
    } else {
      status = integrateStep(owner, operatorData, current, next, temporary, k1,
                             k2, k3, k4, workspaceData, deviceFactor, stepTime,
                             dt, integrator);
    }
    std::swap(current, next);
  }
  if (status == CUDM_SUCCESS)
    status = copyState(current, output, deviceFactor);
  if (deviceFactor)
    cudaFree(deviceFactor);
  for (auto *allocated : scratch)
    cudm_state_destroy(allocated);
  return status;
}

CudmStatus cudm_classical_register(CudmHandle handle, uint32_t function_id,
                                   CudmClassicalHandler handler) {
  auto *run = static_cast<HandleData *>(handle);
  if (!run)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "cudm_classical_register received a null handle");
  if (!handler)
    return fail(CUDM_ERROR_INVALID_STATE,
                "cudm_classical_register received a null handler");
  for (auto &entry : run->classical)
    if (entry.first == function_id) {
      entry.second = handler;
      return CUDM_SUCCESS;
    }
  run->classical.emplace_back(function_id, handler);
  return CUDM_SUCCESS;
}

CudmStatus cudm_classical_call(CudmHandle handle, const void *request,
                               int64_t request_bytes, void *answer,
                               int64_t answer_capacity, int64_t *answer_bytes) {
  using cudaq_pulse::wire::RequestHeader;
  using cudaq_pulse::wire::ResponseHeader;
  auto *run = static_cast<HandleData *>(handle);
  if (!run)
    return fail(CUDM_ERROR_INVALID_HANDLE,
                "cudm_classical_call received a null handle");
  if (!request || !answer || !answer_bytes ||
      request_bytes < static_cast<int64_t>(sizeof(RequestHeader)) ||
      answer_capacity < static_cast<int64_t>(sizeof(ResponseHeader)))
    return fail(CUDM_ERROR_INVALID_STATE,
                "cudm_classical_call needs a request and an answer slot of at "
                "least one header");
  RequestHeader header;
  std::memcpy(&header, request, sizeof header);
  if (header.magic != cudaq_pulse::wire::kRequestMagic ||
      sizeof(RequestHeader) + header.argLen >
          static_cast<uint64_t>(request_bytes))
    return fail(CUDM_ERROR_INVALID_STATE,
                "cudm_classical_call received a malformed request");
  CudmClassicalHandler handler = nullptr;
  for (const auto &entry : run->classical)
    if (entry.first == header.functionId)
      handler = entry.second;
  if (!handler) {
    lastError = "no classical handler registered on this run for function " +
                std::to_string(header.functionId);
    return CUDM_ERROR_INVALID_STATE;
  }
  handler(request, answer, static_cast<size_t>(answer_capacity));
  ResponseHeader response;
  std::memcpy(&response, answer, sizeof response);
  if (response.magic != cudaq_pulse::wire::kResponseMagic ||
      response.requestId != header.requestId ||
      sizeof(ResponseHeader) + response.resultLen >
          static_cast<uint64_t>(answer_capacity))
    return fail(CUDM_ERROR_INVALID_STATE,
                "a classical handler wrote a malformed answer");
  *answer_bytes =
      static_cast<int64_t>(sizeof(ResponseHeader) + response.resultLen);
  return response.status == 0 ? CUDM_SUCCESS : CUDM_ERROR_INVALID_STATE;
}

} // extern "C"
