// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A host implementation of the cuDensityMat calls cudm_runtime.cpp makes. The
// operator action is a scalar model: L(t) rho = s(t) rho, where s(t) sums,
// over every appended term and product, the coefficients times the value of
// the product's scalar callback at t. That keeps the runtime's integrators,
// state handling and callback table real while needing no GPU. A batched
// state holds its members one after another, and each member takes the
// callbacks' value and the scaling factor for its own index. It holds the
// runtime to the library's documented workspace contract: a prepared
// workspace has no memory until some is attached, and an action without it
// fails.

#include <cudensitymat.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Complex = std::complex<double>;

struct Product {
  Complex coefficient;
  cudensitymatScalarCallback_t callback;
};

} // namespace

struct FakeCudmHandle {};
struct FakeCudmWorkspace {
  size_t required = 0;
  bool attached = false;
};
struct FakeCudmElementary {};

struct FakeCudmState {
  size_t size = 0;
  Complex *data = nullptr;
  size_t batch = 1;
};

struct FakeCudmTerm {
  std::vector<Product> products;
};

struct FakeCudmOperator {
  std::vector<std::pair<FakeCudmTerm *, Complex>> terms;
};

namespace {

cudensitymatStatus_t scalarsAt(const FakeCudmOperator *op, double time,
                               std::vector<Complex> &values) {
  std::fill(values.begin(), values.end(), Complex{});
  std::vector<cuDoubleComplex> samples(values.size());
  for (const auto &[term, termCoefficient] : op->terms) {
    for (const Product &product : term->products) {
      std::fill(samples.begin(), samples.end(), make_cuDoubleComplex(1.0, 0.0));
      if (product.callback &&
          product.callback(time, static_cast<int64_t>(samples.size()), 0,
                           nullptr, CUDA_C_64F, samples.data(), nullptr) != 0)
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      for (size_t member = 0; member < values.size(); ++member)
        values[member] += termCoefficient * product.coefficient *
                          Complex(samples[member].x, samples[member].y);
    }
  }
  return CUDENSITYMAT_STATUS_SUCCESS;
}

Complex toComplex(cuDoubleComplex value) { return {value.x, value.y}; }

thread_local FakeCudmOperator *lastOperator = nullptr;

// Of this thread: the host-to-device copies made, the allocations not freed
// yet, and the host-to-device copies left before one fails (none if negative).
thread_local int64_t hostToDeviceCopies = 0;
thread_local int64_t liveAllocations = 0;
thread_local int64_t copiesBeforeFailure = -1;

} // namespace

cudaError_t cudaMalloc(void **pointer, size_t size) {
  *pointer = std::calloc(1, size == 0 ? 1 : size);
  if (!*pointer)
    return cudaErrorMemoryAllocation;
  ++liveAllocations;
  return cudaSuccess;
}

cudaError_t cudaFree(void *pointer) {
  if (pointer)
    --liveAllocations;
  std::free(pointer);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *destination, const void *source, size_t size,
                       cudaMemcpyKind kind) {
  if (kind == cudaMemcpyHostToDevice) {
    if (copiesBeforeFailure == 0) {
      copiesBeforeFailure = -1;
      return cudaErrorUnknown;
    }
    if (copiesBeforeFailure > 0)
      --copiesBeforeFailure;
    ++hostToDeviceCopies;
  }
  std::memcpy(destination, source, size);
  return cudaSuccess;
}

int64_t fakeCudaHostToDeviceCopies() { return hostToDeviceCopies; }

int64_t fakeCudaLiveAllocations() { return liveAllocations; }

void fakeCudaFailHostToDeviceCopy(int64_t after) {
  copiesBeforeFailure = after;
}

const char *cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }

size_t cudensitymatGetVersion() { return 700; }

cudensitymatStatus_t cudensitymatCreate(cudensitymatHandle_t *handle) {
  *handle = new FakeCudmHandle{};
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatDestroy(cudensitymatHandle_t handle) {
  delete handle;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatCreateState(const cudensitymatHandle_t,
                        cudensitymatStatePurity_t purity, int32_t numSpaceModes,
                        const int64_t spaceModeExtents[], int64_t batchSize,
                        cudaDataType_t dataType, cudensitymatState_t *state) {
  if (dataType != CUDA_C_64F)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  size_t size = 1;
  for (int32_t i = 0; i < numSpaceModes; ++i)
    size *= static_cast<size_t>(spaceModeExtents[i]);
  if (purity == CUDENSITYMAT_STATE_PURITY_MIXED)
    size *= size;
  const size_t batch = batchSize > 1 ? static_cast<size_t>(batchSize) : 1;
  *state = new FakeCudmState{size * batch, nullptr, batch};
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatDestroyState(cudensitymatState_t state) {
  delete state;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatStateGetNumComponents(const cudensitymatHandle_t,
                                  const cudensitymatState_t, int32_t *count) {
  *count = 1;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatStateGetComponentStorageSize(
    const cudensitymatHandle_t, const cudensitymatState_t state, int32_t,
    size_t componentBufferSize[]) {
  componentBufferSize[0] = state->size * sizeof(Complex);
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatStateAttachComponentStorage(
    const cudensitymatHandle_t, cudensitymatState_t state, int32_t,
    void *componentBuffer[], const size_t[]) {
  state->data = static_cast<Complex *>(componentBuffer[0]);
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatStateInitializeZero(const cudensitymatHandle_t,
                                                     cudensitymatState_t state,
                                                     cudaStream_t) {
  std::fill(state->data, state->data + state->size, Complex{});
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatStateComputeAccumulation(
    const cudensitymatHandle_t, const cudensitymatState_t stateIn,
    cudensitymatState_t stateOut, const void *scalingFactors, cudaStream_t) {
  const auto *factors = static_cast<const cuDoubleComplex *>(scalingFactors);
  const size_t member = stateOut->size / stateOut->batch;
  for (size_t i = 0; i < stateOut->size; ++i)
    stateOut->data[i] += toComplex(factors[i / member]) * stateIn->data[i];
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatCreateWorkspace(const cudensitymatHandle_t,
                            cudensitymatWorkspaceDescriptor_t *workspace) {
  *workspace = new FakeCudmWorkspace{};
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspace) {
  delete workspace;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatWorkspaceGetMemorySize(
    const cudensitymatHandle_t,
    const cudensitymatWorkspaceDescriptor_t workspace, cudensitymatMemspace_t,
    cudensitymatWorkspaceKind_t, size_t *size) {
  *size = workspace->required;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatWorkspaceSetMemory(
    const cudensitymatHandle_t, cudensitymatWorkspaceDescriptor_t workspace,
    cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, void *buffer,
    size_t size) {
  workspace->attached = buffer && size >= workspace->required;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatCreateOperator(const cudensitymatHandle_t,
                                                int32_t, const int64_t[],
                                                cudensitymatOperator_t *op) {
  *op = new FakeCudmOperator{};
  lastOperator = *op;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatDestroyOperator(cudensitymatOperator_t op) {
  if (lastOperator == op)
    lastOperator = nullptr;
  delete op;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatCreateElementaryOperator(
    const cudensitymatHandle_t, int32_t, const int64_t[],
    cudensitymatElementaryOperatorSparsity_t, int32_t, const int32_t[],
    cudaDataType_t, void *, cudensitymatWrappedTensorCallback_t,
    cudensitymatWrappedTensorGradientCallback_t,
    cudensitymatElementaryOperator_t *elem) {
  *elem = new FakeCudmElementary{};
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elem) {
  delete elem;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatCreateOperatorTerm(const cudensitymatHandle_t, int32_t,
                               const int64_t[],
                               cudensitymatOperatorTerm_t *term) {
  *term = new FakeCudmTerm{};
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t
cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t term) {
  delete term;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProduct(
    const cudensitymatHandle_t, cudensitymatOperatorTerm_t term, int32_t,
    const cudensitymatElementaryOperator_t[], const int32_t[], const int32_t[],
    cuDoubleComplex coefficient, cudensitymatWrappedScalarCallback_t callback,
    cudensitymatWrappedScalarGradientCallback_t) {
  term->products.push_back({toComplex(coefficient), callback.callback});
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatOperatorAppendTerm(
    const cudensitymatHandle_t, cudensitymatOperator_t op,
    cudensitymatOperatorTerm_t term, int32_t, cuDoubleComplex coefficient,
    cudensitymatWrappedScalarCallback_t,
    cudensitymatWrappedScalarGradientCallback_t) {
  op->terms.emplace_back(term, toComplex(coefficient));
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatOperatorPrepareAction(
    const cudensitymatHandle_t, cudensitymatOperator_t,
    const cudensitymatState_t, const cudensitymatState_t,
    cudensitymatComputeType_t computeType, size_t,
    cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t) {
  if (computeType != CUDENSITYMAT_COMPUTE_64F)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  workspace->required = 256;
  workspace->attached = false;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatOperatorComputeAction(
    const cudensitymatHandle_t, cudensitymatOperator_t op, double time,
    int64_t batchSize, int32_t, const double *,
    const cudensitymatState_t stateIn, cudensitymatState_t stateOut,
    cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t) {
  if (!workspace->attached)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  if (static_cast<size_t>(batchSize > 1 ? batchSize : 1) != stateIn->batch ||
      stateIn->batch != stateOut->batch)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  std::vector<Complex> scalars(stateIn->batch);
  if (auto status = scalarsAt(op, time, scalars))
    return status;
  const size_t member = stateOut->size / stateOut->batch;
  for (size_t i = 0; i < stateOut->size; ++i)
    stateOut->data[i] += scalars[i / member] * stateIn->data[i];
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t fakeCudmLastOperatorScalar(double time,
                                                cuDoubleComplex *value) {
  if (!lastOperator)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  std::vector<Complex> scalars(1);
  auto status = scalarsAt(lastOperator, time, scalars);
  *value = make_cuDoubleComplex(scalars[0].real(), scalars[0].imag());
  return status;
}
