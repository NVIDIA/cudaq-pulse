// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A host implementation of the cuDensityMat calls cudm_runtime.cpp makes. The
// operator action is a scalar model: L(t) rho = s(t) rho, where s(t) sums,
// over every appended term and product, the coefficients times the value of
// the product's scalar callback at t. That keeps the runtime's integrators,
// state handling and callback table real while needing no GPU.

#include <cudensitymat.h>

#include <algorithm>
#include <complex>
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
struct FakeCudmWorkspace {};
struct FakeCudmElementary {};

struct FakeCudmState {
  size_t size = 0;
  Complex *data = nullptr;
};

struct FakeCudmTerm {
  std::vector<Product> products;
};

struct FakeCudmOperator {
  std::vector<std::pair<FakeCudmTerm *, Complex>> terms;
};

namespace {

cudensitymatStatus_t scalarAt(const FakeCudmOperator *op, double time,
                              Complex &value) {
  value = 0.0;
  for (const auto &[term, termCoefficient] : op->terms) {
    for (const Product &product : term->products) {
      Complex factor = 1.0;
      if (product.callback) {
        cuDoubleComplex sample{};
        if (product.callback(time, 1, 0, nullptr, CUDA_C_64F, &sample,
                             nullptr) != 0)
          return CUDENSITYMAT_STATUS_INVALID_VALUE;
        factor = Complex(sample.x, sample.y);
      }
      value += termCoefficient * product.coefficient * factor;
    }
  }
  return CUDENSITYMAT_STATUS_SUCCESS;
}

Complex toComplex(cuDoubleComplex value) { return {value.x, value.y}; }

thread_local FakeCudmOperator *lastOperator = nullptr;

} // namespace

cudaError_t cudaMalloc(void **pointer, size_t size) {
  *pointer = std::calloc(1, size == 0 ? 1 : size);
  return *pointer ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFree(void *pointer) {
  std::free(pointer);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *destination, const void *source, size_t size,
                       cudaMemcpyKind) {
  std::memcpy(destination, source, size);
  return cudaSuccess;
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

cudensitymatStatus_t cudensitymatCreateState(const cudensitymatHandle_t,
                                             cudensitymatStatePurity_t purity,
                                             int32_t numSpaceModes,
                                             const int64_t spaceModeExtents[],
                                             int64_t, cudaDataType_t dataType,
                                             cudensitymatState_t *state) {
  if (dataType != CUDA_C_64F)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  size_t size = 1;
  for (int32_t i = 0; i < numSpaceModes; ++i)
    size *= static_cast<size_t>(spaceModeExtents[i]);
  if (purity == CUDENSITYMAT_STATE_PURITY_MIXED)
    size *= size;
  *state = new FakeCudmState{size, nullptr};
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
  const Complex factor =
      toComplex(*static_cast<const cuDoubleComplex *>(scalingFactors));
  for (size_t i = 0; i < stateOut->size; ++i)
    stateOut->data[i] += factor * stateIn->data[i];
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
    const cudensitymatHandle_t, const cudensitymatWorkspaceDescriptor_t,
    cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, size_t *size) {
  *size = 0;
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t cudensitymatWorkspaceSetMemory(
    const cudensitymatHandle_t, cudensitymatWorkspaceDescriptor_t,
    cudensitymatMemspace_t, cudensitymatWorkspaceKind_t, void *, size_t) {
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
    cudensitymatWorkspaceDescriptor_t, cudaStream_t) {
  return computeType == CUDENSITYMAT_COMPUTE_64F
             ? CUDENSITYMAT_STATUS_SUCCESS
             : CUDENSITYMAT_STATUS_INVALID_VALUE;
}

cudensitymatStatus_t cudensitymatOperatorComputeAction(
    const cudensitymatHandle_t, cudensitymatOperator_t op, double time, int64_t,
    int32_t, const double *, const cudensitymatState_t stateIn,
    cudensitymatState_t stateOut, cudensitymatWorkspaceDescriptor_t,
    cudaStream_t) {
  Complex scalar;
  if (auto status = scalarAt(op, time, scalar))
    return status;
  for (size_t i = 0; i < stateOut->size; ++i)
    stateOut->data[i] += scalar * stateIn->data[i];
  return CUDENSITYMAT_STATUS_SUCCESS;
}

cudensitymatStatus_t fakeCudmLastOperatorScalar(double time,
                                                cuDoubleComplex *value) {
  if (!lastOperator)
    return CUDENSITYMAT_STATUS_INVALID_VALUE;
  Complex scalar;
  auto status = scalarAt(lastOperator, time, scalar);
  *value = make_cuDoubleComplex(scalar.real(), scalar.imag());
  return status;
}
