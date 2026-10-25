/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Host-only stand-in for the part of the cuDensityMat API that cudm_runtime.cpp
// calls, with the same signatures. fake_cudensitymat.cpp implements it with a
// scalar model of the operator action; see there.

#pragma once

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

typedef struct FakeCudmHandle *cudensitymatHandle_t;
typedef struct FakeCudmState *cudensitymatState_t;
typedef struct FakeCudmOperator *cudensitymatOperator_t;
typedef struct FakeCudmTerm *cudensitymatOperatorTerm_t;
typedef struct FakeCudmElementary *cudensitymatElementaryOperator_t;
typedef struct FakeCudmWorkspace *cudensitymatWorkspaceDescriptor_t;

typedef enum {
  CUDENSITYMAT_STATUS_SUCCESS = 0,
  CUDENSITYMAT_STATUS_INVALID_VALUE = 7,
} cudensitymatStatus_t;

typedef enum {
  CUDENSITYMAT_STATE_PURITY_PURE = 0,
  CUDENSITYMAT_STATE_PURITY_MIXED = 1,
} cudensitymatStatePurity_t;

typedef enum {
  CUDENSITYMAT_COMPUTE_64F = 1 << 4,
  CUDENSITYMAT_COMPUTE_32F = 1 << 2,
} cudensitymatComputeType_t;

typedef enum {
  CUDENSITYMAT_MEMSPACE_DEVICE = 0,
  CUDENSITYMAT_MEMSPACE_HOST = 1,
} cudensitymatMemspace_t;

typedef enum {
  CUDENSITYMAT_WORKSPACE_SCRATCH = 0,
} cudensitymatWorkspaceKind_t;

typedef enum {
  CUDENSITYMAT_OPERATOR_SPARSITY_NONE = 0,
  CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL = 1,
} cudensitymatElementaryOperatorSparsity_t;

typedef enum {
  CUDENSITYMAT_CALLBACK_DEVICE_CPU = 0,
  CUDENSITYMAT_CALLBACK_DEVICE_GPU = 1,
} cudensitymatCallbackDevice_t;

typedef int32_t (*cudensitymatScalarCallback_t)(
    double time, int64_t batchSize, int32_t numParams, const double *params,
    cudaDataType_t dataType, void *scalarStorage, cudaStream_t stream);

typedef struct {
  cudensitymatScalarCallback_t callback;
  cudensitymatCallbackDevice_t device;
  void *wrapper;
} cudensitymatWrappedScalarCallback_t;

typedef struct {
  void *callback;
  cudensitymatCallbackDevice_t device;
  void *wrapper;
} cudensitymatWrappedScalarGradientCallback_t;

typedef struct {
  void *callback;
  cudensitymatCallbackDevice_t device;
  void *wrapper;
} cudensitymatWrappedTensorCallback_t;

typedef struct {
  void *callback;
  cudensitymatCallbackDevice_t device;
  void *wrapper;
} cudensitymatWrappedTensorGradientCallback_t;

const cudensitymatWrappedScalarCallback_t cudensitymatScalarCallbackNone = {
    nullptr, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr};
const cudensitymatWrappedScalarGradientCallback_t
    cudensitymatScalarGradientCallbackNone = {
        nullptr, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr};
const cudensitymatWrappedTensorCallback_t cudensitymatTensorCallbackNone = {
    nullptr, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr};
const cudensitymatWrappedTensorGradientCallback_t
    cudensitymatTensorGradientCallbackNone = {
        nullptr, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr};

size_t cudensitymatGetVersion();

cudensitymatStatus_t cudensitymatCreate(cudensitymatHandle_t *handle);
cudensitymatStatus_t cudensitymatDestroy(cudensitymatHandle_t handle);

cudensitymatStatus_t
cudensitymatCreateState(const cudensitymatHandle_t handle,
                        cudensitymatStatePurity_t purity, int32_t numSpaceModes,
                        const int64_t spaceModeExtents[], int64_t batchSize,
                        cudaDataType_t dataType, cudensitymatState_t *state);
cudensitymatStatus_t cudensitymatDestroyState(cudensitymatState_t state);
cudensitymatStatus_t
cudensitymatStateGetNumComponents(const cudensitymatHandle_t handle,
                                  const cudensitymatState_t state,
                                  int32_t *numStateComponents);
cudensitymatStatus_t cudensitymatStateGetComponentStorageSize(
    const cudensitymatHandle_t handle, const cudensitymatState_t state,
    int32_t numStateComponents, size_t componentBufferSize[]);
cudensitymatStatus_t cudensitymatStateAttachComponentStorage(
    const cudensitymatHandle_t handle, cudensitymatState_t state,
    int32_t numStateComponents, void *componentBuffer[],
    const size_t componentBufferSize[]);
cudensitymatStatus_t
cudensitymatStateInitializeZero(const cudensitymatHandle_t handle,
                                cudensitymatState_t state, cudaStream_t stream);
cudensitymatStatus_t cudensitymatStateComputeAccumulation(
    const cudensitymatHandle_t handle, const cudensitymatState_t stateIn,
    cudensitymatState_t stateOut, const void *scalingFactors,
    cudaStream_t stream);

cudensitymatStatus_t
cudensitymatCreateWorkspace(const cudensitymatHandle_t handle,
                            cudensitymatWorkspaceDescriptor_t *workspaceDescr);
cudensitymatStatus_t
cudensitymatDestroyWorkspace(cudensitymatWorkspaceDescriptor_t workspaceDescr);
cudensitymatStatus_t cudensitymatWorkspaceGetMemorySize(
    const cudensitymatHandle_t handle,
    const cudensitymatWorkspaceDescriptor_t workspaceDescr,
    cudensitymatMemspace_t memSpace, cudensitymatWorkspaceKind_t workspaceKind,
    size_t *memoryBufferSize);
cudensitymatStatus_t
cudensitymatWorkspaceSetMemory(const cudensitymatHandle_t handle,
                               cudensitymatWorkspaceDescriptor_t workspaceDescr,
                               cudensitymatMemspace_t memSpace,
                               cudensitymatWorkspaceKind_t workspaceKind,
                               void *memoryBuffer, size_t memoryBufferSize);

cudensitymatStatus_t cudensitymatCreateOperator(
    const cudensitymatHandle_t handle, int32_t numSpaceModes,
    const int64_t spaceModeExtents[], cudensitymatOperator_t *superoperator);
cudensitymatStatus_t
cudensitymatDestroyOperator(cudensitymatOperator_t superoperator);
cudensitymatStatus_t cudensitymatCreateElementaryOperator(
    const cudensitymatHandle_t handle, int32_t numSpaceModes,
    const int64_t spaceModeExtents[],
    cudensitymatElementaryOperatorSparsity_t sparsity, int32_t numDiagonals,
    const int32_t diagonalOffsets[], cudaDataType_t dataType, void *tensorData,
    cudensitymatWrappedTensorCallback_t tensorCallback,
    cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback,
    cudensitymatElementaryOperator_t *elemOperator);
cudensitymatStatus_t
cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elem);
cudensitymatStatus_t cudensitymatCreateOperatorTerm(
    const cudensitymatHandle_t handle, int32_t numSpaceModes,
    const int64_t spaceModeExtents[], cudensitymatOperatorTerm_t *operatorTerm);
cudensitymatStatus_t
cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm);
cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProduct(
    const cudensitymatHandle_t handle, cudensitymatOperatorTerm_t operatorTerm,
    int32_t numElemOperators,
    const cudensitymatElementaryOperator_t elemOperators[],
    const int32_t stateModesActedOn[], const int32_t modeActionDuality[],
    cuDoubleComplex coefficient,
    cudensitymatWrappedScalarCallback_t coefficientCallback,
    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);
cudensitymatStatus_t cudensitymatOperatorAppendTerm(
    const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator,
    cudensitymatOperatorTerm_t operatorTerm, int32_t duality,
    cuDoubleComplex coefficient,
    cudensitymatWrappedScalarCallback_t coefficientCallback,
    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);
cudensitymatStatus_t cudensitymatOperatorPrepareAction(
    const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator,
    const cudensitymatState_t stateIn, const cudensitymatState_t stateOut,
    cudensitymatComputeType_t computeType, size_t workspaceSizeLimit,
    cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream);
cudensitymatStatus_t cudensitymatOperatorComputeAction(
    const cudensitymatHandle_t handle, cudensitymatOperator_t superoperator,
    double time, int64_t batchSize, int32_t numParams, const double *params,
    const cudensitymatState_t stateIn, cudensitymatState_t stateOut,
    cudensitymatWorkspaceDescriptor_t workspace, cudaStream_t stream);

// Test hook, not part of cuDensityMat: the scalar by which the action of the
// operator this thread created last multiplies the state at `time`.
cudensitymatStatus_t fakeCudmLastOperatorScalar(double time,
                                                cuDoubleComplex *value);
