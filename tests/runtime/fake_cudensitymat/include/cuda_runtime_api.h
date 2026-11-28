// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host-only stand-in for the CUDA runtime calls of the cuDensityMat runtime:
// "device" memory is host memory.

#pragma once

#include <cstddef>
#include <cstdint>

typedef struct FakeCudaStream *cudaStream_t;

typedef enum cudaError_enum {
  cudaSuccess = 0,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNoDevice = 100,
  cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind_enum {
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
} cudaMemcpyKind;

typedef enum cudaDataType_enum {
  CUDA_C_32F = 4,
  CUDA_C_64F = 5,
} cudaDataType_t;

cudaError_t cudaMalloc(void **pointer, size_t size);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemcpy(void *destination, const void *source, size_t size,
                       cudaMemcpyKind kind);
const char *cudaGetErrorString(cudaError_t error);

// Test hooks, not part of CUDA, each for the calling thread: the
// host-to-device copies made so far; the allocations not freed yet; and
// making one host-to-device copy fail, the one after `after` more succeed.
int64_t fakeCudaHostToDeviceCopies();
int64_t fakeCudaLiveAllocations();
void fakeCudaFailHostToDeviceCopy(int64_t after);
