// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The per-run table of classical handlers of the cuDensityMat runtime, run
// against the host fake in fake_cudensitymat/: concurrent runs that register
// different handlers for one function id each reach their own, and a
// malformed request or answer, or an unknown id, fails.

#include "cudm_runtime.h"

#include "cudaq-pulse/Runtime/WireFormat.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using cudaq_pulse::wire::RequestHeader;
using cudaq_pulse::wire::ResponseHeader;

std::atomic<int> failures{0};

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

constexpr uint32_t kId = cudaq_pulse::wire::fnv1a32("estimate");

// Answer with one byte: the request's first argument byte plus `Offset`.
template <uint8_t Offset>
void addHandler(const void *request, void *answer, size_t) {
  RequestHeader in;
  std::memcpy(&in, request, sizeof in);
  uint8_t value = static_cast<const uint8_t *>(request)[sizeof in];
  ResponseHeader out{cudaq_pulse::wire::kResponseMagic, 0, 1, in.requestId,
                     in.ptpTimestamp};
  std::memcpy(answer, &out, sizeof out);
  static_cast<uint8_t *>(answer)[sizeof out] =
      static_cast<uint8_t>(value + Offset);
}

void badMagicHandler(const void *, void *answer, size_t) {
  ResponseHeader out{0, 0, 0, 0, 0};
  std::memcpy(answer, &out, sizeof out);
}

std::vector<uint8_t> request(uint8_t value, uint32_t id = kId,
                             uint32_t requestId = 7) {
  std::vector<uint8_t> bytes(sizeof(RequestHeader) + 1);
  RequestHeader header{cudaq_pulse::wire::kRequestMagic, id, 1, requestId, 0};
  std::memcpy(bytes.data(), &header, sizeof header);
  bytes[sizeof header] = value;
  return bytes;
}

int call(CudmHandle run, const std::vector<uint8_t> &bytes, uint8_t &result) {
  std::vector<uint8_t> answer(64);
  int64_t size = 0;
  CudmStatus status = cudm_classical_call(
      run, bytes.data(), static_cast<int64_t>(bytes.size()), answer.data(),
      static_cast<int64_t>(answer.size()), &size);
  if (status == CUDM_SUCCESS)
    result = answer[sizeof(ResponseHeader)];
  return status;
}

void eachRunReachesItsOwnHandler() {
  std::thread threads[2];
  uint8_t results[2] = {0, 0};
  for (int k = 0; k < 2; ++k)
    threads[k] = std::thread([k, &results] {
      CudmHandle run = nullptr;
      check(cudm_init(&run) == CUDM_SUCCESS, "cudm_init");
      check(cudm_classical_register(
                run, kId, k == 0 ? &addHandler<1> : &addHandler<100>) ==
                CUDM_SUCCESS,
            "register");
      for (int repeat = 0; repeat < 1000; ++repeat)
        check(call(run, request(5), results[k]) == CUDM_SUCCESS, "call");
      cudm_destroy(run);
    });
  for (auto &thread : threads)
    thread.join();
  check(results[0] == 6, "run 0 answered by its handler");
  check(results[1] == 105, "run 1 answered by its handler");
}

void malformedFramesFail() {
  CudmHandle run = nullptr;
  check(cudm_init(&run) == CUDM_SUCCESS, "cudm_init");
  uint8_t result = 0;
  check(call(run, request(1), result) != CUDM_SUCCESS,
        "an unregistered function id fails");
  check(std::string(cudm_last_error_message()).find("no classical handler") !=
            std::string::npos,
        "the error names the missing handler");
  cudm_classical_register(run, kId, &addHandler<0>);
  auto bad = request(1);
  bad[0] ^= 1;
  check(call(run, bad, result) != CUDM_SUCCESS, "a bad request magic fails");
  cudm_classical_register(run, kId, &badMagicHandler);
  check(call(run, request(1), result) != CUDM_SUCCESS,
        "a bad answer magic fails");
  check(cudm_classical_register(run, kId, nullptr) != CUDM_SUCCESS,
        "a null handler is rejected");
  check(cudm_classical_register(nullptr, kId, &addHandler<0>) != CUDM_SUCCESS,
        "a null run is rejected");
  cudm_destroy(run);
}

} // namespace

int main() {
  eachRunReachesItsOwnHandler();
  malformedFramesFail();
  if (failures) {
    std::fprintf(stderr, "%d classical-table checks failed\n", failures.load());
    return 1;
  }
  std::printf("classical handler table: all checks passed\n");
  return 0;
}
