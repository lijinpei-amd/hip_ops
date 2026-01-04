#include <iostream>
#include <random>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "hip/hip_runtime.h"
#include "hip_ops/utils/error_handling.hpp"

constexpr unsigned FLAG0 = unsigned(1) << 30;
constexpr unsigned FLAG1 = unsigned(1) << 31;

template <bool Atomic>
__global__ void inter_block_cumsum(int* cumsum, const int* local_val,
                                   int num_val) {
  int pid = blockIdx.x;
  if (pid >= num_val) {
    return;
  }
  int val = local_val[pid];
  if constexpr (Atomic) {
    __hip_atomic_store(cumsum + pid, val | FLAG0, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  } else {
    cumsum[pid] = val | FLAG0;
  }
  for (int target_pid = pid - 1; target_pid >= 0; --target_pid) {
    int other_val;
    while (true) {
      other_val = __hip_atomic_load(cumsum + target_pid, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
      if (other_val) {
        break;
      }
    }
    if (other_val & FLAG0) {
      val += other_val & (~FLAG0);
    } else {
      val += other_val & (~FLAG1);
      break;
    }
  }
  if constexpr (Atomic) {
    __hip_atomic_store(cumsum + pid, val | FLAG1, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  } else {
    cumsum[pid] = val | FLAG1;
  }
}

ABSL_FLAG(int, num_val, 10, "Number of values");
ABSL_FLAG(int, max_val, 1000, "Maximum number of input value");
ABSL_FLAG(bool, use_atomic, true, "Use atomic load/store in kernel");

std::vector<int> prepare_inputs(int num_val, int max_val) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(0, max_val);
  std::vector<int> results;
  results.reserve(num_val);
  for (int i = 0; i < num_val; ++i) {
    results.push_back(dist(rng));
  }
  return results;
}

int main(int argc, char* argv[]) {
  auto remainFlags = absl::ParseCommandLine(argc, argv);
  if (int remSize = remainFlags.size(); remSize != 1) {
    std::cout << "unknown flags: ";
    for (int i = 1; i < remSize; ++i) {
      std::cout << remainFlags[i] << ' ';
    }
    std::cout << std::endl;
    return 2;
  }
  int num_val = absl::GetFlag(FLAGS_num_val);
  int max_val = absl::GetFlag(FLAGS_max_val);
  bool use_atomic = absl::GetFlag(FLAGS_use_atomic);
  if (use_atomic) {
    std::cout << "using atomic kernel\n";
  } else {
    std::cout << "using non-atomic kernel\n";
  }

  auto host_input = prepare_inputs(num_val, max_val);
  std::vector<int> host_ref;
  host_ref.reserve(num_val);
  int cumsum = 0;
  for (auto val : host_input) {
    cumsum += val;
    host_ref.push_back(cumsum | FLAG1);
  }
  int* device_input;
  int* device_output;
  CHECK_HIP_ERROR(hipMalloc, (void**)&device_input, sizeof(int) * num_val);
  CHECK_HIP_ERROR(hipMalloc, (void**)&device_output, sizeof(int) * num_val);
  CHECK_HIP_ERROR(hipMemset, device_output, 0, sizeof(int) * num_val);
  CHECK_HIP_ERROR(hipMemcpy, device_input, host_input.data(),
                  sizeof(int) * num_val, hipMemcpyHostToDevice);
  hipStream_t stream;
  CHECK_HIP_ERROR(hipStreamCreate, &stream);
  void* kernel = use_atomic ? (void*)&inter_block_cumsum<true>
                            : (void*)&inter_block_cumsum<false>;
  void* args[] = {(void*)&device_output, (void*)&device_input, (void*)&num_val};
  CHECK_HIP_ERROR(hipLaunchKernel, kernel, dim3(num_val), dim3(1), args, 0,
                  stream);
  CHECK_HIP_ERROR(hipStreamSynchronize, stream);
  int* res_host = (int*)malloc(sizeof(int) * num_val);
  CHECK_HIP_ERROR(hipMemcpy, res_host, device_output, sizeof(int) * num_val,
                  hipMemcpyDeviceToHost);
  CHECK_HIP_ERROR(hipFree, device_input);
  CHECK_HIP_ERROR(hipFree, device_output);
  bool pass = true;
  for (int i = 0; i < num_val; ++i) {
    if (res_host[i] != host_ref[i]) {
      std::cout << "failed at index: " << i << "\n";
      std::cout << "host val: " << host_ref[i] << " device val: " << res_host[i]
                << "\n";
      pass = false;
      break;
    }
  }
  free(res_host);
  std::cout << (pass ? "pass" : "fail") << std::endl;
  return pass ? 0 : 1;
}
