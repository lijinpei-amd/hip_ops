#include <gtest/gtest.h>

#include "hip/hip_runtime.h"
#include "hip_ops/utils/error_handling.hpp"

TEST(RuntimeErrorCheck, NoError) { CHECK_HIP_ERROR(hipInit, 0); }

namespace {
hipError_t fakeHipReturnError(int error) { return hipError_t(error); }
}  // namespace

TEST(RuntimeErrorCheck, HasError) {
  CHECK_HIP_ERROR(fakeHipReturnError, hipErrorInvalidValue);
}
