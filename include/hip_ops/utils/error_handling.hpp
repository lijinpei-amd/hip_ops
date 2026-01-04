#pragma once

#include <string_view>

#include "absl/base/log_severity.h"
#include "hip/hip_runtime.h"

#define CHECK_HIP_ERROR(_API_FUNC, ...)                                    \
  ::hip_ops::check_hip_error_impl(_API_FUNC(__VA_ARGS__),                  \
                                  absl::LogSeverity::kWarning, #_API_FUNC, \
                                  __FILE__, __LINE__)

namespace hip_ops {
void check_hip_error_impl(hipError_t error, absl::LogSeverity log_severity,
                          std::string_view api_name, std::string_view file,
                          int line);
}
