#include "hip_ops/utils/error_handling.hpp"

#include "absl/log/log_streamer.h"
#include "hip/hip_runtime_api.h"

namespace hip_ops {
void check_hip_error_impl(hipError_t error, absl::LogSeverity log_severity,
                          std::string_view api_name, std::string_view file,
                          int line) {
  if (error == hipSuccess) {
    return;
  }
  const char* error_name = hipGetErrorName(error);
  if (!error_name) {
    error_name = "<unknown error>";
  }
  const char* error_desc = hipGetErrorString(error);
  if (!error_desc) {
    error_desc = "<unknown description>";
  }
  absl::LogStreamer(log_severity, file, line).stream()
      << "HIP API " << api_name << " failed with error_code " << (int)error
      << " : " << error_name << " : " << error_desc;
}
}  // namespace hip_ops
