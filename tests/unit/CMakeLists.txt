add_subdirectory(utils)

add_executable(
  unit_test
  test_main.cpp
)
target_link_libraries(
  unit_test
  PRIVATE
  hip_ops_utils_test
  GTest::gtest
  absl::log_initialize
)
