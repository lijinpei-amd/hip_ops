add_library(hip_ops_utils_test OBJECT
        test_error_handling.hip.cpp)
target_link_libraries(hip_ops_utils_test PUBLIC hip_ops_utils GTest::gtest_main)
