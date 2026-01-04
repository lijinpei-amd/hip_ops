add_library(hip_ops_utils
        STATIC
        error_handling.cpp
        )
target_link_libraries(hip_ops_utils
        PUBLIC
        hip::host
        absl::log)
