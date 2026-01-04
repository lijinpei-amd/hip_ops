add_executable(inter_block_cumsum
        inter_block_cumsum.hip.cpp)
set_source_files_properties(inter_block_cumsum.hip.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(inter_block_cumsum
        hip_ops_utils
        absl::flags
        absl::flags_parse)
