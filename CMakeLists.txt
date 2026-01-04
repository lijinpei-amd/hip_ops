cmake_minimum_required(VERSION 3.28)
project(hip_ops LANGUAGES HIP CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
list(APPEND CMAKE_HIP_SOURCE_FILE_EXTENSIONS ".hip.cpp")
find_package(hip REQUIRED CONFIG)

add_subdirectory(3rd_party)

include_directories(include)
add_subdirectory(lib)

if (BUILD_TESTING)
        add_subdirectory(tests)
endif()
