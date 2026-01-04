add_subdirectory(abseil-cpp EXCLUDE_FROM_ALL)
if (BUILD_TESTING)
        add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()
