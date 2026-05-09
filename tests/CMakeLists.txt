include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE transformers_lite GTest::gtest_main)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE transformers_lite GTest::gtest_main)
add_test(NAME test_memory COMMAND test_memory)

add_subdirectory(ops)
add_subdirectory(layers)
