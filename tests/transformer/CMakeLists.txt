find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_FOUND)
    message(STATUS "Python3 not found — skipping test_transformer")
    return()
endif()

# Check numpy is available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE _numpy_ok
    OUTPUT_QUIET ERROR_QUIET
)
if(NOT _numpy_ok EQUAL 0)
    message(STATUS "numpy not found — skipping test_transformer")
    return()
endif()

set(_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
    OUTPUT  ${_DATA_DIR}/weights.bin ${_DATA_DIR}/expected.bin
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_weights.py
    WORKING_DIRECTORY ${_DATA_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_weights.py
    COMMENT "Generating transformer reference data (gen_weights.py)"
)

add_custom_target(transformer_test_data
    DEPENDS ${_DATA_DIR}/weights.bin ${_DATA_DIR}/expected.bin
)

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer PRIVATE transformers_lite GTest::gtest_main)
target_compile_definitions(test_transformer PRIVATE TEST_DATA_DIR="${_DATA_DIR}")
add_dependencies(test_transformer transformer_test_data)
add_test(NAME test_transformer COMMAND test_transformer)
