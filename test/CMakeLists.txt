cmake_minimum_required(VERSION 3.26)
project(AesiMultiprecisionTesting)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION GREATER_EQUAL 12)
        message("-- The CUDA compiler was detected")
        set(CMAKE_CUDA_STANDARD 20)
        execute_process(COMMAND mv main.cpp main.cu
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    else()
        message("-- The CUDA compiler was detected, but the version ${CUDAToolkit_VERSION} is too outdated")
        set(CMAKE_CXX_STANDARD 20)
    endif(CUDAToolkit_VERSION GREATER_EQUAL 12)
else()
    message("-- The CUDA compiler was NOT found in the system")
    set(CMAKE_CXX_STANDARD 20)
endif()

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/8a6feabf04bec8fb125e0df0ad1195c42350725f.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecisionTesting ${Tests} ../AesiMultiprecision.h)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)

target_link_libraries(AesiMultiprecisionTesting GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(AesiMultiprecisionTesting)


