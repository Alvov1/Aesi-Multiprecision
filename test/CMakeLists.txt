cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(MultiprecisionGPU)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/8a6feabf04bec8fb125e0df0ad1195c42350725f.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(MultiprecisionGPU ${Tests} ../AesiMultiprecision.h
        bitwise/bitwise_and.cpp
        bitwise/bitwise_or.cpp
        bitwise/bitwise_xor.cpp)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(gpu)

target_link_libraries(MultiprecisionGPU GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MultiprecisionGPU)


