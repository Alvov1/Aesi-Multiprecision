cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(MetalMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/8a6feabf04bec8fb125e0df0ad1195c42350725f.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(MultiprecisionTesting ${Tests} ../Multiprecision.h ../Multiprecision.cpp
        extended_arithmetic/bitshift.cpp
        extended_arithmetic/boolean.cpp
        extended_arithmetic/bitvise.cpp
        extended_arithmetic/comparison.cpp
        assignment.cpp)
add_subdirectory(basic_arithmetic)
add_subdirectory(extended_arithmetic)

target_link_libraries(MultiprecisionTesting GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MultiprecisionTesting)


