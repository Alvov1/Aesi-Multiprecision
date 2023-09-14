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

file(GLOB Tests *.cpp)
add_executable(Multiprecision ${Tests} ../Multiprecision.h ../Multiprecision.cpp)
add_subdirectory(arithmetic)
add_subdirectory(operations)

target_link_libraries(Multiprecision GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(Multiprecision)


