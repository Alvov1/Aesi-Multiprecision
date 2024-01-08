cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(AesiMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

find_package (SQLite3)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests} ../Aesi.h)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(gpu)
add_subdirectory(boolean)

include_directories(${SQLite3_INCLUDE_DIRS})
target_link_libraries(AesiMultiprecision GTest::gtest_main ${SQLite3_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(AesiMultiprecision)


