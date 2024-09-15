cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(AesiMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests} ../Aesi.h
        speed_comparison/addition.cpp
        speed_comparison/multiplication.cpp
        speed_comparison/subtraction.cpp
        speed_comparison/division.cpp
        speed_comparison/powm.cpp
        speed_comparison/gcd.cpp
)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(gpu)
add_subdirectory(boolean)
add_subdirectory(benchmarks)

option(CRYPTOPP_CMP "Enable CryptoPP comparison" OFF)
if (CRYPTOPP_CMP)
    add_subdirectory(cryptopp-cmake-master)
    set(CryptoppLibraryCall cryptopp)
endif()

include_directories(${SQLite3_INCLUDE_DIRS})
target_link_libraries(AesiMultiprecision GTest::gtest_main ${SQLite3_LIBRARIES} ${CryptoppLibraryCall})

include(GoogleTest)
gtest_discover_tests(AesiMultiprecision)


