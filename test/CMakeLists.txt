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
add_executable(AesiMultiprecision ${Tests} ../Aesi.h
        arithmetic/addition/addition_unsigned.cpp
        arithmetic/modulo/modulo_unsigned.cpp
        arithmetic/multiplication/multiplication_unsigned.cpp
        arithmetic/subtraction/subtraction_unsigned.cpp
        bitwise/and/bitwise_and_unsigned.cpp
        bitwise/or/bitwise_or_unsigned.cpp
        bitwise/xor/bitwise_xor_unsigned.cpp
        bitwise/manipulations/bitwise_manipulations_unsigned.cpp
        boolean/boolean_unsigned.cpp
        boolean/boolean_device_unsigned.cpp
        number-theory/gcd/gcd_unsigned.cpp
        number-theory/lcm/gcd_unsigned.cpp
        number-theory/powm/gcd_unsigned.cpp
        operations/casts/integral_cast_unsigned.cpp
        operations/casts/precision_cast_unsigned.cpp
        operations/display/display_unsigned.cpp
        operations/initialization/basic_initialization_unsigned.cpp
        operations/initialization/initialization_different_notation_unsigned.cpp)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
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


