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
        arithmetic/addition/unsigned.cpp
        arithmetic/modulo/unsigned.cpp
        arithmetic/multiplication/unsigned.cpp
        arithmetic/subtraction/unsigned.cpp
        bitwise/and/unsigned.cpp
        bitwise/or/unsigned.cpp
        bitwise/xor/unsigned.cpp
        bitwise/manipulations/unsigned.cpp
        boolean/spaceship/unsigned.cpp
        boolean/preCpp20/unsigned.cpp
        number-theory/gcd/unsigned.cpp
        number-theory/lcm/unsigned.cpp
        operations/display/unsigned.cpp
        operations/initialization/unsigned.cpp
)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(boolean)

option(CRYPTOPP_CMP "Enable CryptoPP comparison" OFF)
if (CRYPTOPP_CMP)
    add_subdirectory(cryptopp-cmake-master)
    set(CryptoppLibraryCall cryptopp)
endif()

include_directories(${SQLite3_INCLUDE_DIRS})

target_link_libraries(AesiMultiprecision GTest::gtest_main ${SQLite3_LIBRARIES} ${CryptoppLibraryCall})


include(GoogleTest)
gtest_discover_tests(AesiMultiprecision)


