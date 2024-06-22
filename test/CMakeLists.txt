cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(AesiMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)
FetchContent_Declare(
        CryptoPPLib
        GIT_REPOSITORY https://github.com/abdes/cryptopp-cmake.git
        GIT_TAG        master
)
FetchContent_Populate(CryptoPPLib)

find_package (SQLite3)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests} ../Aesi.h
        bitwise/shift/signed.cpp)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(boolean)

add_compile_definitions(CRYPTOPP_USE_STD_SHOWBASE)
add_compile_definitions(AESI_CRYPTOPP_INTEGRATION)
add_subdirectory(${cryptopplib_SOURCE_DIR} ${cryptopplib_BINARY_DIR})

if (PREPROCESSOR_UNSAFE)
    add_definitions(-DAESI_UNSAFE=1)
endif (PREPROCESSOR_UNSAFE)

if (PREPROCESSOR_PRE_CPP_20)
    add_definitions(-DPRE_CPP_20=1)
endif (PREPROCESSOR_PRE_CPP_20)

include_directories(${SQLite3_INCLUDE_DIRS})

target_link_libraries(AesiMultiprecision GTest::gtest_main ${SQLite3_LIBRARIES} cryptopp)

include(GoogleTest)
gtest_discover_tests(AesiMultiprecision)


