cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(AesiMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(GMPXX REQUIRED IMPORTED_TARGET gmpxx)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests} ../Aeu.h ../Aesi.h)
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(boolean)

add_compile_definitions(AESI_GMP_INTEGRATION)

if (PREPROCESSOR_UNSAFE)
    add_definitions(-DAESI_UNSAFE=1)
endif (PREPROCESSOR_UNSAFE)

if (PREPROCESSOR_PRE_CPP_20)
    add_definitions(-DPRE_CPP_20=1)
endif (PREPROCESSOR_PRE_CPP_20)

target_link_libraries(AesiMultiprecision PUBLIC
        GTest::gtest_main
        PkgConfig::GMP
        PkgConfig::GMPXX)
target_include_directories(AesiMultiprecision PUBLIC
        PkgConfig::GMP
        PkgConfig::GMPXX)

include(GoogleTest)
gtest_discover_tests(AesiMultiprecision XML_OUTPUT_DIR ${CMAKE_BINARY_DIR})


