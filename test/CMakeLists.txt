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

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests})
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(boolean)

add_compile_definitions(CRYPTOPP_USE_STD_SHOWBASE)
add_compile_definitions(AESI_CRYPTOPP_INTEGRATION)
add_compile_definitions(AESI_GMP_INTEGRATION)
add_subdirectory(${cryptopplib_SOURCE_DIR} ${cryptopplib_BINARY_DIR})

if (PREPROCESSOR_UNSAFE)
    add_definitions(-DAESI_UNSAFE=1)
endif (PREPROCESSOR_UNSAFE)

if (PREPROCESSOR_PRE_CPP_20)
    add_definitions(-DPRE_CPP_20=1)
endif (PREPROCESSOR_PRE_CPP_20)

target_link_libraries(AesiMultiprecision PUBLIC GTest::gtest_main cryptopp PkgConfig::GMP)
target_include_directories(AesiMultiprecision PUBLIC PkgConfig::GMP)

include(GoogleTest)
gtest_discover_tests(AesiMultiprecision XML_OUTPUT_DIR ${CMAKE_BINARY_DIR})


