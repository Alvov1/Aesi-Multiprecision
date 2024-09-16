cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(SpeedComparison)

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
add_subdirectory(${cryptopplib_SOURCE_DIR})

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(SpeedComparison ${Tests})

target_link_libraries(SpeedComparison GTest::gtest_main cryptopp PkgConfig::GMP)
target_include_directories(SpeedComparison PUBLIC PkgConfig::GMP)

include(GoogleTest)
gtest_discover_tests(SpeedComparison)


