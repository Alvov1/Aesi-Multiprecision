cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(SpeedComparison)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

find_path(CRYPTOPP_INCLUDE_DIR cryptopp/cryptlib.h /usr/include /usr/local/include)
find_library(CRYPTOPP_LIBRARY cryptopp /usr/lib /usr/local/lib)

file(GLOB Tests *.cpp)
add_executable(SpeedComparison ${Tests})

if(CRYPTOPP_INCLUDE_DIR AND CRYPTOPP_LIBRARY)
    include_directories(${CRYPTOPP_INCLUDE_DIR})
    target_link_libraries(SpeedComparison GTest::gtest_main ${CRYPTOPP_LIBRARY} PkgConfig::GMP)
    target_include_directories(SpeedComparison PUBLIC PkgConfig::GMP)
else()
    message(FATAL_ERROR "Crypto++ not found!")
endif()

include(GoogleTest)
gtest_discover_tests(SpeedComparison)


