cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(Benchmarking)

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

find_path(BENCH_INCLUDE benchmark/benchmark.h /usr/include /usr/local/include /opt/homebrew/include)
find_library(BENCH_LIB benchmark /usr/lib /usr/local/lib /opt/homebrew/lib)
if(NOT BENCH_INCLUDE OR NOT BENCH_LIB)
    message(FATAL_ERROR "Google Benchmarking is not found!")
endif()

find_path(CRYPTOPP_INCLUDE cryptopp/cryptlib.h /usr/include /usr/local/include /opt/homebrew/include)
find_library(CRYPTOPP_LIB cryptopp /usr/lib /usr/local/lib /opt/homebrew/lib)
if(NOT CRYPTOPP_INCLUDE OR NOT CRYPTOPP_LIB)
    message(FATAL_ERROR "Crypto++ is not found!")
endif()

file(GLOB Benches *.cpp)
add_executable(Benchmarking ${Benches})

include_directories(${CRYPTOPP_INCLUDE} ${BENCH_INCLUDE})
target_link_libraries(Benchmarking ${BENCH_LIB} ${CRYPTOPP_LIB} PkgConfig::GMP)
target_include_directories(Benchmarking PUBLIC PkgConfig::GMP)