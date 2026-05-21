if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.26)
    project(Benchmarking)
    set(CMAKE_CXX_STANDARD 20)
endif()

# 1. Prepare GMP library
find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

# 2. Prepare Google Benchmarking
find_package(benchmark REQUIRED)

# 3. Prepare Crypto++ library
find_path(CRYPTOPP_INCLUDE cryptopp/cryptlib.h /usr/include /usr/local/include /opt/homebrew/include)
find_library(CRYPTOPP_LIB cryptopp /usr/lib /usr/local/lib /opt/homebrew/lib)
if(NOT CRYPTOPP_INCLUDE OR NOT CRYPTOPP_LIB)
    message(FATAL_ERROR "Crypto++ is not found!")
endif()

file(GLOB Benches *.cpp)
add_executable(Benchmarking ${Benches})

target_link_libraries(Benchmarking PRIVATE
        PkgConfig::GMP
        benchmark::benchmark
        ${CRYPTOPP_LIB})
target_include_directories(Benchmarking PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        PkgConfig::GMP
        benchmark::benchmark
        ${CRYPTOPP_INCLUDE})