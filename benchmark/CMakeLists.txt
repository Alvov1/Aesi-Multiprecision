cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(Benchmarking)

find_package(Benchmark REQUIRED)

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)

find_path(CRYPTOPP_INCLUDE_DIR cryptopp/cryptlib.h /usr/include /usr/local/include /opt/homebrew/include)
find_library(CRYPTOPP_LIBRARY cryptopp /usr/lib /usr/local/lib /opt/homebrew/lib)

file(GLOB Benches *.cpp)
add_executable(Benchmarking ${Benches}
        ../starter.cpp)

if(CRYPTOPP_INCLUDE_DIR AND CRYPTOPP_LIBRARY)
    include_directories(${CRYPTOPP_INCLUDE_DIR})
    target_link_libraries(Benchmarking benchmark::benchmark ${CRYPTOPP_LIBRARY} PkgConfig::GMP)
    target_include_directories(Benchmarking PUBLIC PkgConfig::GMP)
else()
    message(FATAL_ERROR "Crypto++ not found!")
endif()


