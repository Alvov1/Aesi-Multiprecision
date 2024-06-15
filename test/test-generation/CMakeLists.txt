cmake_minimum_required(VERSION 3.27)
project(CryptoPP)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        CryptoPPLib
        GIT_REPOSITORY https://github.com/abdes/cryptopp-cmake.git
        GIT_TAG        master
)
FetchContent_Populate(CryptoPPLib)

message(${cryptopplib_SOURCE_DIR})
add_subdirectory(${cryptopplib_SOURCE_DIR} ${cryptopplib_BINARY_DIR})

add_executable(CryptoPP main.cpp)
target_link_libraries(CryptoPP cryptopp)