cmake_minimum_required(VERSION 3.28)
project(AesiSanitize)

set(CMAKE_CXX_STANDARD 20)

add_executable(AesiSanitize main.cpp)

target_compile_options(AesiSanitize PRIVATE -fsanitize=undefined
        -fsanitize=unsigned-integer-overflow)
target_link_options(AesiSanitize PRIVATE -fsanitize=undefined
        -fsanitize=unsigned-integer-overflow)
