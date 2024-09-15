cmake_minimum_required(VERSION 3.28)
project(AesiSanitize)

set(CMAKE_CXX_STANDARD 20)

add_executable(AesiSanitize main.cpp)

target_compile_options(AesiSanitize PRIVATE -fsanitize=undefined
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize-coverage=trace-pc)
target_link_options(AesiSanitize PRIVATE -fsanitize=undefined
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fsanitize-coverage=trace-pc)
