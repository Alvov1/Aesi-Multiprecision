cmake_minimum_required(VERSION 3.28)
project(AesiSanitize)

option(ENABLE_SANITIZERS "Enable sanitizers" ON)

set(CMAKE_CXX_STANDARD 20)


add_executable(AesiSanitize main.cpp)

if(ENABLE_SANITIZERS)
    target_compile_options(AesiSanitize PRIVATE -fsanitize=undefined
            -fsanitize=address
            -fsanitize-address-use-after-scope)
    target_link_options(AesiSanitize PRIVATE -fsanitize=undefined
            -fsanitize=address
            -fsanitize-address-use-after-scope)
else()
    target_compile_options(AesiSanitize PRIVATE)
    target_link_options(AesiSanitize PRIVATE)
endif()
