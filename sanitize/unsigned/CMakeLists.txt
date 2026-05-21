cmake_minimum_required(VERSION 3.28)
project(AesiSanitize)

option(ENABLE_SANITIZERS "Enable sanitizers" ON)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(primesieve
        GIT_REPOSITORY https://github.com/kimwalisch/primesieve.git
        GIT_TAG master
        GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(primesieve)

add_executable(AesiSanitize main.cpp)
target_include_directories(AesiSanitize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(AesiSanitize PRIVATE primesieve::primesieve)

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
