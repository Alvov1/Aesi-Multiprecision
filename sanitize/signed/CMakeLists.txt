if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.28)
    project(AesiSignedSanitize)
    set(CMAKE_CXX_STANDARD 20)
endif()

option(ENABLE_SANITIZERS "Enable sanitizers" ON)

include(FetchContent)
FetchContent_Declare(primesieve
        GIT_REPOSITORY https://github.com/kimwalisch/primesieve.git
        GIT_TAG v12.14
        GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(primesieve)

add_executable(AesiSignedSanitize main.cpp)
target_include_directories(AesiSignedSanitize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(AesiSignedSanitize PRIVATE primesieve::primesieve)

if(ENABLE_SANITIZERS)
    target_compile_options(AesiSignedSanitize PRIVATE -fsanitize=undefined
            -fsanitize=address
            -fsanitize-address-use-after-scope)
    target_link_options(AesiSignedSanitize PRIVATE -fsanitize=undefined
            -fsanitize=address
            -fsanitize-address-use-after-scope)
else()
    target_compile_options(AesiSignedSanitize PRIVATE)
    target_link_options(AesiSignedSanitize PRIVATE)
endif()
