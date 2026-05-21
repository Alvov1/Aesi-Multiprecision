cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(Build)

include(FetchContent)
FetchContent_Declare(AesiMultiprecision
        GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
        GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)

add_executable(Build main.cpp)
target_include_directories(Build PRIVATE ${AesiMultiprecision_SOURCE_DIR})