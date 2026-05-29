cmake_minimum_required(VERSION 3.26)
project(AesiCudaSmoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include(FetchContent)
FetchContent_Declare(AesiMultiprecision
        GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
        GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)

add_executable(AesiCudaSmoke smoke.cu)
target_link_libraries(AesiCudaSmoke PRIVATE AesiMultiprecision)
