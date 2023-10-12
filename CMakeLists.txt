cmake_minimum_required(VERSION 3.24)
project(DivisorsFinder)

set(CMAKE_CXX_STANDARD 20)

file(GLOB Sources *.cpp)
add_executable(DivisorsFinder ${Sources})