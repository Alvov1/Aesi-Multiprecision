cmake_minimum_required(VERSION 3.26)
project(AesiMultiprecision)
set(CMAKE_CXX_STANDARD 20)
add_library(AesiMultiprecision STATIC AesiMultiprecision.h)
set_target_properties(AesiMultiprecision PROPERTIES LINKER_LANGUAGE CXX)

