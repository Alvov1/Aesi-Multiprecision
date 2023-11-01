cmake_minimum_required(VERSION 3.26)
project(AesiMultiprecision CXX)
set(CMAKE_CXX_STANDARD 20)
set_target_properties(AesiMultiprecision PROPERTIES LINKER_LANGUAGE CXX)
add_library(AesiMultiprecision STATIC AesiMultiprecision.h)


