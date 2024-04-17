file(GLOB NumberTheorySources gcd/*.cpp lcm/*.cpp *.cpp)
target_sources(AesiMultiprecision PRIVATE ${NumberTheorySources})