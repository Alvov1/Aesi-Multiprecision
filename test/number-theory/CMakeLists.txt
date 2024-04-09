file(GLOB NumberTheorySources gcd/*.cpp lcm/*.cpp powm/*.cpp)
target_sources(AesiMultiprecision PRIVATE ${NumberTheorySources})