file(GLOB BitwiseTests *.cpp and/*.cpp manipulations/*.cpp or/*.cpp xor/*.cpp shift/*.cpp)
target_sources(AesiMultiprecision PRIVATE ${BitwiseTests})