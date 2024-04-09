file(GLOB BitwiseTests *.cpp and/*.cpp manipulations/*.cpp or/*.cpp xor/*.cpp)
target_sources(AesiMultiprecision PRIVATE ${BitwiseTests})