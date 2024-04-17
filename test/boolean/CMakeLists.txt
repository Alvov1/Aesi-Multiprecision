file(GLOB BooleanSources *.cpp preCpp20/*.cpp spaceship/*.cpp)
target_sources(AesiMultiprecision PRIVATE ${BooleanSources})