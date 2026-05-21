file(GLOB ArithmeticTests *.cpp addition/*.cpp subtraction/*.cpp division/*.cpp modulo/*.cpp multiplication/*.cpp)
target_sources(AesiTests PRIVATE ${ArithmeticTests})