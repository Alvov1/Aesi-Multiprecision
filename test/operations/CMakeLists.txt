file(GLOB OperationsTesting *.cpp casts/*.cpp display/*.cpp initialization/*.cpp)
target_sources(AesiMultiprecision PRIVATE ${OperationsTesting})