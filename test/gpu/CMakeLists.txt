file(GLOB GpuTests *.cpp)
target_sources(AesiMultiprecision PRIVATE ${GpuTests})