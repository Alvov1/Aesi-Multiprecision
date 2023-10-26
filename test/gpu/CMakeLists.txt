file(GLOB GpuTests *.cpp)
target_sources(AesiMultiprecisionTesting PRIVATE ${GpuTests})