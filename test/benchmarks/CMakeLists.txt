file(GLOB BenchmarksTests benchmarks.cpp)

if(EXISTS cryptopp-cmake-master)
    add_subdirectory(cryptopp-cmake-master)
    target_link_libraries(AesiMultiprecisionTesting cryptopp)
    list(APPEND BenchmarksTests "cryptopp_comparison.cpp")
endif(EXISTS cryptopp-cmake-master)

target_sources(AesiMultiprecisionTesting PRIVATE ${BenchmarksTests})