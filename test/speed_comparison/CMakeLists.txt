cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(SpeedComparison)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(SpeedComparison ${Tests})

add_subdirectory(cryptopp-cmake-master)
set(CryptoppLibraryCall cryptopp)

target_link_libraries(SpeedComparison GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(SpeedComparison)


