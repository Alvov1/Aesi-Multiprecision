cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(AesiMultiprecision)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

find_package (SQLite3)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiMultiprecision ${Tests} ../Aeu.h ../Aesi.h )
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(number-theory)
add_subdirectory(boolean)

option(CRYPTOPP_CMP "Enable CryptoPP comparison" OFF)
if (CRYPTOPP_CMP)
    add_subdirectory(cryptopp-cmake-master)
    set(CryptoppLibraryCall cryptopp)
endif()

if (PREPROCESSOR_UNSAFE)
    add_definitions(-DAESI_UNSAFE=1)
endif (PREPROCESSOR_UNSAFE)

if (PREPROCESSOR_PRE_CPP_20)
    add_definitions(-DPRE_CPP_20=1)
endif (PREPROCESSOR_PRE_CPP_20)

include_directories(${SQLite3_INCLUDE_DIRS})

target_link_libraries(AesiMultiprecision GTest::gtest_main ${SQLite3_LIBRARIES} ${CryptoppLibraryCall})


include(GoogleTest)
gtest_discover_tests(AesiMultiprecision)


