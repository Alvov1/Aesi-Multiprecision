cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(Build)

include(FetchContent)
FetchContent_Declare(AesiMultiprecision
        GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
        GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)


# 1. Build project with Aesi as library
add_executable(Library_Integration main.cpp)
target_link_libraries(Library_Integration PRIVATE AesiMultiprecision)

# 2. Test examples from README with GTest
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_test(NAME library-integration COMMAND Library_Integration)

add_executable(Readme_Examples presentation.cpp)
target_link_libraries(Readme_Examples PRIVATE GTest::gtest_main AesiMultiprecision)

include(GoogleTest)
gtest_discover_tests(Readme_Examples XML_OUTPUT_DIR ${CMAKE_BINARY_DIR})