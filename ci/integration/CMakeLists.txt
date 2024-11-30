cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(Build)

include(FetchContent)
FetchContent_Declare(AesiMultiprecision
        GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
        GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)


# 1. Build project with Aesi as library
add_executable(Library_Integration main.cpp) # Build project as library
target_include_directories(Library_Integration PRIVATE ${AesiMultiprecision_SOURCE_DIR})



# 2. Test examples from README with GTest
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(Readme_Examples presentation.cpp) # Test examples from README
target_link_libraries(Readme_Examples PRIVATE GTest::gtest_main)
target_include_directories(Readme_Examples PRIVATE
        ${AesiMultiprecision_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(Readme_Examples XML_OUTPUT_DIR ${CMAKE_BINARY_DIR})