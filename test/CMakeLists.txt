if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.26)
    project(AesiTests)
    set(CMAKE_CXX_STANDARD 20)

    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
            GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

find_package(PkgConfig)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(GMPXX REQUIRED IMPORTED_TARGET gmpxx)

enable_testing()

file(GLOB Tests *.cpp)
add_executable(AesiTests ${Tests})
add_subdirectory(arithmetic)
add_subdirectory(operations)
add_subdirectory(bitwise)
add_subdirectory(boolean)

target_compile_definitions(AesiTests PRIVATE
        AESI_GMP_INTEGRATION
        $<$<BOOL:${PREPROCESSOR_UNSAFE}>:AESI_UNSAFE=1>
        $<$<BOOL:${PREPROCESSOR_PRE_CPP_20}>:PRE_CPP_20=1>)

target_link_libraries(AesiTests PUBLIC
        GTest::gtest_main
        PkgConfig::GMP
        PkgConfig::GMPXX)
target_include_directories(AesiTests PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        PkgConfig::GMP
        PkgConfig::GMPXX)

target_compile_options(AesiTests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)

if(AESI_BUILD_COVERAGE)
    target_compile_options(AesiTests PRIVATE --coverage)
    target_link_options(AesiTests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(AesiTests XML_OUTPUT_DIR ${CMAKE_BINARY_DIR})


