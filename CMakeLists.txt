cmake_minimum_required(VERSION 3.22)
project(AesiMultiprecision VERSION 2.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_library(AesiMultiprecision INTERFACE)
target_include_directories(AesiMultiprecision INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

option(AESI_BUILD_TESTS      "Build tests"       OFF)
option(AESI_BUILD_BENCHMARKS "Build benchmarks"  OFF)
option(AESI_BUILD_SANITIZERS "Build sanitizers"  OFF)

if(AESI_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
            GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(test)
endif()
if(AESI_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
if(AESI_BUILD_SANITIZERS)
    add_subdirectory(sanitize/unsigned)
endif()

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS AesiMultiprecision
        EXPORT AesiMultiprecisionTargets)
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT AesiMultiprecisionTargets
        FILE AesiMultiprecisionTargets.cmake
        NAMESPACE AesiMultiprecision::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AesiMultiprecision)

configure_package_config_file(
        cmake/AesiMultiprecisionConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/AesiMultiprecisionConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AesiMultiprecision)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/AesiMultiprecisionConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/AesiMultiprecisionConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/AesiMultiprecisionConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AesiMultiprecision)