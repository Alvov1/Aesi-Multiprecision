Aesi Multiprecision 
==================

<p align="center">
    <a href="https://alvov1.github.io/Aesi-Multiprecision/index.html">
        <img src="https://img.shields.io/badge/Documentation-8A2BE2" alt="Documentation"></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/actions/workflows/gtest_multiple_platforms.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/Alvov1/Aesi-Multiprecision/gtest_multiple_platforms.yml" alt="Workflow build status"/></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/actions">
        <img src="https://img.shields.io/github/last-commit/alvov1/Aesi-Multiprecision" alt="Last Commit"></a>
    <a href="https://sonarcloud.io/summary/new_code?id=Alvov1_Aesi-Multiprecision">
        <img src="https://sonarcloud.io/api/project_badges/measure?project=Alvov1_Aesi-Multiprecision&metric=alert_status" alt="SonarCloud Code Quality"></a>
    <a href="https://app.codacy.com/gh/Alvov1/Aesi-Multiprecision/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
        <img src="https://app.codacy.com/project/badge/Grade/1658a07a21cf41dd8ac84ea56d62dd45" alt="Codacy Code Quality"></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/commits/main/">
        <img src="https://img.shields.io/github/commit-activity/y/Alvov1/Aesi-Multiprecision" alt="GitHub commit activity" /></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision">
        <img src="https://img.shields.io/github/languages/code-size/Alvov1/Aesi-Multiprecision" alt="GitHub code size in bytes" /></a>
</p>

The goal of this project is to develop a fast and handy multi-precision library that can be used with GPU parallelization frameworks such as CUDA, OpenCL, and Metal. The library should correspond to modern C++ standards, support constexpr expressions, and move semantics.

> [!IMPORTANT]
> Project is currently in the testing and development stage to support the *Cuda* framework. Please be aware that errors and problems may occur. OpenCL support is next in line for development. Metal support is scheduled after some time, due to the presence of significant differences in the framework from Cuda and OpenCL.
>

## Functionality
Library supports each arithmetic (binary and unary), bitwise, and boolean operations. Various functions from number theory are being added to the library, among which the greatest common divisor, the least common multiplier, and exponentiation by modulo have already been implemented.

## Installation:
Package could be downloaded to project's directory, or accessed directly through CMake:
```
include(FetchContent)
FetchContent_Declare(AesiMultiprecision
    GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
    GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)
...
target_include_directories(Target PRIVATE ${AesiMultiprecision_SOURCE_DIR})
```
Further library could be included in project with standard preprocessor command:
> #include <Aeu.h>

## Usage:
The library is a header only to avoid difficulties while building. In this case, it can be used simultaneously in C++ and CUDA projects without changing the file extension from .cu to .cpp and backwards. Library supports an object-oriented style of data management. Class operators are overloaded for use in expressions. The number's bitness is passed to the class object as a template parameter and has a default value of __*512 bits*__. It should be a multiple of 32.

Number's initialization could be done with numbers, strings, string-views, string literals, or library objects with different precision. User-defined string literals are planned to be released in the future.

Library supports display operations with STD streams (char and wchar_t based only), along with stream modifications (std::showbase, std::uppercase, std::hex, std::dec, std::oct). std::format support is planned to be released in the future.

### Host:
```cpp
#include <iostream>
#include "Aeu.h"

int main() {
    Aeu<512> f = 1u;
    for(unsigned i = 2; i <= 50; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}
```
> 0x49eebc961ed279b02b1ef4f28d19a84f5973a1d2c7800000000000

### Cuda kernel:
```cpp
__global__ void test() {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid != 0) return;

    Aesi<128> amp = 1562144106091796071UL;
    printf("Were in kernel thread and number is %lu\n", amp.integralCast<unsigned long>());
}

int main() {
    test<<<32, 32>>>();
    return cudaSuccess != cudaDeviceSynchronize();
}
```
> Were in kernel thread and number is 1562144106091796071

## About precision cast
It is admissible to use numbers of different precision inside the majority of operations, but it is not recommended cause it leads to redundant copying inside type conversions. Operation-assignment expressions (+=, -=, &=, etc...) require the bitness of the assignment to be greater or equal to the bitness of the assignable. The precision cast operator could be called by a user directly.

```cpp
Aesi<128> base = "10888869450418352160768000001";
Aesi<96> power = "99990001";
Aesi<256> mod = "8683317618811886495518194401279999999";

std::cout << Aesi<256>::powm(base, power, mod) << std::endl;
// Numbers get cast explicitly to bitness 256 

Aesi<128> m128 = "265252859812191058636308479999999";
Aesi<160> m160 = "263130836933693530167218012159999999";

std::cout << m128.precisionCast<256>() * m160 << std::endl; 
// Cast number of 128 bits to 256 bits, than multiply by number of 160 bits
```
> 1142184225164688919052733263067509431086585217025      6680141832773294447513292887050873529

An exception to the rule above is using longer precision boundaries inside functions, susceptible to overflow. As far as the number's precision is fixed on the stage of compilation, functions that require number multiplication or exponentiation may easily lead to overflow:
```cpp
Aesi<128> base = "340199290171201906239764863559915798527",
        power = "340282366920937859000464800151540596704",
        modulo = "338953138925230918806032648491249958912";

std::cout << Aesi<128>::powm(base, power, modulo) << std::endl; // Overflow !!!
std::cout << Aesi<256>::powm(base, power, modulo) << std::endl; // Fine
```
> 201007033690655614485250957754150944769

## Issues
Library is relatively slow in compare to other multiple precision libraries
<img src="https://dub.sh/jNgf79u" alt="Dynamic Image">

