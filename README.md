# Aesi Multiprecision

<p align="center">
    <a href="https://alvov1.github.io/Aesi-Multiprecision/index.html">
        <img src="https://img.shields.io/badge/Documentation-8A2BE2" alt="Documentation"></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/actions/workflows/integration.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/Alvov1/Aesi-Multiprecision/integration.yml" alt="Integration status"/></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-BSD_2--Clause-blue.svg" alt="License: BSD 2-Clause"/></a>
    <a href="https://codecov.io/gh/Alvov1/Aesi-Multiprecision">
        <img src="https://codecov.io/gh/Alvov1/Aesi-Multiprecision/graph/badge.svg" alt="Code coverage"/></a>
    <a href="https://sonarcloud.io/summary/new_code?id=Alvov1_Aesi-Multiprecision">
        <img src="https://sonarcloud.io/api/project_badges/measure?project=Alvov1_Aesi-Multiprecision&metric=alert_status" alt="SonarCloud Code Quality"></a>
    <a href="https://app.codacy.com/gh/Alvov1/Aesi-Multiprecision/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade">
        <img src="https://app.codacy.com/project/badge/Grade/1658a07a21cf41dd8ac84ea56d62dd45" alt="Codacy Code Quality"></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision/commits/main/">
        <img src="https://img.shields.io/github/commit-activity/y/Alvov1/Aesi-Multiprecision" alt="GitHub commit activity" /></a>
    <a href="https://github.com/Alvov1/Aesi-Multiprecision">
        <img src="https://img.shields.io/github/languages/code-size/Alvov1/Aesi-Multiprecision" alt="GitHub code size in bytes" /></a>
</p>

A header-only static-sized multiprecision arithmetic library for C++ and CUDA. Supports `constexpr` expressions, move semantics, and GPU parallelization frameworks in CUDA environments.

## Functionality

Supports all arithmetic (binary and unary), bitwise, and boolean operations. Number theory functions include GCD, LCM, and modular exponentiation.

## Requirements

- C++20 or later
- CMake 3.22 or later

## Installation

**Via CMake FetchContent:**
```cmake
include(FetchContent)
FetchContent_Declare(AesiMultiprecision
    GIT_REPOSITORY https://github.com/Alvov1/Aesi-Multiprecision.git
    GIT_TAG main)
FetchContent_MakeAvailable(AesiMultiprecision)

target_link_libraries(Target PRIVATE AesiMultiprecision)
```

**Via cmake --install / find_package:**
```bash
cmake -B build && cmake --install build
```
```cmake
find_package(AesiMultiprecision REQUIRED)
target_link_libraries(Target PRIVATE AesiMultiprecision::AesiMultiprecision)
```

Then include in your source:
```cpp
#include <AesiMultiprecision/Aeu.h>   // unsigned
#include <AesiMultiprecision/Aesi.h>  // signed
```

## Usage

The library is header-only and works in both C++ and CUDA projects without changing file extensions. The number's bitness is a template parameter with a default of **512 bits** — must be a multiple of 32.

Initialization accepts integers, strings, string views, and library objects of different precision. Display works with STD streams (`char` and `wchar_t`), including `std::showbase`, `std::uppercase`, `std::hex`, `std::dec`, `std::oct`.

### Host

```cpp
#include <iostream>
#include <AesiMultiprecision/Aeu.h>

int main() {
    Aeu<512> f = 1u;
    for(unsigned i = 2; i <= 50; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}
```
> 0x49eebc961ed279b02b1ef4f28d19a84f5973a1d2c7800000000000

### CUDA kernel

```cpp
__global__ void test() {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid != 0) return;

    Aesi<128> amp = 1562144106091796071UL;
    printf("Value in kernel thread: %lu\n", amp.integralCast<unsigned long>());
}

int main() {
    test<<<32, 32>>>();
    return cudaSuccess != cudaDeviceSynchronize();
}
```
> Value in kernel thread: 1562144106091796071

## Precision cast

Numbers of different precision can be mixed in most operations, though it causes implicit copying. Operation-assignment expressions (`+=`, `-=`, `&=`, etc.) require the left-hand bitness to be greater or equal to the right-hand. Use `precisionCast<N>()` to convert explicitly.

```cpp
Aeu<128> base = "10888869450418352160768000001";
Aeu<96>  power = "99990001";
Aeu<256> mod = "8683317618811886495518194401279999999";

cout << Aeu<256>::powm(base.precisionCast<256>(), power.precisionCast<256>(), mod) << endl;
```
> 6680141832773294447513292887050873529

Functions susceptible to overflow (e.g. `powm`) should use a larger precision explicitly:
```cpp
Aeu<128> base  = "340199290171201906239764863559915798527",
         power  = "340282366920937859000464800151540596704",
         modulo = "338953138925230918806032648491249958912";

cout << Aeu<128>::powm(base, power, modulo) << endl;             // Overflow
cout << Aeu<256>::powm(base.precisionCast<256>(),
         power, modulo.precisionCast<256>()) << endl;            // OK
```
> \***Overflowed***&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;201007033690655614485250957754150944769

## Performance

The library is slower than CPU-optimized multiprecision libraries. Benchmark results are published in each [Benchmarking workflow run](https://github.com/Alvov1/Aesi-Multiprecision/actions/workflows/benchmarks.yml) as a job summary.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions, branch naming, and how to run tests, benchmarks, and sanitizers.

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.
