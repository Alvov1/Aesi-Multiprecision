# Aesi multiprecision
The goal of this project is to develop a fast and handy multiprecision library which can be used with GPU parallelization frameworks such as CUDA, OpenCL and Metal. Library should correspond to modern C++ standards, support constexpr expressions and move semantic. Library is header only to avoid difficulties while building.

## Current progress
Project is currently under development and is being tested for support the __Cuda__ framework. OpenCL support is next in line for development. Metal support will be implemented after some time, due to the presence of significant differences of the framework from Cuda and from OpenCL.

## Functionality
Library supports each of arithmetic (binary and unary), bitwise and boolean operations. Various functions from number theory are being added to the library, among which greatest common divisor and exponentiation by modulo have already been implemented.

## Usage instructions: 
Library supports object-oriented style of data management. Class operators are overloaded for use in expressions and universal references are used whenever possible. Number's bitness is passed to the class object as a template parameter and has a default value of __*512 bit*__. It should be a multiple of 32 (bit length of unsigned type, which could be different on your system).

__1. Initialization.__ Number's initialization could be done with numbers, strings, string-views, string literals or library objects with different precision. User-defined string literals are planned to be released in the future.

__2. Display.__ Library supports STD streams along with stream modifications (std::showbase, std::uppercase, std::hex, std::dec, std::oct). std::format support is planned to be released in the future.

### Host:
```cpp
#include <iostream>
#include "Aesi.h"

Aesi<1024> factorial(unsigned n) {
    Aesi<1024> f = 1;
    for(unsigned i = 2; i <= n; ++i)
        f *= i;
    return f;
}

int main() {
    Aesi<1024> f100 = factorial(100);
    std::cout << std::hex << f100 << std::endl;
    return 0;
}
```
>1b30964ec395dc24069528d54bbda40d16e966ef9a70eb21b5b2943a321cdf10391745570cca9420c6ecb3b72ed2ee8b02ea2735c61a000000000000000000000000

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
>Were in kernel thread and number is 1562144106091796071

### OpenCl kernel:
```cpp
```
>

## About precision cast
It is admissible to use numbers of different precision inside the majority of operations, but it is not recommended cause it leads to redundant copying inside type conversions. Operation-assignment expressions (+=, -=, &=, etc...) requires the bitness of assignment to be greater or equal than bitness of assignable.
Precision cast operator could be called by user directly.

```cpp
Aesi<128> base = "10888869450418352160768000001";
Aesi<96> power = "99990001";
Aesi<256> mod = "8683317618811886495518194401279999999";

std::cout << Aesi<256>::powm(base, power, mod) << std::endl;
// Numbers are cast explicitly to bitness 256 

Aesi<128> m128 = "265252859812191058636308479999999";
Aesi<160> m160 = "263130836933693530167218012159999999";

std::cout << m128.precisionCast<256>() * m160 << std::endl; 
// Cast number of 128 bits to 256 bits, than multiply by number of 160 bits
```
>1142184225164688919052733263067509431086585217025     
6680141832773294447513292887050873529

An exception to the rule above is using longer precision boundaries inside functions, susceptible to overflow. As far as the number's precision is fixed on the stage of compilation, functions that require number multiplication or exponentiation may easily lead to overflow:
```cpp
Aesi<128> base = "340199290171201906239764863559915798527",
        power = "340282366920937859000464800151540596704",
        modulo = "338953138925230918806032648491249958912";

std::cout << Aesi<128>::powm(base, power, modulo) << std::endl; // Overflow !!!
std::cout << Aesi<256>::powm(base, power, modulo) << std::endl; // Fine
```
>201007033690655614485250957754150944769

## Implementation notes:
- Sign in bitwise operators is depending on sign of the first operand.
- Tilde operator (~) does __NOT__ affect the sign of number.
- Both bitshift operators do not make any effort if the shift value is greater than the bitness of the number. If the shift is negative, the opposite operator is called with the absolute value of the shift.
- Be careful with exponentiation overflow when using __POWM__ function and similar.
- GetString method for hexadecimal notation returns number representation with letters in __LOWERCASE__
- Both display methods work significantly faster for hexadecimal notation