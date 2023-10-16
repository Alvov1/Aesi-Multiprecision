# Multiprecision for GPU frameworks
***
The goal of this project is to develop a fast and handy multiprecision library which can be used with GPU parallelization frameworks such as CUDA, OpenCL and Metal. Library should correspond to modern C++ standards, support constexpr expressions and move semantic. Library is header only to avoid difficulties while building.

## Current progress
Project is currently under development and supports CUDA framework only.

## Functionality
Library supports each of arithmetic (binary and unary), bitwise and boolean operations. Various functions from number theory are being added to the library, among which greatest common divisor and exponentiation by modulo have already been implemented.

## Usage instructions: 
Library supports object-oriented style of data management. Class operators are overloaded for use in expressions and universal references are used whenever possible. Number's bitness is passed to the class object as a template parameter and has a default value of 512 bit. It should be a multiple of 32 (bit length of unsigned type, which could be different on your system).

__1. Initialization.__ Number's initialization could be done with numbers, strings, string-views, string literals or library objects with different precision. User-defined string literals are planned to be released in the future.

__2. Display.__ Library supports STD streams along with stream modifications (std::showbase, std::uppercase, std::hex, std::dec, std::oct). Printf-like functions and std::format support is planned to be released in the future.

```cpp
#include <iostream>
#include "Multiprecision.h"

Multiprecision<1024> factorial(unsigned n) {
    Multiprecision<1024> f = 1;
    for(unsigned i = 2; i <= n; ++i)
        f *= i;
    return f;
}

int main() {
    Multiprecision<1024> f100 = factorial(100);
    std::cout << std::hex << f100 << std::endl;
    return 0;
}
```
>1b30964ec395dc24069528d54bbda40d16e966ef9a70eb21b5b2943a321cdf10391745570cca9420c6ecb3b72ed2ee8b02ea2735c61a000000000000000000000000

## About precision cast
It is admissible to use numbers of different precision inside the majority of operations, but it is not recommended cause it leads to redundant copying inside type conversions. Operation-assignment expressions (+=, -=, &=, etc...) requires the bitness of assignment to be greater or equal than bitness of assignable.
Precision cast operator could be called by user directly.

## Implementation notes:
- Sign in bitwise operators is depending on sign of the first operand.
- Tilde operator (~) does __NOT__ affect the sign of number.
- Both bitshift operators do not make any effort if the shift value is greater than the bitness of the number. If the shift is negative, the opposite operator is called with the absolute value of the shift.