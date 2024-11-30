#include <iostream>
#include <Aeu.h>

int main() {
    const Aeu512 number = "0xffff'ffff'ffff'ffff'ffff'ffff'ffff'ffff'ffff'ffff";
    std::cout << std::hex << std::showbase << number << std::endl;
    return 0;
}