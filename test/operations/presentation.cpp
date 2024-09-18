#include <gtest/gtest.h>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aesi<1024> f = 1;
    for(unsigned i = 2; i <= 50; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}