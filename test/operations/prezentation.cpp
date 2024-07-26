#include <gtest/gtest.h>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aeu1024 f = 1u;
    for(unsigned i = 2; i <= 100; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}