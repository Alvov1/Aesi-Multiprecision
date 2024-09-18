#include <gtest/gtest.h>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aeu<512> f = 1u;
    for(unsigned i = 2; i <= 50; ++i)
        f *= i;

    std::cout << std::showbase << std::hex << f << std::endl;
}