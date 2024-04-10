#include <gtest/gtest.h>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aeu1024 f = 1u;
    for(unsigned i = 2; i <= 100; ++i)
        f *= i;

    std::stringstream ss {}; ss << f;
    EXPECT_EQ(ss.str(), "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000");

    std::stringstream ss1 {}; ss1 << std::hex << f;
    EXPECT_EQ(ss1.str(), "1b30964ec395dc24069528d54bbda40d16e966ef9a70eb21b5b2943a321cdf10391745570cca9420c6ecb3b72ed2ee8b02ea2735c61a000000000000000000000000");
}