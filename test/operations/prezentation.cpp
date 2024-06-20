#include <gtest/gtest.h>
#include "../../Aeu.h"

TEST(Prezentation, Factorial) {
    Aeu1024 f = 1u;
    for(unsigned i = 2; i <= 100; ++i)
        f *= i;

    std::stringstream ss {}; ss << f;
    EXPECT_EQ(ss.str(), "933262154439441526816992388562"
                        "667004907159682643816214685929"
                        "638952175999932299156089414639"
                        "761565182862536979208272237582"
                        "51185210916864000000000000000000000000");

    std::stringstream ss1 {}; ss1 << std::hex << f;
    EXPECT_EQ(ss1.str(), "1b30964ec395dc24069528d54bbda40d"
                         "16e966ef9a70eb21b5b2943a321cdf10"
                         "391745570cca9420c6ecb3b72ed2ee8b"
                         "02ea2735c61a000000000000000000000000");
}