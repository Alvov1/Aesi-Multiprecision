#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, LeftShift) {
    { Aeu256 v = "0x7c3123a5c28cd0ac794069214ef6f721"; EXPECT_EQ(v << 0u, "0x7c3123a5c28cd0ac794069214ef6f721"); }

    constexpr auto testsAmount = 1024, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(512);
        Aeu<bitness> aeu = value;
        EXPECT_EQ(aeu << (i + 512), value << (i + 512));

        value = Generation::getRandomWithBits(512);
        aeu = value; aeu <<= (i + 512);
        EXPECT_EQ(aeu, value << (i + 512));
    }
}

TEST(Unsigned_Bitwise, RightShift) {
    { Aeu256 v = "0x375c7c6861efbe692a13f3793a5ceb84e1fc458dd3a8471bd3456fea6e"; EXPECT_EQ(v >> 0u, "0x375c7c6861efbe692a13f3793a5ceb84e1fc458dd3a8471bd3456fea6e"); }

    constexpr auto testsAmount = 1024, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(1800);
        Aeu<bitness> aeu = value;
        EXPECT_EQ(aeu >> (i + 512), value >> (i + 512));

        value = Generation::getRandomWithBits(1800);
        aeu = value; aeu >>= (i + 512);
        EXPECT_EQ(aeu, value >> (i + 512));
    }
}