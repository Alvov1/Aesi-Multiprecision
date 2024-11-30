#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, LeftShift) {
    constexpr auto testsAmount = 512, bitness = 1024, constShift = 200;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(200);
        Aeu<bitness> aeu = value;
        EXPECT_EQ(aeu << (i + constShift), value << (i + constShift));

        value = Generation::getRandomWithBits(200);
        aeu = value; aeu <<= (i + constShift);
        EXPECT_EQ(aeu, value << (i + constShift));
    }
}

TEST(Unsigned_Bitwise, RightShift) {
    constexpr auto testsAmount = 512, bitness = 1024, constShift = 256;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(990);
        Aeu<bitness> aeu = value;
        EXPECT_EQ(aeu >> (i + constShift), value >> (i + constShift));

        value = Generation::getRandomWithBits(990);
        aeu = value; aeu >>= (i + constShift);
        EXPECT_EQ(aeu, value >> (i + constShift));
    }
}