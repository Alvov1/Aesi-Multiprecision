#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, LeftShift) {
    constexpr auto testsAmount = 1024, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(512);
        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu << (i + 512), value << (i + 512));

        value = Generation::getRandomWithBits(512);
        aeu = value; aeu <<= (i + 512);
        EXPECT_EQ(aeu, value << (i + 512));
    }
}

TEST(Unsigned_Bitwise, RightShift) {
    constexpr auto testsAmount = 1024, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(1800);
        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu >> (i + 512), value >> (i + 512));

        value = Generation::getRandomWithBits(1800);
        aeu = value; aeu >>= (i + 512);
        EXPECT_EQ(aeu, value >> (i + 512));
    }
}