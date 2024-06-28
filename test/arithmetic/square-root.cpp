#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_SquareRoot, SquareRoot) {
    constexpr auto testsAmount = 2048, bitness = 2048;
    for (unsigned i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(bitness - 20);
        const Aesi<bitness> m = value;
        if(i % 2 == 0)
            EXPECT_EQ(m.squareRoot(), value.SquareRoot());
        else EXPECT_EQ(m.squareRoot(), 0u);
    }
    Aesi<bitness> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}

TEST(Unsigned_SquareRoot, SquareRoot) {
    constexpr auto testsAmount = 2048, bitness = 2048;
    for (unsigned i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(bitness - 20);
        const Aeu<bitness> m = value;
        EXPECT_EQ(m.squareRoot(), value.SquareRoot());
    }

    Aeu<bitness> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}