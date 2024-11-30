#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_SquareRoot, SquareRoot) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (unsigned i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        const Aesi<blocksNumber * 32> m = value;
        if(i % 2 == 0)
            EXPECT_EQ(m.squareRoot(), value.SquareRoot());
        else EXPECT_EQ(m.squareRoot(), 0u);
    }
    Aesi<blocksNumber * 32> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}

TEST(Unsigned_SquareRoot, SquareRoot) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (unsigned i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        const Aeu<blocksNumber * 32> m = value;
        EXPECT_EQ(m.squareRoot(), value.SquareRoot());
    }

    Aeu<blocksNumber * 32> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}