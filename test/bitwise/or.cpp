#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, OR) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto l = Generation::getRandomWithBits(blocksNumber * 32 - 20),
                r = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aeu<blocksNumber * 32> left = l, right = r;
        EXPECT_EQ(left | right, l | r);

        l = Generation::getRandomWithBits(blocksNumber * 32 - 20),
                r = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        left = l, right = r; left |= right;
        EXPECT_EQ(left, l | r);
    }
}