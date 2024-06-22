#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_NumberTheory, LeastCommonMultiplier) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto left = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                right = Generation::getRandomWithBits(blocksNumber * 16 - 10);

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::lcm(l, r), CryptoPP::LCM(left, right));
    }
}