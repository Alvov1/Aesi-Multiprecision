#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_NumberTheory, GreatestCommonDivisor) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto common = Generation::getRandomWithBits(blocksNumber * 8 - 10),
                left = common * Generation::getRandomWithBits(blocksNumber * 24 - 10),
                right = common * Generation::getRandomWithBits(blocksNumber * 24 - 10);

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::gcd(l, r), CryptoPP::GCD(left, right));
    }
}