#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_NumberTheory, GreatestCommonDivisor) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto common = Generation::getRandomWithBits(blocksNumber * 8 - 10),
                left = common * Generation::getRandomWithBits(blocksNumber * 24 - 10),
                right = common * Generation::getRandomWithBits(blocksNumber * 24 - 10);
        std::stringstream leftS, rightS, gcdS;
        leftS << "0x" << std::hex << left;
        rightS << "0x" << std::hex << right;
        gcdS << "0x" << std::hex << CryptoPP::GCD(left, right);

        Aeu<blocksNumber * 32> l = leftS.str(), r = rightS.str();
        EXPECT_EQ(Aeu<blocksNumber * 32>::gcd(l, r), gcdS.str());
    }
}