#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_NumberTheory, LeastCommonMultiplier) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto left = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                right = Generation::getRandomWithBits(blocksNumber * 16 - 10);
        std::stringstream leftS, rightS, lcmS;
        leftS << "0x" << std::hex << left;
        rightS << "0x" << std::hex << right;
        lcmS << "0x" << std::hex << CryptoPP::LCM(left, right);

        Aeu<blocksNumber * 32> l = leftS.str(), r = rightS.str();
        EXPECT_EQ(Aeu<blocksNumber * 32>::lcm(l, r), lcmS.str());
    }
}