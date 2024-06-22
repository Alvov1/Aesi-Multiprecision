#ifndef AESI_CRYPTOPP_INTEGRATION
#define AESI_CRYPTOPP_INTEGRATION
#endif

#include <gtest/gtest.h>
#include <../../../../Aeu.h>
#include "../../generation.h"

TEST(Signed_Initialization, CryptoPP) {
    EXPECT_TRUE(false);
}

TEST(Unsigned_Initialization, CryptoPP) {
    constexpr auto testsAmount = 2048, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto cryptopp = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        const Aeu<blocksNumber * 32> aeu = cryptopp;

        std::stringstream ss, ss2;
        ss << "0x" << std::hex << cryptopp;
        ss2 << std::hex << std::showbase << aeu;
        EXPECT_EQ(ss.str(), ss2.str());
    }
}