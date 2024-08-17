#ifndef AESI_CRYPTOPP_INTEGRATION
#define AESI_CRYPTOPP_INTEGRATION
#endif

#include <gtest/gtest.h>
#include <../../../../Aesi.h>
#include <../../../../Aeu.h>
#include "../../generation.h"

TEST(Signed_Initialization, CryptoPP) {
    constexpr auto testsAmount = 2, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto cryptopp = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        const Aesi<blocksNumber * 32> aeu = cryptopp;

        std::stringstream ss, ss2; ss << cryptopp; ss2 << aeu;
        EXPECT_EQ(ss.str(), ss2.str());
    }
}

TEST(Unsigned_Initialization, CryptoPP) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto cryptopp = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        const Aeu<blocksNumber * 32> aeu = cryptopp;

        std::stringstream ss, ss2; ss << cryptopp; ss2 << aeu;
        EXPECT_EQ(ss.str(), ss2.str());
    }
}