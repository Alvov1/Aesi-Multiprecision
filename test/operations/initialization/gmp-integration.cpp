#ifndef AESI_GMP_INTEGRATION
#define AESI_GMP_INTEGRATION
#endif

#include <gtest/gtest.h>
#include <../../../../Aeu.h>
#include "../../generation.h"

TEST(Signed_Initialization, GMP) {
    constexpr auto testsAmount = 2, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class gmp = Generation::getRandom(blocksNumber * 32 - 20) * (i % 2 == 0 ? 1 : -1);
        const Aeu<blocksNumber * 32> aeu = gmp;

        std::stringstream ss; ss << aeu;
        EXPECT_EQ(ss.str(), gmp.get_str());
    }
}

TEST(Unsigned_Initialization, GMP) {
    constexpr auto testsAmount = 256, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class gmp = Generation::getRandom(blocksNumber * 32 - 20);
        const Aeu<blocksNumber * 32> aeu = gmp;

        std::stringstream ss; ss << aeu;
        EXPECT_EQ(ss.str(), gmp.get_str());
    }
}