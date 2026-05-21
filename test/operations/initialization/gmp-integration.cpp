#ifndef AESI_GMP_INTEGRATION
#define AESI_GMP_INTEGRATION
#endif

#include <gtest/gtest.h>
#include <AesiMultiprecision/Aesi.h>
#include <AesiMultiprecision/Aeu.h>
#include "../../generation.h"

TEST(Signed_Initialization, DISABLED_GMP) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class gmp = Generation::getRandom(N - 20) * (i % 2 == 0 ? 1 : -1);
            const Aesi<N> aeu = gmp;

            std::stringstream ss; ss << aeu;
            EXPECT_EQ(ss.str(), gmp.get_str());
        }
    });
}

TEST(Unsigned_Initialization, DISABLED_GMP) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class gmp = Generation::getRandom(N - 20);
            const Aeu<N> aeu = gmp;

            std::stringstream ss; ss << aeu;
            EXPECT_EQ(ss.str(), gmp.get_str());
        }
    });
}