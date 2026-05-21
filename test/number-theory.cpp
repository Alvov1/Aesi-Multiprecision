#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include "generation.h"

TEST(NumberTheory, PowerByModulo) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandom(blocksNumber * 16 - 10),
                power = Generation::getRandom(blocksNumber * 16 - 10),
                modulo = Generation::getRandom(blocksNumber * 16 - 10);

        mpz_class powm {};
        mpz_powm(powm.get_mpz_t(), base.get_mpz_t(), power.get_mpz_t(), modulo.get_mpz_t());

        Aeu<blocksNumber * 32> b = base, p = power, m = modulo;
        EXPECT_EQ(Aeu<blocksNumber * 32>::powm(b, p, m), powm);
    }
}

TEST(NumberTheory, PowerByModuloDifferentPrecision) {
    constexpr auto testsAmount = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandom(250),
                power = Generation::getRandom(1000),
                modulo = Generation::getRandom(250);

        mpz_class powm {};
        mpz_powm(powm.get_mpz_t(), base.get_mpz_t(), power.get_mpz_t(), modulo.get_mpz_t());

        Aeu<512> b = base, m = modulo, result = powm;
        Aeu<1024> p = power;

        EXPECT_EQ(Aeu<512>::powm<1024>(b, p, m), result);
    }
}

TEST(NumberTheory, LeastCommonMultiplier) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto left = Generation::getRandom(blocksNumber * 16 - 10),
                right = Generation::getRandom(blocksNumber * 16 - 10);

        mpz_class lcm {};
        mpz_lcm(lcm.get_mpz_t(), left.get_mpz_t(), right.get_mpz_t());

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::lcm(l, r), lcm);
    }
}

TEST(NumberTheory, GreatestCommonDivisor) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class common = Generation::getRandom(blocksNumber * 8 - 10),
                left = common * Generation::getRandom(blocksNumber * 24 - 10),
                right = common * Generation::getRandom(blocksNumber * 24 - 10);

        mpz_class gcd {};
        mpz_gcd(gcd.get_mpz_t(), left.get_mpz_t(), right.get_mpz_t());

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::gcd(l, r), gcd);
    }
}