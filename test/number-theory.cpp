#include <gtest/gtest.h>
#include "../Aeu.h"
#include "generation.h"

TEST(NumberTheory, PowerByModulo) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                power = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                modulo = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                powm = CryptoPP::ModularExponentiation(base, power, modulo);

        Aeu<blocksNumber * 32> b = base, p = power, m = modulo;
        EXPECT_EQ(Aeu<blocksNumber * 32>::powm(b, p, m), powm);
    }
}

TEST(NumberTheory, PowerByModuloDifferentPrecision) {
    constexpr auto testsAmount = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto base = Generation::getRandomWithBits(250),
                power = Generation::getRandomWithBits(1000),
                modulo = Generation::getRandomWithBits(250),
                powm = CryptoPP::ModularExponentiation(base, power, modulo);

        Aeu<512> b = base, m = modulo, result = powm;
        Aeu<1024> p = power;

        EXPECT_EQ(Aeu<512>::powm<1024>(b, p, m), result);
    }
}

TEST(NumberTheory, LeastCommonMultiplier) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto left = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                right = Generation::getRandomWithBits(blocksNumber * 16 - 10);

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::lcm(l, r), CryptoPP::LCM(left, right));
    }
}

TEST(NumberTheory, GreatestCommonDivisor) {
    constexpr auto testsAmount = 256, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto common = Generation::getRandomWithBits(blocksNumber * 8 - 10),
                left = common * Generation::getRandomWithBits(blocksNumber * 24 - 10),
                right = common * Generation::getRandomWithBits(blocksNumber * 24 - 10);

        Aeu<blocksNumber * 32> l = left, r = right;
        EXPECT_EQ(Aeu<blocksNumber * 32>::gcd(l, r), CryptoPP::GCD(left, right));
    }
}