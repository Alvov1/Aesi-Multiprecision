#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Multiplication, Basic) {
    Aeu128 zero = 0u, one = 1u;
    Aeu128 m0 = 10919396u;
    EXPECT_EQ(m0 * 0u, 0u);
    EXPECT_EQ(0u * m0, 0u);
    EXPECT_EQ(m0 * 1u, 10919396u);
    EXPECT_EQ(1u * m0, 10919396u);

    EXPECT_EQ(m0 * zero, 0u);
    EXPECT_EQ(m0 * +zero, 0u);
    EXPECT_EQ(m0 * one, 10919396u);
    EXPECT_EQ(m0 * +one, 10919396u);

    EXPECT_EQ(zero * m0, 0u);
    EXPECT_EQ(zero * +m0, 0u);
    EXPECT_EQ(one * m0, 10919396u);
    EXPECT_EQ(one * +m0, 10919396u);

    EXPECT_EQ(+zero * m0, 0u);
    EXPECT_EQ(+zero * +m0, 0u);
    EXPECT_EQ(+one * m0, 10919396u);
    EXPECT_EQ(+one * +m0, 10919396u);

    m0 *= 0u; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= +zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    zero *= m0; EXPECT_EQ(zero, 0u);
    zero *= +m0; EXPECT_EQ(zero, 0u);

    m0 *= 1u; EXPECT_EQ(m0, 10919396u);

    m0 *= one; EXPECT_EQ(m0, 10919396u);
    m0 *= +one; EXPECT_EQ(m0, 10919396u);
    one *= m0; EXPECT_EQ(one, 10919396u); one = 1u;
    one *= +m0; EXPECT_EQ(one, 10919396u);
}

TEST(Unsigned_Multiplication, Huge) {
    constexpr auto testsAmount = 2048, blocksNumber = 64;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 16 - 10),
                r = Generation::getRandomWithBits(blocksNumber * 16 - 20);
        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA * rA, l * r);

        lA *= rA;
        EXPECT_EQ(lA, l * r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto factorU = Generation::getRandom<unsigned>();

        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu * factorU, value * factorU);

        aeu *= factorU;
        EXPECT_EQ(aeu, value * factorU);
    }
}