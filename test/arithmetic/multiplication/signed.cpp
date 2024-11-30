#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Multiplication, Basic) {
    Aesi128 zero = 0u, one = 1u;
    Aesi128 m0 = 10919396u;
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

TEST(Signed_Multiplication, Huge) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        int first = 0, second = 0;
        switch(i % 4) {
        case 0:
            first = 1, second = 1; break;
        case 1:
            first = -1, second = -1; break;
        case 2:
            first = -1, second = 1; break;
        default:
            first = 1, second = -1;
        }
        const auto l = first * Generation::getRandomWithBits(blocksNumber * 16 - 110),
                r = second * Generation::getRandomWithBits(blocksNumber * 16 - 110);

        Aesi<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA * rA, l * r);

        lA *= rA;
        EXPECT_EQ(lA, l * r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto factor = Generation::getRandom<unsigned>();

        Aesi<blocksNumber * 32> aeu = value;
        EXPECT_EQ((aeu * factor) * factor, (value * factor) * factor);

        aeu *= factor; aeu *= factor;
        EXPECT_EQ(aeu, (value * factor) * factor);
    }
}