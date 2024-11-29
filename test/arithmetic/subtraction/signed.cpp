#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Subtraction, Basic) {
    Aesi128 zero = 0l;
    Aesi128 m0 = 14377898l;
    EXPECT_EQ(m0 - 0u, 14377898l);
    EXPECT_EQ(m0 + 0u, 14377898l);
    EXPECT_EQ(m0 + zero, 14377898l);
    EXPECT_EQ(m0 - zero, 14377898l);
    EXPECT_EQ(m0 + +zero, 14377898l);
    EXPECT_EQ(m0 - +zero, 14377898l);
    m0 -= 0u; EXPECT_EQ(m0, 14377898l);
    m0 -= zero; EXPECT_EQ(m0, 14377898l);
    m0 -= -zero; EXPECT_EQ(m0, 14377898l);
    m0 -= +zero; EXPECT_EQ(m0, 14377898l);

    Aesi128 m1 = -42824647l;
    EXPECT_EQ(m1 - 0u, -42824647l);
    EXPECT_EQ(m1 + 0u, -42824647l);
    EXPECT_EQ(m1 + zero, -42824647l);
    EXPECT_EQ(m1 - zero, -42824647l);
    EXPECT_EQ(m1 + +zero, -42824647l);
    EXPECT_EQ(m1 - +zero, -42824647l);
    m1 -= 0u; EXPECT_EQ(m1, -42824647l);
    m1 -= zero; EXPECT_EQ(m1, -42824647l);
    m1 -= +zero; EXPECT_EQ(m1, -42824647l);
}

TEST(Signed_Subtraction, Huge) {
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
        const auto l = first * Generation::getRandomWithBits(blocksNumber * 32 - 110),
                r = second * Generation::getRandomWithBits(blocksNumber * 32 - 110);

        Aesi<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA - rA, l - r);

        lA -= rA;
        EXPECT_EQ(lA, l - r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto sub = Generation::getRandom<long long>();

        Aesi<blocksNumber * 32> aeu = value;
        EXPECT_EQ((aeu - sub) - sub, (value - sub) - sub);

        aeu -= sub; aeu -= sub;
        EXPECT_EQ(aeu, (value - sub) - sub);
    }
}

TEST(Signed_Subtraction, Decrement) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 110);
        Aesi<blocksNumber * 32> value = l;

        const std::size_t decrements = rand() % 100;
        for (std::size_t j = 0; j < decrements * 2; j += 2) {
            EXPECT_EQ(value--, l - j);
            EXPECT_EQ(--value, l - j - 2);
        }
        EXPECT_EQ(value, l - decrements * 2);
    }

    Aesi<blocksNumber * 32> test = 0; --test;
    EXPECT_EQ(test, -1);
    EXPECT_FALSE(test.isZero());
    EXPECT_LT(test, 0);

    test = 0; test--;
    EXPECT_EQ(test, -1);
    EXPECT_FALSE(test.isZero());
    EXPECT_LT(test, 0);

    test = 1; --test;
    EXPECT_EQ(test, 0);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1);

    test = 1; test--;
    EXPECT_EQ(test, 0);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1);
}