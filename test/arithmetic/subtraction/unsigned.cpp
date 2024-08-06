#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Subtraction, Basic) {
    Aeu128 zero = 0u;
    Aeu128 m0 = 14377898u;
    EXPECT_EQ(m0 - 0u, 14377898u);
    EXPECT_EQ(m0 + 0u, 14377898u);
    EXPECT_EQ(m0 + zero, 14377898u);
    EXPECT_EQ(m0 - zero, 14377898u);
    EXPECT_EQ(m0 + +zero, 14377898u);
    EXPECT_EQ(m0 - +zero, 14377898u);
    m0 -= 0u; EXPECT_EQ(m0, 14377898u);
    m0 -= zero; EXPECT_EQ(m0, 14377898u);
    m0 -= -zero; EXPECT_EQ(m0, 14377898u);
    m0 -= +zero; EXPECT_EQ(m0, 14377898u);

    Aeu128 m1 = 42824647u;
    EXPECT_EQ(m1 - 0u, 42824647u);
    EXPECT_EQ(m1 + 0u, 42824647u);
    EXPECT_EQ(m1 + zero, 42824647u);
    EXPECT_EQ(m1 - zero, 42824647u);
    EXPECT_EQ(m1 + +zero, 42824647u);
    EXPECT_EQ(m1 - +zero, 42824647u);
    m1 -= 0u; EXPECT_EQ(m1, 42824647u);
    m1 -= zero; EXPECT_EQ(m1, 42824647u);
    m1 -= +zero; EXPECT_EQ(m1, 42824647u);
}

TEST(Unsigned_Subtraction, Huge) {
    constexpr auto testsAmount = 2048, blocksNumber = 64;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 5),
            r = Generation::getRandomWithBits(blocksNumber * 32 - 32);

        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA - rA, l - r);

        lA -= rA;
        EXPECT_EQ(lA, l - r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 10);
        const auto subU = Generation::getRandom<unsigned>();

        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu - subU, value - subU);

        aeu -= subU;
        EXPECT_EQ(aeu, value - subU);
    }
}

TEST(Unsigned_Subtraction, Decrement) {
    constexpr auto testsAmount = 2, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 110);
        Aeu<blocksNumber * 32> value = l;

        const std::size_t decrements = rand() % 100;
        for (std::size_t j = 0; j < decrements * 2; j += 2) {
            EXPECT_EQ(value--, l - j);
            EXPECT_EQ(--value, l - j - 2);
        }
        EXPECT_EQ(value, l - decrements * 2);
    }

    Aeu<blocksNumber * 32> test = 1u; --test;
    EXPECT_EQ(test, 0u);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1u);

    test = 1u; test--;
    EXPECT_EQ(test, 0u);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1u);
}