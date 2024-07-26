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
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 5),
            r = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        // std::cout << "l: " << l << ", r: " << r << std::endl;
        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA - rA, l - r);

        lA -= rA;
        EXPECT_EQ(lA, l - r);

        const std::size_t decrements = rand() % 1'000;
        for (std::size_t j = 0; j < decrements * 2; j += 2) {
            EXPECT_EQ(rA--, r - j);
            EXPECT_EQ(--rA, r - j - 2);
        }
        EXPECT_EQ(rA, r - decrements * 2);
    }

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
    Aeu512 m0 = 62492992u;
    --m0; --m0; m0--; --m0; m0--; --m0; m0--; --m0; m0--; --m0;
    EXPECT_EQ(m0, 62492982u);
    Aeu512 t0 = m0--, u0 = --m0;
    EXPECT_EQ(t0, 62492982u); EXPECT_EQ(u0, 62492980u); EXPECT_EQ(m0, 62492980u);

    Aeu512 m2 = 77428594u;
    m2--; m2--; --m2; m2--; m2--; m2--; --m2; m2--; m2--; m2--; --m2; m2--; --m2; --m2;
    EXPECT_EQ(m2, 77428580u);
    Aeu512 t2 = m2--, u2 = --m2;
    EXPECT_EQ(t2, 77428580u); EXPECT_EQ(u2, 77428578u); EXPECT_EQ(m2, 77428578u);

    Aeu512 m3 = 77677795u;
    --m3; --m3; --m3; m3--; --m3; m3--; --m3; --m3; m3--; --m3; m3--; --m3; m3--; m3--; m3--; m3--; m3--; --m3;
    EXPECT_EQ(m3, 77677777u);
    Aeu512 t3 = m3--, u3 = --m3;
    EXPECT_EQ(t3, 77677777u); EXPECT_EQ(u3, 77677775u); EXPECT_EQ(m3, 77677775u);
}