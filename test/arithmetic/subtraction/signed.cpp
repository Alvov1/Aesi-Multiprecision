#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Subtraction, Basic) {
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

TEST(Signed_Subtraction, Huge) {
    constexpr auto testsAmount = 2, blocksNumber = 64;
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

        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA - rA, l - r);

        lA -= rA;
        EXPECT_EQ(lA, l - r);

        const std::size_t decrements = rand() % 100;
        for (std::size_t j = 0; j < decrements * 2; j += 2) {
            EXPECT_EQ(rA--, r - j);
            EXPECT_EQ(--rA, r - j - 2);
        }
        EXPECT_EQ(rA, r - decrements * 2);
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto subU = Generation::getRandom<unsigned>();
        const auto subULL = Generation::getRandom<uint64_t>();

        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ((aeu - subU) - subULL, (value - subU) - subULL);

        aeu -= subU; aeu -= subULL;
        EXPECT_EQ(aeu, (value - subU) - subULL);
    }
}

TEST(Signed_Subtraction, Decrement) {

}