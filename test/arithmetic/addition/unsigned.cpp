#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Addition, Basic) {
    {
        Aeu128 zero = 0u, m0 = 26359343u;
        EXPECT_EQ(zero + m0, 26359343u);
        EXPECT_EQ(m0 + zero, 26359343u);
        EXPECT_EQ(zero + +m0, 26359343u);
        EXPECT_EQ(m0 + +zero, 26359343u);
        EXPECT_EQ(+zero + m0, 26359343u);
        EXPECT_EQ(+m0 + zero, 26359343u);
        EXPECT_EQ(+zero + +m0, 26359343u);
        EXPECT_EQ(+m0 + +zero, 26359343u);
        m0 += zero; EXPECT_EQ(m0, 26359343u); m0 += +zero; EXPECT_EQ(m0, 26359343u); +m0 += zero; EXPECT_EQ(m0, 26359343u); +m0 += +zero; EXPECT_EQ(m0, 26359343u);
    }
    {
        Aeu128 zero = 0u, m1 = 14670384u, m2 = 55908622u;
        EXPECT_EQ(m1 + 0u, 14670384u); EXPECT_EQ(0u + m1, 14670384u); EXPECT_EQ(m1 + zero, 14670384u); EXPECT_EQ(m1 + +zero, 14670384u);
        EXPECT_EQ(m1 + +zero, 14670384u); EXPECT_EQ(zero + m1, 14670384u); EXPECT_EQ(+zero + m1, 14670384u); m1 += 0u; EXPECT_EQ(m1, 14670384u);
        m1 += zero; EXPECT_EQ(m1, 14670384u); m1 += +zero; EXPECT_EQ(m1, 14670384u);

        EXPECT_EQ(m2 + 0u, 55908622u); EXPECT_EQ(0u + m2, 55908622u); EXPECT_EQ(m2 + zero, 55908622u); EXPECT_EQ(m2 + +zero, 55908622u);
        EXPECT_EQ(m2 + +zero, 55908622u); EXPECT_EQ(zero + m2, 55908622u); EXPECT_EQ(+zero + m2, 55908622u); m2 += 0u; EXPECT_EQ(m2, 55908622u);
        m2 += zero; EXPECT_EQ(m2, 55908622u); m2 += +zero; EXPECT_EQ(m2, 55908622u);
    }
    {
        Aeu128 s1 = 0x24DFBE889u, s2 = 0x193E161Cu, s3 = 0x51CDFC6u, s4 = 0x1706808355u;
        EXPECT_EQ(s1 + s1, 0x49BF7D112u);
        EXPECT_EQ(s1 + s2, 0x26739fea5u);
        EXPECT_EQ(s1 + s3, 0x25318c84fu);
        EXPECT_EQ(s1 + s4, 0x19547c6bdeu);
        EXPECT_EQ(s2 + s1, 0x26739fea5u);
        EXPECT_EQ(s2 + s2, 0x327c2c38u);
        EXPECT_EQ(s2 + s3, 0x1e5af5e2u);
        EXPECT_EQ(s2 + s4, 0x171fbe9971u);
        EXPECT_EQ(s3 + s1, 0x25318c84fu);
        EXPECT_EQ(s3 + s2, 0x1e5af5e2u);
        EXPECT_EQ(s3 + s3, 0xa39bf8cu);
        EXPECT_EQ(s3 + s4, 0x170b9d631bu);
        EXPECT_EQ(s4 + s1, 0x19547c6bdeu);
        EXPECT_EQ(s4 + s2, 0x171fbe9971u);
        EXPECT_EQ(s4 + s3, 0x170b9d631bu);
        EXPECT_EQ(s4 + s4, 0x2e0d0106aau);
    }
}

TEST(Unsigned_Addition, Huge) {
    constexpr auto testsAmount = 2048, blocksNumber = 64;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 31 - 5),
            r = Generation::getRandomWithBits(blocksNumber * 16 - 32);

        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA + rA, l + r);

        lA += rA;
        EXPECT_EQ(lA, l + r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 10);
        const auto subU = Generation::getRandom<unsigned>();

        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu + subU, value + subU);

        aeu += subU;
        EXPECT_EQ(aeu, value + subU);
    }
}

TEST(Unsigned_Addition, Increment) {
    constexpr auto testsAmount = 2, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 110);
        Aeu<blocksNumber * 32> value = l;

        const std::size_t increments = rand() % 100;
        for (std::size_t j = 0; j < increments * 2; j += 2) {
            EXPECT_EQ(value++, l + j);
            EXPECT_EQ(++value, l + j + 2);
        }
        EXPECT_EQ(value, l + increments * 2);
    }
}
