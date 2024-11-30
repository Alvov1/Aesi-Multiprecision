#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Modulo, Basic) {
    Aesi128 one = 1u, mOne = -1, zero = 0u, ten = 10u, two = 2u;
    EXPECT_EQ(one % mOne, 0);
    EXPECT_EQ(one % zero, 1);
    EXPECT_EQ(one % ten, 1);
    EXPECT_EQ(one % two, 1);
    EXPECT_EQ(mOne % zero, -1);
    EXPECT_EQ(mOne % ten, -1);
    EXPECT_EQ(mOne % two, -1);
    EXPECT_EQ(zero % ten, 0);
    EXPECT_EQ(zero % two, 0);
    EXPECT_EQ(ten % two, 0);

    EXPECT_EQ(two % ten, 2);
    EXPECT_EQ(two % zero, 2);
    EXPECT_EQ(two % mOne, 0);
    EXPECT_EQ(two % one, 0);
    EXPECT_EQ(ten % zero, 10);
    EXPECT_EQ(ten % mOne, 0);
    EXPECT_EQ(ten % one, 0);
    EXPECT_EQ(zero % mOne, 0);
    EXPECT_EQ(zero % one, 0);
    EXPECT_EQ(mOne % one, 0);
}

TEST(Signed_Modulo, Huge) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 110),
                r = Generation::getRandomWithBits(blocksNumber * 16 - 110);

        Aesi<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA % rA, l % r);

        lA %= rA;
        EXPECT_EQ(lA, l % r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto mod = Generation::getRandom<unsigned long>();

        Aesi<blocksNumber * 32> aesi = value;
        EXPECT_EQ(aesi % mod, value % mod);

        aesi %= mod; aesi %= mod;
        EXPECT_EQ(aesi, value % mod);
    }
}