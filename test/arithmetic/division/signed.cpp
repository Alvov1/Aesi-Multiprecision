#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Division, Basic) {
    Aesi128 one = 1, mOne = -1, zero = 0, ten = 10, two = 2;
    EXPECT_EQ(one / mOne, -1);
    EXPECT_EQ(one / zero, 0);
    EXPECT_EQ(one / ten, 0);
    EXPECT_EQ(one / two, 0);
    EXPECT_EQ(mOne / zero, 0);
    EXPECT_EQ(mOne / ten, 0);
    EXPECT_EQ(mOne / two, 0);
    EXPECT_EQ(zero / ten, 0);
    EXPECT_EQ(zero / two, 0);
    EXPECT_EQ(ten / two, 5);

    EXPECT_EQ(two / ten, 0);
    EXPECT_EQ(two / zero, 0);
    EXPECT_EQ(two / mOne, -2);
    EXPECT_EQ(two / one, 2);
    EXPECT_EQ(ten / zero, 0);
    EXPECT_EQ(ten / mOne, -10);
    EXPECT_EQ(ten / one, 10);
    EXPECT_EQ(zero / mOne, 0);
    EXPECT_EQ(zero / one, 0);
    EXPECT_EQ(mOne / one, -1);
}

TEST(Signed_Division, Huge) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        int first, second;
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
        r = second * Generation::getRandomWithBits(blocksNumber * 16 - 110);

        Aesi<blocksNumber * 32> lA = l, rA = r;
        if(first == 1) {
            EXPECT_EQ(lA / rA, l / r );
            lA /= rA;
            EXPECT_EQ(lA, l / r);
        } else {
            auto result = -1 * ((l * -1) / r); // Encountered some errors in Cryptopp library LOL!!!
            EXPECT_EQ(lA / rA, result);
            lA /= rA;
            EXPECT_EQ(lA, result);
        }
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto mod = Generation::getRandom<unsigned long>();

        Aesi<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu / mod, value / mod);

        aeu /= mod;
        EXPECT_EQ(aeu, value / mod);
    }
}