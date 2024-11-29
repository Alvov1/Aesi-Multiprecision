#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Modulo, Basic) {
    Aeu128 one = 1u, zero = 0u, ten = 10u, two = 2u;
    EXPECT_EQ(zero % one, zero);
    EXPECT_EQ(ten % two, 0u);
    EXPECT_EQ(two % ten, 2u);
    EXPECT_EQ(ten % one, 0u);
    EXPECT_EQ(one % ten, 1u);
}

TEST(Unsigned_Modulo, Huge) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    /* Composite numbers. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandomWithBits(blocksNumber * 32 - 5),
            r = Generation::getRandomWithBits(blocksNumber * 16 - 32);

        Aeu<blocksNumber * 32> lA = l, rA = r;
        EXPECT_EQ(lA % rA, l % r);

        lA %= rA;
        EXPECT_EQ(lA, l % r);
    }

    /* Built-in types. */
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 200);
        const auto mod = Generation::getRandom<unsigned>();

        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(aeu % mod, value % mod);

        aeu %= mod;
        EXPECT_EQ(aeu, value % mod);
    }
}