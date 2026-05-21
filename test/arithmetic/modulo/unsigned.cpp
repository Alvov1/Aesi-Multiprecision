#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
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
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        /* Composite numbers. */
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto l = Generation::getRandom(N - 5),
                r = Generation::getRandom(N / 2 - 32);

            Aeu<N> lA = l, rA = r;
            EXPECT_EQ(lA % rA, l % r);

            lA %= rA;
            EXPECT_EQ(lA, l % r);
        }

        /* Built-in types. */
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 200);
            const auto mod = Generation::getRandom<unsigned>();

            Aeu<N> aeu = value;
            EXPECT_EQ(aeu % mod, value % mod);

            aeu %= mod;
            EXPECT_EQ(aeu, value % mod);
        }
    });
}