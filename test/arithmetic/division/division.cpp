#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

TEST(Unsigned_Division, Basic) {
    Aeu128 one = 1u, zero = 0u, ten = 10u, two = 2u;
    EXPECT_EQ(zero / one, zero);
    EXPECT_EQ(ten / two, 5u);
    EXPECT_EQ(two / ten, 0u);
}

TEST(Unsigned_Division, Huge) {
    Generation::forEachPrecision([]<std::size_t N>() {
        Generation::runCompositeTest<Aeu, N>(N - 5, N / 2 - 32,
            [](auto a, auto b) { return a / b; },
            [](auto& a, const auto& b) { a /= b; });

        /* Built-in types. */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 10);
            const auto subU = Generation::getRandom<unsigned>();

            Aeu<N> aeu = value;
            EXPECT_EQ(aeu / subU, value / subU);

            aeu /= subU;
            EXPECT_EQ(aeu, value / subU);
        }
    });
}

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
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
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
            const mpz_class l = first * Generation::getRandom(N - 110),
            r = second * Generation::getRandom(N / 2 - 110);

            Aesi<N> lA = l, rA = r;
            if(first == 1) {
                EXPECT_EQ(lA / rA, l / r );
                lA /= rA;
                EXPECT_EQ(lA, l / r);
            } else {
                const mpz_class result = -1 * ((l * -1) / r); // Encountered some errors in Cryptopp library LOL!!!
                EXPECT_EQ(lA / rA, result);
                lA /= rA;
                EXPECT_EQ(lA, result);
            }
        }

        /* Built-in types. */
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 200);
            const auto mod = Generation::getRandom<unsigned long>();

            Aesi<N> aeu = value;
            EXPECT_EQ(aeu / mod, value / mod);

            aeu /= mod;
            EXPECT_EQ(aeu, value / mod);
        }
    });
}