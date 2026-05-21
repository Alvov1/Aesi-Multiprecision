#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_SquareRoot, SquareRoot) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (unsigned i = 0; i < testsAmount; ++i) {
            const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(N - 20);
            const Aesi<N> m = value;
            if(i % 2 == 0) {
                mpz_class expected; mpz_sqrt(expected.get_mpz_t(), value.get_mpz_t());
                EXPECT_EQ(m.squareRoot(), expected);
            } else EXPECT_EQ(m.squareRoot(), 0u);
        }
    });
    Aesi<1024> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}

TEST(Unsigned_SquareRoot, SquareRoot) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (unsigned i = 0; i < testsAmount; ++i) {
            const mpz_class value = Generation::getRandom(N - 20);
            const Aeu<N> m = value;
            mpz_class expected; mpz_sqrt(expected.get_mpz_t(), value.get_mpz_t());
            EXPECT_EQ(m.squareRoot(), expected);
        }
    });

    Aeu<1024> zero = 0u;
    EXPECT_EQ(zero.squareRoot(), 0u);
}