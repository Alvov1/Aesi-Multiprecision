#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_OddEven, Basic) {
    Aesi256 zero = 0u; EXPECT_EQ(zero.isOdd(), 0); EXPECT_EQ(zero.isEven(), 1);

    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 10);

            Aesi<N> aesi = value;
            EXPECT_EQ(mpz_even_p(value.get_mpz_t()), aesi.isEven());
            EXPECT_EQ(mpz_odd_p(value.get_mpz_t()), aesi.isOdd());
        }
    });
}

TEST(Unsigned_OddEven, Basic) {
    Aeu256 zero = 0u; EXPECT_EQ(zero.isOdd(), 0); EXPECT_EQ(zero.isEven(), 1);

    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 10);

            Aeu<N> aeu = value;
            EXPECT_EQ(mpz_even_p(value.get_mpz_t()), aeu.isEven());
            EXPECT_EQ(mpz_odd_p(value.get_mpz_t()), aeu.isOdd());
        }
    });
}