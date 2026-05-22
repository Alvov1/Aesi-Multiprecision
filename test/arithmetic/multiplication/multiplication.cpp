#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

template <typename T128>
void testMultiplicationBasic() {
    T128 zero = 0u, one = 1u;
    T128 m0 = 10919396u;
    EXPECT_EQ(m0 * 0u, 0u);
    EXPECT_EQ(0u * m0, 0u);
    EXPECT_EQ(m0 * 1u, 10919396u);
    EXPECT_EQ(1u * m0, 10919396u);

    EXPECT_EQ(m0 * zero, 0u);
    EXPECT_EQ(m0 * +zero, 0u);
    EXPECT_EQ(m0 * one, 10919396u);
    EXPECT_EQ(m0 * +one, 10919396u);

    EXPECT_EQ(zero * m0, 0u);
    EXPECT_EQ(zero * +m0, 0u);
    EXPECT_EQ(one * m0, 10919396u);
    EXPECT_EQ(one * +m0, 10919396u);

    EXPECT_EQ(+zero * m0, 0u);
    EXPECT_EQ(+zero * +m0, 0u);
    EXPECT_EQ(+one * m0, 10919396u);
    EXPECT_EQ(+one * +m0, 10919396u);

    m0 *= 0u; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    m0 *= +zero; EXPECT_EQ(m0, 0u); m0 = 10919396u;
    zero *= m0; EXPECT_EQ(zero, 0u);
    zero *= +m0; EXPECT_EQ(zero, 0u);

    m0 *= 1u; EXPECT_EQ(m0, 10919396u);

    m0 *= one; EXPECT_EQ(m0, 10919396u);
    m0 *= +one; EXPECT_EQ(m0, 10919396u);
    one *= m0; EXPECT_EQ(one, 10919396u); one = 1u;
    one *= +m0; EXPECT_EQ(one, 10919396u);
}

TEST(Unsigned_Multiplication, Basic) { testMultiplicationBasic<Aeu128>(); }
TEST(Signed_Multiplication, Basic)   { testMultiplicationBasic<Aesi128>(); }

TEST(Unsigned_Multiplication, Huge) {
    Generation::forEachPrecision([]<std::size_t N>() {
        Generation::runCompositeTest<Aeu, N>(N / 2 - 10, N / 2 - 20,
            [](auto a, auto b) { return a * b; },
            [](auto& a, const auto& b) { a *= b; });

        /* Built-in types. */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 200);
            const auto factorU = Generation::getRandom<unsigned>();

            Aeu<N> aeu = value;
            EXPECT_EQ(aeu * factorU, value * factorU);

            aeu *= factorU;
            EXPECT_EQ(aeu, value * factorU);
        }
    });
}

TEST(Signed_Multiplication, Huge) {
    Generation::forEachPrecision([]<std::size_t N>() {
        Generation::runSignedCompositeTest<Aesi, N>(N / 2 - 110, N / 2 - 110,
            [](auto a, auto b) { return a * b; },
            [](auto& a, const auto& b) { a *= b; });

        /* Built-in types. */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 200);
            const auto factor = Generation::getRandom<unsigned>();

            Aesi<N> aeu = value;
            EXPECT_EQ((aeu * factor) * factor, (value * factor) * factor);

            aeu *= factor; aeu *= factor;
            EXPECT_EQ(aeu, (value * factor) * factor);
        }
    });
}
