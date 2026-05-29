#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

template <typename T, std::size_t N>
void runDecrementLoop(const mpz_class& l) {
    T value = l;
    const std::size_t decrements = static_cast<std::size_t>(rand()) % 100;
    for (std::size_t j = 0; j < decrements * 2; j += 2) {
        EXPECT_EQ(value--, l - j);
        EXPECT_EQ(--value, l - j - 2);
    }
    EXPECT_EQ(value, l - decrements * 2);
}

TEST(Unsigned_Subtraction, Basic) {
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

TEST(Signed_Subtraction, Basic) {
    Aesi128 zero = 0l;
    Aesi128 m0 = 14377898l;
    EXPECT_EQ(m0 - 0u, 14377898l);
    EXPECT_EQ(m0 + 0u, 14377898l);
    EXPECT_EQ(m0 + zero, 14377898l);
    EXPECT_EQ(m0 - zero, 14377898l);
    EXPECT_EQ(m0 + +zero, 14377898l);
    EXPECT_EQ(m0 - +zero, 14377898l);
    m0 -= 0u; EXPECT_EQ(m0, 14377898l);
    m0 -= zero; EXPECT_EQ(m0, 14377898l);
    m0 -= -zero; EXPECT_EQ(m0, 14377898l);
    m0 -= +zero; EXPECT_EQ(m0, 14377898l);

    Aesi128 m1 = -42824647l;
    EXPECT_EQ(m1 - 0u, -42824647l);
    EXPECT_EQ(m1 + 0u, -42824647l);
    EXPECT_EQ(m1 + zero, -42824647l);
    EXPECT_EQ(m1 - zero, -42824647l);
    EXPECT_EQ(m1 + +zero, -42824647l);
    EXPECT_EQ(m1 - +zero, -42824647l);
    m1 -= 0u; EXPECT_EQ(m1, -42824647l);
    m1 -= zero; EXPECT_EQ(m1, -42824647l);
    m1 -= +zero; EXPECT_EQ(m1, -42824647l);
}

TEST(Unsigned_Subtraction, Huge) {
    Generation::forEachPrecision([]<std::size_t N>() {
        Generation::runCompositeTest<Aeu, N>(N - 5, N - 32,
            [](auto a, auto b) { return a - b; },
            [](auto& a, const auto& b) { a -= b; });
        /* Built-in types. */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 10);
            const auto subU = Generation::getRandom<unsigned>();
            Aeu<N> aeu = value;
            EXPECT_EQ(aeu - subU, value - subU);
            aeu -= subU;
            EXPECT_EQ(aeu, value - subU);
        }
    });
}

TEST(Signed_Subtraction, Huge) {
    Generation::forEachPrecision([]<std::size_t N>() {
        Generation::runSignedCompositeTest<Aesi, N>(N - 110, N - 110,
            [](auto a, auto b) { return a - b; },
            [](auto& a, const auto& b) { a -= b; });
        /* Built-in types (signed). */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class value = Generation::getRandom(N - 200);
            const auto sub = Generation::getRandom<long>();
            Aesi<N> aeu = value;
            EXPECT_EQ((aeu - sub) - sub, (value - sub) - sub);
            aeu -= sub; aeu -= sub;
            EXPECT_EQ(aeu, (value - sub) - sub);
        }
    });
}

TEST(Unsigned_Subtraction, Decrement) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i)
            runDecrementLoop<Aeu<N>, N>(Generation::getRandom(N - 110));
    });

    Aeu<1024> test = 1u; --test;
    EXPECT_EQ(test, 0u);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1u);

    test = 1u; test--;
    EXPECT_EQ(test, 0u);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1u);

    Aeu<512> test2 = "0x"
    "0000_0000'0000_0000'0000_0000'0000_0000"
    "0000_0000'0000_0000'0000_0000'0000_0000"
    "0000_0000'0000_0000'0000_0000'0000_0000"
    "0000_0001'0000_0000'0000_0000'0000_0000";
    EXPECT_EQ(--test2, "0x"
        "0000_0000'0000_0000'0000_0000'0000_0000"
        "0000_0000'0000_0000'0000_0000'0000_0000"
        "0000_0000'0000_0000'0000_0000'0000_0000"
        "0000_0000'FFFF_FFFF'FFFF_FFFF'FFFF_FFFF");
}

TEST(Signed_Subtraction, Decrement) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i)
            runDecrementLoop<Aesi<N>, N>((i % 2 == 0 ? 1 : -1) * Generation::getRandom(N - 110));
    });

    Aesi<1024> test = 0; --test;
    EXPECT_EQ(test, -1);
    EXPECT_FALSE(test.isZero());
    EXPECT_LT(test, 0);

    test = 0; test--;
    EXPECT_EQ(test, -1);
    EXPECT_FALSE(test.isZero());
    EXPECT_LT(test, 0);

    test = 1; --test;
    EXPECT_EQ(test, 0);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1);

    test = 1; test--;
    EXPECT_EQ(test, 0);
    EXPECT_TRUE(test.isZero());
    EXPECT_LT(test, 1);
}
