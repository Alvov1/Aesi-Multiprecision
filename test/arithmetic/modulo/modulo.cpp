#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

template<template<std::size_t> class T, std::size_t N>
void testModuloComposite(std::size_t lBits, std::size_t rBits) {
    constexpr auto testsAmount = 256;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto l = Generation::getRandom(lBits),
                r = Generation::getRandom(rBits);
        T<N> lA = l, rA = r;
        EXPECT_EQ(lA % rA, l % r);
        lA %= rA;
        EXPECT_EQ(lA, l % r);
    }
}

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
        testModuloComposite<Aeu, N>(N - 5, N / 2 - 32);

        /* Built-in types. */
        constexpr auto testsAmount = 256;
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
    Generation::forEachPrecision([]<std::size_t N>() {
        testModuloComposite<Aesi, N>(N - 110, N / 2 - 110);

        /* Built-in types. */
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 200);
            const auto mod = Generation::getRandom<unsigned long>();

            Aesi<N> aesi = value;
            EXPECT_EQ(aesi % mod, value % mod);

            aesi %= mod; aesi %= mod;
            EXPECT_EQ(aesi, value % mod);
        }
    });
}