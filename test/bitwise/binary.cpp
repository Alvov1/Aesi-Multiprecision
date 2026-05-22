#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include "../generation.h"

template <std::size_t N, typename Op, typename AssignOp>
void runBitwiseTest(Op op, AssignOp assignOp) {
    constexpr auto testsAmount = 256;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto l = Generation::getRandom(N - 20),
                r = Generation::getRandom(N - 20);
        Aeu<N> left = l, right = r;
        EXPECT_EQ(op(left, right), op(l, r));

        l = Generation::getRandom(N - 20);
        r = Generation::getRandom(N - 20);
        left = l; right = r; assignOp(left, right);
        EXPECT_EQ(left, op(l, r));
    }
}

TEST(Unsigned_Bitwise, AND) {
    Generation::forEachPrecision([]<std::size_t N>() {
        runBitwiseTest<N>(
            [](auto a, auto b) { return a & b; },
            [](auto& a, const auto& b) { a &= b; });
    });
}

TEST(Unsigned_Bitwise, OR) {
    Generation::forEachPrecision([]<std::size_t N>() {
        runBitwiseTest<N>(
            [](auto a, auto b) { return a | b; },
            [](auto& a, const auto& b) { a |= b; });
    });
}

TEST(Unsigned_Bitwise, XOR) {
    Generation::forEachPrecision([]<std::size_t N>() {
        runBitwiseTest<N>(
            [](auto a, auto b) { return a ^ b; },
            [](auto& a, const auto& b) { a ^= b; });
    });
}
