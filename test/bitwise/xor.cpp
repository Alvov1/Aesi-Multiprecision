#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include "../generation.h"

TEST(Unsigned_Bitwise, XOR) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            auto l = Generation::getRandom(N - 20),
                r = Generation::getRandom(N - 20);
            Aeu<N> left = l, right = r;
            EXPECT_EQ(left ^ right, l ^ r);

            l = Generation::getRandom(N - 20),
                r = Generation::getRandom(N - 20);
            left = l, right = r; left ^= right;
            EXPECT_EQ(left, l ^ r);
        }
    });
}