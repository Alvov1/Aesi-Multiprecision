#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_OddEven, Basic) {
    Aesi256 zero = 0u; EXPECT_EQ(zero.isOdd(), 0); EXPECT_EQ(zero.isEven(), 1);

    constexpr auto testsAmount = 2, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 10);
        Aesi<blocksNumber * 32> aesi = value;
        EXPECT_EQ(value.IsEven(), aesi.isEven());
        EXPECT_EQ(value.IsOdd(), aesi.isOdd());
    }
}

TEST(Unsigned_OddEven, Basic) {
    Aeu256 zero = 0u; EXPECT_EQ(zero.isOdd(), 0); EXPECT_EQ(zero.isEven(), 1);

    constexpr auto testsAmount = 2048, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 10);
        Aeu<blocksNumber * 32> aeu = value;
        EXPECT_EQ(value.IsEven(), aeu.isEven());
        EXPECT_EQ(value.IsOdd(), aeu.isOdd());
    }
}