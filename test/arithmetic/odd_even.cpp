#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"

TEST(Signed, OddEven) { EXPECT_TRUE(false); }

TEST(Unsigned, OddEven) {
    Aeu256 zero = 0u; EXPECT_EQ(zero.isOdd(), 0); EXPECT_EQ(zero.isEven(), 1);
    Aeu256 b0 = "1504499442701679703283353229512838841."; EXPECT_EQ(b0.isOdd(), 1); EXPECT_EQ(b0.isEven(), 0);
}