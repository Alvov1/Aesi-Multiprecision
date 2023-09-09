#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Multiplication, Small) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 * small3, 5757576); // --
    EXPECT_EQ(small2 * small1, -36031556); // +-
    EXPECT_EQ(small3 * small2, -2876754);  // +-
    EXPECT_EQ(small2 * small4, 9856489);  // ++
}

TEST(Multiplication, Huge) {
    Multiprecision m1 = "347332704752475459", m2 = "2777605119999997", m3 = "-86833176188118864", m4 = "-18194401279999999";
    EXPECT_EQ(m1 * m2, "964753099063923125594635822573623");        // ++
    EXPECT_EQ(m2 * m3, "-241188274765980779315455115643408");       // +-
    EXPECT_EQ(m3 * m1, "-30160001947667571815110868534958576");     // -+
    EXPECT_EQ(m3 * m4, "1579877651983575293120569731881136");       // --
}