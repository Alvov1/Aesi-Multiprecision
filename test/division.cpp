#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Division, Small) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 / small3, 12); // --
    EXPECT_EQ(small2 / small3, -6); // +-
    EXPECT_EQ(small1 / small4, -3);  // -+
    EXPECT_EQ(small2 / small4, 1);  // ++
}

TEST(Division, Huge) {
    Multiprecision m1 = "347332704752475459", m2 = "-2777605119999997", m3 = "8683317", m4 = "-181944012";
    EXPECT_EQ(m1 / m3, "40000002850");  // ++
    EXPECT_EQ(m2 / m3, "-319878350");   // -+
    EXPECT_EQ(m1 / m4, "-1909008716");  // +-
    EXPECT_EQ(m2 / m4, "15266262");     // --
}