#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Addition, Small) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 + small3, -9170);
    EXPECT_EQ(small3 + small1, -9170);

    EXPECT_EQ(small2 + small1, -4249);
    EXPECT_EQ(small1 + small2, -4249);

    EXPECT_EQ(small2 + small3, 3565);
    EXPECT_EQ(small3 + small2, 3565);

    EXPECT_EQ(small2 + small4, 6566);
    EXPECT_EQ(small4 + small2, 6566);
}

TEST(Addition, Huge) {
    Multiprecision huge = "8683317618811886495518194401279999999", negativeHuge = "-8683317618811886495518194401279999999";
    EXPECT_EQ(huge + negativeHuge, 0);
    EXPECT_EQ(huge + huge + huge, "26049952856435659486554583203839999997");

    Multiprecision huge2 = "26049952856435659486554583203839999997";
    huge += huge2;
    EXPECT_EQ(huge, "34733270475247545982072777605119999996");

    Multiprecision huge3 = "-489133282872437279";
    huge2 += huge3;
    EXPECT_EQ(huge2, "26049952856435659486065449920967562718");
}