#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Subtraction, Small) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 - small3, -7814);
    EXPECT_EQ(small3 - small1, 7814);

    EXPECT_EQ(small2 - small1, 12735);
    EXPECT_EQ(small1 - small2, -12735);

    EXPECT_EQ(small2 - small3, 4921);
    EXPECT_EQ(small3 - small2, -4921);

    EXPECT_EQ(small2 - small4, 1920);
    EXPECT_EQ(small4 - small2, -1920);
}

TEST(Subtraction, Huge) {
    Multiprecision huge = "34733270475247545982072777605119999996", greater = "34733270475247545982072777605119999997";
    EXPECT_EQ(huge - greater, 0);

    Multiprecision mega = "8683317618811886495518194401279999999";
    EXPECT_EQ(mega - huge - huge, "-60783223331683205468627360808959999993");

    mega -= huge;
    EXPECT_EQ(mega, "-26049952856435659486554583203839999997");

    Multiprecision huge2 = "-19175002942688032928599";
    huge -= huge2;
    EXPECT_EQ(huge2, "34733270475247526807069834917087071397");
}