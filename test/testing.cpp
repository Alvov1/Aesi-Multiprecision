#include <gtest/gtest.h>
#include "../Multiprecision.h"

TEST(Multiprecision, DefaultInitialization) {
    Multiprecision m {};
    EXPECT_EQ(m, 0);
}

TEST(Multiprecision, BasicInitialization) {
    Multiprecision i00 = 0;
    EXPECT_EQ(i00, 0);

    Multiprecision i01 = 1;
    EXPECT_EQ(i01, 1);

    Multiprecision i02 = -1, i03 = 127, i04 = -127, i05 = -128, i06 = +127;
    EXPECT_EQ(i01, i00 + 1);
    EXPECT_EQ(i02, -i01);
    EXPECT_EQ(i03, -i04);
    EXPECT_EQ(-(i05 + 1), i06);
}

TEST(Multiprecision, Bitness16) {
    Multiprecision i10 = 0, i11 = 1, i12 = -1, i13 = 32767, i14 = -32767, i15 = -32768, i16 = +32767;
    EXPECT_EQ(i11, i10 + 1);
    EXPECT_EQ(i12, -i11);
    EXPECT_EQ(i13, -i14);
    EXPECT_EQ(-(i15 + 1), i16);
}

TEST(Multiprecision, Bitness32) {
    Multiprecision i20 = 0, i21 = 1, i22 = -1, i23 = 2147483647, i24 = -2147483647, i25 = -2147483648, i26 = +2147483647;
    EXPECT_EQ(i21, i20 + 1);
    EXPECT_EQ(i22, -i21);
    EXPECT_EQ(i23, -i24);
    EXPECT_EQ(-(i25 + 1), i26);
    EXPECT_EQ(i23, (1 << 31) - 1);
}

TEST(Multiprecision, Bitness64) {
    Multiprecision i30 = 0, i31 = 1, i32 = -1, i33 = 9223372036854775807, i34 = -9223372036854775807;
    Multiprecision i35 = 9223372036854775808U; i35 += 1;
    Multiprecision i36 = +9223372036854775807;
    EXPECT_EQ(+i31, i30 + 1);
    EXPECT_EQ(i32, -i31);
    EXPECT_EQ(+i33, -i34);
    EXPECT_EQ(-(i35 + 1), i36);
    EXPECT_EQ(i33, (1ULL << 63) - 1);
}

TEST(Multiprecision, SmallStringInitialization) {
    Multiprecision ten = "10", negativeTen = "-10", hexTenLC = "0xa", hexTenHC = "0xA", negativeHexTenLC = "-0xa", negativeHexTenHC = "-0xA";
    EXPECT_EQ(ten, 10);
    EXPECT_EQ(hexTenLC, 10);
    EXPECT_EQ(hexTenHC, 10);
    EXPECT_EQ(negativeTen, -10);
    EXPECT_EQ(negativeHexTenLC, -10);
    EXPECT_EQ(negativeHexTenHC, -10);
}

TEST(Multiprecision, HugeStringInitialization) {
    using namespace std::string_view_literals;

    Multiprecision huge = "8683317618811886495518194401279999999", negativeHuge = "-8683317618811886495518194401279999999";
    std::stringstream ss1 {}; ss1 << huge << negativeHuge;
    EXPECT_EQ(ss1.str(), "8683317618811886495518194401279999999-8683317618811886495518194401279999999"sv);

    Multiprecision hugeHexLC = "0x688589cc0e9505e2f2fee557fffffff", hugeHexHC = "0x688589CC0E9505E2F2FEE557FFFFFFF";
    std::stringstream ss2 {}; ss2 << hugeHexLC << hugeHexHC;
    EXPECT_EQ(ss2.str(), "0x688589cc0e9505e2f2fee557fffffff0x688589CC0E9505E2F2FEE557FFFFFFF"sv);

    Multiprecision hugeNegativeHexLC = "-0x688589cc0e9505e2f2fee557fffffff", hugeNegativeHexHC = "-0x688589CC0E9505E2F2FEE557FFFFFFF";
    std::stringstream ss3 {}; ss3 << hugeNegativeHexLC << hugeNegativeHexHC;
    EXPECT_EQ(ss3.str(), "-0x688589cc0e9505e2f2fee557fffffff-0x688589CC0E9505E2F2FEE557FFFFFFF"sv);
}