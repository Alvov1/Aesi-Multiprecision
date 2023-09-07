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

    i00 + 1;

    Multiprecision i02 = -1, i03 = 127, i04 = -127, i05 = -128, i06 = +127;
    i05 = 0;
    i05 + 1;
    EXPECT_EQ(i01, i00 + 1);
    EXPECT_EQ(i02, -i01);
    EXPECT_EQ(i03, -i04);
    EXPECT_EQ(-(i05 + 1), i06);
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

TEST(Multiprecision, StringInitializationDifferentNotations) {
    Multiprecision binPositive = "0b1010101010101010101010101", binNegative = "-0b10101001100010101001";
    EXPECT_EQ(binPositive, 22369621);
    EXPECT_EQ(binNegative, -694441);

    Multiprecision octPositive = "0106274176273174613", octNegative = "-0642054234601645202742";
    EXPECT_EQ(octPositive, 2475842268363147);
    EXPECT_EQ(octNegative, -7531577461358003682);

    Multiprecision hexPositive = "0x191347024000932", hexNegative = "-0x1066340417491710";
    EXPECT_EQ(hexPositive, 112929121905936690);
    EXPECT_EQ(hexNegative, -1181689144406513424);
}

TEST(Multiprecision, Display) {
    Multiprecision complex = "340282366920938463463374607435256044433";
    std::stringstream notations {}; notations << std::oct << complex << std::dec << complex << std::hex << complex;
    EXPECT_EQ(notations.str(), "4000000000000000000000000000000031771015621\n340282366920938463463374607435256044433\n1000000000000000000000000CFE41B91");
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

TEST(Multiprecision, HugeStringInitialization) {
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    Multiprecision huge = "8683317618811886495518194401279999999", negativeHuge = "-8683317618811886495518194401279999999";
    std::stringstream ss1 {}; ss1 << huge << negativeHuge;
    EXPECT_EQ(ss1.str(), "8683317618811886495518194401279999999-8683317618811886495518194401279999999");

    Multiprecision hugeHexLC = "0x688589cc0e9505e2f2fee557fffffff", hugeHexHC = "0x688589CC0E9505E2F2FEE557FFFFFFF";
    std::stringstream ss2 {}; ss2 << hugeHexLC << hugeHexHC;
    EXPECT_EQ(ss2.str(), "0x688589cc0e9505e2f2fee557fffffff0x688589CC0E9505E2F2FEE557FFFFFFF");

    Multiprecision hugeNegativeHexLC = "-0x688589cc0e9505e2f2fee557fffffff", hugeNegativeHexHC = "-0x688589CC0E9505E2F2FEE557FFFFFFF";
    std::stringstream ss3 {}; ss3 << hugeNegativeHexLC << hugeNegativeHexHC;
    EXPECT_EQ(ss3.str(), "-0x688589cc0e9505e2f2fee557fffffff-0x688589CC0E9505E2F2FEE557FFFFFFF");

    Multiprecision str2 = "8683317618811886495518194401279999999"sv, str3 = "8683317618811886495518194401279999999"s;
    std::stringstream ss4 {}; ss4 << str2 << str3;
    EXPECT_EQ(ss4.str(), "86833176188118864955181944012799999998683317618811886495518194401279999999");
}

TEST(Multiprecision, Addition) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 + small3, -9170);
    EXPECT_EQ(small3 + small1, -9170);

    EXPECT_EQ(small2 + small1, -4249);
    EXPECT_EQ(small1 + small2, -4249);

    EXPECT_EQ(small2 + small3, 3565);
    EXPECT_EQ(small3 + small2, 3565);

    EXPECT_EQ(small2 + small4, 6566);
    EXPECT_EQ(small4 + small2, 6566);

    Multiprecision huge = "8683317618811886495518194401279999999", negativeHuge = "-8683317618811886495518194401279999999";
    EXPECT_EQ(huge + negativeHuge, 0);
    EXPECT_EQ(huge + huge + huge, "26049952856435659486554583203839999997");

    Multiprecision huge2 = "26049952856435659486554583203839999997";
    huge += huge2;
    EXPECT_EQ(huge, "34733270475247545982072777605119999996");
}

TEST(Multiprecision, Subtraction) {
    Multiprecision small1 = "-8492", small2 = "4243", small3 = "-678", small4 = "2323";
    EXPECT_EQ(small1 - small3, -7814);
    EXPECT_EQ(small3 - small1, 7814);

    EXPECT_EQ(small2 - small1, 12735);
    EXPECT_EQ(small1 - small2, -12735);

    EXPECT_EQ(small2 - small3, 4921);
    EXPECT_EQ(small3 - small2, -4921);

    EXPECT_EQ(small2 - small4, 1920);
    EXPECT_EQ(small4 - small2, -1920);

    Multiprecision huge = "34733270475247545982072777605119999996", greater = "34733270475247545982072777605119999997";
    EXPECT_EQ(huge - greater, 0);

    Multiprecision mega = "8683317618811886495518194401279999999";
    EXPECT_EQ(mega - huge - huge, "-60783223331683205468627360808959999993");

    mega -= huge;
    EXPECT_EQ(mega, "-26049952856435659486554583203839999997");
}

TEST(Multiprecision, Complex) {
    Multiprecision a("123456789012345678901234567890");
    Multiprecision b("987654321098765432109876543210");
    EXPECT_EQ(a + b, "111111111011111111011111111100");
    EXPECT_EQ(a - b, "-86419753108641975310864197520");
    EXPECT_EQ(a * b, "12193263112862162186216216216919326311286216216216");
    EXPECT_EQ(a / b, "124");

    Multiprecision c("123");
    Multiprecision d("123456789012345678901234567890");
    EXPECT_EQ(d / c, "1003712536650411410484961790650");

    Multiprecision e("123456789012345678901234567890");
    Multiprecision f("987");
    EXPECT_EQ(e * f, "121932631128621621862162162170");

    Multiprecision g("-123456789012345678901234567890");
    Multiprecision h("-987654321098765432109876543210");
    EXPECT_EQ(g + h, "-111111111011111111011111111100");
    EXPECT_EQ(g - h, "86419753108641975310864197520");
    EXPECT_EQ(g * h, "12193263112862162186216216216919326311286216216216");
    EXPECT_EQ(g / h, "124");

    Multiprecision i("0");
    Multiprecision j("123456789012345678901234567890");
    EXPECT_EQ(j / i, "0");

    Multiprecision r("987654321098765432109876543210");
    Multiprecision s("123456789012345678901234567890");
    EXPECT_EQ(r % s, "987654321098765432109876543210");

    Multiprecision t("-987654321098765432109876543210");
    Multiprecision u("123456789012345678901234567890");
    EXPECT_EQ(t % u, "-987654321098765432109876543210");

    Multiprecision complex = ((((a + b) * c / d % e) - f * 4096) / 2) + (g * h - j) % ((r + s + t + u) / 40);
    EXPECT_EQ(complex, "");
}