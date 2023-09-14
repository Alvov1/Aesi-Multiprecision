#include <gtest/gtest.h>
#include "../../Multiprecision.h"

TEST(Initialization, ZeroInitialization) {
    Multiprecision m0 {};
    EXPECT_EQ(m0, 0);

    Multiprecision m1(0);
    EXPECT_EQ(m1, 0);

    Multiprecision m2 = 0;
    EXPECT_EQ(m2, 0);

    Multiprecision m3 = Multiprecision(0);
    EXPECT_EQ(m3, 0);

    Multiprecision m4 = {};
    EXPECT_EQ(m4, 0);
}

TEST(Initialization, BasicInitialization) {
    Multiprecision i01 = 1;
    EXPECT_EQ(i01, 1);

    Multiprecision i02 = -1, i03 = 127, i04 = -127, i05 = -128, i06 = +127;
    EXPECT_EQ(i02, -1);
    EXPECT_EQ(i03, 127);
    EXPECT_EQ(i04, -127);
    EXPECT_EQ(i05, -128);
    EXPECT_EQ(i06, 127);
}

TEST(Initialization, SmallCharArrayInitialization) {
    Multiprecision ten = "10", negativeTen = "-10";
    EXPECT_EQ(ten, 10);
    EXPECT_EQ(negativeTen, -10);

    Multiprecision hexTenLC = "0xa", hexTenHC = "0xA", negativeHexTenLC = "-0xa", negativeHexTenHC = "-0xA";
    EXPECT_EQ(hexTenLC, 10);
    EXPECT_EQ(hexTenHC, 10);
    EXPECT_EQ(negativeHexTenLC, -10);
    EXPECT_EQ(negativeHexTenHC, -10);
}

TEST(Initialization, StringStringViewInitialization) {
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    Multiprecision d0 = "489133282872437279"s;
    EXPECT_EQ(d0, 489133282872437279);
    Multiprecision d1 = "63018038201"sv;
    EXPECT_EQ(d1, 63018038201);
    Multiprecision d2 = "-489133282872437279"s;
    EXPECT_EQ(d2, -489133282872437279);
    Multiprecision d3 = "-63018038201"sv;
    EXPECT_EQ(d3, -63018038201);

    Multiprecision b0 = "0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b0, 489133282872437279);
    Multiprecision b1 = "0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b1, 63018038201);
    Multiprecision b2 = "-0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b2, -489133282872437279);
    Multiprecision b3 = "-0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b3, -63018038201);

    Multiprecision o0 = "0o106274176273174613"s;
    EXPECT_EQ(o0, 2475842268363147);
    Multiprecision o1 = "0o642054234601645202742"sv;
    EXPECT_EQ(o1, 7531577461358003682);
    Multiprecision o2 = "-0o106274176273174613"s;
    EXPECT_EQ(o2, -2475842268363147);
    Multiprecision o3 = "-0o642054234601645202742"sv;
    EXPECT_EQ(o3, -7531577461358003682);

    Multiprecision h0 = "0x688589CC0E9505E2"s;
    EXPECT_EQ(h0, 7531577461358003682);
    Multiprecision h1 = "0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h1, 1066340417491710);
    Multiprecision h2 = "-0x688589CC0E9505E2"s;
    EXPECT_EQ(h2, -7531577461358003682);
    Multiprecision h3 = "-0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h3, -1066340417491710);
    Multiprecision h4 = "0x688589cc0e9505e2"s;
    EXPECT_EQ(h4, 7531577461358003682);
    Multiprecision h5 = "0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h5, 1066340417491710);
    Multiprecision h6 = "-0x688589cc0e9505e2"s;
    EXPECT_EQ(h6, -7531577461358003682);
    Multiprecision h7 = "-0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h7, -1066340417491710);
}

TEST(Initialization, Binary) {
    Multiprecision m0 = 0b1111111111111111111111111111111111111111111111111111111111111111;
    EXPECT_EQ(m0, 18446744073709551615ULL);

    Multiprecision m1 = -0b100001100011011110111101000001011010111101101;
    EXPECT_EQ(m1, -18446744073709);

    Multiprecision m2 = "0b11011001001110000000010000100010101011011101010101000011111";
    EXPECT_EQ(m2, 489133282872437279);

    Multiprecision m3 = "-0b111010101100001010101111001110111001";
    EXPECT_EQ(m3, -63018038201);

    Multiprecision m4 = "0b1010101010101010101010101";
    EXPECT_EQ(m4, 22369621);

    Multiprecision m5 = "-0b10101001100010101001";
    EXPECT_EQ(m5, -694441);
}

TEST(Initialization, Decimal) {
    Multiprecision m0 = 99194853094755497;
    EXPECT_EQ(m0, 99194853094755497);

    Multiprecision m1 = -2971215073;
    EXPECT_EQ(m1, -2971215073);

    Multiprecision m2 = "2475842268363147";
    EXPECT_EQ(m2, 2475842268363147);

    Multiprecision m3 = "-7531577461358003682";
    EXPECT_EQ(m3, -7531577461358003682);

    Multiprecision d = "18446744073709551615";
    EXPECT_EQ(d, 18446744073709551615ULL);
}

TEST(Initialization, Octal) {
    Multiprecision m0 = 05403223057620506251;
    EXPECT_EQ(m0, 99194853094755497);

    Multiprecision m1 = 026106222341;
    EXPECT_EQ(m1, 2971215073);

    Multiprecision m2 = "0o106274176273174613";
    EXPECT_EQ(m2, 2475842268363147);

    Multiprecision m3 = "-0o642054234601645202742";
    EXPECT_EQ(m3, -7531577461358003682);

    Multiprecision o = "0o1777777777777777777777";
    EXPECT_EQ(o, 18446744073709551615ULL);
}

TEST(Initialization, Hexadecimal) {
    Multiprecision m0 = 0xFFFFFFFFFFFFFFFF;
    EXPECT_EQ(m0, 18446744073709551615ULL);

    Multiprecision m1 = 0x191347024000932;
    EXPECT_EQ(m1, 112929121905936690);

    Multiprecision m2 = "-0x688589CC0E9505E2";
    EXPECT_EQ(m2, -7531577461358003682);

    Multiprecision m3 = "-0x3C9D4B9CB52FE";
    EXPECT_EQ(m3, -1066340417491710);

    Multiprecision m4 = "-0x688589cc0e9505e2";
    EXPECT_EQ(m4, -7531577461358003682);

    Multiprecision m5 = "-0x3c9d4b9cb52fe";
    EXPECT_EQ(m5, -1066340417491710);

    Multiprecision m6 = "-0x688589Cc0E9505e2";
    EXPECT_EQ(m6, -7531577461358003682);

    Multiprecision m7 = "-0x3C9d4B9Cb52Fe";
    EXPECT_EQ(m7, -1066340417491710);

    Multiprecision m8 = "0x688589CC0E9505E2";
    EXPECT_EQ(m8, 7531577461358003682);

    Multiprecision m9 = "0x3C9D4B9CB52FE";
    EXPECT_EQ(m9, 1066340417491710);
}