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

    Multiprecision m5 = "0";
    EXPECT_EQ(m5, 0);

    Multiprecision m6 = "0.";
    EXPECT_EQ(m6, 0);

    Multiprecision m7 = "Somebody once told me The world is gonna roll me I ain't the sharpest tool in the shed";
    EXPECT_EQ(m7, 0);
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

TEST(Multiplication, DifferentPrecisions) {
    long long iValue0 = 3218136187561313218;
    Multiprecision<64> o00 = iValue0;
    Multiprecision<96> o01 = iValue0; Multiprecision<128> o02 = iValue0; Multiprecision<160> o03 = iValue0; Multiprecision<192> o04 = iValue0; Multiprecision<224> o05 = iValue0; Multiprecision<256> o06 = iValue0; Multiprecision<288> o07 = iValue0; Multiprecision<320> o08 = iValue0; Multiprecision<352> o09 = iValue0;
    Multiprecision<384> o010 = iValue0; Multiprecision<416> o011 = iValue0; Multiprecision<448> o012 = iValue0; Multiprecision<480> o013 = iValue0; Multiprecision<512> o014 = iValue0; Multiprecision<544> o015 = iValue0; Multiprecision<576> o016 = iValue0; Multiprecision<608> o017 = iValue0; Multiprecision<640> o018 = iValue0;
    Multiprecision<672> o019 = iValue0; Multiprecision<704> o020 = iValue0; Multiprecision<736> o021 = iValue0; Multiprecision<768> o022 = iValue0; Multiprecision<800> o023 = iValue0; Multiprecision<832> o024 = iValue0; Multiprecision<864> o025 = iValue0; Multiprecision<896> o026 = iValue0; Multiprecision<928> o027 = iValue0;
    Multiprecision<960> o028 = iValue0; Multiprecision<992> o029 = iValue0; Multiprecision<1024> o030 = iValue0; Multiprecision<1056> o031 = iValue0; Multiprecision<1088> o032 = iValue0; Multiprecision<1120> o033 = iValue0; Multiprecision<1152> o034 = iValue0; Multiprecision<1184> o035 = iValue0; Multiprecision<1216> o036 = iValue0;
    Multiprecision<1248> o037 = iValue0; Multiprecision<1280> o038 = iValue0; Multiprecision<1312> o039 = iValue0;
    EXPECT_EQ(o00, iValue0);
    EXPECT_EQ(o01, iValue0); EXPECT_EQ(o02, iValue0); EXPECT_EQ(o03, iValue0); EXPECT_EQ(o04, iValue0); EXPECT_EQ(o05, iValue0); EXPECT_EQ(o06, iValue0); EXPECT_EQ(o07, iValue0); EXPECT_EQ(o08, iValue0); EXPECT_EQ(o09, iValue0);
    EXPECT_EQ(o010, iValue0); EXPECT_EQ(o011, iValue0); EXPECT_EQ(o012, iValue0); EXPECT_EQ(o013, iValue0); EXPECT_EQ(o014, iValue0); EXPECT_EQ(o015, iValue0); EXPECT_EQ(o016, iValue0); EXPECT_EQ(o017, iValue0); EXPECT_EQ(o018, iValue0);
    EXPECT_EQ(o019, iValue0); EXPECT_EQ(o020, iValue0); EXPECT_EQ(o021, iValue0); EXPECT_EQ(o022, iValue0); EXPECT_EQ(o023, iValue0); EXPECT_EQ(o024, iValue0); EXPECT_EQ(o025, iValue0); EXPECT_EQ(o026, iValue0); EXPECT_EQ(o027, iValue0);
    EXPECT_EQ(o028, iValue0); EXPECT_EQ(o029, iValue0); EXPECT_EQ(o030, iValue0); EXPECT_EQ(o031, iValue0); EXPECT_EQ(o032, iValue0); EXPECT_EQ(o033, iValue0); EXPECT_EQ(o034, iValue0); EXPECT_EQ(o035, iValue0); EXPECT_EQ(o036, iValue0);
    EXPECT_EQ(o037, iValue0); EXPECT_EQ(o038, iValue0); EXPECT_EQ(o039, iValue0);

    long long iValue2 = -380464553884730375;
    Multiprecision<64> o20 = iValue2;
    Multiprecision<96> o21 = iValue2; Multiprecision<128> o22 = iValue2; Multiprecision<160> o23 = iValue2; Multiprecision<192> o24 = iValue2; Multiprecision<224> o25 = iValue2; Multiprecision<256> o26 = iValue2; Multiprecision<288> o27 = iValue2; Multiprecision<320> o28 = iValue2; Multiprecision<352> o29 = iValue2;
    Multiprecision<384> o210 = iValue2; Multiprecision<416> o211 = iValue2; Multiprecision<448> o212 = iValue2; Multiprecision<480> o213 = iValue2; Multiprecision<512> o214 = iValue2; Multiprecision<544> o215 = iValue2; Multiprecision<576> o216 = iValue2; Multiprecision<608> o217 = iValue2; Multiprecision<640> o218 = iValue2;
    Multiprecision<672> o219 = iValue2; Multiprecision<704> o220 = iValue2; Multiprecision<736> o221 = iValue2; Multiprecision<768> o222 = iValue2; Multiprecision<800> o223 = iValue2; Multiprecision<832> o224 = iValue2; Multiprecision<864> o225 = iValue2; Multiprecision<896> o226 = iValue2; Multiprecision<928> o227 = iValue2;
    Multiprecision<960> o228 = iValue2; Multiprecision<992> o229 = iValue2; Multiprecision<1024> o230 = iValue2; Multiprecision<1056> o231 = iValue2; Multiprecision<1088> o232 = iValue2; Multiprecision<1120> o233 = iValue2; Multiprecision<1152> o234 = iValue2; Multiprecision<1184> o235 = iValue2; Multiprecision<1216> o236 = iValue2;
    Multiprecision<1248> o237 = iValue2; Multiprecision<1280> o238 = iValue2; Multiprecision<1312> o239 = iValue2;
    EXPECT_EQ(o20, iValue2);
    EXPECT_EQ(o21, iValue2); EXPECT_EQ(o22, iValue2); EXPECT_EQ(o23, iValue2); EXPECT_EQ(o24, iValue2); EXPECT_EQ(o25, iValue2); EXPECT_EQ(o26, iValue2); EXPECT_EQ(o27, iValue2); EXPECT_EQ(o28, iValue2); EXPECT_EQ(o29, iValue2);
    EXPECT_EQ(o210, iValue2); EXPECT_EQ(o211, iValue2); EXPECT_EQ(o212, iValue2); EXPECT_EQ(o213, iValue2); EXPECT_EQ(o214, iValue2); EXPECT_EQ(o215, iValue2); EXPECT_EQ(o216, iValue2); EXPECT_EQ(o217, iValue2); EXPECT_EQ(o218, iValue2);
    EXPECT_EQ(o219, iValue2); EXPECT_EQ(o220, iValue2); EXPECT_EQ(o221, iValue2); EXPECT_EQ(o222, iValue2); EXPECT_EQ(o223, iValue2); EXPECT_EQ(o224, iValue2); EXPECT_EQ(o225, iValue2); EXPECT_EQ(o226, iValue2); EXPECT_EQ(o227, iValue2);
    EXPECT_EQ(o228, iValue2); EXPECT_EQ(o229, iValue2); EXPECT_EQ(o230, iValue2); EXPECT_EQ(o231, iValue2); EXPECT_EQ(o232, iValue2); EXPECT_EQ(o233, iValue2); EXPECT_EQ(o234, iValue2); EXPECT_EQ(o235, iValue2); EXPECT_EQ(o236, iValue2);
    EXPECT_EQ(o237, iValue2); EXPECT_EQ(o238, iValue2); EXPECT_EQ(o239, iValue2);

    long long iValue4 = -2577490965723039550;
    Multiprecision<64> o40 = iValue4;
    Multiprecision<96> o41 = iValue4; Multiprecision<128> o42 = iValue4; Multiprecision<160> o43 = iValue4; Multiprecision<192> o44 = iValue4; Multiprecision<224> o45 = iValue4; Multiprecision<256> o46 = iValue4; Multiprecision<288> o47 = iValue4; Multiprecision<320> o48 = iValue4; Multiprecision<352> o49 = iValue4;
    Multiprecision<384> o410 = iValue4; Multiprecision<416> o411 = iValue4; Multiprecision<448> o412 = iValue4; Multiprecision<480> o413 = iValue4; Multiprecision<512> o414 = iValue4; Multiprecision<544> o415 = iValue4; Multiprecision<576> o416 = iValue4; Multiprecision<608> o417 = iValue4; Multiprecision<640> o418 = iValue4;
    Multiprecision<672> o419 = iValue4; Multiprecision<704> o420 = iValue4; Multiprecision<736> o421 = iValue4; Multiprecision<768> o422 = iValue4; Multiprecision<800> o423 = iValue4; Multiprecision<832> o424 = iValue4; Multiprecision<864> o425 = iValue4; Multiprecision<896> o426 = iValue4; Multiprecision<928> o427 = iValue4;
    Multiprecision<960> o428 = iValue4; Multiprecision<992> o429 = iValue4; Multiprecision<1024> o430 = iValue4; Multiprecision<1056> o431 = iValue4; Multiprecision<1088> o432 = iValue4; Multiprecision<1120> o433 = iValue4; Multiprecision<1152> o434 = iValue4; Multiprecision<1184> o435 = iValue4; Multiprecision<1216> o436 = iValue4;
    Multiprecision<1248> o437 = iValue4; Multiprecision<1280> o438 = iValue4; Multiprecision<1312> o439 = iValue4;
    EXPECT_EQ(o40, iValue4);
    EXPECT_EQ(o41, iValue4); EXPECT_EQ(o42, iValue4); EXPECT_EQ(o43, iValue4); EXPECT_EQ(o44, iValue4); EXPECT_EQ(o45, iValue4); EXPECT_EQ(o46, iValue4); EXPECT_EQ(o47, iValue4); EXPECT_EQ(o48, iValue4); EXPECT_EQ(o49, iValue4);
    EXPECT_EQ(o410, iValue4); EXPECT_EQ(o411, iValue4); EXPECT_EQ(o412, iValue4); EXPECT_EQ(o413, iValue4); EXPECT_EQ(o414, iValue4); EXPECT_EQ(o415, iValue4); EXPECT_EQ(o416, iValue4); EXPECT_EQ(o417, iValue4); EXPECT_EQ(o418, iValue4);
    EXPECT_EQ(o419, iValue4); EXPECT_EQ(o420, iValue4); EXPECT_EQ(o421, iValue4); EXPECT_EQ(o422, iValue4); EXPECT_EQ(o423, iValue4); EXPECT_EQ(o424, iValue4); EXPECT_EQ(o425, iValue4); EXPECT_EQ(o426, iValue4); EXPECT_EQ(o427, iValue4);
    EXPECT_EQ(o428, iValue4); EXPECT_EQ(o429, iValue4); EXPECT_EQ(o430, iValue4); EXPECT_EQ(o431, iValue4); EXPECT_EQ(o432, iValue4); EXPECT_EQ(o433, iValue4); EXPECT_EQ(o434, iValue4); EXPECT_EQ(o435, iValue4); EXPECT_EQ(o436, iValue4);
    EXPECT_EQ(o437, iValue4); EXPECT_EQ(o438, iValue4); EXPECT_EQ(o439, iValue4);

    long long iValue6 = -7225109388162562138;
    Multiprecision<64> o60 = iValue6;
    Multiprecision<96> o61 = iValue6; Multiprecision<128> o62 = iValue6; Multiprecision<160> o63 = iValue6; Multiprecision<192> o64 = iValue6; Multiprecision<224> o65 = iValue6; Multiprecision<256> o66 = iValue6; Multiprecision<288> o67 = iValue6; Multiprecision<320> o68 = iValue6; Multiprecision<352> o69 = iValue6;
    Multiprecision<384> o610 = iValue6; Multiprecision<416> o611 = iValue6; Multiprecision<448> o612 = iValue6; Multiprecision<480> o613 = iValue6; Multiprecision<512> o614 = iValue6; Multiprecision<544> o615 = iValue6; Multiprecision<576> o616 = iValue6; Multiprecision<608> o617 = iValue6; Multiprecision<640> o618 = iValue6;
    Multiprecision<672> o619 = iValue6; Multiprecision<704> o620 = iValue6; Multiprecision<736> o621 = iValue6; Multiprecision<768> o622 = iValue6; Multiprecision<800> o623 = iValue6; Multiprecision<832> o624 = iValue6; Multiprecision<864> o625 = iValue6; Multiprecision<896> o626 = iValue6; Multiprecision<928> o627 = iValue6;
    Multiprecision<960> o628 = iValue6; Multiprecision<992> o629 = iValue6; Multiprecision<1024> o630 = iValue6; Multiprecision<1056> o631 = iValue6; Multiprecision<1088> o632 = iValue6; Multiprecision<1120> o633 = iValue6; Multiprecision<1152> o634 = iValue6; Multiprecision<1184> o635 = iValue6; Multiprecision<1216> o636 = iValue6;
    Multiprecision<1248> o637 = iValue6; Multiprecision<1280> o638 = iValue6; Multiprecision<1312> o639 = iValue6;
    EXPECT_EQ(o60, iValue6);
    EXPECT_EQ(o61, iValue6); EXPECT_EQ(o62, iValue6); EXPECT_EQ(o63, iValue6); EXPECT_EQ(o64, iValue6); EXPECT_EQ(o65, iValue6); EXPECT_EQ(o66, iValue6); EXPECT_EQ(o67, iValue6); EXPECT_EQ(o68, iValue6); EXPECT_EQ(o69, iValue6);
    EXPECT_EQ(o610, iValue6); EXPECT_EQ(o611, iValue6); EXPECT_EQ(o612, iValue6); EXPECT_EQ(o613, iValue6); EXPECT_EQ(o614, iValue6); EXPECT_EQ(o615, iValue6); EXPECT_EQ(o616, iValue6); EXPECT_EQ(o617, iValue6); EXPECT_EQ(o618, iValue6);
    EXPECT_EQ(o619, iValue6); EXPECT_EQ(o620, iValue6); EXPECT_EQ(o621, iValue6); EXPECT_EQ(o622, iValue6); EXPECT_EQ(o623, iValue6); EXPECT_EQ(o624, iValue6); EXPECT_EQ(o625, iValue6); EXPECT_EQ(o626, iValue6); EXPECT_EQ(o627, iValue6);
    EXPECT_EQ(o628, iValue6); EXPECT_EQ(o629, iValue6); EXPECT_EQ(o630, iValue6); EXPECT_EQ(o631, iValue6); EXPECT_EQ(o632, iValue6); EXPECT_EQ(o633, iValue6); EXPECT_EQ(o634, iValue6); EXPECT_EQ(o635, iValue6); EXPECT_EQ(o636, iValue6);
    EXPECT_EQ(o637, iValue6); EXPECT_EQ(o638, iValue6); EXPECT_EQ(o639, iValue6);

    long long iValue8 = -2599822390419074042;
    Multiprecision<64> o80 = iValue8;
    Multiprecision<96> o81 = iValue8; Multiprecision<128> o82 = iValue8; Multiprecision<160> o83 = iValue8; Multiprecision<192> o84 = iValue8; Multiprecision<224> o85 = iValue8; Multiprecision<256> o86 = iValue8; Multiprecision<288> o87 = iValue8; Multiprecision<320> o88 = iValue8; Multiprecision<352> o89 = iValue8;
    Multiprecision<384> o810 = iValue8; Multiprecision<416> o811 = iValue8; Multiprecision<448> o812 = iValue8; Multiprecision<480> o813 = iValue8; Multiprecision<512> o814 = iValue8; Multiprecision<544> o815 = iValue8; Multiprecision<576> o816 = iValue8; Multiprecision<608> o817 = iValue8; Multiprecision<640> o818 = iValue8;
    Multiprecision<672> o819 = iValue8; Multiprecision<704> o820 = iValue8; Multiprecision<736> o821 = iValue8; Multiprecision<768> o822 = iValue8; Multiprecision<800> o823 = iValue8; Multiprecision<832> o824 = iValue8; Multiprecision<864> o825 = iValue8; Multiprecision<896> o826 = iValue8; Multiprecision<928> o827 = iValue8;
    Multiprecision<960> o828 = iValue8; Multiprecision<992> o829 = iValue8; Multiprecision<1024> o830 = iValue8; Multiprecision<1056> o831 = iValue8; Multiprecision<1088> o832 = iValue8; Multiprecision<1120> o833 = iValue8; Multiprecision<1152> o834 = iValue8; Multiprecision<1184> o835 = iValue8; Multiprecision<1216> o836 = iValue8;
    Multiprecision<1248> o837 = iValue8; Multiprecision<1280> o838 = iValue8; Multiprecision<1312> o839 = iValue8;
    EXPECT_EQ(o80, iValue8);
    EXPECT_EQ(o81, iValue8); EXPECT_EQ(o82, iValue8); EXPECT_EQ(o83, iValue8); EXPECT_EQ(o84, iValue8); EXPECT_EQ(o85, iValue8); EXPECT_EQ(o86, iValue8); EXPECT_EQ(o87, iValue8); EXPECT_EQ(o88, iValue8); EXPECT_EQ(o89, iValue8);
    EXPECT_EQ(o810, iValue8); EXPECT_EQ(o811, iValue8); EXPECT_EQ(o812, iValue8); EXPECT_EQ(o813, iValue8); EXPECT_EQ(o814, iValue8); EXPECT_EQ(o815, iValue8); EXPECT_EQ(o816, iValue8); EXPECT_EQ(o817, iValue8); EXPECT_EQ(o818, iValue8);
    EXPECT_EQ(o819, iValue8); EXPECT_EQ(o820, iValue8); EXPECT_EQ(o821, iValue8); EXPECT_EQ(o822, iValue8); EXPECT_EQ(o823, iValue8); EXPECT_EQ(o824, iValue8); EXPECT_EQ(o825, iValue8); EXPECT_EQ(o826, iValue8); EXPECT_EQ(o827, iValue8);
    EXPECT_EQ(o828, iValue8); EXPECT_EQ(o829, iValue8); EXPECT_EQ(o830, iValue8); EXPECT_EQ(o831, iValue8); EXPECT_EQ(o832, iValue8); EXPECT_EQ(o833, iValue8); EXPECT_EQ(o834, iValue8); EXPECT_EQ(o835, iValue8); EXPECT_EQ(o836, iValue8);
    EXPECT_EQ(o837, iValue8); EXPECT_EQ(o838, iValue8); EXPECT_EQ(o839, iValue8);
}