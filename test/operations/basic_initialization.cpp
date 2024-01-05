#include <gtest/gtest.h>
#include "../../Aesi.h"

TEST(Initialization, ZeroInitialization) {
    Aesi512 m0 {};
    EXPECT_EQ(m0, 0);

    Aesi512 m1(0);
    EXPECT_EQ(m1, 0);

    Aesi512 m2 = 0;
    EXPECT_EQ(m2, 0);

    Aesi512 m3 = Aesi512(0);
    EXPECT_EQ(m3, 0);

    Aesi512 m4 = {};
    EXPECT_EQ(m4, 0);

    Aesi512 m5 = "0";
    EXPECT_EQ(m5, 0);

    Aesi512 m6 = "0.";
    EXPECT_EQ(m6, 0);

    Aesi512 m7 = "Somebody once told me The world is gonna roll me I ain't the sharpest tool in the shed";
    EXPECT_EQ(m7, 0);
}

TEST(Initialization, BasicInitialization) {
    Aesi512 i01 = 1;
    EXPECT_EQ(i01, 1);

    Aesi512 i02 = -1, i03 = 127, i04 = -127, i05 = -128, i06 = +127;
    EXPECT_EQ(i02, -1); EXPECT_EQ(i03, 127); EXPECT_EQ(i04, -127); EXPECT_EQ(i05, -128); EXPECT_EQ(i06, 127);
}

TEST(Initialization, SmallCharArrayInitialization) {
    Aesi512 ten = "10", negativeTen = "-10";
    EXPECT_EQ(ten, 10);
    EXPECT_EQ(negativeTen, -10);

    Aesi512 hexTenLC = "0xa", hexTenHC = "0xA", negativeHexTenLC = "-0xa", negativeHexTenHC = "-0xA";
    EXPECT_EQ(hexTenLC, 10);
    EXPECT_EQ(hexTenHC, 10); EXPECT_EQ(negativeHexTenLC, -10); EXPECT_EQ(negativeHexTenHC, -10);
}

TEST(Initialization, StringStringViewInitialization) {
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    Aesi512 d0 = "489133282872437279"s;
    EXPECT_EQ(d0, 489133282872437279);
    Aesi512 d1 = "63018038201"sv;
    EXPECT_EQ(d1, 63018038201);
    Aesi512 d2 = "-489133282872437279"s;
    EXPECT_EQ(d2, -489133282872437279);
    Aesi512 d3 = "-63018038201"sv;
    EXPECT_EQ(d3, -63018038201);

    Aesi512 b0 = "0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b0, 489133282872437279);
    Aesi512 b1 = "0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b1, 63018038201);
    Aesi512 b2 = "-0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b2, -489133282872437279);
    Aesi512 b3 = "-0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b3, -63018038201);

    Aesi512 o0 = "0o106274176273174613"s;
    EXPECT_EQ(o0, 2475842268363147);
    Aesi512 o1 = "0o642054234601645202742"sv;
    EXPECT_EQ(o1, 7531577461358003682);
    Aesi512 o2 = "-0o106274176273174613"s;
    EXPECT_EQ(o2, -2475842268363147);
    Aesi512 o3 = "-0o642054234601645202742"sv;
    EXPECT_EQ(o3, -7531577461358003682);

    Aesi512 h0 = "0x688589CC0E9505E2"s;
    EXPECT_EQ(h0, 7531577461358003682);
    Aesi512 h1 = "0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h1, 1066340417491710);
    Aesi512 h2 = "-0x688589CC0E9505E2"s;
    EXPECT_EQ(h2, -7531577461358003682);
    Aesi512 h3 = "-0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h3, -1066340417491710);
    Aesi512 h4 = "0x688589cc0e9505e2"s;
    EXPECT_EQ(h4, 7531577461358003682);
    Aesi512 h5 = "0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h5, 1066340417491710);
    Aesi512 h6 = "-0x688589cc0e9505e2"s;
    EXPECT_EQ(h6, -7531577461358003682);
    Aesi512 h7 = "-0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h7, -1066340417491710);

    d0 = L"489133282872437279"s;
    EXPECT_EQ(d0, 489133282872437279);
    d1 = L"63018038201"sv;
    EXPECT_EQ(d1, 63018038201);
    d2 = L"-489133282872437279"s;
    EXPECT_EQ(d2, -489133282872437279);
    d3 = L"-63018038201"sv;
    EXPECT_EQ(d3, -63018038201);

    b0 = L"0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b0, 489133282872437279);
    b1 = L"0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b1, 63018038201);
    b2 = L"-0b11011001001110000000010000100010101011011101010101000011111"s;
    EXPECT_EQ(b2, -489133282872437279);
    b3 = L"-0b111010101100001010101111001110111001"sv;
    EXPECT_EQ(b3, -63018038201);

    o0 = L"0o106274176273174613"s;
    EXPECT_EQ(o0, 2475842268363147);
    o1 = L"0o642054234601645202742"sv;
    EXPECT_EQ(o1, 7531577461358003682);
    o2 = L"-0o106274176273174613"s;
    EXPECT_EQ(o2, -2475842268363147);
    o3 = L"-0o642054234601645202742"sv;
    EXPECT_EQ(o3, -7531577461358003682);

    h0 = L"0x688589CC0E9505E2"s;
    EXPECT_EQ(h0, 7531577461358003682);
    h1 = L"0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h1, 1066340417491710);
    h2 = L"-0x688589CC0E9505E2"s;
    EXPECT_EQ(h2, -7531577461358003682);
    h3 = L"-0x3C9D4B9CB52FE"sv;
    EXPECT_EQ(h3, -1066340417491710);
    h4 = L"0x688589cc0e9505e2"s;
    EXPECT_EQ(h4, 7531577461358003682);
    h5 = L"0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h5, 1066340417491710);
    h6 = L"-0x688589cc0e9505e2"s;
    EXPECT_EQ(h6, -7531577461358003682);
    h7 = L"-0x3c9d4b9cb52fe"sv;
    EXPECT_EQ(h7, -1066340417491710);
}

TEST(Initialization, DifferentPrecisions) {
    long long iValue0 = 3218136187561313218;
    Aesi < 96 > o00 = iValue0;
    Aesi < 96 > o01 = iValue0;

    Aesi < 128 > o02 = iValue0; Aesi < 160 > o03 = iValue0; Aesi < 192 > o04 = iValue0; Aesi < 224 > o05 = iValue0; Aesi < 256 > o06 = iValue0; Aesi < 288 > o07 = iValue0; Aesi < 320 > o08 = iValue0; Aesi < 352 > o09 = iValue0;
    Aesi < 384 > o010 = iValue0; Aesi < 416 > o011 = iValue0; Aesi < 448 > o012 = iValue0; Aesi < 480 > o013 = iValue0; Aesi < 512 > o014 = iValue0; Aesi < 544 > o015 = iValue0; Aesi < 576 > o016 = iValue0; Aesi < 608 > o017 = iValue0; Aesi < 640 > o018 = iValue0;
    Aesi < 672 > o019 = iValue0; Aesi < 704 > o020 = iValue0; Aesi < 736 > o021 = iValue0; Aesi < 768 > o022 = iValue0; Aesi < 800 > o023 = iValue0; Aesi < 832 > o024 = iValue0; Aesi < 864 > o025 = iValue0; Aesi < 896 > o026 = iValue0; Aesi < 928 > o027 = iValue0;
    Aesi < 960 > o028 = iValue0; Aesi < 992 > o029 = iValue0; Aesi < 1024 > o030 = iValue0; Aesi < 1056 > o031 = iValue0; Aesi < 1088 > o032 = iValue0; Aesi < 1120 > o033 = iValue0; Aesi < 1152 > o034 = iValue0; Aesi < 1184 > o035 = iValue0; Aesi < 1216 > o036 = iValue0;
    Aesi < 1248 > o037 = iValue0; Aesi < 1280 > o038 = iValue0; Aesi < 1312 > o039 = iValue0;
    EXPECT_EQ(o00, iValue0);
    EXPECT_EQ(o01, iValue0); EXPECT_EQ(o02, iValue0); EXPECT_EQ(o03, iValue0); EXPECT_EQ(o04, iValue0); EXPECT_EQ(o05, iValue0); EXPECT_EQ(o06, iValue0); EXPECT_EQ(o07, iValue0); EXPECT_EQ(o08, iValue0); EXPECT_EQ(o09, iValue0);
    EXPECT_EQ(o010, iValue0); EXPECT_EQ(o011, iValue0); EXPECT_EQ(o012, iValue0); EXPECT_EQ(o013, iValue0); EXPECT_EQ(o014, iValue0); EXPECT_EQ(o015, iValue0); EXPECT_EQ(o016, iValue0); EXPECT_EQ(o017, iValue0); EXPECT_EQ(o018, iValue0);
    EXPECT_EQ(o019, iValue0); EXPECT_EQ(o020, iValue0); EXPECT_EQ(o021, iValue0); EXPECT_EQ(o022, iValue0); EXPECT_EQ(o023, iValue0); EXPECT_EQ(o024, iValue0); EXPECT_EQ(o025, iValue0); EXPECT_EQ(o026, iValue0); EXPECT_EQ(o027, iValue0);
    EXPECT_EQ(o028, iValue0); EXPECT_EQ(o029, iValue0); EXPECT_EQ(o030, iValue0); EXPECT_EQ(o031, iValue0); EXPECT_EQ(o032, iValue0); EXPECT_EQ(o033, iValue0); EXPECT_EQ(o034, iValue0); EXPECT_EQ(o035, iValue0); EXPECT_EQ(o036, iValue0);
    EXPECT_EQ(o037, iValue0); EXPECT_EQ(o038, iValue0); EXPECT_EQ(o039, iValue0);

    long long iValue2 = -380464553884730375;
    Aesi < 96 > o20 = iValue2;
    Aesi < 96 > o21 = iValue2; Aesi < 128 > o22 = iValue2; Aesi < 160 > o23 = iValue2; Aesi < 192 > o24 = iValue2; Aesi < 224 > o25 = iValue2; Aesi < 256 > o26 = iValue2; Aesi < 288 > o27 = iValue2; Aesi < 320 > o28 = iValue2; Aesi < 352 > o29 = iValue2;
    Aesi < 384 > o210 = iValue2; Aesi < 416 > o211 = iValue2; Aesi < 448 > o212 = iValue2; Aesi < 480 > o213 = iValue2; Aesi < 512 > o214 = iValue2; Aesi < 544 > o215 = iValue2; Aesi < 576 > o216 = iValue2; Aesi < 608 > o217 = iValue2; Aesi < 640 > o218 = iValue2;
    Aesi < 672 > o219 = iValue2; Aesi < 704 > o220 = iValue2; Aesi < 736 > o221 = iValue2; Aesi < 768 > o222 = iValue2; Aesi < 800 > o223 = iValue2; Aesi < 832 > o224 = iValue2; Aesi < 864 > o225 = iValue2; Aesi < 896 > o226 = iValue2; Aesi < 928 > o227 = iValue2;
    Aesi < 960 > o228 = iValue2; Aesi < 992 > o229 = iValue2; Aesi < 1024 > o230 = iValue2; Aesi < 1056 > o231 = iValue2; Aesi < 1088 > o232 = iValue2; Aesi < 1120 > o233 = iValue2; Aesi < 1152 > o234 = iValue2; Aesi < 1184 > o235 = iValue2; Aesi < 1216 > o236 = iValue2;
    Aesi < 1248 > o237 = iValue2; Aesi < 1280 > o238 = iValue2; Aesi < 1312 > o239 = iValue2;
    EXPECT_EQ(o20, iValue2);
    EXPECT_EQ(o21, iValue2); EXPECT_EQ(o22, iValue2); EXPECT_EQ(o23, iValue2); EXPECT_EQ(o24, iValue2); EXPECT_EQ(o25, iValue2); EXPECT_EQ(o26, iValue2); EXPECT_EQ(o27, iValue2); EXPECT_EQ(o28, iValue2); EXPECT_EQ(o29, iValue2);
    EXPECT_EQ(o210, iValue2); EXPECT_EQ(o211, iValue2); EXPECT_EQ(o212, iValue2); EXPECT_EQ(o213, iValue2); EXPECT_EQ(o214, iValue2); EXPECT_EQ(o215, iValue2); EXPECT_EQ(o216, iValue2); EXPECT_EQ(o217, iValue2); EXPECT_EQ(o218, iValue2);
    EXPECT_EQ(o219, iValue2); EXPECT_EQ(o220, iValue2); EXPECT_EQ(o221, iValue2); EXPECT_EQ(o222, iValue2); EXPECT_EQ(o223, iValue2); EXPECT_EQ(o224, iValue2); EXPECT_EQ(o225, iValue2); EXPECT_EQ(o226, iValue2); EXPECT_EQ(o227, iValue2);
    EXPECT_EQ(o228, iValue2); EXPECT_EQ(o229, iValue2); EXPECT_EQ(o230, iValue2); EXPECT_EQ(o231, iValue2); EXPECT_EQ(o232, iValue2); EXPECT_EQ(o233, iValue2); EXPECT_EQ(o234, iValue2); EXPECT_EQ(o235, iValue2); EXPECT_EQ(o236, iValue2);
    EXPECT_EQ(o237, iValue2); EXPECT_EQ(o238, iValue2); EXPECT_EQ(o239, iValue2);

    long long iValue4 = -2577490965723039550;
    Aesi < 96 > o40 = iValue4;
    Aesi < 96 > o41 = iValue4; Aesi < 128 > o42 = iValue4; Aesi < 160 > o43 = iValue4; Aesi < 192 > o44 = iValue4; Aesi < 224 > o45 = iValue4; Aesi < 256 > o46 = iValue4; Aesi < 288 > o47 = iValue4; Aesi < 320 > o48 = iValue4; Aesi < 352 > o49 = iValue4;
    Aesi < 384 > o410 = iValue4; Aesi < 416 > o411 = iValue4; Aesi < 448 > o412 = iValue4; Aesi < 480 > o413 = iValue4; Aesi < 512 > o414 = iValue4; Aesi < 544 > o415 = iValue4; Aesi < 576 > o416 = iValue4; Aesi < 608 > o417 = iValue4; Aesi < 640 > o418 = iValue4;
    Aesi < 672 > o419 = iValue4; Aesi < 704 > o420 = iValue4; Aesi < 736 > o421 = iValue4; Aesi < 768 > o422 = iValue4; Aesi < 800 > o423 = iValue4; Aesi < 832 > o424 = iValue4; Aesi < 864 > o425 = iValue4; Aesi < 896 > o426 = iValue4; Aesi < 928 > o427 = iValue4;
    Aesi < 960 > o428 = iValue4; Aesi < 992 > o429 = iValue4; Aesi < 1024 > o430 = iValue4; Aesi < 1056 > o431 = iValue4; Aesi < 1088 > o432 = iValue4; Aesi < 1120 > o433 = iValue4; Aesi < 1152 > o434 = iValue4; Aesi < 1184 > o435 = iValue4; Aesi < 1216 > o436 = iValue4;
    Aesi < 1248 > o437 = iValue4; Aesi < 1280 > o438 = iValue4; Aesi < 1312 > o439 = iValue4;
    EXPECT_EQ(o40, iValue4);
    EXPECT_EQ(o41, iValue4); EXPECT_EQ(o42, iValue4); EXPECT_EQ(o43, iValue4); EXPECT_EQ(o44, iValue4); EXPECT_EQ(o45, iValue4); EXPECT_EQ(o46, iValue4); EXPECT_EQ(o47, iValue4); EXPECT_EQ(o48, iValue4); EXPECT_EQ(o49, iValue4);
    EXPECT_EQ(o410, iValue4); EXPECT_EQ(o411, iValue4); EXPECT_EQ(o412, iValue4); EXPECT_EQ(o413, iValue4); EXPECT_EQ(o414, iValue4); EXPECT_EQ(o415, iValue4); EXPECT_EQ(o416, iValue4); EXPECT_EQ(o417, iValue4); EXPECT_EQ(o418, iValue4);
    EXPECT_EQ(o419, iValue4); EXPECT_EQ(o420, iValue4); EXPECT_EQ(o421, iValue4); EXPECT_EQ(o422, iValue4); EXPECT_EQ(o423, iValue4); EXPECT_EQ(o424, iValue4); EXPECT_EQ(o425, iValue4); EXPECT_EQ(o426, iValue4); EXPECT_EQ(o427, iValue4);
    EXPECT_EQ(o428, iValue4); EXPECT_EQ(o429, iValue4); EXPECT_EQ(o430, iValue4); EXPECT_EQ(o431, iValue4); EXPECT_EQ(o432, iValue4); EXPECT_EQ(o433, iValue4); EXPECT_EQ(o434, iValue4); EXPECT_EQ(o435, iValue4); EXPECT_EQ(o436, iValue4);
    EXPECT_EQ(o437, iValue4); EXPECT_EQ(o438, iValue4); EXPECT_EQ(o439, iValue4);

    long long iValue6 = -7225109388162562138;
    Aesi < 96 > o60 = iValue6;
    Aesi < 96 > o61 = iValue6; Aesi < 128 > o62 = iValue6; Aesi < 160 > o63 = iValue6; Aesi < 192 > o64 = iValue6; Aesi < 224 > o65 = iValue6; Aesi < 256 > o66 = iValue6; Aesi < 288 > o67 = iValue6; Aesi < 320 > o68 = iValue6; Aesi < 352 > o69 = iValue6;
    Aesi < 384 > o610 = iValue6; Aesi < 416 > o611 = iValue6; Aesi < 448 > o612 = iValue6; Aesi < 480 > o613 = iValue6; Aesi < 512 > o614 = iValue6; Aesi < 544 > o615 = iValue6; Aesi < 576 > o616 = iValue6; Aesi < 608 > o617 = iValue6; Aesi < 640 > o618 = iValue6;
    Aesi < 672 > o619 = iValue6; Aesi < 704 > o620 = iValue6; Aesi < 736 > o621 = iValue6; Aesi < 768 > o622 = iValue6; Aesi < 800 > o623 = iValue6; Aesi < 832 > o624 = iValue6; Aesi < 864 > o625 = iValue6; Aesi < 896 > o626 = iValue6; Aesi < 928 > o627 = iValue6;
    Aesi < 960 > o628 = iValue6; Aesi < 992 > o629 = iValue6; Aesi < 1024 > o630 = iValue6; Aesi < 1056 > o631 = iValue6; Aesi < 1088 > o632 = iValue6; Aesi < 1120 > o633 = iValue6; Aesi < 1152 > o634 = iValue6; Aesi < 1184 > o635 = iValue6; Aesi < 1216 > o636 = iValue6;
    Aesi < 1248 > o637 = iValue6; Aesi < 1280 > o638 = iValue6; Aesi < 1312 > o639 = iValue6;
    EXPECT_EQ(o60, iValue6);
    EXPECT_EQ(o61, iValue6); EXPECT_EQ(o62, iValue6); EXPECT_EQ(o63, iValue6); EXPECT_EQ(o64, iValue6); EXPECT_EQ(o65, iValue6); EXPECT_EQ(o66, iValue6); EXPECT_EQ(o67, iValue6); EXPECT_EQ(o68, iValue6); EXPECT_EQ(o69, iValue6);
    EXPECT_EQ(o610, iValue6); EXPECT_EQ(o611, iValue6); EXPECT_EQ(o612, iValue6); EXPECT_EQ(o613, iValue6); EXPECT_EQ(o614, iValue6); EXPECT_EQ(o615, iValue6); EXPECT_EQ(o616, iValue6); EXPECT_EQ(o617, iValue6); EXPECT_EQ(o618, iValue6);
    EXPECT_EQ(o619, iValue6); EXPECT_EQ(o620, iValue6); EXPECT_EQ(o621, iValue6); EXPECT_EQ(o622, iValue6); EXPECT_EQ(o623, iValue6); EXPECT_EQ(o624, iValue6); EXPECT_EQ(o625, iValue6); EXPECT_EQ(o626, iValue6); EXPECT_EQ(o627, iValue6);
    EXPECT_EQ(o628, iValue6); EXPECT_EQ(o629, iValue6); EXPECT_EQ(o630, iValue6); EXPECT_EQ(o631, iValue6); EXPECT_EQ(o632, iValue6); EXPECT_EQ(o633, iValue6); EXPECT_EQ(o634, iValue6); EXPECT_EQ(o635, iValue6); EXPECT_EQ(o636, iValue6);
    EXPECT_EQ(o637, iValue6); EXPECT_EQ(o638, iValue6); EXPECT_EQ(o639, iValue6);

    long long iValue8 = -2599822390419074042;
    Aesi < 96 > o80 = iValue8;
    Aesi < 96 > o81 = iValue8; Aesi < 128 > o82 = iValue8; Aesi < 160 > o83 = iValue8; Aesi < 192 > o84 = iValue8; Aesi < 224 > o85 = iValue8; Aesi < 256 > o86 = iValue8; Aesi < 288 > o87 = iValue8; Aesi < 320 > o88 = iValue8; Aesi < 352 > o89 = iValue8;
    Aesi < 384 > o810 = iValue8; Aesi < 416 > o811 = iValue8; Aesi < 448 > o812 = iValue8; Aesi < 480 > o813 = iValue8; Aesi < 512 > o814 = iValue8; Aesi < 544 > o815 = iValue8; Aesi < 576 > o816 = iValue8; Aesi < 608 > o817 = iValue8; Aesi < 640 > o818 = iValue8;
    Aesi < 672 > o819 = iValue8; Aesi < 704 > o820 = iValue8; Aesi < 736 > o821 = iValue8; Aesi < 768 > o822 = iValue8; Aesi < 800 > o823 = iValue8; Aesi < 832 > o824 = iValue8; Aesi < 864 > o825 = iValue8; Aesi < 896 > o826 = iValue8; Aesi < 928 > o827 = iValue8;
    Aesi < 960 > o828 = iValue8; Aesi < 992 > o829 = iValue8; Aesi < 1024 > o830 = iValue8; Aesi < 1056 > o831 = iValue8; Aesi < 1088 > o832 = iValue8; Aesi < 1120 > o833 = iValue8; Aesi < 1152 > o834 = iValue8; Aesi < 1184 > o835 = iValue8; Aesi < 1216 > o836 = iValue8;
    Aesi < 1248 > o837 = iValue8; Aesi < 1280 > o838 = iValue8; Aesi < 1312 > o839 = iValue8;
    EXPECT_EQ(o80, iValue8);
    EXPECT_EQ(o81, iValue8); EXPECT_EQ(o82, iValue8); EXPECT_EQ(o83, iValue8); EXPECT_EQ(o84, iValue8); EXPECT_EQ(o85, iValue8); EXPECT_EQ(o86, iValue8); EXPECT_EQ(o87, iValue8); EXPECT_EQ(o88, iValue8); EXPECT_EQ(o89, iValue8);
    EXPECT_EQ(o810, iValue8); EXPECT_EQ(o811, iValue8); EXPECT_EQ(o812, iValue8); EXPECT_EQ(o813, iValue8); EXPECT_EQ(o814, iValue8); EXPECT_EQ(o815, iValue8); EXPECT_EQ(o816, iValue8); EXPECT_EQ(o817, iValue8); EXPECT_EQ(o818, iValue8);
    EXPECT_EQ(o819, iValue8); EXPECT_EQ(o820, iValue8); EXPECT_EQ(o821, iValue8); EXPECT_EQ(o822, iValue8); EXPECT_EQ(o823, iValue8); EXPECT_EQ(o824, iValue8); EXPECT_EQ(o825, iValue8); EXPECT_EQ(o826, iValue8); EXPECT_EQ(o827, iValue8);
    EXPECT_EQ(o828, iValue8); EXPECT_EQ(o829, iValue8); EXPECT_EQ(o830, iValue8); EXPECT_EQ(o831, iValue8); EXPECT_EQ(o832, iValue8); EXPECT_EQ(o833, iValue8); EXPECT_EQ(o834, iValue8); EXPECT_EQ(o835, iValue8); EXPECT_EQ(o836, iValue8);
    EXPECT_EQ(o837, iValue8); EXPECT_EQ(o838, iValue8); EXPECT_EQ(o839, iValue8);
}

TEST(Initialization, CopyConstruction) {
    Aesi < 320 > l0 = "856969574457709690462967066638280185610032288787864641536747837575250187179350463621579392244871."; Aesi < 640 > r0 = l0;
    EXPECT_EQ(r0, "856969574457709690462967066638280185610032288787864641536747837575250187179350463621579392244871.");
    Aesi < 192 > l1 = "1817464682414554777634009954620732846679878578413134368229."; Aesi < 576 > r1 = l1;
    EXPECT_EQ(r1, "1817464682414554777634009954620732846679878578413134368229.");
    Aesi < 288 > l2 = "273585059134613895932021029292354478649861471872302821549505743650758591411133430451408."; Aesi < 512 > r2 = l2;
    EXPECT_EQ(r2, "273585059134613895932021029292354478649861471872302821549505743650758591411133430451408.");
    Aesi < 96 > l3 = "16848821812072936343963016744."; Aesi < 576 > r3 = l3;
    EXPECT_EQ(r3, "16848821812072936343963016744.");
    Aesi < 256 > l4 = "89425012188150576676147569796603714593119677771166340023586622845972385164563."; Aesi < 608 > r4 = l4;
    EXPECT_EQ(r4, "89425012188150576676147569796603714593119677771166340023586622845972385164563.");
    Aesi < 128 > l5 = "295375145424713840341444584487141559072."; Aesi < 352 > r5 = l5;
    EXPECT_EQ(r5, "295375145424713840341444584487141559072.");
    Aesi < 288 > l6 = "339499853026571984978143860631483264809110095600112495957713684212166920280720856537507."; Aesi < 512 > r6 = l6;
    EXPECT_EQ(r6, "339499853026571984978143860631483264809110095600112495957713684212166920280720856537507.");
    Aesi < 224 > l7 = "25304709703351498872407921001491752489198569317853002340745585561130."; Aesi < 512 > r7 = l7;
    EXPECT_EQ(r7, "25304709703351498872407921001491752489198569317853002340745585561130.");
    Aesi < 256 > l8 = "113561681989557162025191006677416332468760445588423923562428981656155174119038."; Aesi < 544 > r8 = l8;
    EXPECT_EQ(r8, "113561681989557162025191006677416332468760445588423923562428981656155174119038.");
    Aesi < 160 > l9 = "492906146523434665001337396377408716037187766713."; Aesi < 544 > r9 = l9;
    EXPECT_EQ(r9, "492906146523434665001337396377408716037187766713.");
    Aesi < 320 > l10 = "1298781122230094131066140892218880004926912293801695693149260137199378490485067000212313855524472."; Aesi < 448 > r10 = l10;
    EXPECT_EQ(r10, "1298781122230094131066140892218880004926912293801695693149260137199378490485067000212313855524472.");
    Aesi < 192 > l11 = "85527839737441623387596737943574069844258992761886791264."; Aesi < 512 > r11 = l11;
    EXPECT_EQ(r11, "85527839737441623387596737943574069844258992761886791264.");
    Aesi < 128 > l12 = "22115878739988382109358311456734219104."; Aesi < 544 > r12 = l12;
    EXPECT_EQ(r12, "22115878739988382109358311456734219104.");
    Aesi < 96 > l13 = "49475628611456277257242336001."; Aesi < 544 > r13 = l13;
    EXPECT_EQ(r13, "49475628611456277257242336001.");
    Aesi < 288 > l14 = "56671084328196451986849244082695509909817145587284637777749032936926262765481285182134."; Aesi < 416 > r14 = l14;
    EXPECT_EQ(r14, "56671084328196451986849244082695509909817145587284637777749032936926262765481285182134.");
    Aesi < 192 > l15 = "3544498225942911223590252920351612557423834077300495237770."; Aesi < 352 > r15 = l15;
    EXPECT_EQ(r15, "3544498225942911223590252920351612557423834077300495237770.");
    Aesi < 192 > l16 = "3820599076383766096263039217201155696262014013686177183435."; Aesi < 416 > r16 = l16;
    EXPECT_EQ(r16, "3820599076383766096263039217201155696262014013686177183435.");
    Aesi < 96 > l17 = "57599326491869816621963559440."; Aesi < 640 > r17 = l17;
    EXPECT_EQ(r17, "57599326491869816621963559440.");
    Aesi < 160 > l18 = "398621409139070063759305124012533025860694318871."; Aesi < 608 > r18 = l18;
    EXPECT_EQ(r18, "398621409139070063759305124012533025860694318871.");
    Aesi < 320 > l19 = "248316872826048034227114052633289726014974581360870214968041723567293448101375940146375519025610."; Aesi < 352 > r19 = l19;
    EXPECT_EQ(r19, "248316872826048034227114052633289726014974581360870214968041723567293448101375940146375519025610.");

    Aesi < 608 > l20 = "480748306912701683551026904120451207851157046496025514603903733409124634475655295590263438367593131754374865349097113977359141809844045727298373624458221965327208187046764425885989375.";
    Aesi < 96 > r20 = l20; EXPECT_EQ(r20, "7315839924642923755123456511.");
    Aesi < 384 > l21 = "24732712984347763863686063342763357230152897395584545075070515616089388988095271904791330759237060982754926527148659.";
    Aesi < 128 > r21 = l21; EXPECT_EQ(r21, "276593900546481545361984775786702991987.");
    Aesi < 384 > l22 = "25265080169480331943217304297776392543015169251561211370971697299949969674628417070404543397222311206461399768507413.";
    Aesi < 192 > r22 = l22; EXPECT_EQ(r22, "5427574012535137818718507120413538908796299709273432400917.");
    Aesi < 544 > l23 = "51745623400592303273488368459034916606144660476304319387847981608773906660582998678767323241739412296092733302695015801193856715621466668157630426069181035184679156.";
    Aesi < 128 > r23 = l23; EXPECT_EQ(r23, "48277504082481759000353990028204468468.");
    Aesi < 640 > l24 = "2483659815302037713246132972687567200992533059356141579944568310265888324863696400805396970092208056614613672365514009958609839094050108024653994093041593083792431501631630944271578577629096184.";
    Aesi < 192 > r24 = l24; EXPECT_EQ(r24, "2479783778105728395870394836207134923910444741150009993464.");
    Aesi < 480 > l25 = "2575444764979228256302972756555363258983850034223003439688891561318782119534456833559884731570576183810759624412115817433201040035214828423182040.";
    Aesi < 160 > r25 = l25; EXPECT_EQ(r25, "678578983352461974029650041212752611069554161368.");
    Aesi < 352 > l26 = "6447444378744195218140254845048874447093869507850770757516522265101456245432283131966757618231045311275742.";
    Aesi < 96 > r26 = l26; EXPECT_EQ(r26, "28425401103843126580300325598.");
    Aesi < 480 > l27 = "994436160823845050116679985411979610820612109057253361937912071612950806651752613535023596299841105763992170915210059597788845237818885529822885.";
    Aesi < 256 > r27 = l27; EXPECT_EQ(r27, "84749325263055339442963923121992410225629138360736796140135959251605670749861.");
    Aesi < 512 > l28 = "11675072289569369370637177529427320811004043093738351423810014974101379207770749479775777216322323111873129448755953546751654776241818136245314659030835976.";
    Aesi < 192 > r28 = l28; EXPECT_EQ(r28, "2976371073428599466223411603937197263712729513112299545352.");
    Aesi < 448 > l29 = "219815026951655971061040887493048537575925564844708161716220465836587034762546971034698314904015255075296497325403919563298509665586491.";
    Aesi < 128 > r29 = l29; EXPECT_EQ(r29, "295116307160859791396010484912615050555.");
    Aesi < 640 > l30 = "1830881751462381294753718092869867337284273557348663989645594378121754617183058738777858020089460582636217187006034826757571951771627291712145118998043450989696100518065114441444487605022011180.";
    Aesi < 256 > r30 = l30; EXPECT_EQ(r30, "8958889727000130083485224995263070146065328843751127247768740464026098249516.");
    Aesi < 448 > l31 = "296932102348854834261141373273572906583691861479751578338365647160839113748367350360938651050110927850688650658575786620891664204314149.";
    Aesi < 288 > r31 = l31; EXPECT_EQ(r31, "396450197407333239118602857604013002149492692004445494305179535751775133157549266581029.");
    Aesi < 576 > l32 = "241958040428506665397698526697657254487495059616873909245932880996748501717028189503013348796772267784641030910857023369803763662693806273473556437047102016711022006537966850.";
    Aesi < 128 > r32 = l32; EXPECT_EQ(r32, "295702489753947751894198331394795394306.");
    Aesi < 480 > l33 = "1310351347503661216897612434294681162938417202645981065894559841116932759717150274773175006485302032046996163492417938457974972276983262045728603.";
    Aesi < 160 > r33 = l33; EXPECT_EQ(r33, "351113845345664714346131860917895061411534925659.");
    Aesi < 352 > l34 = "1515829831287265514002624168838526063861459069032586902881374788429757813036157336399942353423383649894443.";
    Aesi < 256 > r34 = l34; EXPECT_EQ(r34, "18289382377494143608247833819288947325624502588516708023779061379520056311851.");
    Aesi < 512 > l35 = "8042235538536782266783946070420140382943166349620298785958420047897761190898834350374170437023348481199234025976425566442360741411104540457455789544927007.";
    Aesi < 128 > r35 = l35; EXPECT_EQ(r35, "139039142689008298202686827231104560927.");
    Aesi < 448 > l36 = "162907801127962717251082455693406896878007900268804686977536204715384900250930480801348587978712333516922017803477107543643677590789771.";
    Aesi < 160 > r36 = l36; EXPECT_EQ(r36, "481047254507856721807620347743073032727556759179.");
    Aesi < 448 > l37 = "96057392907500153858333509155852412483644876791355916222379185306505865121174693102600304587232372621281510615110305873205540709529902.";
    Aesi < 192 > r37 = l37; EXPECT_EQ(r37, "2162594323210120305985950610806219485640440144750136850734.");
    Aesi < 544 > l38 = "5757066267316319477508877363463974751181033059683780026672648777320786034173156480174153805057640005656918250673476649133642684366906989776616930253727649075889040.";
    Aesi < 256 > r38 = l38; EXPECT_EQ(r38, "106933706339764000878509010506267008707527642239213681219687340160313450627984.");
    Aesi < 416 > l39 = "86605218842175287861295312762812733538876615816577435089539372437460337976422690938739296779379732680027937815494246618893146.";
    Aesi < 320 > r39 = l39; EXPECT_EQ(r39, "1000120221928044347333617422540787789504562475931143059065186465720343014110720116619366564105050.");
}