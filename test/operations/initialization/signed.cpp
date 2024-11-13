#include <gtest/gtest.h>
#include <bitset>
#include "../../../Aesi.h"
#include "../../generation.h"

constexpr auto testsAmount = 2,
    blocksNumber = 64;

TEST(Signed_Initialization, Basic) {
    {
        Aesi128 m2 = 0,
            m3 = Aesi128(0),
            m4 = {},
            m5 = "0",
            m6 = "-0",
            m7 = "Somebody once told me...";
        EXPECT_EQ(Aesi128 {}, 0); EXPECT_EQ(Aesi128(0), 0); EXPECT_EQ(m2, 0); EXPECT_EQ(m3, 0);
        EXPECT_EQ(m4, 0); EXPECT_EQ(m5, 0); EXPECT_EQ(m6, 0); EXPECT_EQ(m7, 0); EXPECT_EQ(Aesi128(), 0);
    }
    {
        Aesi128 i01 = 1, i02 = -1, i03 = 127, i04 = -127, i05 = -128, i06 = +127;
        EXPECT_EQ(i01, 1); EXPECT_EQ(i02, -1); EXPECT_EQ(i03, 127);
        EXPECT_EQ(i04, -127); EXPECT_EQ(i05, -128); EXPECT_EQ(i06, 127);
    }
    {
        EXPECT_EQ(Aesi128("10"), 10); EXPECT_EQ(Aesi128("-10"), -10);
        EXPECT_EQ(Aesi128("50"), 50); EXPECT_EQ(Aesi128("-50"), -50);
        EXPECT_EQ(Aesi128("0o24"), 20); EXPECT_EQ(Aesi128("-0o24"), -20);
        EXPECT_EQ(Aesi128("0xa"), 10); EXPECT_EQ(Aesi128("0xA"), 10);
        EXPECT_EQ(Aesi128("-0xa"), -10); EXPECT_EQ(Aesi128("-0xA"), -10);
    }
    {
        using namespace std::string_literals; using namespace std::string_view_literals;
        Aesi512 d0 = "489133282872437279"s, d1 = "63018038201"sv, d2 = "-489133282872437279"s, d3 = "-63018038201"sv;
        EXPECT_EQ(d0, 489133282872437279); EXPECT_EQ(d1, 63018038201); EXPECT_EQ(d2, -489133282872437279); EXPECT_EQ(d3, -63018038201);

        Aesi512 b0 = "0b11011001001110000000010000100010101011011101010101000011111"s;  EXPECT_EQ(b0, 489133282872437279);
        Aesi512 b1 = "0b111010101100001010101111001110111001"sv;                        EXPECT_EQ(b1, 63018038201);
        Aesi512 b2 = "-0b11011001001110000000010000100010101011011101010101000011111"s; EXPECT_EQ(b2, -489133282872437279);
        Aesi512 b3 = "-0b111010101100001010101111001110111001"sv;                       EXPECT_EQ(b3, -63018038201);

        Aesi512 o0 = "0o106274176273174613"s, o1 = "0o642054234601645202742"sv, o2 = "-0o106274176273174613"s, o3 = "-0o642054234601645202742"sv;
        EXPECT_EQ(o0, 2475842268363147); EXPECT_EQ(o1, 7531577461358003682); EXPECT_EQ(o2, -2475842268363147); EXPECT_EQ(o3, -7531577461358003682);

        Aesi512 h0 = "0x688589CC0E9505E2"s, h1 = "0x3C9D4B9CB52FE"sv, h2 = "-0x688589CC0E9505E2"s, h3 = "-0x3C9D4B9CB52FE"sv;
        EXPECT_EQ(h0, 7531577461358003682); EXPECT_EQ(h1, 1066340417491710); EXPECT_EQ(h2, -7531577461358003682); EXPECT_EQ(h3, -1066340417491710);
        Aesi512 h4 = "0x688589cc0e9505e2"s, h5 = "0x3c9d4b9cb52fe"sv, h6 = "-0x688589cc0e9505e2"s, h7 = "-0x3c9d4b9cb52fe"sv;
        EXPECT_EQ(h4, 7531577461358003682); EXPECT_EQ(h5, 1066340417491710); EXPECT_EQ(h6, -7531577461358003682); EXPECT_EQ(h7, -1066340417491710);

        d0 = L"489133282872437279"s, d1 = L"63018038201"sv, d2 = L"-489133282872437279"s, d3 = L"-63018038201"sv;
        EXPECT_EQ(d0, 489133282872437279); EXPECT_EQ(d1, 63018038201); EXPECT_EQ(d2, -489133282872437279); EXPECT_EQ(d3, -63018038201);

        b0 = L"0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279);
        b1 = L"0b111010101100001010101111001110111001"sv;                           EXPECT_EQ(b1, 63018038201);
        b2 = L"-0b11011001001110000000010000100010101011011101010101000011111"s;    EXPECT_EQ(b2, -489133282872437279);
        b3 = L"-0b111010101100001010101111001110111001"sv;                          EXPECT_EQ(b3, -63018038201);

        b0 = L"-0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"--0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"---0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"----0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"-----0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"-------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"--------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"---------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"----------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"-----------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"-------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"--------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"---------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"----------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);
        b0 = L"-----------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, -489133282872437279LL);
        b0 = L"------------------0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279LL);

        o0 = L"0o106274176273174613"s, o1 = L"0o642054234601645202742"sv, o2 = L"-0o106274176273174613"s, o3 = L"-0o642054234601645202742"sv;
        EXPECT_EQ(o0, 2475842268363147); EXPECT_EQ(o1, 7531577461358003682); EXPECT_EQ(o2, -2475842268363147); EXPECT_EQ(o3, -7531577461358003682);

        h0 = L"0x688589CC0E9505E2"s, h1 = L"0x3C9D4B9CB52FE"sv, h2 = L"-0x688589CC0E9505E2"s, h3 = L"-0x3C9D4B9CB52FE"sv;
        EXPECT_EQ(h0, 7531577461358003682); EXPECT_EQ(h1, 1066340417491710); EXPECT_EQ(h2, -7531577461358003682); EXPECT_EQ(h3, -1066340417491710);
        h4 = L"0x688589cc0e9505e2"s, h5 = L"0x3c9d4b9cb52fe"sv, h6 = L"-0x688589cc0e9505e2"s, h7 = L"-0x3c9d4b9cb52fe"sv;
        EXPECT_EQ(h4, 7531577461358003682); EXPECT_EQ(h5, 1066340417491710); EXPECT_EQ(h6, -7531577461358003682); EXPECT_EQ(h7, -1066340417491710);
    }
}

TEST(Signed_Initialization, Different_precisions) {
    {
        uint64_t iValue0 = 3218136187561313218u;
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
    }
    {
        uint64_t iValue0 = -3218136187561313218u;
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
    }
}

TEST(Signed_Initialization, Binary) {
    Aesi<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::string binary {};
        for (auto byte = value.GetByte(value.ByteCount() - 1); byte; byte >>= 1)
            binary += (byte & 1 ? '1' : '0');
        std::stringstream ss {};
        ss << (i % 2 == 0 ? "" : "-") << "0b" << std::string(binary.rbegin(), binary.rend());
        for(long long j = value.ByteCount() - 2; j >= 0; --j)
            ss << std::bitset<8>(value.GetByte(j));
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Signed_Initialization, Decimal) {
    Aesi<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {}; ss << std::dec << value;
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Signed_Initialization, Octal) {
    Aesi<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {}; ss << (i % 2 == 0 ? "" : "-") << "0o" << std::oct << (i % 2 == 0 ? value : value * -1);
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Signed_Initialization, Hexadecimal) {
    Aesi<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {};
        if(i % 2 == 0)
            ss << (i % 2 == 0 ? "" : "-") << "0x" << std::hex << std::uppercase << (i % 2 == 0 ? value : value * -1);
        else
            ss << (i % 2 == 0 ? "" : "-") << "0x" << std::hex << std::nouppercase << (i % 2 == 0 ? value : value * -1);
        record = ss.str(); EXPECT_EQ(record, value);
    }
}
