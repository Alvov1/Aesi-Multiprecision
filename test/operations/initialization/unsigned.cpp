#include <gtest/gtest.h>
#include <bitset>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Initialization, Basic) {
    {
        Aeu128 m0{}, m1(0u), m2 = 0u, m3 = Aeu128(0u), m4 = {}, m5 = "0", m6 = "-0", m7 = "Somebody once told me...";
        EXPECT_EQ(m0, 0u); EXPECT_EQ(m1, 0u); EXPECT_EQ(m2, 0u); EXPECT_EQ(m3, 0u);
        EXPECT_EQ(m4, 0u); EXPECT_EQ(m5, 0u); EXPECT_EQ(m6, 0u); EXPECT_EQ(m7, 0u);
    }
    {
        Aeu128 i01 = 1u, i03 = 127u;
        EXPECT_EQ(i01, 1u); EXPECT_EQ(i03, 127u);
    }
    {
        Aeu128 ten = "10", fifty = "50";
        EXPECT_EQ(ten, 10u);
        EXPECT_EQ(fifty, 50u);

        Aeu128 hexTenLC = "0xa", hexTenHC = "0xA";
        EXPECT_EQ(hexTenLC, 10u); EXPECT_EQ(hexTenHC, 10u);

        Aeu128 octTwentyLC = "0o24", octTwentyHC = "0o24";
        EXPECT_EQ(octTwentyLC, 20u);
        EXPECT_EQ(octTwentyHC, 20u);
    }
    {
        using namespace std::string_literals; using namespace std::string_view_literals;
        Aeu512 d0 = "489133282872437279"s, d1 = "63018038201"sv;
        EXPECT_EQ(d0, 489133282872437279u); EXPECT_EQ(d1, 63018038201u);

        Aeu512 b0 = "0b11011001001110000000010000100010101011011101010101000011111"s;  EXPECT_EQ(b0, 489133282872437279u);
        Aeu512 b1 = "0b111010101100001010101111001110111001"sv;                        EXPECT_EQ(b1, 63018038201u);

        Aeu512 o0 = "0o106274176273174613"s, o1 = "0o642054234601645202742"sv;
        EXPECT_EQ(o0, 2475842268363147u); EXPECT_EQ(o1, 7531577461358003682u);

        Aeu512 h0 = "0x688589CC0E9505E2"s, h1 = "0x3C9D4B9CB52FE"sv;
        EXPECT_EQ(h0, 7531577461358003682u); EXPECT_EQ(h1, 1066340417491710u);
        Aeu512 h4 = "0x688589cc0e9505e2"s, h5 = "0x3c9d4b9cb52fe"sv;
        EXPECT_EQ(h4, 7531577461358003682u); EXPECT_EQ(h5, 1066340417491710u);

        d0 = L"489133282872437279"s, d1 = L"63018038201"sv;
        EXPECT_EQ(d0, 489133282872437279u); EXPECT_EQ(d1, 63018038201u);

        b0 = L"0b11011001001110000000010000100010101011011101010101000011111"s;     EXPECT_EQ(b0, 489133282872437279u);
        b1 = L"0b111010101100001010101111001110111001"sv;                           EXPECT_EQ(b1, 63018038201u);

        o0 = L"0o106274176273174613"s, o1 = L"0o642054234601645202742"sv;
        EXPECT_EQ(o0, 2475842268363147u); EXPECT_EQ(o1, 7531577461358003682u);

        h0 = L"0x688589CC0E9505E2"s, h1 = L"0x3C9D4B9CB52FE"sv;
        EXPECT_EQ(h0, 7531577461358003682u); EXPECT_EQ(h1, 1066340417491710u);
        h4 = L"0x688589cc0e9505e2"s, h5 = L"0x3c9d4b9cb52fe"sv;
        EXPECT_EQ(h4, 7531577461358003682u); EXPECT_EQ(h5, 1066340417491710u);
    }
}

TEST(Unsigned_Initialization, Different_precisions) {
    uint64_t iValue0 = 3218136187561313218u;
    Aeu < 96 > o00 = iValue0;
    Aeu < 96 > o01 = iValue0;

    Aeu < 128 > o02 = iValue0; Aeu < 160 > o03 = iValue0; Aeu < 192 > o04 = iValue0; Aeu < 224 > o05 = iValue0; Aeu < 256 > o06 = iValue0; Aeu < 288 > o07 = iValue0; Aeu < 320 > o08 = iValue0; Aeu < 352 > o09 = iValue0;
    Aeu < 384 > o010 = iValue0; Aeu < 416 > o011 = iValue0; Aeu < 448 > o012 = iValue0; Aeu < 480 > o013 = iValue0; Aeu < 512 > o014 = iValue0; Aeu < 544 > o015 = iValue0; Aeu < 576 > o016 = iValue0; Aeu < 608 > o017 = iValue0; Aeu < 640 > o018 = iValue0;
    Aeu < 672 > o019 = iValue0; Aeu < 704 > o020 = iValue0; Aeu < 736 > o021 = iValue0; Aeu < 768 > o022 = iValue0; Aeu < 800 > o023 = iValue0; Aeu < 832 > o024 = iValue0; Aeu < 864 > o025 = iValue0; Aeu < 896 > o026 = iValue0; Aeu < 928 > o027 = iValue0;
    Aeu < 960 > o028 = iValue0; Aeu < 992 > o029 = iValue0; Aeu < 1024 > o030 = iValue0; Aeu < 1056 > o031 = iValue0; Aeu < 1088 > o032 = iValue0; Aeu < 1120 > o033 = iValue0; Aeu < 1152 > o034 = iValue0; Aeu < 1184 > o035 = iValue0; Aeu < 1216 > o036 = iValue0;
    Aeu < 1248 > o037 = iValue0; Aeu < 1280 > o038 = iValue0; Aeu < 1312 > o039 = iValue0;

    EXPECT_EQ(o00, iValue0);
    EXPECT_EQ(o01, iValue0); EXPECT_EQ(o02, iValue0); EXPECT_EQ(o03, iValue0); EXPECT_EQ(o04, iValue0); EXPECT_EQ(o05, iValue0); EXPECT_EQ(o06, iValue0); EXPECT_EQ(o07, iValue0); EXPECT_EQ(o08, iValue0); EXPECT_EQ(o09, iValue0);
    EXPECT_EQ(o010, iValue0); EXPECT_EQ(o011, iValue0); EXPECT_EQ(o012, iValue0); EXPECT_EQ(o013, iValue0); EXPECT_EQ(o014, iValue0); EXPECT_EQ(o015, iValue0); EXPECT_EQ(o016, iValue0); EXPECT_EQ(o017, iValue0); EXPECT_EQ(o018, iValue0);
    EXPECT_EQ(o019, iValue0); EXPECT_EQ(o020, iValue0); EXPECT_EQ(o021, iValue0); EXPECT_EQ(o022, iValue0); EXPECT_EQ(o023, iValue0); EXPECT_EQ(o024, iValue0); EXPECT_EQ(o025, iValue0); EXPECT_EQ(o026, iValue0); EXPECT_EQ(o027, iValue0);
    EXPECT_EQ(o028, iValue0); EXPECT_EQ(o029, iValue0); EXPECT_EQ(o030, iValue0); EXPECT_EQ(o031, iValue0); EXPECT_EQ(o032, iValue0); EXPECT_EQ(o033, iValue0); EXPECT_EQ(o034, iValue0); EXPECT_EQ(o035, iValue0); EXPECT_EQ(o036, iValue0);
    EXPECT_EQ(o037, iValue0); EXPECT_EQ(o038, iValue0); EXPECT_EQ(o039, iValue0);
}

TEST(Unsigned_Initialization, Binary) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;

    Aeu<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {};
        std::string binary {};
        for (auto byte = value.GetByte(value.ByteCount() - 1); byte; byte >>= 1)
            binary += (byte & 1 ? '1' : '0');
        ss << "0b" << std::string(binary.rbegin(), binary.rend());
        for(long long j = value.ByteCount() - 2; j >= 0; --j)
            ss << std::bitset<8>(value.GetByte(j));
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Unsigned_Initialization, Decimal) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;

    Aeu<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {}; ss << std::dec << value;
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Unsigned_Initialization, Octal) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;

    Aeu<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {}; ss << "0o" << std::oct << value;
        record = ss.str(); EXPECT_EQ(record, value);
    }
}

TEST(Unsigned_Initialization, Hexadecimal) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;

    Aeu<blocksNumber * 32> record {};
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        record = value; EXPECT_EQ(record, value);

        std::stringstream ss {};
        if(i % 2 == 0)
            ss << "0x" << std::hex << std::uppercase << value;
        else ss << "0x" << std::hex << std::nouppercase << value;
        record = ss.str(); EXPECT_EQ(record, value);
    }
}
