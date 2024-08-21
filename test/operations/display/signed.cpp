#include <gtest/gtest.h>
#include <format>

#ifndef AESI_CRYPTOPP_INTEGRATION
#define AESI_CRYPTOPP_INTEGRATION
#endif
#include "../../../Aesi.h"
#include "../../generation.h"

/* Output tester for zero values */
TEST(Signed_Display, Zero) {
    Aesi128 m = 0u;

    { std::stringstream ss1{}; ss1 << m << +m; EXPECT_EQ(ss1.str(), "00"); }
    { std::stringstream ss2{}; ss2 << std::dec << m << +m; EXPECT_EQ(ss2.str(), "00"); }
    { std::stringstream ss3{}; ss3 << std::oct << m << +m; EXPECT_EQ(ss3.str(), "00"); }
    { std::stringstream ss4{}; ss4 << std::hex << m << +m; EXPECT_EQ(ss4.str(), "00"); }
    { std::stringstream ss5{}; m = 8u; ss5 << m - 8u << +(m - 8u); m -= 8u; ss5 << m; EXPECT_EQ(ss5.str(), "000"); }

    {
        m = 0u;
        using namespace std::string_view_literals;

        std::array<char, 10> askii{};
        std::array<wchar_t, 10> utf{};
        auto size = m.getString<10>(askii.data(), 10, false);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        size = m.getString<2>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0b0"sv);

        askii = {};
        size = m.getString<8>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0o0"sv);

        askii = {};
        size = m.getString<10>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        size = m.getString<16>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0x0"sv);

        size = m.getString<10>(utf.data(), 10);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        size = m.getString<2>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0b0"sv);

        utf = {};
        size = m.getString<8>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0o0"sv);

        utf = {};
        size = m.getString<10>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        size = m.getString<16>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0x0"sv);
    }
}

/* Output tester for std::streams and std::wstreams with decimal notation */
TEST(Signed_Display, DecimalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss, ss2; ss << std::dec << std::noshowbase << value;
        ss2 << std::dec << std::noshowbase << Aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss; ss << std::dec << std::noshowbase << value;
        std::wstringstream ss2; ss2 << std::dec << std::noshowbase << Aesi;
        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with octal notation */
TEST(Signed_Display, OctalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> aesi = value;

        std::stringstream ss, ss2; ss << std::oct << std::noshowbase << value;
        ss2 << std::oct << std::noshowbase << aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss; ss << std::oct << std::noshowbase << value;
        std::wstringstream ss2; ss2 << std::oct << std::noshowbase << Aesi;
        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with hexadecimal notation */
TEST(Signed_Display, HexadecimalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss, ss2; ss << std::hex << std::noshowbase << value;
        ss2 << std::hex << std::noshowbase << Aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss; ss << std::hex << std::noshowbase << value;
        std::wstringstream ss2; ss2 << std::hex << std::noshowbase << Aesi;
        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for ASKII c-style arrays with different notations
 * Generates:
 *  - ASKII Octal
 *  - ASKII Hexadecimal, lowercase
 *  - ASKII Hexadecimal, uppercase
 *  - ASKII Decimal
 *  - ASKII Binary
 *  */
TEST(Signed_Display, FormatAskii) {
    constexpr auto testsAmount = 20, blocksNumber = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch(i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                const auto size = Aesi.getString<8>(askii.data(), askii.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = Aesi.getString<16>(askii.data(), askii.size(), false, false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                const auto size = Aesi.getString<16>(askii.data(), askii.size(), false, true);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = Aesi.getString<10>(askii.data(), askii.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << (i % 2 == 0 ? "" : "-") << std::format("{:b}", value.GetByte(value.ByteCount() - 1));
                for(long long j = value.ByteCount() - 2; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                Aesi.getString<2>(askii.data(), askii.size(), false);
            }
        }

        EXPECT_EQ(std::string_view(askii.data()), ss.str());
    }
}

/* Output tester for UTF c-style arrays with different notations
 * Generates:
 *  - UTF Octal
 *  - UTF Hexadecimal, lowercase
 *  - UTF Hexadecimal, uppercase
 *  - UTF Decimal
 *  - UTF Binary
 *  */
TEST(Signed_Display, FormatUtf) {
    constexpr auto testsAmount = 20, blocksNumber = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf{};

        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch (i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                const auto size = Aesi.getString<8>(utf.data(), utf.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = Aesi.getString<16>(utf.data(), utf.size(), false, false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                const auto size = Aesi.getString<16>(utf.data(), utf.size(), false, true);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = Aesi.getString<10>(utf.data(), utf.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << (i % 2 == 0 ? "" : "-") << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for (long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                Aesi.getString<2>(utf.data(), utf.size(), false);
            }
        }

        const auto &ref = ss.str();
        const std::wstring comparative(ref.begin(), ref.end());
        EXPECT_EQ(std::wstring_view(utf.data()), comparative);
    }
}

/* Output tester for SHOWBASE option for ASKII
 * Generates:
 * - Std::streams octal
 * - Std::streams hexadecimal, lowercase
 * - Std::streams hexadecimal, uppercase
 * - Std::streams decimal
 *
 *  - C-style ASKII Octal
 *  - C-style ASKII Hexadecimal, lowercase
 *  - C-style ASKII Hexadecimal, uppercase
 *  - C-style ASKII Decimal
 *  - C-style ASKII Binary
 *  */
TEST(Signed_Display, ShowBaseAskii) {
    constexpr auto testsAmount = 36, blocksNumber = 2;
    for (std::size_t i = 8; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss, ss2;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::oct << std::showbase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::hex << std::showbase << std::nouppercase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::uppercase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::hex << std::showbase << std::uppercase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 3: {   /* Std::streams decimal */
                ss << std::dec << value;
                ss2 << std::dec << std::showbase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 4: {   /* C-style ASKII Octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<8>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<16>(askii.data(), askii.size(), true, false);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<16>(askii.data(), askii.size(), true, false);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = Aesi.getString<10>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            default: {   /* C-style ASKII Binary */
                ss << (i % 2 == 0 ? "0b" : "-0b") << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for(long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                Aesi.getString<2>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
            }
        }
    }
}

/* Output tester for SHOWBASE option for UTF
 * Generates:
 * - Std::Wstreams octal
 * - Std::Wstreams hexadecimal, lowercase
 * - Std::Wstreams hexadecimal, uppercase
 * - Std::Wstreams decimal
 *
 *  - C-style UTF Octal
 *  - C-style UTF Hexadecimal, lowercase
 *  - C-style UTF Hexadecimal, uppercase
 *  - C-style UTF Decimal
 *  - C-style UTF Binary
 *  */
TEST(Signed_Display, ShowBaseUtf) {
    constexpr auto testsAmount = 36, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf {};

        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        std::wstringstream ss2;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::oct << std::showbase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x")  << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::hex << std::showbase << std::nouppercase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x")  << std::hex << std::noshowbase << std::uppercase << (i % 2 == 0 ? value : value * -1);
                ss2 << std::hex << std::showbase << std::uppercase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 3: {   /* Std::streams decimal */
                ss << std::dec << value;
                ss2 << std::dec << std::showbase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 4: {   /* C-style ASKII Octal */
                ss << (i % 2 == 0 ? "0o" : "-0o")  << std::oct << std::noshowbase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<8>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x")  << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<16>(utf.data(), utf.size(), true, false);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x")  << std::hex << std::noshowbase << std::nouppercase << (i % 2 == 0 ? value : value * -1);
                const auto size = Aesi.getString<16>(utf.data(), utf.size(), true, false);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = Aesi.getString<10>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            default: {   /* C-style ASKII Binary */
                ss << (i % 2 == 0 ? "0b" : "-0b")  << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for(long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                Aesi.getString<2>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
            }
        }
    }
}
