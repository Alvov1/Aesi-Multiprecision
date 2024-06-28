#include <gtest/gtest.h>
#include <format>

#ifndef AESI_CRYPTOPP_INTEGRATION
#define AESI_CRYPTOPP_INTEGRATION
#endif
#include "../../../Aeu.h"
#include "../../generation.h"

/* Output tester for zero values */
TEST(Unsigned_Display, Zero) {
    Aeu128 m = 0u;

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
TEST(Unsigned_Display, DecimalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss, ss2; ss << std::dec << std::noshowbase << value;
        ss2 << std::dec << std::noshowbase << aeu;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss; ss << std::dec << std::noshowbase << value;
        std::wstringstream ss2; ss2 << std::dec << std::noshowbase << aeu;
        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with octal notation */
TEST(Unsigned_Display, OctalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss, ss2; ss << "0o" << std::oct << std::noshowbase << value;
        ss2 << "0o" << std::oct << std::noshowbase << aeu;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss; ss << "0o" << std::oct << std::noshowbase << value;
        std::wstringstream ss2; ss2 << "0o" << std::oct << std::noshowbase << aeu;
        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with hexadecimal notation */
TEST(Unsigned_Display, HexadecimalStreams) {
    constexpr auto testsAmount = 30, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss, ss2; ss << "0x" << std::hex << std::noshowbase << value;
        ss2 << "0x" << std::hex << std::noshowbase << aeu;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss; ss << "0x" << std::hex << std::noshowbase << value;
        std::wstringstream ss2; ss2 << "0x" << std::hex << std::noshowbase << aeu;
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
TEST(Unsigned_Display, FormatAskii) {
    constexpr auto testsAmount = 20, blocksNumber = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aeu<blocksNumber * 32> aeu = value;

        switch(i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                const auto size = aeu.getString<8>(askii.data(), askii.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(askii.data(), askii.size(), false, false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                const auto size = aeu.getString<16>(askii.data(), askii.size(), false, true);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = aeu.getString<10>(askii.data(), askii.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << std::format("{:b}", value.GetByte(value.ByteCount() - 1));
                for(long long j = value.ByteCount() - 2; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                aeu.getString<2>(askii.data(), askii.size(), false);
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
TEST(Unsigned_Display, FormatUtf) {
    constexpr auto testsAmount = 20, blocksNumber = 2;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf{};

        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aeu<blocksNumber * 32> aeu = value;

        switch (i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                const auto size = aeu.getString<8>(utf.data(), utf.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(utf.data(), utf.size(), false, false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                const auto size = aeu.getString<16>(utf.data(), utf.size(), false, true);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = aeu.getString<10>(utf.data(), utf.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for (long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                aeu.getString<2>(utf.data(), utf.size(), false);
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
TEST(Unsigned_Display, ShowBaseAskii) {
    constexpr auto testsAmount = 36, blocksNumber = 2;
    for (std::size_t i = 8; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = CryptoPP::Integer(0b101100111111000100111010010010);//Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss, ss2;
        const Aeu<blocksNumber * 32> aeu = value;

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << "0o" << std::oct << std::noshowbase << value;
                ss2 << std::oct << std::showbase << aeu;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                ss2 << std::hex << std::showbase << std::nouppercase << aeu;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::uppercase << value;
                ss2 << std::hex << std::showbase << std::uppercase << aeu;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 3: {   /* Std::streams decimal */
                ss << std::dec << value;
                ss2 << std::dec << std::showbase << aeu;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 4: {   /* C-style ASKII Octal */
                ss << "0o" << std::oct << std::noshowbase << value;
                const auto size = aeu.getString<8>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(askii.data(), askii.size(), true, false);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(askii.data(), askii.size(), true, false);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = aeu.getString<10>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            default: {   /* C-style ASKII Binary */
                ss << "0b" << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for(long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                aeu.getString<2>(askii.data(), askii.size(), true);
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
TEST(Unsigned_Display, ShowBaseUtf) {
    constexpr auto testsAmount = 36, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf {};

        const auto value = Generation::getRandomWithBits(blocksNumber * 32 - 32);
        std::stringstream ss;
        std::wstringstream ss2;
        const Aeu<blocksNumber * 32> aeu = value;

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << "0o" << std::oct << std::noshowbase << value;
                ss2 << std::oct << std::showbase << aeu;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                ss2 << std::hex << std::showbase << std::nouppercase << aeu;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::uppercase << value;
                ss2 << std::hex << std::showbase << std::uppercase << aeu;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 3: {   /* Std::streams decimal */
                ss << std::dec << value;
                ss2 << std::dec << std::showbase << aeu;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 4: {   /* C-style ASKII Octal */
                ss << "0o" << std::oct << std::noshowbase << value;
                const auto size = aeu.getString<8>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(utf.data(), utf.size(), true, false);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                const auto size = aeu.getString<16>(utf.data(), utf.size(), true, false);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                const auto size = aeu.getString<10>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            default: {   /* C-style ASKII Binary */
                ss << "0b" << std::format("{:b}", value.GetByte((value.BitCount() - 1) / 8));
                for(long long j = (value.BitCount() - 1) / 8 - 1; j >= 0; --j)
                    ss << std::bitset<8>(value.GetByte(j));
                aeu.getString<2>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
            }
        }
    }
}
