#include <gtest/gtest.h>
#include <bitset>

#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

static std::string gmpToBinaryString(const mpz_class& v) {
    const std::size_t bitCount = mpz_sizeinbase(v.get_mpz_t(), 2);
    const std::size_t byteCount = (bitCount + 7) / 8;
    auto getByte = [&](std::size_t k) -> unsigned char {
        return static_cast<unsigned char>(mpz_class(v >> (8 * k)).get_ui() & 0xFF);
    };
    std::string msb;
    for (auto byte = getByte((bitCount - 1) / 8); byte; byte >>= 1)
        msb += (byte & 1 ? '1' : '0');
    std::string result(msb.rbegin(), msb.rend());
    for (std::size_t j = byteCount - 1; j-- > 0;)
        result += std::bitset<8>(getByte(j)).to_string();
    return result;
}

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
        m.getString<10>(askii.data(), 10, false);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        m.getString<2>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0b0"sv);

        askii = {};
        m.getString<8>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0o0"sv);

        askii = {};
        m.getString<10>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        m.getString<16>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0x0"sv);

        m.getString<10>(utf.data(), 10);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        m.getString<2>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0b0"sv);

        utf = {};
        m.getString<8>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0o0"sv);

        utf = {};
        m.getString<10>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        m.getString<16>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0x0"sv);
    }
}

/* Output tester for std::streams and std::wstreams with decimal notation */
TEST(Unsigned_Display, DecimalStreams) {
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss, ss2;
        ss << std::dec << std::noshowbase << value;
        ss2 << std::dec << std::noshowbase << aeu;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss;
        ss << std::dec << std::noshowbase << value;
        std::wstringstream ss2;
        ss2 << std::dec << std::noshowbase << aeu;

        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());
        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with octal notation */
TEST(Unsigned_Display, OctalStreams) {
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss {};
        ss << "0o" << std::oct << std::noshowbase << value;

        std::stringstream ss2 {};
        ss2 << "0o" << std::oct << std::noshowbase << aeu;

        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss {};
        ss << "0o" << std::oct << std::noshowbase << value;

        std::wstringstream ss2 {};
        ss2 << "0o" << std::oct << std::noshowbase << aeu;

        const auto& ref = ss.str();
        std::wstring wstring (ref.begin(), ref.end());

        EXPECT_EQ(ss2.str(), wstring);
    }
}

/* Output tester for std::streams and std::wstreams with hexadecimal notation */
TEST(Unsigned_Display, HexadecimalStreams) {
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss {};
        ss << "0x" << std::hex << std::noshowbase << value;

        std::stringstream ss2 {};
        ss2 << "0x" << std::hex << std::noshowbase << aeu;

        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        const Aeu<blocksNumber * 32> aeu = value;

        std::stringstream ss;
        ss << "0x" << std::hex << std::noshowbase << value;

        std::wstringstream ss2;
        ss2 << "0x" << std::hex << std::noshowbase << aeu;

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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aeu<blocksNumber * 32> aeu = value;

        switch(i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                aeu.getString<8>(askii.data(), askii.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(askii.data(), askii.size(), false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                aeu.getString<16, true>(askii.data(), askii.size(), false);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                aeu.getString<10>(askii.data(), askii.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << gmpToBinaryString(value);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf{};

        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aeu<blocksNumber * 32> aeu = value;

        switch (i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                aeu.getString<8>(utf.data(), utf.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(utf.data(), utf.size(), false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                aeu.getString<16, true>(utf.data(), utf.size(), false);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                aeu.getString<10>(utf.data(), utf.size(), false);
                break;
            }
            default: {  /* Binary */
                ss << gmpToBinaryString(value);
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
    constexpr auto testsAmount = 36, blocksNumber = 16;
    for (std::size_t i = 8; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
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
                aeu.getString<8>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                aeu.getString<10>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            default: {   /* C-style ASKII Binary */
                ss << "0b" << gmpToBinaryString(value);
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
    constexpr auto testsAmount = 36, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf {};

        const auto value = Generation::getRandom(blocksNumber * 32 - 32);
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
                aeu.getString<8>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << "0x" << std::hex << std::noshowbase << std::nouppercase << value;
                aeu.getString<16>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                aeu.getString<10>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            default: {   /* C-style UTF Binary */
                ss << "0b" << gmpToBinaryString(value);
                aeu.getString<2>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
            }
        }
    }
}

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
        m.getString<10>(askii.data(), 10, false);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        m.getString<2>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0b0"sv);

        askii = {};
        m.getString<8>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0o0"sv);

        askii = {};
        m.getString<10>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0"sv);

        askii = {};
        m.getString<16>(askii.data(), 10, true);
        EXPECT_EQ(std::string_view(askii.data()), "0x0"sv);

        m.getString<10>(utf.data(), 10);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        m.getString<2>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0b0"sv);

        utf = {};
        m.getString<8>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0o0"sv);

        utf = {};
        m.getString<10>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0"sv);

        utf = {};
        m.getString<16>(utf.data(), 10, true);
        EXPECT_EQ(std::wstring_view(utf.data()), L"0x0"sv);
    }
}

/* Output tester for std::streams and std::wstreams with decimal notation */
TEST(Signed_Display, DecimalStreams) {
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss, ss2; ss << std::dec << std::noshowbase << value;
        ss2 << std::dec << std::noshowbase << Aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> aesi = value;

        std::stringstream ss, ss2; ss << std::oct << std::noshowbase << value;
        ss2 << std::oct << std::noshowbase << aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        const Aesi<blocksNumber * 32> Aesi = value;

        std::stringstream ss, ss2; ss << std::hex << std::noshowbase << value;
        ss2 << std::hex << std::noshowbase << Aesi;
        EXPECT_EQ(ss2.str(), ss.str());
    }

    for (std::size_t i = 0; i < testsAmount; ++i) {
        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch(i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                Aesi.getString<8>(askii.data(), askii.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                Aesi.getString<16>(askii.data(), askii.size(), false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                Aesi.getString<16, true>(askii.data(), askii.size(), false);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                Aesi.getString<10>(askii.data(), askii.size(), false);
                break;
            }
            default: {  /* Binary */
                mpz_class absVal; mpz_abs(absVal.get_mpz_t(), value.get_mpz_t());
                ss << (i % 2 == 0 ? "" : "-") << gmpToBinaryString(absVal);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf{};

        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss;
        const Aesi<blocksNumber * 32> Aesi = value;

        switch (i % 5) {
            case 0: {   /* Octal */
                ss << std::oct << std::noshowbase << value;
                Aesi.getString<8>(utf.data(), utf.size(), false);
                break;
            }
            case 1: {   /* Hexadecimal, lowercase */
                ss << std::hex << std::noshowbase << std::nouppercase << value;
                Aesi.getString<16>(utf.data(), utf.size(), false);
                break;
            }
            case 2: {   /* Hexadecimal, uppercase */
                ss << std::hex << std::noshowbase << std::uppercase << value;
                Aesi.getString<16, true>(utf.data(), utf.size(), false);
                break;
            }
            case 3: {  /* Decimal */
                ss << std::dec << std::noshowbase << value;
                Aesi.getString<10>(utf.data(), utf.size(), false);
                break;
            }
            default: {  /* Binary */
                mpz_class absVal; mpz_abs(absVal.get_mpz_t(), value.get_mpz_t());
                ss << (i % 2 == 0 ? "" : "-") << gmpToBinaryString(absVal);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 8; i < testsAmount; ++i) {
        static std::array<char, blocksNumber * 32 + 2> askii {};

        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss, ss2;
        const Aesi<blocksNumber * 32> Aesi = value;
        mpz_class absValue; mpz_abs(absValue.get_mpz_t(), value.get_mpz_t());

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << absValue;
                ss2 << std::oct << std::showbase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                ss2 << std::hex << std::showbase << std::nouppercase << Aesi;
                EXPECT_EQ(ss2.str(), ss.str());
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::uppercase << absValue;
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
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << absValue;
                Aesi.getString<8>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 5: {   /* C-style ASKII Hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                Aesi.getString<16>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 6: {   /* C-style ASKII Hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                Aesi.getString<16>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            case 7: {   /* C-style ASKII Decimal */
                ss << std::dec << std::noshowbase << value;
                Aesi.getString<10>(askii.data(), askii.size(), true);
                EXPECT_EQ(std::string_view(askii.data()), ss.str());
                break;
            }
            default: {   /* C-style Binary */
                ss << (i % 2 == 0 ? "0b" : "-0b") << gmpToBinaryString(absValue);
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
    constexpr auto testsAmount = 64, blocksNumber = 16;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        static std::array<wchar_t, blocksNumber * 32 + 2> utf {};

        const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(blocksNumber * 32 - 32);
        std::stringstream ss;
        std::wstringstream ss2;
        const Aesi<blocksNumber * 32> Aesi = value;
        mpz_class absValue; mpz_abs(absValue.get_mpz_t(), value.get_mpz_t());

        switch (i % 9) {
            case 0: {   /* Std::streams octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << absValue;
                ss2 << std::oct << std::showbase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 1: {   /* Std::streams hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                ss2 << std::hex << std::showbase << std::nouppercase << Aesi;
                const auto& ref = ss.str();
                std::wstring wstring (ref.begin(), ref.end());
                EXPECT_EQ(ss2.str(), wstring);
                break;
            }
            case 2: {   /* Std::streams hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::uppercase << absValue;
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
            case 4: {   /* C-style UTF Octal */
                ss << (i % 2 == 0 ? "0o" : "-0o") << std::oct << std::noshowbase << absValue;
                Aesi.getString<8>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 5: {   /* C-style UTF Hexadecimal, lowercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                Aesi.getString<16>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 6: {   /* C-style UTF Hexadecimal, uppercase */
                ss << (i % 2 == 0 ? "0x" : "-0x") << std::hex << std::noshowbase << std::nouppercase << absValue;
                Aesi.getString<16>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            case 7: {   /* C-style UTF Decimal */
                ss << std::dec << std::noshowbase << value;
                Aesi.getString<10>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
                break;
            }
            default: {   /* C-style UTF Binary */
                ss << (i % 2 == 0 ? "0b" : "-0b");
                std::string binary {};
                const std::size_t bitCount = mpz_sizeinbase(absValue.get_mpz_t(), 2);
                const std::size_t byteCount = (bitCount + 7) / 8;
                auto getByteGmp = [&](std::size_t k) -> unsigned char {
                    return static_cast<unsigned char>(mpz_class(absValue >> (8 * k)).get_ui() & 0xFF);
                };
                for (auto byte = getByteGmp((bitCount - 1) / 8); byte; byte >>= 1)
                    binary += (byte & 1 ? '1' : '0');
                ss << std::string(binary.rbegin(), binary.rend());
                for (std::size_t j = byteCount - 1; j-- > 0;)
                    ss << std::bitset<8>(getByteGmp(j));
                Aesi.getString<2>(utf.data(), utf.size(), true);
                const auto &ref = ss.str();
                const std::wstring comparative(ref.begin(), ref.end());
                EXPECT_EQ(std::wstring_view(utf.data()), comparative);
            }
        }
    }
}