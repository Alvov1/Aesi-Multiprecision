#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <iostream>
#include <array>

namespace {
    using block = uint32_t;
    constexpr auto blockBitLength = sizeof(block) * 8;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;
}

template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Multiprecision {
    static_assert(bitness > 64, "Use built-in types for numbers 64-bit or less.");

    static constexpr std::size_t blocksNumber = bitness / blockBitLength;
    using blockLine = std::array<block, blocksNumber>;
    enum Sign { Zero = 0, Positive = 1, Negative = 2 };

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    Sign sign { Zero };
    /* ----------------------------------------------------------------------- */

    /* --------------------------- Helper functions. ------------------------- */
    static constexpr auto addLine(blockLine& dst, const blockLine& src) noexcept -> uint64_t {
        uint64_t carryOut = 0;
        for (std::size_t i = 0; i < blocksNumber; ++i) {
            uint64_t sum = static_cast<uint64_t>(dst[i])
                                     + static_cast<uint64_t>(src[i]) + carryOut;
            carryOut = sum / blockBase;
            dst[i] = sum % blockBase;
        }
        return carryOut;
    }
    static constexpr auto makeComplement(const blockLine& line) noexcept -> blockLine {
        blockLine result {};

        uint64_t carryOut = 1;
        for(std::size_t i = 0; i < blocksNumber; ++i) {
            const uint64_t sum = blockBase - 1ULL - static_cast<uint64_t>(line[i]) + carryOut;
            carryOut = sum / blockBase; result[i] = sum % blockBase;
        }

        return result;
    }
    static constexpr auto isLineEmpty(const blockLine& line) noexcept -> bool {
        return lineLength(line) == 0;
    }
    static constexpr auto lineLength(const blockLine& line) noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(line[i]) return i + 1;
        return 0;
    }
    static constexpr auto divide(const Multiprecision& number, const Multiprecision& divisor) noexcept -> std::pair<Multiprecision, Multiprecision> {
        const Multiprecision divAbs = divisor.abs();
        const auto ratio = number.abs().operator<=>(divAbs);

        std::pair<Multiprecision, Multiprecision> results = { 0, 0 };
        auto& [quotient, remainder] = results;

        if(ratio == std::strong_ordering::greater) {
            const auto bitsUsed = lineLength(number.blocks) * blockBitLength;
            for(long long i = bitsUsed - 1; i >= 0; --i) {
                remainder <<= 1;
                remainder.setBit(0, number.getBit(i));

                if(remainder >= divAbs) {
                    remainder -= divAbs;
                    quotient.setBit(i, true);
                }
            }

            if(isLineEmpty(quotient.blocks))
                quotient.sign = Zero; else if(number.sign != divisor.sign) quotient = -quotient;
            if(isLineEmpty(remainder.blocks))
                remainder.sign = Zero; else if(number.sign == Negative) remainder = -remainder;
        } else if(ratio == std::strong_ordering::less)
            remainder = number; else quotient = 1;

        return results;
    }
    /* ----------------------------------------------------------------------- */

public:
    /* ----------------------- Different constructors. ----------------------- */
    constexpr Multiprecision() noexcept = default;
    constexpr Multiprecision(const Multiprecision& copy) noexcept = default;
    constexpr Multiprecision(Multiprecision&& move) noexcept = default;

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {
        if(value != 0) {
            uint64_t tValue {};
            if (value < 0) {
                sign = Negative;
                tValue = static_cast<uint64_t>(value * -1);
            } else {
                sign = Positive;
                tValue = static_cast<uint64_t>(value);
            }

            for (std::size_t i = 0; i < blocksNumber; ++i) {
                blocks[i] = static_cast<block>(tValue % blockBase);
                tValue /= blockBase;
            }
        }
    }

    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    constexpr Multiprecision(const Char (&array)[arrayLength]) noexcept : Multiprecision(std::basic_string_view<Char>(array)) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    constexpr Multiprecision(String&& stringView) noexcept {
        if(stringView.size() == 0) return;

        constexpr struct {
            Char minus = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return '-';
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return L'-';
                if constexpr (std::is_same_v<char8_t, Char>)
                    return u8'-';
                if constexpr (std::is_same_v<char16_t, Char>)
                    return u'-';
                if constexpr (std::is_same_v<char32_t, Char>)
                    return U'-';
            } ();
            Char zero = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return '0';
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return L'0';
                if constexpr (std::is_same_v<char8_t, Char>)
                    return u8'0';
                if constexpr (std::is_same_v<char16_t, Char>)
                    return u'0';
                if constexpr (std::is_same_v<char32_t, Char>)
                    return U'0';
            } ();
            Char nine = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return '9';
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return L'9';
                if constexpr (std::is_same_v<char8_t, Char>)
                    return u8'9';
                if constexpr (std::is_same_v<char16_t, Char>)
                    return u'9';
                if constexpr (std::is_same_v<char32_t, Char>)
                    return U'9';
            } ();
            std::pair<Char, Char> a = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return std::pair { 'a', 'A' };
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return std::pair { L'a', L'A' };
                if constexpr (std::is_same_v<char8_t, Char>)
                    return std::pair { u8'a', u8'A' };
                if constexpr (std::is_same_v<char16_t, Char>)
                    return std::pair { u'a', u'A' };
                if constexpr (std::is_same_v<char32_t, Char>)
                    return std::pair { U'a', U'A' };
            } ();
            std::pair<Char, Char> f = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return std::pair { 'f', 'F' };
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return std::pair { L'f', L'F' };
                if constexpr (std::is_same_v<char8_t, Char>)
                    return std::pair { u8'f', u8'F' };
                if constexpr (std::is_same_v<char16_t, Char>)
                    return std::pair { u'f', u'F' };
                if constexpr (std::is_same_v<char32_t, Char>)
                    return std::pair { U'f', U'F' };
            } ();
            std::pair<Char, Char> octal = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return std::pair { 'o', 'O' };
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return std::pair { L'o', L'O' };
                if constexpr (std::is_same_v<char8_t, Char>)
                    return std::pair { u8'o', u8'O' };
                if constexpr (std::is_same_v<char16_t, Char>)
                    return std::pair { u'o', u'O' };
                if constexpr (std::is_same_v<char32_t, Char>)
                    return std::pair { U'o', U'O' };
            } ();
            std::pair<Char, Char> binary = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return std::pair { 'b', 'B' };
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return std::pair { L'b', L'B' };
                if constexpr (std::is_same_v<char8_t, Char>)
                    return std::pair { u8'b', u8'B' };
                if constexpr (std::is_same_v<char16_t, Char>)
                    return std::pair { u'b', u'B' };
                if constexpr (std::is_same_v<char32_t, Char>)
                    return std::pair { U'b', U'B' };
            } ();
            std::pair<Char, Char> hexadecimal = [] {
                if constexpr (std::is_same_v<char, Char>)
                    return std::pair { 'x', 'X' };
                if constexpr (std::is_same_v<wchar_t, Char>)
                    return std::pair { L'x', L'X' };
                if constexpr (std::is_same_v<char8_t, Char>)
                    return std::pair { u8'x', u8'X' };
                if constexpr (std::is_same_v<char16_t, Char>)
                    return std::pair { u'x', u'X' };
                if constexpr (std::is_same_v<char32_t, Char>)
                    return std::pair { U'x', u'X' };
            } ();
        } characters;
        std::size_t position = 0;

        bool negative = false;
        if(stringView[position] == characters.minus) {
            negative = true; ++position;
        }

        const auto base = [&stringView, &position, &characters] {
            if (stringView[position] == characters.zero && stringView.size() > position + 1) {
                switch (stringView[position + 1]) {
                    case characters.binary.first:
                    case characters.binary.second:
                        position += 2;
                        return 2;
                    case characters.octal.first:
                    case characters.octal.second:
                        position += 2;
                        return 8;
                    case characters.hexadecimal.first:
                    case characters.hexadecimal.second:
                        position += 2;
                        return 16;
                    default:
                        return 10;
                }
            } else return 10;
        } ();
        for(; position < stringView.size(); ++position) {
            const auto digit = [&characters] (Char ch) {
                if(characters.zero <= ch && ch <= characters.nine)
                    return static_cast<int>(ch) - static_cast<int>(characters.zero);
                if(characters.a.first <= ch && ch <= characters.f.first)
                    return static_cast<int>(ch) - static_cast<int>(characters.a.first) + 10;
                if(characters.a.second <= ch && ch <= characters.f.second)
                    return static_cast<int>(ch) - static_cast<int>(characters.a.second) + 10;
                return 99;
            } (stringView[position]);

            if(digit < base) {
                this->operator*=(base);
                this->operator+=(digit);
            }
        }

        if(negative) sign = Negative;
    }

    constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        blocks = other.blocks; sign = other.sign; return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* ------------------------ Arithmetic operators. ------------------------ */
    constexpr Multiprecision operator+() const noexcept { return *this; }
    constexpr Multiprecision operator-() const noexcept {
        if(sign == Zero) return Multiprecision();
        Multiprecision result = *this;
        result.sign = (result.sign == Positive ? Negative : Positive); return result;
    }

    constexpr Multiprecision& operator++() noexcept { return this->operator+=(1); }
    constexpr Multiprecision operator++(int) & noexcept {
        Multiprecision old = *this; operator++(); return old;
    }
    constexpr Multiprecision& operator--() noexcept { return this->operator-=(1); }
    constexpr Multiprecision operator--(int) & noexcept {
        Multiprecision old = *this; operator--(); return old;
    }

    constexpr Multiprecision operator+(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result += value; return result;
    }
    constexpr Multiprecision& operator+=(const Multiprecision& value) noexcept {
        if(sign == Zero) return this->operator=(value);
        if(value.sign == Zero) return *this;

        if (sign != value.sign) {
            if (sign == Negative)
                blocks = makeComplement(blocks);
            const uint64_t carryOut = (value.sign != Negative ?
                                       addLine(blocks, value.blocks) : addLine(blocks, makeComplement(value.blocks)));
            if (carryOut == 0) {
                blocks = makeComplement(blocks);
                sign = Negative;
            } else sign = Positive;
        } else
            addLine(blocks, value.blocks);

        if (isLineEmpty(blocks))
            sign = Zero;

        return *this;
    }

    constexpr Multiprecision operator-(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result -= value; return result;
    }
    constexpr Multiprecision& operator-=(const Multiprecision& value) noexcept {
        return this->operator+=(-value);
    }

    constexpr Multiprecision operator*(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result *= value; return result;
    }
    constexpr Multiprecision& operator*=(const Multiprecision& value) noexcept {
        if(sign == Zero) return *this;
        if(value.sign == Zero)
            return this->operator=(Multiprecision());
        sign = (sign != value.sign ? Negative : Positive);

        constexpr auto multiplyLines = [] (const blockLine& longerLine, std::size_t longerLength,
                const blockLine& smallerLine, std::size_t smallerLength) {
            blockLine buffer {};

            for(std::size_t i = 0; i < longerLength; ++i) {
                uint64_t tBlock = longerLine[i], carryOut = 0;
                for(std::size_t j = 0; j < smallerLength; ++j) {
                    const auto product = tBlock * static_cast<uint64_t>(smallerLine[j]) + carryOut;
                    const auto block = static_cast<uint64_t>(buffer[i + j]) + (product % blockBase);
                    carryOut = product / blockBase + block / blockBase;
                    buffer[i + j] = block % blockBase;
                }
                if(smallerLength < blocksNumber)
                    buffer[smallerLength + i] += carryOut;
            }

            return buffer;
        };

        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
        if(thisLength > valueLength)
            blocks = multiplyLines(blocks, thisLength, value.blocks, valueLength);
        else
            blocks = multiplyLines(value.blocks, valueLength, blocks, thisLength);

        return *this;
    }

    constexpr Multiprecision operator/(const Multiprecision& divisor) const noexcept {
        return divide(*this, divisor).first;
    }
    constexpr Multiprecision& operator/=(const Multiprecision& divisor) noexcept {
        return this->operator=(divide(*this, divisor).first);
    }

    constexpr Multiprecision operator%(const Multiprecision& divisor) const noexcept {
        return divide(*this, divisor).second;
    }
    constexpr Multiprecision& operator%=(const Multiprecision& divisor) noexcept {
        return this->operator=(divide(*this, divisor).second);
    }
    /* ----------------------------------------------------------------------- */


    /* ------------------------- Bitwise operators. -------------------------- */
    constexpr Multiprecision operator~() const noexcept {
        Multiprecision result {};
        for(std::size_t i = 0; i < blocksNumber; ++i)
            result.blocks[i] = ~blocks[i];
        if(isLineEmpty(result.blocks))
            result.sign = Zero; else result.sign = sign;
        return result;
    }

    constexpr Multiprecision operator^(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result ^= value; return result;
    }
    constexpr Multiprecision& operator^=(const Multiprecision& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] ^= value.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
        return *this;
    }

    constexpr Multiprecision operator&(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result &= value; return result;
    }
    constexpr Multiprecision& operator&=(const Multiprecision& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] &= value.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
        return *this;
    }

    constexpr Multiprecision operator|(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result |= value; return result;
    }
    constexpr Multiprecision& operator|=(const Multiprecision& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] |= value.blocks[i];
        return *this;
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision operator<<(Integral bitShift) const noexcept {
        Multiprecision result = *this; result <<= bitShift; return result;
    }
    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision& operator<<=(Integral bitShift) noexcept {
        /*                    3                                               2                                     1                                     0
           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0011 0010 0110 0100 0011 0110 || 0010 0001 0001 1011 1011 0110 1001 0111 << 48
           0000 0000 0000 0000|0000 0000 0011 0010 || 0110 0100 0011 0110|0010 0001 0001 1011 || 1011 0110 1001 0111|0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000
            Правая половина 2   Левая половина 1   || Правая половина 1    Левая половина 0   || Правая половина 0   ---- ---- ---- ---- || ---- ---- ---- ---- ---- ---- ---- ----
           3992422065038923586049945894912 = 0x( 0000 0032 || 6436 211B || B697 0000 || 0000 0000 )

           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 1011 1000 0011 0110 1010 1101 || 1101 1100 0110 0101 0010 0110 1110 0100 << 40
           ---- ---- ---- ---- ---- ----|0000 0000 || 1011 1000 0011 0110 1010 1101|1101 1100 || 0110 0101 0010 0110 1110 0100|---- ---- || ---- ---- ---- ---- ---- ---- ---- ----
                      Право 2              Лево 1  ||            Право 1              Лево 0  ||            Право 0            ---- ---- || ---- ---- ---- ---- ---- ---- ---- ----
           -57011344836360679021114032128 = -0x( 0000 0000 || B836 ADDC || 6526 E400 || 0000 0000)

           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000
           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 << 32
           2852520100991549003783995392 = 0x( 0000 0000 || 0937 8CA0 || 8716 1000 || 0000 0000 )

           0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 << 64
           12251480544941320143633640456854700032 = 0x( 0937 8CA0 || 8716 1000 || 0000 0000 || 0000 0000 )
        */

        if(bitShift < 0)
            return this->operator>>=(-bitShift);

        if(bitShift < bitness && bitShift > 0) {
            const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
            const block stamp = (1UL << (blockBitLength - remainder)) - 1;

            for (long long i = blocksNumber - 1; i >= (quotient + (remainder ? 1 : 0)); --i)
                blocks[i] = ((blocks[i - quotient] & stamp) << remainder) | ((blocks[i - quotient - (remainder ? 1 : 0)] & ~stamp) >> (blockBitLength - remainder));

            blocks[quotient] = (blocks[0] & stamp) << remainder;

            for (std::size_t i = 0; i < quotient; ++i)
                blocks[i] = 0;

            if(isLineEmpty(blocks)) sign = Zero;
        }

        return *this;
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision operator>>(Integral bitShift) const noexcept {
        Multiprecision result = *this; result >>= bitShift; return result;
    }
    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision& operator>>=(Integral bitShift) noexcept {
        /*                    3                                               2                                     1                                     0
           0000 0000 0000 0000 0000 0000 0011 0010 || 0110 0100 0011 0110 0010 0001 0001 1011 || 1011 0110 1001 0111 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 >> 48
           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0011 0010|0110 0100 0011 0110 || 0010 0001 0001 1011|1011 0110 1001 0111
           ---- ---- ---- ---- ---- ---- ---- ---- || ---- ---- ---- ----    Левая часть 3    ||    Правая часть 3      Левая часть 2    ||    Правая часть 2       Левая часть 1
           14183932482008727 = 0x( 0000 0000 || 0000 0000 || 0032 6436 || 211B B697 )

           0000 0000 0000 0000 0000 0000 0000 0000 || 1011 1000 0011 0110 1010 1101 1101 1100 || 0110 0101 0010 0110 1110 0100 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 >> 40
           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000|1011 1000 0011 0110 1010 1101 || 1101 1100|0110 0101 0010 0110 1110 0100
           ---- ---- ---- ---- ---- ---- ---- ---- || ---- ---- ---- ---- ---- ---- ---- ---- ||  Право 3             Лево 2             ||  Право 2             Лево 1
           51851516069619428 = 0x( 0000 0000 || 0000 0000 || 00B8 36AD || DC65 26E4 )

           0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000
           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 >> 32
           2852520100991549003783995392 = 0x( 0000 0000 || 0937 8CA0 || 8716 1000 || 0000 0000 )

           0000 0000 0000 0000 0000 0000 0000 0000 || 0000 0000 0000 0000 0000 0000 0000 0000 || 0000 1001 0011 0111 1000 1100 1010 0000 || 1000 0111 0001 0110 0001 0000 0000 0000 >> 64
           664154091149463552 = 0x( 0000 0000 || 0000 0000 || 0937 8CA0 || 8716 1000 )
        */

        if(bitShift < 0)
            return this->operator<<=(-bitShift);

        if(bitShift < bitness && bitShift > 0) {
            const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
            const block stamp = (1UL << remainder) - 1;

            for(std::size_t i = 0; i < blocksNumber - (quotient + (remainder ? 1 : 0)); ++i)
                blocks[i] = ((blocks[i + quotient + (remainder ? 1 : 0)] & stamp) << (blockBitLength - remainder)) | ((blocks[i + quotient] & ~stamp) >> remainder);

            blocks[blocksNumber - 1 - quotient] = (blocks[blocksNumber - 1] & ~stamp) >> remainder;

            for(long long i = blocksNumber - quotient; i < blocksNumber; ++i)
                blocks[i] = 0;

            if(isLineEmpty(blocks)) sign = Zero;
        }

        return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* ----------------------- Comparison operators. ------------------------- */
    constexpr bool operator==(const Multiprecision& value) const noexcept = default;
    constexpr std::strong_ordering operator<=>(const Multiprecision& value) const noexcept {
        switch (sign) {
            case Zero:
                switch (value.sign) {
                    case Zero: return std::strong_ordering::equal;
                    case Positive: return std::strong_ordering::less;
                    case Negative: return std::strong_ordering::greater;
                    default: return std::strong_ordering::equivalent;
                }
            case Positive:
                switch (value.sign) {
                    case Positive: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength) return thisLength <=> valueLength;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i]) return blocks[i] <=> value.blocks[i];

                        return std::strong_ordering::equal;
                    }
                    case Zero:
                    case Negative: return std::strong_ordering::greater;
                    default: return std::strong_ordering::equivalent;
                }
            case Negative:
                switch (value.sign) {
                    case Negative: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength) return (static_cast<long long>(thisLength) * -1) <=> (static_cast<long long>(valueLength) * -1);

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i]) return (static_cast<long>(blocks[i]) * -1) <=> (static_cast<long>(value.blocks[i]) * -1);

                        return std::strong_ordering::equal;
                    }
                    case Zero:
                    case Positive: return std::strong_ordering::less;
                    default: return std::strong_ordering::equivalent;
                }
            default: return std::strong_ordering::equivalent;
        }
    };
    /* ----------------------------------------------------------------------- */


    /* ------------------------ Arithmetic methods. -------------------------- */
    [[nodiscard]] constexpr auto getBit(std::size_t index) const noexcept -> bool {
        if(index >= bitness) return false;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        return blocks[blockNumber] & (1U << bitNumber);
    }
    constexpr auto setBit(std::size_t index, bool value) noexcept -> void {
        if(index >= bitness) return;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        if(value) {
            blocks[blockNumber] |= (1U << bitNumber);
            if(sign == Zero && !isLineEmpty(blocks))
                sign = Positive;
        } else {
            blocks[blockNumber] &= (~(1U << bitNumber));
            if(sign != Zero && isLineEmpty(blocks))
                sign = Zero;
        }
    }
    [[nodiscard]] constexpr auto abs() const noexcept -> Multiprecision {
        if(sign == Zero)
            return *this;
        Multiprecision result = *this; result.sign = Positive; return result;
    }
    /* ----------------------------------------------------------------------- */

    constexpr friend std::ostream& operator<<(std::ostream& ss, const Multiprecision& value) noexcept {
        auto flags = ss.flags();

        if(value.sign != Zero) {
            if (value.sign == Negative) ss.write("-", 1);

            const auto base = [] (long baseField, std::ostream& ss, bool showbase) {
                auto base = (baseField == std::ios::hex ? 16 : (baseField == std::ios::oct ? 8 : 10));
                if(showbase && base != 10) ss.write(base == 8 ? "0o" : "0x", 2);
                return base;
            } (flags & std::ios::basefield, ss, flags & std::ios::showbase);


            auto iter = value.blocks.rbegin();
            for(; *iter == 0 && iter != value.blocks.rend(); ++iter);

            if(base == 16) {
                ss << *iter++;
                for (; iter != value.blocks.rend(); ++iter) {
                    ss.fill('0'); ss.width(8); ss << std::right << *iter;
                }
            } else {
                /* Well, here we use a pre-calculated magic number to ratio the lengths of numbers in decimal or octal notation according to bitness.
                 * It is 2.95-98 for octal and 3.2 for decimal. */
                constexpr auto bufferSize = static_cast<std::size_t>(static_cast<double>(bitness) / 2.95);
                std::array<char, bufferSize> buffer {}; std::size_t filled = 0;

                Multiprecision copy = value;
                while(copy != 0) {
                    auto [quotient, remainder] = divide(copy, base);
                    buffer[filled++] = '0' + remainder.template integralCast<unsigned long>();
                    copy = quotient;
                }

                for(; filled > 0; --filled)
                    ss << buffer[filled - 1];
            }
        } else ss.write("0", 1);

        return ss;
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr auto integralCast() const noexcept -> Integral {
        const uint64_t value = (static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0]);
        if constexpr (std::is_signed_v<Integral>)
            return static_cast<Integral>(value) * (sign == Negative ? -1 : 1); else return static_cast<Integral>(value);
    }

    template <std::size_t length>
    constexpr explicit Multiprecision(const std::array<block, length>& data) noexcept {
        /* FIXME: Remove this function, change precision cast to use bitness. */
        for(std::size_t i = 0; i < blocksNumber && i < length; ++i)
            blocks[i] = data[i]; sign = Positive;
        /* FIXME */
    }

    template <std::size_t newBitness> requires (newBitness > bitness)
    constexpr auto precisionCast() const noexcept -> Multiprecision<newBitness> {
        Multiprecision<newBitness> result (blocks);
        if(sign == Negative) return -result; return result;
    }
    /*  TODO: Fixed version of precision cast
        template <std::size_t newBitness> requires (newBitness > bitness)
        constexpr auto precisionCast() const noexcept -> Multiprecision<newBitness> {
            Multiprecision<newBitness> result;
            for(block block: blocks) {
                result <<= blockBitLength;
                result |= block;
            }

            if(sign == Negative)
                return -result;
            return result;
        }
    */
};

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr bool operator==(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        Multiprecision<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return (first == reducedSecond);
    } else {
        Multiprecision<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return (reducedFirst == second);
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr std::strong_ordering operator<=>(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        Multiprecision<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return (first <=> reducedSecond);
    } else {
        Multiprecision<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return (reducedFirst <=> second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision addition ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator+(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) + value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator+(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) noexcept
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        Multiprecision<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return first + reducedSecond;
    } else {
        Multiprecision<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return reducedFirst + second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator+=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator+=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator-(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) - value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator-(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first - second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() - second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator-=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator-=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator*(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) * value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator*(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first * second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() * second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator*=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator*=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator/(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) / value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator/(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first / second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() / second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator/=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator/=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator%(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) % value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator%(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first % second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() % second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator%=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator%=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator^(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) ^ value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator^(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first ^ second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() ^ second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator^=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator^=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator&(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) & value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator&(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first & second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() & second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator&=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator&=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
constexpr auto operator|(Integral number, const Multiprecision<bitness>& value) noexcept {
    return Multiprecision<bitness>(number) | value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
constexpr auto operator|(const Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Multiprecision<bFirst>, Multiprecision<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first | second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() | second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
constexpr auto operator|=(Multiprecision<bFirst>& first, const Multiprecision<bSecond>& second) -> Multiprecision<bFirst>& {
    return first.operator|=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //METALMULTIPRECISION_MULTIPRECISION_H
