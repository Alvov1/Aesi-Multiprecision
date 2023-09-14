#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <iostream>
#include <array>

namespace {
    using block = uint32_t;
    constexpr auto blockBitLength = sizeof(block) * 8;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;
    constexpr uint64_t blockMax = std::numeric_limits<block>::max();
}

template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Multiprecision {
    static constexpr std::size_t blocksNumber = bitness / blockBitLength;
    using blockLine = std::array<block, blocksNumber>;
    enum Sign { Zero = 0, Positive = 1, Negative = 2 };

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    Sign sign { Zero };
    /* ----------------------------------------------------------------------- */

    /* --------------------------- Helper functions. ------------------------- */
    template <typename Char>
    static constexpr auto strlen(const Char *str) noexcept -> std::size_t {
        const char *s = str;
        for (; *s; ++s);
        return(s - str);
    }
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
        for(std::size_t i = 0; i < blocksNumber; ++i)
            if(line[i] != 0) return false;
        return true;
    }
    static constexpr auto lineLength(const blockLine& line) noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(line[i]) return i + 1;
        return 0;
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

    template <typename Char, std::size_t arrayLength> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    constexpr Multiprecision(const Char (&array)[arrayLength]) noexcept : Multiprecision(std::basic_string_view<Char>(array)) {}

    template <typename String, typename Char = typename String::value_type>
    requires (std::is_same_v<std::basic_string<Char>, typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
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
            if (stringView[position] == characters.zero) {
                switch (stringView[++position]) {
                    case characters.binary.first:
                    case characters.binary.second:
                        ++position;
                        return 2;
                    case characters.octal.first:
                    case characters.octal.second:
                        ++position;
                        return 8;
                    case characters.hexadecimal.first:
                    case characters.hexadecimal.second:
                        ++position;
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
            this->operator*=(base); this->operator+=(digit);
        }

        if(negative) sign = Negative;
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

    constexpr Multiprecision operator/(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result /= value; return result;
    }
    constexpr Multiprecision& operator/=(const Multiprecision& value) noexcept {
        return *this;
    }

    constexpr Multiprecision operator%(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result %= value; return result;
    }
    constexpr Multiprecision& operator%=(const Multiprecision& value) noexcept { return *this; }
    /* ----------------------------------------------------------------------- */


    /* ------------------------- Bitwise operators. -------------------------- */
    constexpr Multiprecision operator~() const noexcept {}

    constexpr Multiprecision operator^(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result ^= value; return result;
    }
    constexpr Multiprecision& operator^=(const Multiprecision& value) noexcept { return *this; }

    constexpr Multiprecision operator&(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result &= value; return result;
    }
    constexpr Multiprecision& operator&=(const Multiprecision& value) noexcept { return *this; }

    constexpr Multiprecision operator|(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result |= value; return result;
    }
    constexpr Multiprecision& operator|=(const Multiprecision& value) noexcept { return *this; }

    constexpr Multiprecision operator<<(const Multiprecision& bitShift) const noexcept {
        Multiprecision result = *this; result <<= bitShift; return result;
    }
    constexpr Multiprecision& operator<<=(const Multiprecision& bitShift) noexcept {
//        if(bitShift >= bitness) return *this;

//        const std::size_t blockShift = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
//        const block stamp = [] (std::size_t remainder) {
//            block value {};
//            for(uint8_t i = 0; i < remainder; ++i) {
//                value <<= 1;
//                value |= 1;
//            }
//            return value;
//        } (remainder);
//        for(long long i = blocksNumber - 1; i >= 0; --i);

//        if(bitShift != blockBitLength) {
//        } else
//            for(auto iter = blocks.rbegin(); iter != (blocks.rend() - 1); ++iter)
//                *iter = *(iter + 1);

        return *this;
    }

    constexpr Multiprecision operator>>(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result >>= value; return result;
    }
    constexpr Multiprecision& operator>>=(const Multiprecision& value) noexcept { return *this; }
    /* ----------------------------------------------------------------------- */

    /* ----------------------- Comparison operators. ------------------------- */
    constexpr auto operator<=>(const Multiprecision& value) const noexcept = default;
    /* ----------------------------------------------------------------------- */

    constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        blocks = other.blocks; sign = other.sign; return *this;
    }

    auto outputDecimal(std::ostream& toStream) const noexcept -> void {}
    auto outputOctal(std::ostream& toStream) const noexcept -> void {}
    auto outputHexadecimal(std::ostream& toStream) const noexcept -> void {
        if(sign == Negative) toStream.write("-0x", 3);
            else toStream.write("0x", 2);

        bool firstFilledBlockWritten = false;
        for(auto iter = blocks.rbegin(); iter != blocks.rend(); ++iter) {
            firstFilledBlockWritten = firstFilledBlockWritten || *iter;
            if(*iter || firstFilledBlockWritten)
                toStream << std::hex << *iter;
        }
    }

    friend std::ostream& operator<<(std::ostream& stream, const Multiprecision& value) noexcept {
        value.outputHexadecimal(stream);
        return stream;
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
        if(sign == Negative) result = -result; return result;
    }
};

template <std::size_t length, typename Integral> requires (std::is_integral_v<Integral>)
auto operator+(Integral number, const Multiprecision<length>& value) noexcept {
    return Multiprecision<length>(number) + value;
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst != lSecond)
auto operator+(const Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) noexcept
-> typename std::conditional<(lFirst > lSecond), Multiprecision<lFirst>, Multiprecision<lSecond>>::type {
    if constexpr (lFirst > lSecond) {
        Multiprecision<lFirst> reducedSecond = second.template precisionCast<lFirst>();
        return first + reducedSecond;
    } else {
        Multiprecision<lSecond> reducedFirst = first.template precisionCast<lSecond>();
        return reducedFirst + second;
    }
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator+=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator+=(second.template precisionCast<lFirst>());
}


template <std::size_t length, typename Integral> requires (std::is_integral_v<Integral>)
auto operator-(Integral number, const Multiprecision<length>& value) noexcept {
    return Multiprecision<length>(number) - value;
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst != lSecond)
auto operator-(const Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second)
-> typename std::conditional<(lFirst > lSecond), Multiprecision<lFirst>, Multiprecision<lSecond>>::type {
    if constexpr (lFirst > lSecond) {
        return first - second.template precisionCast<lFirst>();
    } else {
        return first.template precisionCast<lSecond>() - second;
    }
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator-=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator-=(second.template precisionCast<lFirst>());
}


template <std::size_t length, typename Integral> requires (std::is_integral_v<Integral>)
auto operator*(Integral number, const Multiprecision<length>& value) noexcept {
    return Multiprecision<length>(number) * value;
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst != lSecond)
auto operator*(const Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second)
-> typename std::conditional<(lFirst > lSecond), Multiprecision<lFirst>, Multiprecision<lSecond>>::type {
    if constexpr (lFirst > lSecond) {
        return first * second.template precisionCast<lFirst>();
    } else {
        return first.template precisionCast<lSecond>() * second;
    }
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator*=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator*=(second.template precisionCast<lFirst>());
}


template <std::size_t length, typename Integral> requires (std::is_integral_v<Integral>)
auto operator/(Integral number, const Multiprecision<length>& value) noexcept {
    return Multiprecision<length>(number) / value;
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst != lSecond)
auto operator/(const Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second)
-> typename std::conditional<(lFirst > lSecond), Multiprecision<lFirst>, Multiprecision<lSecond>>::type {
    if constexpr (lFirst > lSecond) {
        return first / second.template precisionCast<lFirst>();
    } else {
        return first.template precisionCast<lSecond>() / second;
    }
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator/=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator/=(second.template precisionCast<lFirst>());
}


template <std::size_t length, typename Integral> requires (std::is_integral_v<Integral>)
auto operator%(Integral number, const Multiprecision<length>& value) noexcept {
    return Multiprecision<length>(number) % value;
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst != lSecond)
auto operator%(const Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second)
-> typename std::conditional<(lFirst > lSecond), Multiprecision<lFirst>, Multiprecision<lSecond>>::type {
    if constexpr (lFirst > lSecond) {
        return first / second.template precisionCast<lFirst>();
    } else {
        return first.template precisionCast<lSecond>() / second;
    }
}
template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator%=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator/=(second.template precisionCast<lFirst>());
}

#endif //METALMULTIPRECISION_MULTIPRECISION_H
