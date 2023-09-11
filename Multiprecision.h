#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <iostream>
#include <array>

namespace {
    using block = unsigned;
    constexpr auto blockBitLength = sizeof(block) * 8;
    constexpr unsigned long long blockBase = 1ULL << blockBitLength;
    constexpr unsigned long long blockMax = std::numeric_limits<block>::max();
}

template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Multiprecision {
    static constexpr std::size_t blocksCount = bitness / blockBitLength;
    using blockLine = std::array<block, blocksCount>;
    enum Sign { Zero = 0, Positive = 1, Negative = 2 };

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    Sign sign { Zero };
    /* --------------------------- Helper functions. ------------------------- */
    template <typename Char>
    static constexpr size_t strlen(const Char *str) noexcept {
        const char *s = str;
        for (; *s; ++s);
        return(s - str);
    }

    /* ----------------------------------------------------------------------- */
    static constexpr auto sumLines(blockLine& dst, const blockLine& src) noexcept -> unsigned long long {
        unsigned long long carryOut = 0;
        for (std::size_t i = 0; i < blocksCount; ++i) {
            unsigned long long sum = static_cast<unsigned long long>(dst[i])
                                     + static_cast<unsigned long long>(src[i]) + carryOut;
            carryOut = sum / blockBase;
            dst[i] = sum % blockBase;
        }
        return carryOut;
    }
    static constexpr auto complement(const blockLine& line) noexcept -> blockLine {
        blockLine result {};

        unsigned long long carryOut = 1;
        for(std::size_t i = 0; i < blocksCount; ++i) {
            const unsigned long long sum = blockBase - 1ULL - static_cast<unsigned long long>(line[i]) + carryOut;
            carryOut = sum / blockBase; result[i] = sum % blockBase;
        }

        return result;
    }
    static constexpr auto emptyLine(const blockLine& line) noexcept -> bool {
        for(std::size_t i = 0; i < line.size(); ++i)
            if(line[i] != 0) return false;
        return true;
    }
    /* ----------------------------------------------------------------------- */


public:
    /* ----------------------- Different constructors. ----------------------- */
    constexpr Multiprecision() noexcept = default;
    constexpr Multiprecision(const Multiprecision& copy) noexcept = default;
    constexpr Multiprecision(Multiprecision&& move) noexcept = default;

    template<std::size_t length>
    constexpr explicit Multiprecision(const std::array<block, length>& data) noexcept {
        for(std::size_t i = 0; i < blocksCount && i < length; ++i)
            blocks[i] = data[i];
        sign = Positive;
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {
        if(value != 0) {
            unsigned long long tValue {};
            if (value < 0) {
                sign = Negative;
                tValue = static_cast<unsigned long long>(value * -1);
            } else {
                sign = Positive;
                tValue = static_cast<unsigned long long>(value);
            }

            for (std::size_t i = 0; i < blocksCount; ++i) {
                blocks[i] = static_cast<unsigned>(tValue % blockBase);
                tValue /= blockBase;
            }
        }
    }

    template <typename String>
    requires (std::is_same<std::string, typename std::decay<String>::type>::value or std::is_same<std::string_view, typename std::decay<String>::type>::value)
    constexpr Multiprecision(String&& value) noexcept {};

    template <typename Char, std::size_t arrayLength>
    constexpr Multiprecision(const Char (&from)[arrayLength]) noexcept {
//        unsigned position = 0;
//        if(from[position] == '-') {
//            sign = Negative;
//            ++position;
//        } else sign = Positive;
//
//        uint8_t base = 10;
//        if(from[position] == '0') {
//            base = 8;
//            ++position;
//            if(from[position] == 'x') {
//                base = 16;
//                ++position;
//            } else if(from[position] == 'b') {
//                base = 2;
//                ++position;
//            }
//        }
//
//        const unsigned charactersPerDigit = [] (uint8_t base) {
//            switch(base) {
//                case 2: return 32;
//                case 8: return 11;
//                case 10: return 10;
//                case 16: return 8;
//                default: return 10;
//            }
//        } (base);
//        std::array<Char, blockBitLength + 1> buffer {};
//
//        const unsigned blocksTotal = (arrayLength - 1 - position + charactersPerDigit - 1) / charactersPerDigit;
//        bool allZeros = true;
//        for(unsigned i = 0; i < blocksTotal; ++i) {
//            long shift = arrayLength - 1 - (i + 1) * charactersPerDigit;
//            if(shift < position) shift = position;
//
//            for(unsigned j = 0; j < charactersPerDigit; ++j)
//                buffer[j] = (shift + j < arrayLength ? from[shift + j] : '\0');
//
//            blocks[i] = strtoul(buffer.data(), nullptr, base);
//            allZeros = allZeros && (blocks[i] == 0);
//        }
//
//        if(allZeros) sign = Zero;
    }
    /* ----------------------------------------------------------------------- */

    constexpr Multiprecision operator+() const noexcept { return *this; }
    constexpr Multiprecision operator-() const noexcept {
        if(sign == Zero) return Multiprecision();
        Multiprecision result = *this;
        result.sign = (result.sign == Positive ? Negative : Positive); return result;
    }

    constexpr Multiprecision operator+(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result += value; return result;
    }
    constexpr Multiprecision& operator+=(const Multiprecision& value) noexcept {
        if(sign == Zero) return this->operator=(value);
        if(value.sign == Zero) return *this;

        if (sign != value.sign) {
            if (sign == Negative)
                blocks = complement(blocks);
            const unsigned long long carryOut = (value.sign != Negative ?
                         sumLines(blocks, value.blocks) : sumLines(blocks, complement(value.blocks)));
            if (carryOut == 0) {
                blocks = complement(blocks);
                sign = Negative;
            } else sign = Positive;
        } else
            sumLines(blocks, value.blocks);

        if (emptyLine(blocks))
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
    constexpr Multiprecision& operator*=(const Multiprecision& value) noexcept { return *this;}

    constexpr Multiprecision operator/(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result /= value; return result;
    }
    constexpr Multiprecision& operator/=(const Multiprecision& value) noexcept { return *this; }

    constexpr Multiprecision operator%(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result %= value; return result;
    }
    constexpr Multiprecision& operator%=(const Multiprecision& value) noexcept { return *this; }

    constexpr bool operator==(const Multiprecision& value) const noexcept {
        return sign == value.sign && blocks == value.blocks;
    }
    constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        blocks = other.blocks; sign = other.sign; return *this;
    }

    auto outputDecimal(std::ostream& toStream) const noexcept -> void {}
    auto outputOctal(std::ostream& toStream) const noexcept -> void {}
    auto outputHexadecimal(std::ostream& toStream) const noexcept -> void {
        if(sign == Negative) toStream.write("-0x", 3);
            else toStream.write("0x", 2);
        for(auto iter = blocks.rbegin(); iter != blocks.rend(); ++iter)
            toStream << std::hex << *iter;
    }

    friend std::ostream& operator<<(std::ostream& stream, const Multiprecision& value) noexcept {
        value.outputHexadecimal(stream);
        return stream;
    }

    template <std::size_t toPrecision> requires (toPrecision > blocksCount * blockBitLength)
    constexpr auto precisionCast() const noexcept -> Multiprecision<toPrecision> {
        Multiprecision<toPrecision> result(blocks); if(sign == Negative) result = -result; return result;
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
