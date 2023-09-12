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
struct Multiprecision {
    static constexpr std::size_t blocksNumber = bitness / blockBitLength;
    using blockLine = std::array<block, blocksNumber>;
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
    static constexpr auto sumLines(blockLine& dst, const blockLine& src) noexcept -> uint64_t {
        uint64_t carryOut = 0;
        for (std::size_t i = 0; i < blocksNumber; ++i) {
            uint64_t sum = static_cast<uint64_t>(dst[i])
                                     + static_cast<uint64_t>(src[i]) + carryOut;
            carryOut = sum / blockBase;
            dst[i] = sum % blockBase;
        }
        return carryOut;
    }
    static constexpr auto complement(const blockLine& line) noexcept -> blockLine {
        blockLine result {};

        uint64_t carryOut = 1;
        for(std::size_t i = 0; i < blocksNumber; ++i) {
            const uint64_t sum = blockBase - 1ULL - static_cast<uint64_t>(line[i]) + carryOut;
            carryOut = sum / blockBase; result[i] = sum % blockBase;
        }

        return result;
    }
    static constexpr auto emptyLine(const blockLine& line) noexcept -> bool {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            if(line[i] != 0) return false;
        return true;
    }
    static constexpr auto longerLineLength(const blockLine& first, const blockLine& second) noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(first[i] || second[i]) return i + 1;
        return 0;
    }
    /* ----------------------------------------------------------------------- */


public:
    /* ----------------------- Different constructors. ----------------------- */
    constexpr Multiprecision() noexcept = default;
    constexpr Multiprecision(const Multiprecision& copy) noexcept = default;
    constexpr Multiprecision(Multiprecision&& move) noexcept = default;

    template<std::size_t length>
    constexpr explicit Multiprecision(const std::array<block, length>& data) noexcept {
        for(std::size_t i = 0; i < blocksNumber && i < length; ++i)
            blocks[i] = data[i];
        sign = Positive;
    }

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

    template <typename String, typename Char = typename String::value_type>
    requires (std::is_same_v<std::basic_string<Char>, typename std::decay<String>::type>)
    constexpr Multiprecision(String&& string) noexcept {};

    template <typename StringView, typename Char = typename StringView::value_type>
    requires (std::is_same_v<std::basic_string_view<Char>, typename std::decay<StringView>::type>)
    constexpr Multiprecision(StringView&& stringView) noexcept {}

    template <typename Char, std::size_t arrayLength> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    constexpr Multiprecision(const Char (&array)[arrayLength]) noexcept {}
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
            const uint64_t carryOut = (value.sign != Negative ?
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
    constexpr Multiprecision& operator*=(const Multiprecision& value) noexcept { 
        if(sign == Zero) return *this;
        if(value.sign == Zero)
            return this->operator=(Multiprecision());
        sign = (sign != value.sign ? Negative : Positive);
        
        const auto longerLength = longerLineLength(blocks, value.blocks);
        blockLine buffer {};
        for(std::size_t i = 0; i < longerLength; ++i) {
            uint64_t carryOut = 0;
            for(std::size_t j = 0; j < longerLength; ++j) {
                auto multiplication = static_cast<uint64_t>(blocks[j]) * static_cast<uint64_t>(value.blocks[i]) + carryOut;
                auto tBlock = static_cast<uint64_t>(buffer[i + j]) + (multiplication % blockBase);
                carryOut = multiplication / blockBase + tBlock / blockBase;
                buffer[i + j] = tBlock % blockBase;
            }
            if(longerLength < blocksNumber)
                buffer[longerLength + i] += carryOut;
        }
        blocks = buffer;

        return *this;
    }

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

    template <std::size_t toPrecision> requires (toPrecision > blocksNumber * blockBitLength)
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
