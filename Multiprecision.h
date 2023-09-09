#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <iostream>
#include <array>

template <std::size_t blocksCount = 8>
class Multiprecision {
    static constexpr auto Positive = 1, Negative = 0;
    using block = unsigned;
    using blockLine = std::array<block, blocksCount>;
    static constexpr auto blockBase = 1ULL << (sizeof(block) * 8);

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    uint8_t sign { Positive };
    /* ----------------------------------------------------------------------- */

    /* ------------------------ Static helper functions. --------------------- */
    template <typename Char>
    static constexpr size_t strlen(const Char *str) noexcept {
        const char *s = str;
        for (; *s; ++s);
        return(s - str);
    }
    /* ----------------------------------------------------------------------- */


public:
    /* ----------------------- Different constructors. ----------------------- */
    constexpr Multiprecision() noexcept = default;
    constexpr Multiprecision(const Multiprecision& copy) noexcept = default;
    constexpr Multiprecision(Multiprecision&& move) noexcept = default;

    template <typename String>
    requires (std::is_same<std::string, typename std::decay<String>::type>::value or std::is_same<std::string_view, typename std::decay<String>::type>::value)
    constexpr Multiprecision(String&& value) noexcept {};

/**/template <typename Char, std::size_t arrayLength>
    constexpr Multiprecision(const Char (&from)[arrayLength]) noexcept {
        unsigned position = 0;
        if(from[position] == '-') {
            sign = Negative;
            ++position;
        }

        uint8_t base = 10;
        if(from[position] == '0') {
            base = 8;
            ++position;
            if(from[position] == 'x') {
                base = 16;
                ++position;
            } else if(from[position] == 'b') {
                base = 2;
                ++position;
            }
        }

        const unsigned charactersPerDigit = [] (uint8_t base) {
            switch(base) {
                case 2: return 32;
                case 8: return 11;
                case 10: return 10;
                case 16: return 8;
                default: return 10;
            }
        } (base);
        std::array<Char, 33> buffer {};

        const unsigned blocksTotal = (arrayLength - 1 - position + charactersPerDigit - 1) / charactersPerDigit;
        bool allZeros = true;
        for(unsigned i = 0; i < blocksTotal; ++i) {
            long shift = arrayLength - 1 - (i + 1) * charactersPerDigit;
            if(shift < position) shift = position;

            for(unsigned j = 0; j < charactersPerDigit; ++j)
                buffer[j] = (shift + j < arrayLength ? from[shift + j] : '\0');

            blocks[i] = strtoul(buffer.data(), nullptr, base);
            allZeros = allZeros && (blocks[i] == 0);
        }

        if(allZeros)
            sign = Positive;
    }

/**/template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {
        unsigned long long tValue {};
        if(value < 0) {
            sign = Negative;
            tValue = static_cast<unsigned long long>(value * -1);
        } else
            tValue = static_cast<unsigned long long>(value);

        for(unsigned i = 0; i < blocksCount; ++i) {
            blocks[i] = static_cast<unsigned>(tValue % blockBase);
            tValue /= blockBase;
        }
    }
    /* ----------------------------------------------------------------------- */

    constexpr Multiprecision operator+() const noexcept {
        return Multiprecision(*this);
    }
    constexpr Multiprecision operator-() const noexcept {
        Multiprecision result = *this;
        result.setSign(result.sign == Positive ? Negative : Positive);
        return result;
    }

/**/constexpr Multiprecision operator+(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result += value; return result;
    }
    constexpr Multiprecision& operator+=(const Multiprecision& value) noexcept {

        return *this;
    }

/**/constexpr Multiprecision operator-(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result -= value; return result;
    }
    constexpr Multiprecision& operator-=(const Multiprecision& value) noexcept { return *this; }

/**/constexpr Multiprecision operator*(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result *= value; return result;
    }
    constexpr Multiprecision& operator*=(const Multiprecision& value) noexcept { return *this;}

/**/constexpr Multiprecision operator/(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result /= value; return result;
    }
    constexpr Multiprecision& operator/=(const Multiprecision& value) noexcept { return *this; }

/**/constexpr Multiprecision operator%(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result %= value; return result;
    }
    constexpr Multiprecision& operator%=(const Multiprecision& value) noexcept { return *this; }

/**/constexpr bool operator==(const Multiprecision& value) const noexcept {
        return sign == value.sign && blocks == value.blocks;
    }
/**/constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        for(uint8_t i = 0; i < blocks.size(); ++i)
            blocks[i] = (i < other.getCapacity()) ? other.blocks[i] : 0;
        setSign(other.sign);
        return *this;
    }

    auto outputDecimal(std::ostream& toStream) const noexcept -> void {}
    auto outputOctal(std::ostream& toStream) const noexcept -> void {}
    auto outputHexadecimal(std::ostream& toStream) const noexcept -> void {}

/**/friend std::ostream& operator<<(std::ostream& stream, const Multiprecision<>& value) noexcept;

    template<std::size_t length>
    Multiprecision(const std::array<block, length>& data) {
        for(std::size_t i = 0; i < blocksCount && i < length; ++i) blocks[i] = data[i];
    }

    template <std::size_t toLength> requires (toLength > blocksCount)
    constexpr auto reduce() const noexcept -> Multiprecision<toLength> {
        Multiprecision<toLength> result(blocks);
        if(sign == Negative) result *= -1;
        return result;
    }
};

template <std::size_t tFirst, std::size_t tSecond> requires (tFirst != tSecond)
auto operator+(const Multiprecision<tFirst>& first, const Multiprecision<tSecond>& second)
-> typename std::conditional<(tFirst > tSecond), Multiprecision<tFirst>, Multiprecision<tSecond>>::type {
    if constexpr (tFirst > tSecond) {
        return first + second.template reduce<tFirst>();
    } else {
        return first.template reduce<tSecond>() + second;
    }
}

template <std::size_t lFirst, std::size_t lSecond> requires (lFirst > lSecond)
auto operator+=(Multiprecision<lFirst>& first, const Multiprecision<lSecond>& second) -> Multiprecision<lFirst>& {
    return first.operator+=(second.template reduce<lFirst>());
}


#endif //METALMULTIPRECISION_MULTIPRECISION_H
