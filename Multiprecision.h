#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <array>

template <unsigned blocksCount = 8>
class Multiprecision {
    static constexpr auto Positive = 1, Negative = 0;
    using block = unsigned;
    using blockLine = std::array<unsigned, blocksCount>;
    static constexpr auto blockBase = 1ULL << (sizeof(block) * 8);

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    uint8_t sign { Positive }, capacity { blocksCount };
    /* ----------------------------------------------------------------------- */

    /* ------------------------ Static helper functions. --------------------- */
    template <typename Char>
    static constexpr size_t strlen(const Char *str) noexcept {
        const char *s = str;
        for (; *s; ++s);
        return(s - str);
    }
    /* ----------------------------------------------------------------------- */

    /* -------------------------- Useful operators. -------------------------- */
    [[nodiscard]]
    constexpr auto getSign() const noexcept -> uint8_t { return sign; };
    constexpr auto setSign(uint8_t newSign) noexcept -> void { sign = newSign; }
    [[nodiscard]]
    constexpr auto getCapacity() const noexcept -> uint8_t { return capacity; }
    constexpr auto setCapacity(uint8_t newCapacity) noexcept -> void { capacity = newCapacity; }
    [[nodiscard]]
    constexpr auto emptyCapacity() const -> bool { return getCapacity() == 0; }
    constexpr auto negate() noexcept -> void {
        if(emptyCapacity()) setSign(Positive);
            else setSign((getSign() == Negative) ? Positive : Negative);
    }

    constexpr auto blocksAdd(const blockLine& from) noexcept -> block {
        block carryOut {};
        for(uint8_t i = 0; i < blocksCount; ++i) {
            block a = blocks[i];
            block b = (i < from.size()) ? from[i] : 0;
        }
        return block();
    };
    constexpr auto blocksComplement() noexcept -> void {

    };
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
            setSign(Negative);
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
            setSign(Positive);
    }

/**/template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {
        unsigned long long tValue {};
        if(value < 0) {
            setSign(Negative);
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
        result.setSign(result.getSign() == Positive ? Negative : Positive);
        return result;
    }

/**/constexpr Multiprecision operator+(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result += value; return result;
    }
    constexpr Multiprecision& operator+=(const Multiprecision& value) noexcept {
        /*if(getSign() != Negative || value.getSign() != Negative) {
            if(getSign() == Negative) blocksComplement();
                else value.blocksComplement();

            uint8_t newSign {};
            if(blocksAdd(value.blocks) == 0 && (getSign() == Negative || value.getSign() == Negative)) {
                blocksComplement();
                newSign = Negative;
            } else
                newSign = Positive;

            if(getSign() == Negative) blocksComplement();
            if(value.getSign()) value.blocksComplement();
            setSign(newSign);
        } else
            if(blocksAdd(value.blocks) != 0)
                std::cerr << "Attention: overflow!!!" << std::endl;*/
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
        return getSign() == value.getSign() && getCapacity() == value.getCapacity() && blocks == value.blocks;
    }
/**/constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        for(uint8_t i = 0; i < getCapacity(); ++i)
            blocks[i] = (i < other.getCapacity()) ? other.blocks[i] : 0;
        setSign(other.getSign());
        return *this;
    }

/**/friend std::ostream& operator<<(std::ostream& stream, const Multiprecision<>& value) noexcept {
        if (value.getSign() == Negative)
            stream << '-';

        int print_zeroes = 0; // don't print leading 0s
        for (int i = value.capacity - 1; i >= 0; i--) {
            unsigned digit = value.blocks[i];

            static constexpr auto hexDigits = "0123456789abcdef";
            if (digit != 0 || print_zeroes) {
                if (!print_zeroes) {
                    char buffer[9] = {'0', '0', '0', '0', '0', '0', '0', '0', '\0'};
                    uint8_t bufferPosition = 0;

                    while(digit > 0 && bufferPosition < 8) {
                        buffer[bufferPosition++] = hexDigits[digit % 16];
                        digit /= 16;
                    }

                    for(uint8_t j = 0; j < bufferPosition; ++j)
                        stream << buffer[bufferPosition - j - 1];
                } else {
                    char buffer[9] {};
                    uint8_t bufferPosition = 0;

                    while(digit > 0 && bufferPosition < 8) {
                        buffer[bufferPosition++] = hexDigits[digit % 16];
                        digit /= 16;
                    }

                    for(uint8_t j = 0; j < 8; ++j)
                        stream << buffer[7 - j];
                }
                print_zeroes = 1;
            }
        }

        /* the number is zero */
        if (print_zeroes == 0)
            stream << '0';
        return stream;
    };
};


#endif //METALMULTIPRECISION_MULTIPRECISION_H
