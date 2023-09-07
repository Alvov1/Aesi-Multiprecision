#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <array>

template <uint8_t blocksCount = 8>
class Multiprecision {
    static constexpr auto Positive = 1, Negative = 0;
    using block = unsigned;
    using blockLine = std::array<unsigned, blocksCount>;
    static constexpr auto blockBase = 1ULL << (sizeof(block) * 8);

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    uint8_t sign { Positive }, capacity { blocksCount };
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

    template <typename String> requires (std::convertible_to<String&&, std::string> or std::convertible_to<String&&, std::string_view>)
    constexpr Multiprecision(String&& from) noexcept {
        unsigned position = 0;
        if(from[0] == '-') {
            setSign(Negative);
            ++position;
        }
    }

    template <typename Char, std::size_t arrayLength>
    constexpr Multiprecision(const Char (&from)[arrayLength]) {
        unsigned position = 0;
        if(from[0] == '-') {
            setSign(Negative);
            ++position;
        }
        const unsigned charactersPerDigit = sizeof(block) * 8 / 4;
        const unsigned digits = (arrayLength + charactersPerDigit - 1) / charactersPerDigit;

        for(unsigned i = 0; i < digits; ++i) {

        }
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {
        unsigned long long tValue {};
        if(value > 0) {
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

    constexpr Multiprecision operator+(const Multiprecision& value) const noexcept {
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

    constexpr Multiprecision operator-(const Multiprecision& value) const noexcept {
        Multiprecision result = *this; result -= value; return result;
    }
    constexpr Multiprecision& operator-=(const Multiprecision& value) noexcept {
        return *this;
    }

    constexpr Multiprecision operator*(const Multiprecision& value) const noexcept {}
    constexpr Multiprecision& operator*=(const Multiprecision& value) noexcept {}

    constexpr Multiprecision operator/(const Multiprecision& value) const noexcept {}
    constexpr Multiprecision& operator/=(const Multiprecision& value) noexcept {}

    constexpr Multiprecision operator%(const Multiprecision& value) const noexcept {}
    constexpr Multiprecision& operator%=(const Multiprecision& value) noexcept {}

    constexpr Multiprecision& operator=(const Multiprecision& other) noexcept {
        for(uint8_t i = 0; i < getCapacity(); ++i)
            blocks[i] = (i < other.getCapacity()) ? other.blocks[i] : 0;
        setSign(other.getSign());
        return *this;
    }
    constexpr bool operator==(const Multiprecision& value) const noexcept { return true; }
};

template <typename Object> requires (std::convertible_to<Object&&, Multiprecision<>>)
std::ostream& operator<<(std::ostream& stream, Object&& value) noexcept {
    return stream;
};


#endif //METALMULTIPRECISION_MULTIPRECISION_H
