#ifndef METALMULTIPRECISION_MULTIPRECISION_H
#define METALMULTIPRECISION_MULTIPRECISION_H

#include <array>

template <uint8_t blocksCount = 8>
class Multiprecision {
    static constexpr auto Positive = 1, Negative = 0;
    using block = unsigned;

    /* ---------------------------- Class members. --------------------------- */
    std::array<block, blocksCount> blocks {};
    uint8_t sign {}, capacity { blocksCount };
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
        setSign((getSign() == Negative) ? Positive : Negative);
        if(emptyCapacity()) setSign(Positive);
    }
    /* ----------------------------------------------------------------------- */


public:
    /* ----------------------- Different constructors. ----------------------- */
    constexpr Multiprecision() noexcept = default;
    constexpr Multiprecision(const Multiprecision& copy) noexcept = default;
    constexpr Multiprecision(Multiprecision&& move) noexcept = default;

    template <typename String> requires (std::convertible_to<String&&, std::string>)
    constexpr Multiprecision(String&& from) noexcept {
        unsigned position = 0;
        if(from[0] == '-') {
            setSign(Negative);
            ++position;
        } else
            setSign(Positive);
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    constexpr Multiprecision(Integral value) noexcept {};
    /* ----------------------------------------------------------------------- */

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr Multiprecision operator+(Object&& value) const noexcept {
        Multiprecision result = *this; result += value; return result;
    }

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr Multiprecision& operator+=(Object&& value) noexcept {
        return *this;
    }

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr Multiprecision operator-(Object&& other) const noexcept {
        Multiprecision result(this); result -= other; return result;
    }

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr Multiprecision& operator-=(Object&& value) noexcept {
        return *this;
    }

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr Multiprecision& operator=(Object&& other) noexcept {
        for(uint8_t i = 0; i < getCapacity(); ++i)
            blocks[i] = (i < other.getCapacity()) ? other.blocks[i] : 0;
        setSign(other.getSign());
        return *this;
    }

    template <typename Object> requires (std::convertible_to<Object&&, Multiprecision>)
    constexpr bool operator==(Object&& value) const noexcept { return true; }

    constexpr Multiprecision operator+() const noexcept {
        return Multiprecision(*this);
    }
    constexpr Multiprecision operator-() const noexcept {
        Multiprecision result = *this;
        result.setSign(result.getSign() == Positive ? Negative : Positive);
        return result;
    }
};

template <typename Object> requires (std::convertible_to<Object&&, Multiprecision<>>)
std::ostream& operator<<(std::ostream& stream, Object&& value) noexcept {
    return stream;
};


#endif //METALMULTIPRECISION_MULTIPRECISION_H
