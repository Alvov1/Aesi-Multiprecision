#ifndef AESI_MULTIPRECISION
#define AESI_MULTIPRECISION

/// @cond HIDE_INCLUDES
#include "Aeu.h"
/// @endcond

/**
 * @file Aesi.h
 * @brief Long precision signed integer with arithmetic operations
 */

namespace {
    enum class Sign { Zero = 0, Positive = 1, Negative = -1 };
}

/**
 * @class Aesi
 * @brief Long precision integer with arithmetic operations
 * @details May be used to represent positive and negative integers. Number precision is set in template parameter bitness.
 */
template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aesi final {
    /* -------------------------- @name Class members. ----------------------- */
    /**
     * @brief Number's sign
     */
    Sign sign;

    /**
     * @brief Number's unsigned base
     */
    Aeu<bitness> base;
    /* ----------------------------------------------------------------------- */
public:
    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aesi() noexcept = default;

    /**
     * @brief Copy constructor
     */
    gpu constexpr Aesi(const Aesi& copy) noexcept = default;

    /**
     * @brief Copy assignment operator
     */
    gpu constexpr Aesi& operator=(const Aesi& other) noexcept = default; //{ blocks = other.blocks; return *this; }

    /**
     * @brief Integral constructor
     * @param value Integral
     * @details Accepts built-in integral types signed and unsigned.
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept : base(static_cast<uint64_t>(abs(value))){
        if(value < 0) sign = Sign::Negative; else sign = Sign::Positive;
    }

    /**
     * @brief Pointer-based character constructor
     * @param Char* pointer
     * @param Size_t size
     * @details Accepts decimal literals along with binary (starting with 0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept : base(ptr, size) {
        if(size > 0 && ptr[0] == [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ())
            sign = Sign::Negative;
        else sign = Sign::Positive;
    }

    /**
     * @brief C-style string literal constructor
     * @param Char[] literal
     */
    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&literal)[arrayLength]) noexcept : Aesi(literal, arrayLength) {}

    /**
     * @brief String or string-view based constructor
     * @param String/String-View sv
     * @details Constructs object from STD::Basic_String or STD::Basic_String_View. Accepts objects based on char or wchar_t
     */
    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /**
    * @brief Unary plus operator
    * @return Aesi
    * @note Does basically nothing
    */
    gpu constexpr auto operator+() const noexcept -> Aesi { return *this; }

    /**
     * @brief Unary minus operator
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator-() const noexcept -> Aesi {
        Aesi result = *this; result.inverse(); return result;
    }

    /**
     * @brief Prefix increment
     * @return Aesi&
     */
    gpu constexpr auto operator++() noexcept -> Aesi& { return this->operator+=(1); }

    /**
     * @brief Postfix increment
     * @return Aesi
     */
    gpu constexpr auto operator++(int) & noexcept -> Aesi {
        Aeu old = *this; operator++(); return old;
    }

    /**
     * @brief Prefix decrement
     * @return Aesi&
     */
    gpu constexpr auto operator--() noexcept -> Aesi& { return this->operator-=(1); }

    /**
     * @brief Postfix decrement
     * @return Aesi
     */
    gpu constexpr auto operator--(int) & noexcept -> Aesi {
        Aeu old = *this; operator--(); return old;
    }

    /**
     * @brief Addition operator
     * @param Aesi addendum
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator+(const Aesi& addendum) const noexcept -> Aesi {
        if(sign == Sign::Zero) return addendum;         // 0 + 10 = 10;
        if(addendum.sign == Sign::Zero) return *this;   // 10 + 0 = 10;

        Aesi result = *this;
        if(sign != addendum.sign) {
            // TODO: Figure out with complements
        } else
            base.operator+=(addendum.base);

        if(base.isZero())
            result.sign = Sign::Zero;
    }

    /**
     * @brief Assignment addition operator
     * @param Aesi addendum
     * @return Aesi&
     */
    gpu constexpr auto operator+=(const Aesi& addendum) noexcept -> Aesi& {
        return this->operator=(this->operator+(addendum));
    }

    /**
     * @brief Subtraction operator
     * @param Aesi subtrahend
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator-(const Aesi& subtrahend) const noexcept -> Aesi {
        if(sign == Sign::Zero) return Aesi { Sign::Negative, subtrahend.base };     // 0 - (10) = -10;
        if(subtrahend.sign == Sign::Zero) return *this;                             // 10 - 0 = 10;
        return this->operator+(-subtrahend);
    }

    /**
     * @brief Assignment subtraction operator
     * @param Aesi subtrahend
     * @return Aesi&
     */
    gpu constexpr auto operator-=(const Aesi& subtrahend) noexcept -> Aesi& {
        return this->operator=(this->operator-(subtrahend));
    }

    /**
     * @brief Multiplication operator
     * @param Aesi factor
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi {
        if(sign == Sign::Zero || factor.sign == Sign::Zero) return Aesi { Sign::Zero, {} }; // 0 * 10 = 0; 10 * 0 = 0;

        Aesi result = *this; result.base->operator*=(factor.base);

        if(!result.base.isZero()){
            if (sign != factor.sign) {
                result.sign = Sign::Negative;
            } else
                result.sign = Sign::Positive;
        } else result.sign = Sign::Zero;

        return result;
    }

    /**
     * @brief Assignment multiplication operator
     * @param Aesi factor
     * @return Aesi&
     */
    gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& {
        return this->operator=(this->operator*(factor));
    }

    /**
     * @brief Division operator
     * @param Aesi divisor
     * @return Aesi
     * @note Returns zero in case of division by zero
     */
    [[nodiscard]]
    gpu constexpr auto operator/(const Aesi& divisor) const noexcept -> Aesi {
        if(sign == Sign::Zero || divisor.sign == Sign::Zero) return Aesi { Sign::Zero, {} }; // 0 / 10 = 0; 10 / 0 = UB -> 0;

        Aesi quotient; Aeu _;

        Aeu<bitness>::divide(base, divisor.base, quotient.base, _);
        if(!quotient.base.isZero()) {
            if (sign != divisor.sign)
                quotient.sign = Sign::Negative;
            else quotient.sign = Sign::Positive;
        } else quotient.sign = Sign::Zero;

        return quotient;
    }

    /**
     * @brief Assignment division operator
     * @param Aesi divisor
     * @return Aesi&
     */
    gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& {
        return this->operator=(this->operator/(divisor));
    }

    /**
     * @brief Modulo operator
     * @param Aesi modulo
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi {
        if(sign == Sign::Zero) return { Sign::Zero, {} };   // 0 % 10 = 0;
        if(modulo.sign == Sign::Zero) return *this;         // 10 % 0 = 10;

        Aeu _; Aesi remainder;

        Aeu<bitness>::divide(base, modulo.base, _, remainder.base);
        if(!remainder.base.isZero()) {
            if (sign != modulo.sign)
                remainder.sign = Sign::Negative;
            else remainder.sign = Sign::Positive;
        } else remainder.sign = Sign::Zero;

        return remainder;
    }

    /**
     * @brief Assignment modulo operator
     * @param Aesi modulo
     * @return Aesi&
     */
    gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {
        return this->operator=(this->operator%(modulo));
    }
    /* ----------------------------------------------------------------------- */


    /* ----------------------- @name Bitwise operators. ---------------------- */
    /**
     * @brief Bitwise complement operator
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator~() const noexcept -> Aesi { return { sign, base.operator~() }; }

    /**
     * @brief Bitwise XOR operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator^(const Aesi& other) const noexcept -> Aesi {
        Aesi result = *this; result.base.operator^=(other.base); return result;
    }

    /**
     * @brief Assignment bitwise XOR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator^=(const Aesi& other) noexcept -> Aesi& {
        base.operator^=(other.base); return *this;
    }

    /**
     * @brief Bitwise AND operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator&(const Aesi& other) const noexcept -> Aesi {
        Aesi result = *this; result.base.operator&=(other.base); return result;
    }

    /**
     * @brief Assignment bitwise AND operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator&=(const Aesi& other) noexcept -> Aesi& {
        base.operator&=(other.base); return *this;
    }

    /**
     * @brief Bitwise OR operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator|(const Aesi& other) const noexcept -> Aesi {
        Aesi result = *this; result.base.operator|=(other.base); return result;
    }

    /**
     * @brief Assignment bitwise OR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator|=(const Aesi& other) noexcept -> Aesi& {
        base.operator|=(other.base); return *this;
    }

    /**
     * @brief Left shift operator
     * @param Integral bit_shift
     * @return Aesi
     * @note Does right shift (>>) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator<<(Integral bitShift) const noexcept -> Aesi {
        Aesi result = *this; result.base.operator<<=(bitShift); return result;
    }

    /**
     * @brief Left shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does right shift (>>=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator<<=(Integral bitShift) noexcept -> Aesi& {
        base.operator<<=(bitShift); return *this;
    }

    /**
     * @brief Right shift operator
     * @param Integral bit_shift
     * @return Aesi
     * @note Does left shift (<<) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator>>(Integral bitShift) const noexcept -> Aesi {
        Aesi result = *this; result.base.operator>>=(bitShift); return result;
    }

    /**
     * @brief Right shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does left shift (<<=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator>>=(Integral bitShift) noexcept -> Aesi& {
        base.operator>>=(bitShift); return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Comparison operators. --------------------- */
    /**
     * @brief Comparison operator
     * @param Aesi other
     * @return Bool
     */
    gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool { return sign == other.sign && base == other.base; };

    /**
     * @brief Internal comparison operator
     * @param Aesi other
     * @return Comparison
     * @note Should almost never return Comparison::Equivalent
     */
    [[nodiscard]]
    gpu constexpr auto compareTo(const Aesi& other) const noexcept -> Comparison {
        if(sign == Sign::Zero && other.sign == Sign::Zero)
            return Comparison::equal;

        if(sign == other.sign)
            return base.compareTo(other.base);
        else if(sign == Sign::Positive)
            return Comparison::greater;     // (positive, negative) -> greater; (positive, zero) -> greater
        else if(sign == Sign::Negative)
            return Comparison::less;        // (negative, positive) -> less; (negative, zero) -> less
        else if(other.sign == Sign::Positive)
            return Comparison::less;        // (0, positive) -> less
        else
            return Comparison::greater;     // (0, negative) -> greater;
    };
    /* ----------------------------------------------------------------------- */


    /* ---------------------- @name Supporting methods. ---------------------- */
    /**
     * @brief Inverse number's sign
     */
    gpu constexpr auto inverse() noexcept -> void {
        if(sign != Sign::Zero)
            sign = (sign == Sign::Positive ? Sign::Negative : Sign::Positive);
    }
    /* ----------------------------------------------------------------------- */


    /* -------------- @name Public arithmetic and number theory. ------------- */
    /* ----------------------------------------------------------------------- */

    /* ------------ @name Arithmetic and number theory overrides. ------------ */
    /* ----------------------------------------------------------------------- */

    /* ----------------- @name Public input-output operators. ---------------- */
    /* ----------------------------------------------------------------------- */
};

#endif //AESI_MULTIPRECISION
