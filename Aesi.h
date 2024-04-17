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
    using Base = Aeu<bitness>;
    Base base;
    /* ----------------------------------------------------------------------- */

    /**
     * @brief Private constructor with members
     */
    gpu constexpr Aesi(Sign withSign, Base withBase): sign { withSign }, base { withBase } {};

public:
    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aesi() noexcept = default;

    /**
     * @brief Copy constructor
     */
    gpu constexpr Aesi(const Aesi& copy) noexcept {
        sign = copy.sign; if(copy.sign != Sign::Zero) base = copy.base;
    }

    /**
     * @brief Copy assignment operator
     */
    gpu constexpr Aesi& operator=(const Aesi& other) noexcept {
        sign = other.sign; if(other.sign != Sign::Zero) base = other.base; return *this;
    }

    /**
     * @brief Integral constructor
     * @param value Integral
     * @details Accepts built-in integral types signed and unsigned.
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept : base(static_cast<uint64_t>(abs(value))){
        if(value == 0)
            sign = Sign::Zero;
        else if(value < 0)
            sign = Sign::Negative; else sign = Sign::Positive;
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

        if(base.isZero())
            sign = Sign::Zero;
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
    gpu constexpr auto operator++(int) & noexcept -> Aesi { Aesi old = *this; operator++(); return old; }

    /**
     * @brief Prefix decrement
     * @return Aesi&
     */
    gpu constexpr auto operator--() noexcept -> Aesi& { return this->operator-=(1); }

    /**
     * @brief Postfix decrement
     * @return Aesi
     */
    gpu constexpr auto operator--(int) & noexcept -> Aesi { Aesi old = *this; operator--(); return old; }

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
        if(sign == Sign::Zero || factor.sign == Sign::Zero) return Aesi { Sign::Zero }; // 0 * 10 = 0; 10 * 0 = 0;

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
        Aesi quotient, _; divide(this, divisor, quotient, _); return quotient;
    }

    /**
     * @brief Assignment division operator
     * @param Aesi divisor
     * @return Aesi&
     */
    gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& {
        Aesi quotient, _; divide(this, divisor, quotient, _); return this->operator=(quotient);
    }

    /**
     * @brief Modulo operator
     * @param Aesi modulo
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi {
        Aesi _, remainder; divide(this, modulo, _, remainder); return remainder;
    }

    /**
     * @brief Assignment modulo operator
     * @param Aesi modulo
     * @return Aesi&
     */
    gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {
        Aesi _, remainder; divide(this, modulo, _, remainder); return this->operator=(remainder);
    }
    /* ----------------------------------------------------------------------- */


    /* ----------------------- @name Bitwise operators. ---------------------- */
    /**
     * @brief Bitwise complement operator
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator~() const noexcept -> Aesi {
        Aesi result { sign, base.operator~() };
        if(result.base.isZero())
            result.sign = Sign::Zero;
        return result;
    }

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
        base.operator^=(other.base);
        if(base.isZero())
            sign = Sign::Zero;
        else if(sign == Sign::Zero)
            sign = Sign::Positive;

        return *this;
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
        base.operator&=(other.base); if(base.isZero()) sign = Sign::Zero; return *this;
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
        base.operator|=(other.base); if(sign == Sign::Zero && !base.isZero()) sign = Sign::Positive; return *this;
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
        base.operator<<=(bitShift);
        if(sign == Sign::Zero && !base.isZero())
            sign = Sign::Positive;
        return *this;
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
        base.operator>>=(bitShift);
        if(sign != Sign::Zero && base.isZero())
            sign = Sign::Zero;
        return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Comparison operators. --------------------- */
    /**
     * @brief Comparison operator
     * @param Aesi other
     * @return Bool
     */
    gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool {
        return sign == other.sign && base == other.base;
    };

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
     * @brief Set bit in number by index starting from the right
     * @param Size_t index
     * @param Bool bit
     * @note Does nothing for index out of range
     */
    gpu constexpr auto setBit(std::size_t index, bool bit) noexcept -> void {
        base.setBit(index, bit);
        if(!bit && sign != Sign::Zero && base.isZero()) // Positive/Negative, set 0 -> 0
            sign = Sign::Zero;
        if(bit && sign == Sign::Zero && !base.isZero()) // Zero, set 1 -> Positive
            sign = Sign::Positive;
    }

    /**
     * @brief Get bit in number by index staring from the right
     * @param Size_t index
     * @return Bool
     * @note Returns zero for index out of range
     */
    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool { return base.getBit(index); }

    /**
 * @brief Set byte in number by index starting from the right
 * @param Size_t index
 * @param Byte byte
 * @note Does nothing for index out of range
 */
    gpu constexpr auto setByte(std::size_t index, byte byte) noexcept -> void {
        base.setByte(index, byte);
        if(byte == 0 && sign != Sign::Zero && base.isZero())    // Positive/Negative, set Zero -> 0
            sign = Sign::Zero;
        if(byte != 0 && sign == Sign::Zero && !base.isZero())   // Zero, set Positive -> Positive
            sign = Sign::Positive;
    }

    /**
     * @brief Get byte in number by index starting from the right
     * @param Size_t index
     * @return Byte
     * @note Returns zero for index out of range
     */
    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> byte { return base.getByte(index); }

    /**
     * @brief Set block in number by index starting from the right
     * @param Size_t index
     * @param Block byte
     * @note Does nothing for index out of range
     */
    gpu constexpr auto setBlock(std::size_t index, block block) noexcept -> void {
        base.setBlock(index, block);
        if(block == 0 && sign != Sign::Zero && base.isZero())    // Positive/Negative, set Zero -> 0
            sign = Sign::Zero;
        if(block != 0 && sign == Sign::Zero && !base.isZero())   // Zero, set Positive -> Positive
            sign = Sign::Positive;
    }

    /**
     * @brief Get block in number by index starting from the right
     * @param Size_t index
     * @return Block
     * @note Returns zero for index out of range
     */
    [[nodiscard]]
    gpu constexpr auto getBlock(std::size_t index) const noexcept -> block { return base.getBlock(index); }

    /**
     * @brief Get amount of non-empty bytes in number right to left
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto byteCount() const noexcept -> std::size_t { return base.byteCount(); }

    /**
     * @brief Get amount of non-empty bits in number right to left
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto bitCount() const noexcept -> std::size_t { return base.bitCount(); }

    /**
     * @brief Check whether number is odd
     * @return Bool - true is number is odd and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return base.isOdd(); }

    /**
     * @brief Check whether number is even
     * @return Bool - true is number is even and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return base.isEven(); }

    /**
 * @brief Check whether number is zero
 * @return Bool - true is number is zero and false otherwise
 */
    [[nodiscard]]
    gpu constexpr auto isZero() const noexcept -> bool { return sign == Sign::Zero; }

    /**
     * @brief Get number's precision
     * @return Size_t
     */
    [[nodiscard]]
    gpu static constexpr auto getBitness() noexcept -> std::size_t { return bitness; }

    /**
     * @brief Get number of blocks inside object
     * @return Size_t
     */
    [[nodiscard]]
    gpu static constexpr auto getBlocksNumber() noexcept -> std::size_t { return Aeu<bitness>::getBlocksNumber(); }

    /**
     * @brief Make swap between two objects
     * @param Aesi other
     */
    gpu constexpr auto swap(Aesi& other) noexcept -> void {
        Sign t = sign; sign = other.sign; other.sign = t; base.swap(other.base);
    }

    /**
     * @brief Inverse number's sign
     */
    gpu constexpr auto inverse() noexcept -> void {
        if(sign != Sign::Zero)
            sign = (sign == Sign::Positive ? Sign::Negative : Sign::Positive);
    }
    /* ----------------------------------------------------------------------- */


    /* -------------- @name Public arithmetic and number theory. ------------- */
    /**
     * @brief Integer division. Returns results by reference
     * @param Aesi number
     * @param Aesi divisor
     * @param Aesi quotient OUT
     * @param Aesi remainder OUT
     * @return Returns Quotient and remainder by reference
     */
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor, Aesi& quotient, Aesi& remainder) noexcept -> void {
        /*  (0, Pos). Quotient: 0, Remainder: 0     - done
            (0, Neg). Quotient: 0, Remainder: 0     - done
            (0, 0).   Quotient: 0, Remainder: 0     - done
            (Pos, 0). Quotient: 0, Remainder: 105   - done
            (Neg, 0). Quotient: 0, Remainder: -105  - done */

        if(number.sign == Sign::Zero)
            return [&quotient, &remainder] { quotient.sign = Sign::Zero; remainder.sign = Sign::Zero; } ();
        if(divisor.sign == Sign::Zero)
            return [&number, &quotient, &remainder] { quotient.sign = Sign::Zero; remainder = number; } ();

        Aeu<bitness>::divide(number.base, divisor.base, quotient.base, remainder.base);

        /*  (Neg, Pos). Quotient: -10, Remainder: -5    - done
            (Pos, Neg). Quotient: -10, Remainder: 5     - done
            (Pos, Pos). Quotient: 10, Remainder: 5      - done
            (Neg, Neg). Quotient: 10, Remainder: -5     - done */

        if(number.sign != divisor.sign)
            return [&number, &quotient, &remainder] { quotient.sign = Sign::Negative; remainder.sign = number.sign; } ();
        quotient.sign = Sign::Positive; remainder.sign = divisor.sign;
    }

    /**
     * @brief Extended Euclidean algorithm for greatest common divisor
     * @param Aesi first
     * @param Aesi second
     * @param Aesi bezoutX OUT
     * @param Aesi bezoutY OUT
     * @return Aesi
     * @details Counts Bézout coefficients along with the greatest common divisor. Returns coefficients by reference. Their signs are positive
     */
    [[nodiscard]]
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second, Aesi& bezoutX, Aesi& bezoutY) noexcept -> Aesi {
        Aeu<bitness>::gcd(first.base, second.base, bezoutX.base, bezoutY.base);
        bezoutX.sign = Sign::Positive; bezoutY.sign = Sign::Positive;
    }

    /**
     * @brief Least common multiplier
     * @param Aesi first
     * @param Aesi second
     * @return Aesi
     * @details Return value's sign is always positive
     */
    [[nodiscard]]
    gpu static constexpr auto lcm(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        return Aesi { Sign::Positive, Aeu<bitness>::lcm(first.base, second.base) };
    }

    /**
     * @brief Fast exponentiation for powers of 2
     * @param Size_t power
     * @return Aesi
     * @details Returns zero for power greater than current bitness
     */
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aesi {
        Aesi result { Sign::Positive, {} }; result.setBit(e, true); return result;
    }

    /**
     * @brief Get square root
     * @return Aesi
     * @note Returns zero for negative value or zero
     */
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {
        if(sign == Sign::Positive)
            return Aesi { Sign::Positive, base.squareRoot() };
        return Aesi { Sign::Zero };
    }
    /* ----------------------------------------------------------------------- */

    /**
     * @brief Built-in integral type cast operator
     * @param Type integral_type TEMPLATE
     * @return Integral
     * @details Takes the lowes part of Aesi for conversion. Accepts signed and unsigned types
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        if(sign == Sign::Zero)
            return Integral();
        if constexpr (std::is_signed_v<Integral>) {
            if(sign == Sign::Negative)
                return base.template integralCast<Integral>() * -1;
        } else {
            return base.template integralCast<Integral>();
        }
    }

    /**
     * @brief Precision cast operator
     * @param Size_t new_bitness TEMPLATE
     * @return Aesi<new_bitness>
     * @details If required precision greater than current precision, remaining blocks are filled with zeros.
     * Otherwise - number is cropped inside smaller blocks array
     * @note This method is used in all manipulations between numbers of different precision. Using this method is not recommended,
     * cause it leads to redundant copying and may be slow
     */
    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> { return Aesi { sign, base.template precisionCast<newBitness>() }; }

    /* ----------------- @name Public input-output operators. ---------------- */
    /**
     * @brief Character buffer output operator
     * @param Byte base TEMPLATE
     * @param Char* buffer
     * @param Size_t buffer_size
     * @param Bool show_number_base
     * @param Bool use_hexadecimal_uppercase
     * @return Size_t - amount of symbols written
     * @details Places the maximum possible amount of number's characters in buffer. Base parameter should be 2, 8, 10, or 16
     * @note Works significantly faster for hexadecimal notation
     */
    template <byte notation, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (notation == 2 || notation == 8 || notation == 10 || notation == 16))
    gpu constexpr auto getString(Char* const buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t {
        if(bufferSize < 2) return 0;

        if(sign == Sign::Zero)
            return [&buffer] { buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } }(); buffer[1] = Char(); } ();
        else if(sign == Sign::Negative)
            [&buffer, &bufferSize] { *(buffer++) = [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } }(); --bufferSize; } ();

        return base.getString(buffer, bufferSize, showBase, hexUppercase);
    }

    /**
     * @brief STD stream output operator
     * @param Ostream stream
     * @param Aesi number
     * @return Ostream
     * @details Writes number in stream. Accepts STD streams based on char or wchar_t. Supports stream manipulators:
     * - Number's notation (std::hex, std::dec, std::oct);
     * - Number's base (std::showbase);
     * - Hexadecimal letters case (std::uppercase, std::lowercase)
     * @note Works significantly faster for hexadecimal notation
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) -> std::basic_ostream<Char>& {
        if(value.sign == Sign::Zero)
            return ss << [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
        else
            if(value.sign == Sign::Negative)
                ss << [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ();
        return ss << value.base;
    }
    /* ----------------------------------------------------------------------- */

#if defined __CUDACC__ || defined DOXYGEN_SKIP
    /**
     * @brief Atomicity-oriented object assignment operator
     * @param Aesi assigning
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for assigning encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicSet(const Aesi& value) noexcept -> void {
        sign = value.sign; base.tryAtomicSet(value.base);
    }

    /**
     * @brief Atomicity-oriented object exchangement operator
     * @param Aesi exchangeable
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for exchanging encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicExchange(const Aesi& value) noexcept -> void {
        Sign tSign = sign; sign = value.sign; value.sign = tSign;
        base.tryAtomicExchange(value.base);
    }
#endif
};

/* -------------------------------------------- @name Type-definitions  ------------------------------------------- */
/**
 * @typedef Aesi128
 * @brief Number with precision 128-bit. */
using Aesi128 = Aesi<128>;

/**
 * @typedef Aesi256
 * @brief Number with precision 128-bit. */
using Aesi256 = Aesi<256>;

/**
 * @typedef Aesi512
 * @brief Number with precision 512-bit. */
using Aesi512 = Aesi<512>;

/**
 * @typedef Aesi768
 * @brief Number with precision 768-bit. */
using Aesi768 = Aesi<768>;

/**
 * @typedef Aesi1024
 * @brief Number with precision 1024-bit. */
using Aesi1024 = Aesi<1024>;

/**
 * @typedef Aesi1536
 * @brief Number with precision 1536-bit. */
using Aesi1536 = Aesi<1536>;

/**
 * @typedef Aesi2048
 * @brief Number with precision 2048-bit. */
using Aesi2048 = Aesi<2048>;

/**
 * @typedef Aesi3072
 * @brief Number with precision 3072-bit. */
using Aesi3072 = Aesi<3072>;

/**
 * @typedef Aesi4096
 * @brief Number with precision 4096-bit. */
using Aesi4096 = Aesi<4096>;

/**
 * @typedef Aesi6144
 * @brief Number with precision 6144-bit. */
using Aesi6144 = Aesi<6144>;

/**
 * @typedef Aesi8192
 * @brief Number with precision 8192-bit. */
using Aesi8192 = Aesi<8192>;
/* ---------------------------------------------------------------------------------------------------------------- */

/* ------------------------------------------ @name Integral conversions  ----------------------------------------- */
/**
 * @brief Integral conversion addition operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) + value; }

/**
 * @brief Integral conversion subtraction operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) - value; }

/**
 * @brief Integral conversion multiplication operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) * value; }

/**
 * @brief Integral conversion division operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) / value; }

/**
 * @brief Integral conversion modulo operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) % value; }

/**
 * @brief Integral conversion bitwise XOR operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) ^ value; }

/**
 * @brief Integral conversion bitwise AND operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) & value; }

/**
 * @brief Integral conversion bitwise OR operator
 * @param Integral number
 * @param Aesi value
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) | value; }
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //AESI_MULTIPRECISION
