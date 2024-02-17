#ifndef AESSI_MULTIPRECISION
#define AESSI_MULTIPRECISION

/// @cond HIDE_INCLUDES
#include <utility>
#include "Aeu.h"
/// @endcond

/**
 * @file Aessi.h
 * @brief Long precision signed integer with arithmetic operations
 */

/**
 * @class Aessi
 * @brief Long precision SIGNED integer with arithmetic operations
 * @details May be used to represent positive or negative integers. Number precision is set in template parameter bitness.
 */

template <std::size_t bitness = 512>
class Aessi final {
    /* -------------------------- @name Class members. ----------------------- */
    /**
     * @brief Unsigned number base
     */
    using CurrentBase = Aeu<bitness>;
    CurrentBase base;

    template <typename T1, typename T2>
    using pair = std::pair<T1, T2>;

    /**
     * @enum Aesi::Sign
     * @brief Specifies sign of the number. Should be Positive, Negative or Zero
     */
    enum Sign { Zero = 0, Positive = 1, Negative = 2 } sign;
    /* ----------------------------------------------------------------------- */

public:

    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aessi() noexcept = default;

    /**
     * @brief Copy constructor
     */
    gpu constexpr Aessi(const Aessi& copy) noexcept {
        sign = copy.sign; if(copy.sign != Zero) base = copy.base;
    };

    /**
     * @brief Copy assignment operator
     */
    gpu constexpr Aessi& operator=(const Aessi& other) noexcept {
        base = other.base; sign = other.sign; return *this;
    }

    /**
     * @brief Integral constructor
     * @param value Integral
     * @details Accepts each integral built-in type signed and unsigned
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aessi(Integral value) noexcept {
        if(value != 0) {
            base = Aeu(value);
            sign = (value < 0 ? Negative : Positive);
        } else {
            base = {};
            sign = Zero;
        }
    }

    /**
     * @brief Pointer-based character constructor
     * @param Char* pointer
     * @param Size_t size
     * @details Accepts decimal literals along with binary (starting with 0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aessi(const Char* ptr, std::size_t size) noexcept : Aessi {} {
        if(size == 0) return;

        if(*ptr == [] { if constexpr (std::is_same_v<char, Char>) { return '-'; } else { return L'-'; } } ()) {
            sign = Negative;
            base = Aeu(ptr + 1, size - 1);
        } else
            base = Aeu(ptr, size);
    }

    /**
     * @brief C-style string literal constructor
     * @param Char[] literal
     */
    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aessi(const Char (&literal)[arrayLength]) noexcept : Aessi(literal, arrayLength) {}

    /**
     * @brief String or string-view based constructor
     * @param String/String-View sv
     * @details Constructs object from STD::Basic_String or STD::Basic_String_View. Accepts objects based on char or wchar_t
     */
    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aessi(String&& stringView) noexcept : Aessi(stringView.data(), stringView.size()) {}
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /**
     * @brief Unary plus operator
     * @return Aesi
     * @note Does basically nothing
     */
    gpu constexpr auto operator+() const noexcept -> Aessi { return *this; }

    /**
     * @brief Unary minus operator
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator-() const noexcept -> Aessi {
        if(sign == Zero) return Aesi {};
        Aesi result = *this; result.inverse(); return result;
    }

    /**
     * @brief Prefix increment
     * @return Aesi&
     */
    gpu constexpr auto operator++() noexcept -> Aessi& { return this->operator+=(1); }

    /**
     * @brief Postfix increment
     * @return Aesi
     */
    gpu constexpr auto operator++(int) & noexcept -> Aessi {
        Aesi old = *this; operator++(); return old;
    }

    /**
     * @brief Prefix decrement
     * @return Aesi&
     */
    gpu constexpr auto operator--() noexcept -> Aessi& { return this->operator-=(1); }

    /**
     * @brief Postfix decrement
     * @return Aesi
     */
    gpu constexpr auto operator--(int) & noexcept -> Aessi {
        Aesi old = *this; operator--(); return old;
    }

    /**
     * @brief Addition operator
     * @param Aesi addendum
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator+(const Aessi& addendum) const noexcept -> Aessi {
        Aesi result = *this; result += addendum; return result;
    }

    /**
     * @brief Assignment addition operator
     * @param Aesi addendum
     * @return Aesi&
     */
    gpu constexpr auto operator+=(const Aessi& addendum) noexcept -> Aessi& {
        if(sign == Zero) return this->operator=(addendum);
        if(addendum.sign == Zero) return *this;

        if (sign != addendum.sign) {
            if (sign == Negative)
                blocks = makeComplement(blocks);
            const uint64_t carryOut = (addendum.sign != Negative ?
                                       addLine(blocks, addendum.blocks) : addLine(blocks, makeComplement(addendum.blocks)));
            if (carryOut == 0) {
                blocks = makeComplement(blocks);
                sign = Negative;
            } else sign = Positive;
        } else
            addLine(blocks, addendum.blocks);

        if (isLineEmpty(blocks))
            sign = Zero;

        return *this;
    }

    /**
     * @brief Subtraction operator
     * @param Aesi subtrahend
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator-(const Aessi& subtrahend) const noexcept -> Aessi {
        Aesi result = *this; result -= subtrahend; return result;
    }

    /**
     * @brief Assignment subtraction operator
     * @param Aesi subtrahend
     * @return Aesi&
     */
    gpu constexpr auto operator-=(const Aessi& subtrahend) noexcept -> Aessi& {
        return this->operator+=(-subtrahend);
    }

    /**
     * @brief Multiplication operator
     * @param Aesi factor
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator*(const Aessi& factor) const noexcept -> Aessi {
        Aesi result = *this; result *= factor; return result;
    }

    /**
     * @brief Assignment multiplication operator
     * @param Aesi factor
     * @return Aesi&
     */
    gpu constexpr auto operator*=(const Aessi& factor) noexcept -> Aessi& {
        if(sign == Zero) return *this;
        if(factor.sign == Zero) return this->operator=(Aesi {});

        sign = (sign != factor.sign ? Negative : Positive);
        base *= factor.base;

        return *this;
    }

    /**
     * @brief Division operator
     * @param Aesi divisor
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator/(const Aessi& divisor) const noexcept -> Aessi {
        Aesi quotient, _; divide(*this, divisor, quotient, _); return quotient;
    }

    /**
     * @brief Assignment division operator
     * @param Aesi divisor
     * @return Aesi&
     */
    gpu constexpr auto operator/=(const Aessi& divisor) noexcept -> Aessi& {
        return this->operator=(divide(*this, divisor).first);
    }

    /**
     * @brief Modulo operator
     * @param Aesi modulo
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator%(const Aessi& modulo) const noexcept -> Aessi {
        Aesi _, remainder; divide(*this, modulo, _, remainder); return remainder;
    }

    /**
     * @brief Assignment modulo operator
     * @param Aesi modulo
     * @return Aesi&
     */
    gpu constexpr auto operator%=(const Aessi& modulo) noexcept -> Aessi& {
        return this->operator=(divide(*this, modulo).second);
    }
    /* ----------------------------------------------------------------------- */


    /* ----------------------- @name Bitwise operators. ---------------------- */
    /**
     * @brief Bitwise complement operator
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator~() const noexcept -> Aessi {
        Aessi result;

        result.base = ~base;
        if(result.base.isZero())
            result.sign = Zero; else result.sign = sign;

        return result;
    }

    /**
     * @brief Bitwise XOR operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator^(const Aessi& other) const noexcept -> Aessi {
        Aessi result = *this; result ^= other; return result;
    }

    /**
     * @brief Assignment bitwise XOR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator^=(const Aessi& other) noexcept -> Aessi& {
        base ^= other.base; if(base.isZero()) sign = Zero; return *this;
    }

    /**
     * @brief Bitwise AND operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator&(const Aessi& other) const noexcept -> Aessi {
        Aessi result = *this; result &= other; return result;
    }

    /**
     * @brief Assignment bitwise AND operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator&=(const Aessi& other) noexcept -> Aessi& {
        base &= other.base; if(base.isZero()) sign = Zero; return *this;
    }

    /**
     * @brief Bitwise OR operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator|(const Aessi& other) const noexcept -> Aessi {
        Aessi result = *this; result |= other; return result;
    }

    /**
     * @brief Assignment bitwise OR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator|=(const Aessi& other) noexcept -> Aessi& {
        base |= other.base; if(sign == Zero && !base.isZero()) sign = Positive; return *this;
    }

    /**
     * @brief Left shift operator
     * @param Integral bit_shift
     * @return Aesi
     * @note Does right shift (>>) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator<<(Integral bitShift) const noexcept -> Aessi {
        Aessi result = *this; result.operator<<=(bitShift); return result;
    }

    /**
     * @brief Left shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does right shift (>>=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator<<=(Integral bitShift) noexcept -> Aessi& {
        base <<= bitShift; if(base.isZero()) sign = Zero; return *this;
    }

    /**
     * @brief Right shift operator
     * @param Integral bit_shift
     * @return Aesi
     * @note Does left shift (<<) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator>>(Integral bitShift) const noexcept -> Aessi {
        Aessi result = *this; result >>= bitShift; return result;
    }

    /**
     * @brief Right shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does left shift (<<=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator>>=(Integral bitShift) noexcept -> Aessi& {
        base >>= bitShift; if(base.isZero()) sign = Zero; return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Comparison operators. --------------------- */
    /**
     * @brief Comparison operator
     * @param Aesi other
     * @return Bool
     */
    gpu constexpr auto operator==(const Aessi& other) const noexcept -> bool {
        if(sign != Zero || other.sign != Zero)
            return (sign == other.sign && base == other.base); else return true;
    };

    /**
     * @brief Internal comparison operator
     * @param Aesi other
     * @return AesiCMP
     * @note Should almost never return AesiCMP::Equivalent
     */
    [[nodiscard]]
    gpu constexpr auto compareTo(const Aessi& other) const noexcept -> Ordering {
        switch (sign) {
            case Zero:
                switch (other.sign) {
                    case Zero:
                        return Ordering::equal;
                    case Positive:
                        return Ordering::less;
                    case Negative:
                        return Ordering::greater;
                    default:
                        return Ordering::equivalent;
                }
            case Positive:
                switch (other.sign) {
                    case Positive:
                        return base.compareTo(other.base);
                    case Zero:
                    case Negative:
                        return Ordering::greater;
                    default:
                        return Ordering::equivalent;
                }
            case Negative:
                switch (other.sign) {
                    case Negative: {
                        const auto ratio = base.compareTo(other.base);
                        if(ratio == Ordering::greater)
                            return Ordering::less; else if(ratio == Ordering::less) return Ordering::greater; else return ratio;
                    }
                    case Zero:
                    case Positive:
                        return Ordering::less;
                    default:
                        return Ordering::equivalent;
                }
            default:
                return Ordering::equivalent;
        }
    };

#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)) && !defined DOXYGEN_SKIP
    /**
     * @brief Oldstyle comparison operator(s). Used inside CUDA cause it does not support <=> on device
     */
    gpu constexpr auto operator!=(const Aessi& value) const noexcept -> bool { return !this->operator==(value); }
    gpu constexpr auto operator<(const Aessi& value) const noexcept -> bool { return this->compareTo(value) == Ordering::less; }
    gpu constexpr auto operator<=(const Aessi& value) const noexcept -> bool { return !this->operator>(value); }
    gpu constexpr auto operator>(const Aessi& value) const noexcept -> bool { return this->compareTo(value) == Ordering::greater; }
    gpu constexpr auto operator>=(const Aessi& value) const noexcept -> bool { return !this->operator<(value); }
#else
    /**
     * @brief Three-way comparison operator
     * @param Aesi other
     * @return Std::Strong_ordering
     * @note Should almost never return Strong_ordering::Equivalent
     */
    gpu constexpr auto operator<=>(const Aessi& other) const noexcept -> std::strong_ordering {
        const auto ratio = this->compareTo(other);
        assert(ratio == Ordering::less || ratio == Ordering::greater || ratio == Ordering::equal);

        switch(ratio) {
            case Ordering::less:
                return std::strong_ordering::less;
            case Ordering::greater:
                return std::strong_ordering::greater;
            case Ordering::equal:
                return std::strong_ordering::equal;
            default:
                return std::strong_ordering::equivalent;
        }
    };
#endif
    /* ----------------------------------------------------------------------- */


    /* ---------------------- @name Supporting methods. ---------------------- */
    /**
     * @brief Set bit in number by index starting from the right
     * @param Size_t index
     * @param Bool bit
     * @note Does nothing for index out of range
     */
    gpu constexpr auto setBit(std::size_t index, bool bit) noexcept -> void {
        if(index >= bitness) return;

        base.setBit(index, bit);
        if(bit) {
            if(sign == Zero && base.getBlock(index / blockBitLength) != 0)
                sign = Positive;
        } else {
            if(sign != Zero && base.isZero())
                sign = Zero;
        }
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
        if(sign == Zero && base.getBlock(index / sizeof(block)) != 0) sign = Positive;
        if(sign != Zero && base.isZero()) sign = Zero;
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
        if(sign == Zero && block != 0) sign = Positive;
        if(sign != Zero && base.isZero()) sign = Zero;
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
     * @brief Get number's absolute value
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto abs() const noexcept -> Aessi {
        if(sign != Negative)
            return *this;
        Aesi result = *this; result.sign = Positive; return result;
    }

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
    gpu constexpr auto isZero() const noexcept -> bool { return sign == Zero; }

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
    gpu static constexpr auto getBlocksNumber() noexcept -> std::size_t { return CurrentBase ::getBlocksNumber(); }

    /**
     * @brief Get square root
     * @return Aesi
     * @note Returns zero for negative value or zero
     */
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aessi {
        if(sign != Positive) return Aesi {}; return base.squareRoot();
    }

    /**
     * @brief Make swap between two objects
     * @param Aesi other
     */
    gpu constexpr auto swap(Aessi& other) noexcept -> void {
        Aesi t = other; other.operator=(*this); this->operator=(t);
    }

    /**
     * @brief Inverse number's sign
     * @param Aesi value
     */
    gpu constexpr auto inverse() noexcept -> void {
        if(sign != Zero)
            sign = (sign == Positive ? Negative : Positive);
    }
    /* ----------------------------------------------------------------------- */

    /* -------------- @name Public arithmetic and number theory. ------------- */
    /**
     * @brief Integer division. Returns results by reference
     * @param Aesi number
     * @param Aesi divisor
     * @param Aesi quotient OUT
     * @param Aesi remainder OUT
     * @return Quotient and remainder by reference
     */
    gpu static constexpr auto divide(const Aessi& number, const Aessi& divisor, Aessi& quotient, Aessi& remainder) noexcept -> void {
        CurrentBase::divide(number.base, divisor.base, quotient.base, remainder.base);
        if(quotient.base.isZero())
            quotient.sign = Zero; else if(number.sign != divisor.sign) quotient.sign = (quotient.sign == Positive ? Negative : Positive);
        if(remainder.base.isZero())
            remainder.sign = Zero; else if(number.sign == Negative) remainder.sign = (remainder.sign == Positive ? Negative : Positive);
    }

    /**
     * @brief Integer division. Returns results by value in pair
     * @param Aesi number
     * @param Aesi divisor
     * @return Pair(Quotient, Remainder)
     */
    [[nodiscard]]
    gpu static constexpr auto divide(const Aessi& number, const Aessi& divisor) noexcept -> pair<Aessi, Aessi> {
        auto results = CurrentBase::divide(number.base, divisor.base);
        if(results.first.base.isZero())
            quotient.sign = Zero; else if(number.sign != divisor.sign) quotient.sign = (quotient.sign == Positive ? Negative : Positive);
        if(remainder.base.isZero())
            remainder.sign = Zero; else if(number.sign == Negative) remainder.sign = (remainder.sign == Positive ? Negative : Positive);
    }

    /**
     * @brief Extended Euclidean algorithm for greatest common divisor
     * @param Aesi first
     * @param Aesi second
     * @param Aesi bezoutX OUT
     * @param Aesi bezoutY OUT
     * @return Aesi
     * @details Counts Bézout coefficients along with the greatest common divisor. Returns coefficients by reference
     */
    [[nodiscard]]
    gpu static constexpr auto gcd(const Aessi& first, const Aessi& second, Aessi& bezoutX, Aessi& bezoutY) noexcept -> Aessi {
        /* TODO: Check sign */
        return CurrentBase::gcd(first.base, second.base, bezoutX.base, bezoutY.base);
    }

    /**
     * @brief Greatest common divisor
     * @param Aesi first
     * @param Aesi second
     * @return Aesi
     */
    [[nodiscard]]
    gpu static constexpr auto gcd(const Aessi& first, const Aessi& second) noexcept -> Aessi {
        /* TODO: Check sign */
        return CurrentBase::gcd(first.base, second.base);
    }

    /**
     * @brief Least common multiplier
     * @param Aesi first
     * @param Aesi second
     * @return Aesi
     */
    [[nodiscard]]
    gpu static constexpr auto lcm(const Aessi& first, const Aessi& second) noexcept -> Aessi {
        return CurrentBase::lcm(first.base, second.base);
    }

    /**
     * @brief Exponentiation by modulo
     * @param Aesi base
     * @param Aesi power
     * @param Aesi modulo
     * @return
     * Aesi
     * @note Be aware of overflow
     */
    [[nodiscard]]
    gpu static constexpr auto powm(const Aessi& base, const Aessi& power, const Aessi& mod) noexcept -> Aessi {
        return CurrentBase::powm(base.base, power.base, mod.base);
    }

    /**
     * @brief Fast exponentiation for powers of 2
     * @param Size_t power
     * @return Aesi
     * @details Returns zero for power greater than current bitness
     */
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aesi {
        Aesi result {}; result.setBit(e, true); return result;
    }
    /* ----------------------------------------------------------------------- */

};

#endif //AESSI_MULTIPRECISION
