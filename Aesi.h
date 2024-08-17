#ifndef AESI_MULTIPRECISION
#define AESI_MULTIPRECISION

/// @cond HIDE_INCLUDES
#include "Aeu.h"
/// @endcond

/**
 * @file Aesi.h
 * @brief Long precision signed integer with arithmetic operations
 */

/**
 * @class Aesi
 * @brief Long precision signed integer
 * @details May be used to represent positive and negative integers. Number precision is set in template parameter bitness.
 */
template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aesi final {
    /* -------------------------- @name Class members. ----------------------- */
    enum class Sign { Zero = 0, Positive = 1, Negative = -1 } sign;

    using Base = Aeu<bitness>;
    Base base;
    /* ----------------------------------------------------------------------- */

    gpu constexpr Aesi(Sign withSign, Base withBase): sign { withSign }, base { withBase } {};

public:
    /* --------------------- TODO: COMPLETED! @name Different constructors. ------------------- */
    gpu constexpr Aesi() noexcept = default;

    gpu constexpr Aesi(const Aesi& copy) noexcept = default;

    template <typename Integral> requires (std::is_signed_v<Integral>)
    gpu constexpr Aesi& operator=(Integral value) noexcept {
        if(value != 0) {
            if(value < 0) {
                sign = Sign::Negative;
                value *= -1;
            } else sign = Sign::Positive;
            base = static_cast<unsigned long long>(value);
        } else sign = Sign::Zero;
        return *this;
    }

    gpu constexpr Aesi& operator=(const Aesi& other) noexcept { base = other.base; sign = other.sign; return *this; }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept {
        if(value < 0) {
            value *= -1;
            base = Base(static_cast<unsigned long long>(value));
            sign = Sign::Negative;
        } else if(value > 0) {
            base = Base(static_cast<unsigned long long>(value));
            sign = Sign::Positive;
        } else
            sign = Sign::Zero;
    }

    /**
     * @brief Pointer-based character constructor
     * @param ptr Char*
     * @param size Size_t
     * @details Accepts decimal strings (no prefix), binary (0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     * @note An odd number of dashes makes the number negative
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept : base(ptr, size) {
        if(!base.isZero()) {
            uint8_t positive = 1;

            const auto dash = [] {
                if constexpr (std::is_same_v<Char, char>) {
                    return '-';
                } else {
                    return L'-';
                }
            } ();
            for(std::size_t i = 0; i < size; ++i)
                if(ptr[i] == dash) positive ^= 1;

            sign = (positive ? Sign::Positive : Sign::Negative);
        } else sign = Sign::Zero;
    }

    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&literal)[arrayLength]) noexcept : Aesi(literal, arrayLength) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(const String& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    explicit gpu constexpr Aesi(const Aeu<bitness>& value) : sign(Sign::Positive), base(value) {}

#ifdef AESI_CRYPTOPP_INTEGRATION
    constexpr Aesi(const CryptoPP::Integer& value) {
        if(value.IsZero())
            sign = Sign::Zero;
        else {
            base = value;
            if(value.IsNegative())
                sign = Sign::Negative;
            else sign = Sign::Positive;
        }
    }
#endif

#ifdef AESI_GMP_INTEGRATION
    constexpr Aesi(const mpz_class& value) {
        if(value == 0)
            sign = Sign::Zero;
        else {
            base = value;
            if(value < 0)
                sign = Sign::Negative;
            else sign = Sign::Positive;
        }
    }
#endif
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /* ------------------------- TODO: COMPLETED! @name Unary operators. -------------------------- */
        gpu constexpr auto operator+() const noexcept -> Aesi { return *this; }

        [[nodiscard]]
        gpu constexpr auto operator-() const noexcept -> Aesi { Aesi copy = *this; copy.inverse(); return copy; }

        gpu constexpr auto operator++() noexcept -> Aesi& {
            if(sign == Sign::Negative) {
                --base; if(base.isZero()) sign = Sign::Zero;
            } else if(sign == Sign::Positive) {
                ++base;
            } else { base = 1u; sign = Sign::Positive; }
            return *this;
    }

        gpu constexpr auto operator++(int) & noexcept -> Aesi { Aesi old = *this; operator++(); return old; }

        gpu constexpr auto operator--() noexcept -> Aesi& {
            if(sign == Sign::Negative) {
                ++base;
            } else if(sign == Sign::Positive) {
                --base; if(base.isZero()) sign = Sign::Zero;
            } else { base = 1u; sign = Sign::Negative; }
            return *this;
        }

        gpu constexpr auto operator--(int) & noexcept -> Aesi { Aesi old = *this; operator--(); return old; }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ TODO: Complete @name Addition operators. ------------------------ */
        [[nodiscard]]
        gpu constexpr auto operator+(const Aesi& addendum) const noexcept -> Aesi { Aesi result = *this; result += addendum; return result; }

        gpu constexpr auto operator+=(const Aesi& addendum) noexcept -> Aesi& {
            if(addendum.sign == Sign::Zero) /* Any += Zero; */
                return *this;
            if(sign == Sign::Zero) /* Zero += Any; */
                return this->operator=(addendum);

            if(sign == addendum.sign) { /* Positive += Positive; */
                base += addendum.base;
                return *this;
            }

            if(sign == Sign::Positive) { /* Positive += Negative; */
                /*  +       -
                 * 100 + (-80) -> 20
                 * 100 + (-150) -> -50
                 * 100 + (-100) -> 0
                 * TODO: REMOVE COMMENT
                 */
                const auto ratio = base.compareTo(addendum.base);
                switch(ratio) {
                    case Comparison::greater: {
                        base -= addendum.base;
                        return *this;
                    }
                    case Comparison::less: {
                        base = addendum.base - base;
                        sign = Sign::Negative;
                        return *this;
                    }
                    default: {
                        sign = Sign::Zero;
                        return *this;
                    }
                }
            } else { /* Negative += Positive; */
                /*    -      +
                 * (-150) + 100 -> -50
                 * (-80) + 100 -> 20
                 * (-100) + 100 -> 0
                 * TODO: REMOVE COMMENT
                 */
                const auto ratio = base.compareTo(addendum.base);
                switch(ratio) {
                    case Comparison::greater: {
                        base -= addendum.base;
                        return *this;
                    }
                    case Comparison::less: {
                        base = addendum.base - base;
                        sign = Sign::Positive;
                        return *this;
                    }
                    default: {
                        sign = Sign::Zero;
                        return *this;
                    }
                }
            }
        }
    /* --------------------------------------------------------------------------- */

    /* ----------------------- TODO: Complete @name Subtraction operators. ---------------------- */
        [[nodiscard]]
        gpu constexpr auto operator-(const Aesi& subtrahend) const noexcept -> Aesi { Aesi result = *this; result -= subtrahend; return result; }

        gpu constexpr auto operator-=(const Aesi& subtrahend) noexcept -> Aesi& {
            if(subtrahend.sign == Sign::Zero) /* Any -= Zero; */
                return *this;
            if(sign == Sign::Zero) { /* Zero -= Any; */
                *this = subtrahend;
                this->inverse();
                return *this;
            }

            if(sign == Sign::Positive) {
                if(subtrahend.sign == Sign::Positive) { /* Positive -= Positive; */
                    /*  +       +
                     * 100 - (80) -> 20
                     * 100 - (150) -> -50
                     * 100 - (100) -> 0
                     * TODO: REMOVE COMMENT
                     */
                    const auto ratio = base.compareTo(subtrahend.base);
                    switch(ratio) {
                        case Comparison::greater: {
                            base -= subtrahend.base;
                            return *this;
                        }
                        case Comparison::less: {
                            base = subtrahend.base - base;
                            sign = Sign::Negative;
                            return *this;
                        }
                        default: {
                            sign = Sign::Zero;
                            return *this;
                        }
                    }
                } else { /* Positive -= Negative; */
                    base += subtrahend.base;
                    return *this;
                }
            } else {
                if(subtrahend.sign == Sign::Negative) { /* Negative -= Negative; */
                    /*    -      -
                     * (-150) - (-100) -> -50
                     * (-80) - (-100) -> 20
                     * (-100) - (-100) -> 0
                     * TODO: REMOVE COMMENT
                     */
                    const auto ratio = base.compareTo(subtrahend.base);
                    switch(ratio) {
                        case Comparison::greater: {
                            base -= subtrahend.base;
                            return *this;
                        }
                        case Comparison::less: {
                            base = subtrahend.base - base;
                            sign = Sign::Positive;
                            return *this;
                        }
                        default: {
                            sign = Sign::Zero;
                            return *this;
                        }
                    }
                } else { /* Negative -= Positive; */
                    base += subtrahend.base;
                    return *this;
                }
            }
        }
    /* --------------------------------------------------------------------------- */

    /* --------------------- TODO: COMPLETED! @name Multiplication operators. --------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator*(Integral factor) const noexcept -> Aesi { Aesi result = *this; result *= factor; return result; }

        [[nodiscard]]
        gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi { Aesi result = *this; result *= factor; return result; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator*=(Integral factor) noexcept -> Aesi& {
            if(factor == 0) {
                sign = Sign::Zero;
            } else {
                if(factor < 0) {
                    this->inverse();
                    factor *= -1;
                }
                base.operator*=(static_cast<unsigned long long>(factor));
            }
            return *this;
        }

        gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& {
            if(factor == 0) {
                sign = Sign::Zero;
            } else {
                if(factor < 0)
                    this->inverse();
                base.operator*=(factor.base);
            }
            return *this;
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ TODO: COMPLETED! @name Division operators. ------------------------ */
        /**
         * @brief Division operator for built-in integral types
         * @param divisor Integral
         * @return Aesi
         * @note Undefined behaviour for division by zero
         */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator/(Integral divisor) const noexcept -> Aesi { Aesi result = *this; result /= divisor; return result; }

        /**
         * @brief Division operator
         * @param divisor Aesi
         * @return Aesi
         * @note Undefined behaviour for division by zero
         */
        [[nodiscard]]
        gpu constexpr auto operator/(const Aesi& divisor) const noexcept -> Aesi { Aesi result = *this; result /= divisor; return result; }

        /**
         * @brief Assignment division operator for built-in integral types
         * @param divisor Integral
         * @return Aesi&
         * @note Undefined behaviour for division by zero
         */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator/=(Integral divisor) noexcept -> Aesi& {
            if(divisor == 0) {
                sign = Sign::Zero;
            } else {
                if(divisor < 0) {
                    this->inverse();
                    divisor *= -1;
                }
                base.operator/=(static_cast<unsigned long long>(divisor));
            }
            return *this;
        }

        /**
         * @brief Assignment division operator
         * @param divisor Aesi
         * @return Aesi&
         * @note Undefined behaviour for division by zero
         */
        gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& {
            if(divisor == 0) {
                sign = Sign::Zero;
            } else {
                if(divisor < 0)
                    this->inverse();
                base.operator/=(divisor.base);
            }
            return *this;
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------- TODO: COMPLETED! @name Modulo operators. ------------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator%(Integral modulo) const noexcept -> Aesi { Aesi result = *this; result %= modulo; return result; }

        [[nodiscard]]
        gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi { Aesi result = *this; result %= modulo; return result; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator%=(Integral modulo) noexcept -> Aesi& {
            if(modulo == 0) {
                sign = Sign::Zero;
            } else {
                if(modulo < 0) {
                    this->inverse();
                    modulo *= -1;
                }
                base.operator%=(static_cast<unsigned long long>(modulo));
            }
            return *this;
        }

        gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {
            if(modulo == 0) {
                sign = Sign::Zero;
            } else {
                if(modulo < 0)
                    this->inverse();
                base.operator%=(modulo.base);
            }
            return *this;
        }
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */

    /* --------------------- TODO: COMPLETED! @name Comparison operators. --------------------- */
    /* ------------------------ @name Equality operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator==(Integral value) const noexcept -> bool { return compareTo(value) == Comparison::equal; }

        gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool {
            return sign == other.sign && base == other.base;
        }

        template <std::size_t otherBitness> requires (otherBitness != bitness)
        gpu constexpr auto operator==(const Aesi<otherBitness>& other) const noexcept -> bool {
            return precisionCast<otherBitness>() == other;
        }
    /* --------------------------------------------------------------------------- */


    /* ----------------------- @name Comparison operators. ----------------------- */
        /* TODO: COMPLETED! */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto compareTo(Integral value) const noexcept -> Comparison {
            if(value == 0) {
                switch(sign) {
                    case Sign::Positive:
                        return Comparison::greater;
                    case Sign::Negative:
                        return Comparison::less;
                    default:
                        return Comparison::equal;
                }
            } else if(value < 0) {
                switch(sign) {
                    case Sign::Negative:
                        switch(base.compareTo(static_cast<unsigned long long>(value))) {
                            case Comparison::greater:
                                return Comparison::less;
                            case Comparison::less:
                                return Comparison::greater;
                            default:
                                return Comparison::equal;
                        }
                    case Sign::Positive:    // + > -
                    default:                // 0 > -
                        return Comparison::greater;
                }
            } else {
                switch(sign) {
                    case Sign::Positive:
                        return base.compareTo(static_cast<unsigned long long>(value));
                    case Sign::Negative:    // - < +
                    default:                // 0 < +
                        return Comparison::less;
                }
            }
        }

        /* TODO: COMPLETED! */
        template <std::size_t otherBitness = bitness> [[nodiscard]]
        gpu constexpr auto compareTo(const Aesi<otherBitness>& value) const noexcept -> Comparison { return precisionCast<otherBitness>().compareTo(value); }

        /* TODO: COMPLETED! */
        [[nodiscard]]
        gpu constexpr auto compareTo(const Aesi& value) const noexcept -> Comparison {
            if(value == 0) {
                switch(sign) {
                    case Sign::Positive:
                        return Comparison::greater;
                    case Sign::Negative:
                        return Comparison::less;
                    default:
                        return Comparison::equal;
                }
            } else if(value < 0) {
                switch(sign) {
                    case Sign::Negative:
                        switch(base.compareTo(value.base)) {
                            case Comparison::greater:
                                return Comparison::less;
                            case Comparison::less:
                                return Comparison::greater;
                            default:
                                return Comparison::equal;
                        }
                    case Sign::Positive:    // + > -
                        default:            // 0 > -
                            return Comparison::greater;
                }
            } else {
                switch(sign) {
                    case Sign::Positive:
                        return base.compareTo(value.base);
                    case Sign::Negative:    // - < +
                        default:            // 0 < +
                            return Comparison::less;
                }
            }
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Spaceship operators. ----------------------- */
#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (PRE_CPP_20)) && !defined DOXYGEN_SKIP
        /* TODO: COMPLETED! */
        /**
         * @brief Oldstyle comparison operator(s). Used inside CUDA cause it does not support <=> operator.
         */
        gpu constexpr auto operator!=(const Aeu& value) const noexcept -> bool { return !this->operator==(value); }
        gpu constexpr auto operator<(const Aeu& value) const noexcept -> bool { return this->compareTo(value) == Comparison::less; }
        gpu constexpr auto operator<=(const Aeu& value) const noexcept -> bool { return !this->operator>(value); }
        gpu constexpr auto operator>(const Aeu& value) const noexcept -> bool { return this->compareTo(value) == Comparison::greater; }
        gpu constexpr auto operator>=(const Aeu& value) const noexcept -> bool { return !this->operator<(value); }
#else
        /* TODO: COMPLETED! */
        /**
         * @brief Three-way comparison operator
         * @param other Aesi
         * @return Std::Strong_ordering
         * @note Available from C++20 standard and further. Should almost never return Strong_ordering::Equivalent
         */
        gpu constexpr auto operator<=>(const Aesi& other) const noexcept -> std::strong_ordering {
            const auto ratio = this->compareTo(other);
            switch(ratio) {
            case Comparison::less:
                return std::strong_ordering::less;
            case Comparison::greater:
                return std::strong_ordering::greater;
            case Comparison::equal:
                return std::strong_ordering::equal;
            default:
                return std::strong_ordering::equivalent;
        }
        };

        /* TODO: COMPLETED! */
        /**
         * @brief Three-way comparison operator for numbers of different precision and built-in integral types
         * @param other Unsigned
         * @return Std::Strong_ordering
         * @note Available from C++20 standard and further. Should almost never return Strong_ordering::Equivalent
         */
        template <typename Unsigned>
        gpu constexpr auto operator<=>(const Unsigned& other) const noexcept -> std::strong_ordering {
            const auto ratio = this->compareTo(other);
            switch(ratio) {
                case Comparison::less:
                    return std::strong_ordering::less;
                case Comparison::greater:
                    return std::strong_ordering::greater;
                case Comparison::equal:
                    return std::strong_ordering::equal;
                default:
                    return std::strong_ordering::equivalent;
            }
        };
#endif
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */

    /* TODO: COMPLETED! */
    /* ---------------------- @name Supporting methods. ---------------------- */
    /**
     * @brief Set bit in number by index starting from the right
     * @param index Size_t
     * @param bit Boolean
     * @note Does nothing for index out of range. Does not affect sign
     */
    gpu constexpr auto setBit(std::size_t index, bool bit) noexcept -> void { return base.setBit(index, bit); }

    /**
     * @brief Get bit in number by index starting from the right
     * @param index Size_t
     * @return Boolean
     * @note Returns zero for index out of range. Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool { return base.getBit(index); }

    /**
     * @brief Set byte in number by index starting from the right
     * @param index Size_t
     * @param byte Byte
     * @note Does nothing for index out of range. Does not affect the sign
     */
    gpu constexpr auto setByte(std::size_t index, byte byte) noexcept -> void { return base.setByte(index, byte); }

    /**
     * @brief Get byte in number by index starting from the right
     * @param index Size_t
     * @return Byte
     * @note Returns zero for index out of range. Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> byte { return base.getByte(index); }

    /**
     * @brief Set block in number by index starting from the right
     * @param index Size_t
     * @param block Block
     * @note Does nothing for index out of range. Does not affect the sign
     */
    gpu constexpr auto setBlock(std::size_t index, block block) noexcept -> void { return base.setBlock(index, block); }

    /**
     * @brief Get block in number by index starting from the right
     * @param index Size_t
     * @return Block
     * @note Returns zero for index out of range. Does not affect the sign
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
     * @return Boolean: true if the number is odd and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return base.isOdd(); }

    /**
     * @brief Check whether number is even
     * @return Boolean: true if the number is even and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return base.isEven(); }

    /**
     * @brief Check whether number is zero
     * @return Boolean: true if the number is zero and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isZero() const noexcept -> bool { return sign == Sign::Zero; }

    /**
     * @brief Get number of non-empty blocks inside object starting from the right
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto filledBlocksNumber() const noexcept -> std::size_t { return base.filledBlocksNumber(); }

    /**
     * @brief Get number's precision
     * @return Size_t
     */
    [[nodiscard]]
    gpu static constexpr auto getBitness() noexcept -> std::size_t { return bitness; }

    /**
     * @brief Get the number of blocks (length of array of uint32_t integers) inside object
     * @return Size_t
     */
    [[nodiscard]]
    gpu static constexpr auto totalBlocksNumber() noexcept -> std::size_t { return Aeu<bitness>::totalBlocksNumber(); }

    /**
     * @brief Make swap between two objects
     * @param other Aesi
     */
    gpu constexpr auto swap(Aesi& other) noexcept -> void {
        Sign tSign = sign; sign = other.sign; other.sign = tSign;
        base.swap(other.base);
    }

    /**
     * @brief Invertes number's bitness
     * @details Turns negative to positive and otherwise. Leaves zero unchanges
     */
    gpu constexpr auto inverse() noexcept -> void { sign = (sign == Sign::Zero ? Sign::Zero : (sign == Sign::Negative ? Sign::Positive : Sign::Negative)); }
    /* ----------------------------------------------------------------------- */


    /* TODO: COMPLETED! */
    /* -------------- @name Public arithmetic and number theory. ------------- */
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor, Aesi& quotient, Aesi& remainder) noexcept -> void {
        if(number.sign == Sign::Zero || divisor.sign == Sign::Zero) {
            quotient.sign = Sign::Zero;
            remainder.sign = Sign::Zero;
            return;
        }

        Base::divide(number.base, divisor.base, quotient.base, remainder.base);
        if(number.sign == Sign::Positive) {
            if(divisor.sign == Sign::Positive) {
                // (+; +) -> (+; +)
                quotient.sign = Sign::Positive;
                remainder.sign = Sign::Positive;
            } else {
                // (+; -) -> (-; +)
                quotient.sign = Sign::Negative;
                remainder.sign = Sign::Positive;
            }
        } else {
            if(divisor.sign == Sign::Positive) {
                // (-; +) -> (-; -)
                quotient.sign = Sign::Negative;
                remainder.sign = Sign::Negative;
            } else {
                // (-; -) -> (+; -)
                quotient.sign = Sign::Positive;
                remainder.sign = Sign::Negative;
            }
        }

        if(quotient.base.isZero())
            quotient.sign = Sign::Zero;
        if(remainder.base.isZero())
            remainder.sign = Sign::Zero;
    }

    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {
        if(sign == Sign::Positive)
            return Aesi { Sign::Positive, base.squareRoot() };
        Aesi result; result.sign = Sign::Zero; return result;
    }

    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t power) noexcept -> Aesi {
        Aesi result { Sign::Positive, Aeu<bitness>::power2(power) };
        if(result.base.isZero())
            result.sign = Sign::Zero;
        return result;
    }
    /* ----------------------------------------------------------------------- */

    /* TODO: COMPLETED! */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        if(sign == Sign::Zero)
            return Integral(0);

        if constexpr (std::is_signed_v<Integral>) {
            if(sign == Sign::Negative)
                return base.template integralCast<Integral>() * -1;
        }

        return base.template integralCast<Integral>();
    }

    /* TODO: COMPLETED! */
    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> {
        if(sign != Sign::Zero) {
            Aesi<newBitness> result;

            const std::size_t blockBoarder = (newBitness > bitness ? Aesi<bitness>::totalBlocksNumber() : Aesi<newBitness>::totalBlocksNumber());
            for(std::size_t blockIdx = 0; blockIdx < blockBoarder; ++blockIdx)
                result.setBlock(blockIdx, getBlock(blockIdx));

            return result;
        } else return Aesi<newBitness> {};
    }

    /* TODO: COMPLETED! */
    gpu constexpr auto unsignedCast() const noexcept -> Aeu<bitness> { return base; }

    /* TODO: COMPLETED! */
    /* ----------------- @name Public input-output operators. ---------------- */
    template <byte notation, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (notation == 2 || notation == 8 || notation == 10 || notation == 16))
    gpu constexpr auto getString(Char* buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t {
        if(sign == Sign::Negative && bufferSize > 0) {
            *buffer++ = [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ();
            --bufferSize;
        }
        return base.template getString<notation, Char>(buffer, bufferSize, showBase, hexUppercase);
    }

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) -> std::basic_ostream<Char>& {
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
    __device__ constexpr auto tryAtomicSet(const Aesi& value) noexcept -> void {}

    /**
     * @brief Atomicity-oriented object exchangement operator
     * @param Aesi exchangeable
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for exchanging encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicExchange(const Aesi& value) noexcept -> void {}
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
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) + value; }

/**
 * @brief Integral conversion subtraction operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) - value; }

/**
 * @brief Integral conversion multiplication operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) * value; }

/**
 * @brief Integral conversion division operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) / value; }

/**
 * @brief Integral conversion modulo operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) % value; }

/**
 * @brief Integral conversion bitwise XOR operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) ^ value; }

/**
 * @brief Integral conversion bitwise AND operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) & value; }

/**
 * @brief Integral conversion bitwise OR operator
 * @param number Integral
 * @param value Aesi
 * @return Aesi
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aesi<bitness>& value) noexcept { return Aesi<bitness>(number) | value; }
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //AESI_MULTIPRECISION
