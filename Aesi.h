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
 * @brief Long precision signed integer
 * @details May be used to represent positive and negative integers. Number precision is set in template parameter bitness.
 */
template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aesi final {
    /* -------------------------- @name Class members. ----------------------- */
    Sign sign;

    using Base = Aeu<bitness>;
    Base base;
    /* ----------------------------------------------------------------------- */

    gpu constexpr Aesi(Sign withSign, Base withBase): sign { withSign }, base { withBase } {};

public:
    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aesi() noexcept = default;

    /**
     * @brief Destructor
     */
    gpu constexpr ~Aesi() noexcept = default;

    /**
     * @brief Copy constructor
     * @param copy Aesi&
     */
    gpu constexpr Aesi(const Aesi& copy) noexcept = default;

    /**
     * @brief Integral constructor
     * @param value Integral
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept {
        using enum Sign;
        if(value < 0) {
            value *= -1;
            base = Base(static_cast<unsigned long long>(value));
            sign = Negative;
        } else if(value > 0) {
            base = Base(static_cast<unsigned long long>(value));
            sign = Positive;
        } else
            sign = Zero;
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
        using enum Sign;
        if(!base.isZero()) {
            std::byte positive { 1 };

            const auto dash = [] {
                if constexpr (std::is_same_v<Char, char>) {
                    return '-';
                } else {
                    return L'-';
                }
            } ();
            for(std::size_t i = 0; i < size; ++i)
                if(ptr[i] == dash) positive ^= std::byte {1};

            sign = (positive == std::byte {1} ? Positive : Negative);
        } else sign = Zero;
    }

    /**
     * @brief C-style string literal constructor
     * @param literal Char[]
     */
    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&literal)[arrayLength]) noexcept : Aesi(literal, arrayLength) {}

    /**
     * @brief String / String-view constructor
     * @param stringView String
     */
    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
        std::decay_t<String>> || std::is_same_v<std::basic_string_view<Char>, std::decay_t<String>>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
        std::decay_t<String>> || std::is_same_v<std::basic_string_view<Char>, std::decay_t<String>>)
    gpu constexpr Aesi(const String& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    /**
     * @brief Unsigned integer conversion
     * @param value Aeu&
     */
    explicit gpu constexpr Aesi(const Aeu<bitness>& value) : sign(Sign::Positive), base(value) {}

#ifdef AESI_CRYPTOPP_INTEGRATION
    /**
     * @brief Crypto++ library integer constructor
     * @param number CryptoPP::Integer&
     */
    constexpr Aesi(const CryptoPP::Integer& number) {
        using enum Sign;
        if(number.IsZero())
            sign = Zero;
        else {
            base = number;
            if(number.IsNegative())
                sign = Negative;
            else sign = Positive;
        }
    }
#endif

#ifdef AESI_GMP_INTEGRATION
    /**
     * @brief GMP library integer constructor
     * @param number mpz_class&
     */
    constexpr Aesi(const mpz_class& number) {
        using enum Sign;
        if(number == 0)
            sign = Zero;
        else {
            base = number;
            if(number < 0)
                sign = Negative;
            else sign = Positive;
        }
    }
#endif

    /**
     * @brief Integral assignment operator
     * @param value Integral
     */
    template <typename Integral> requires (std::is_signed_v<Integral>)
    gpu constexpr Aesi& operator=(Integral value) noexcept {
        using enum Sign;
        if(value != 0) {
            if(value < 0) {
                sign = Negative;
                value *= -1;
            } else sign = Positive;
            base = static_cast<unsigned long long>(value);
        } else sign = Zero;
        return *this;
    }

    /**
     * @brief Copy assignment operator
     * @param other Aesi&
     */
    gpu constexpr Aesi& operator=(const Aesi& other) noexcept { base = other.base; sign = other.sign; return *this; }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /* ------------------------- @name Unary operators. -------------------------- */
        /**
         * @brief Unary plus operator
         * @details Does basically nothing
         * @return Aesi
         */
        gpu constexpr auto operator+() const noexcept -> Aesi { return *this; }

        /**
         * @brief Unary minus operator
         * @return Aesi
         */
        [[nodiscard]]
        gpu constexpr auto operator-() const noexcept -> Aesi { Aesi copy = *this; copy.inverse(); return copy; }

        /**
         * @brief Prefix increment operator
         * @return Aesi&
         */
        gpu constexpr auto operator++() noexcept -> Aesi& {
            using enum Sign;
            if(sign == Negative) {
                --base; if(base.isZero()) sign = Zero;
            } else if(sign == Positive) {
                ++base;
            } else { base = 1u; sign = Positive; }
            return *this;
        }

        /**
         * @brief Postfix increment operator
         * @return Aesi
         */
        gpu constexpr auto operator++(int) & noexcept -> Aesi { Aesi old = *this; operator++(); return old; }

        /**
         * @brief Prefix decrement operator
         * @return Aesi&
         */
        gpu constexpr auto operator--() noexcept -> Aesi& {
            using enum Sign;
            if(sign == Negative) {
                ++base;
            } else if(sign == Positive) {
                --base; if(base.isZero()) sign = Zero;
            } else { base = 1u; sign = Negative; }
            return *this;
        }

        /**
         * @brief Postfix decrement operator
         * @return Aesi
         */
        gpu constexpr auto operator--(int) & noexcept -> Aesi { Aesi old = *this; operator--(); return old; }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Addition operators. ------------------------ */
        /**
         * @brief Addition operator
         * @param base Aesi
         * @param addendum Aesi&
         * @return Aesi
         */
        [[nodiscard]]
        gpu constexpr friend auto operator+(const Aesi& base, const Aesi& addendum) noexcept -> Aesi {
            Aesi result = base; result += addendum; return result;
        }

        /**
         * @brief Addition assignment operator
         * @param left Aesi
         * @param right Aesi&
         * @return Aesi&
         */
        gpu constexpr friend auto operator+=(Aesi& left, const Aesi& right) noexcept -> Aesi& {
            using enum Sign;
            Sign& lSign = left.sign; const Sign& rSign = right.sign;
            Base& lBase = left.base; const Base& rBase = right.base;

            if(rSign == Zero)
                return left;
            else if(lSign == Zero)
                return left = right;
            else if(lSign == rSign) {
                lBase += rBase;
                return left;
            }

            if(lSign == Positive) {
                switch(lBase.compareTo(rBase)) {
                    using enum Comparison;
                    case greater: {
                        lBase -= rBase;
                        return left;
                    }
                    case less: {
                        lBase = rBase - lBase;
                        lSign = Negative;
                        return left;
                    }
                    default: {
                        lSign = Zero;
                        return left;
                    }
                }
            } else {
                switch(const auto ratio = lBase.compareTo(rBase)) {
                    using enum Comparison;
                    case greater: {
                        lBase -= rBase;
                        return left;
                    }
                    case less: {
                        lBase = rBase - lBase;
                        lSign = Positive;
                        return left;
                    }
                    default: {
                        lSign = Zero;
                        return left;
                    }
                }
            }
        }
    /* --------------------------------------------------------------------------- */

    /* ----------------------- @name Subtraction operators. ---------------------- */
        /**
         * @brief Subtraction operator
         * @param base Aesi
         * @param subtrahend Aesi&
         * @return Aesi
         */
        [[nodiscard]]
        gpu constexpr friend auto operator-(const Aesi& base, const Aesi& subtrahend) noexcept -> Aesi {
            Aesi result = base; result -= subtrahend; return result;
        }

        /**
         * @brief Subtraction assignment operator
         * @param left Aesi
         * @param right Aesi&
         * @return Aesi&
         */
        gpu constexpr friend auto operator-=(Aesi& left, const Aesi& right) noexcept -> Aesi& {
            using enum Sign;
            Sign& lSign = left.sign; const Sign& rSign = right.sign;
            Base& lBase = left.base; const Base& rBase = right.base;

            if(rSign == Zero)
                return left;
            if(lSign == Zero) {
                left = right;
                left.inverse();
                return left;
            }

            if(lSign == Positive) {
                if(rSign == Positive) {
                    switch(lBase.compareTo(rBase)) {
                        using enum Comparison;
                        case greater: {
                            lBase -= rBase;
                            return left;
                        }
                        case less: {
                            lBase = rBase - lBase;
                            lSign = Negative;
                            return left;
                        }
                        default: {
                            lSign = Zero;
                            return left;
                        }
                    }
                } else {
                    lBase += rBase;
                    return left;
                }
            } else {
                if(rSign == Negative) {
                    switch(lBase.compareTo(rBase)) {
                        using enum Comparison;
                        case greater: {
                            lBase -= rBase;
                            return left;
                        }
                        case less: {
                            lBase = rBase - lBase;
                            lSign = Positive;
                            return left;
                        }
                        default: {
                            lSign = Zero;
                            return left;
                        }
                    }
                } else {
                    lBase += rBase;
                    return left;
                }
            }
        }
    /* --------------------------------------------------------------------------- */

    /* --------------------- @name Multiplication operators. --------------------- */
        /**
         * @brief Multiplication operator for built-in types
         * @param factor Integral
         * @return Aesi
         */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator*(Integral factor) const noexcept -> Aesi { Aesi result = *this; result *= factor; return result; }

        /**
         * @brief Multiplication operator
         * @param factor Aesi&
         * @return Aesi
         */
        [[nodiscard]]
        gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi { Aesi result = *this; result *= factor; return result; }

        /**
         * @brief Multiplication assignment operator for built-in types
         * @param factor Integral
         * @return Aesi&
         */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator*=(Integral factor) noexcept -> Aesi& {
            using enum Sign;
            if(factor == 0) {
                sign = Zero;
            } else {
                if(factor < 0) {
                    this->inverse();
                    factor *= -1;
                }
                base *= static_cast<unsigned long long>(factor);
            }
            return *this;
        }

        /**
         * @brief Multiplication assignment operator
         * @param factor Aesi&
         * @return Aesi&
         */
        gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& {
            using enum Sign;
            if(factor.isZero()) {
                sign = Zero;
            } else {
                if(factor.isNegative())
                    this->inverse();
                base *= factor.base;
            }
            return *this;
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Division operators. ------------------------ */
        /**
         * @brief Division operator for built-in integral types
         * @param divisor Integral
         * @return Aesi
         * @note Undefined behaviour for division by zero
         */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator/(Integral divisor) const noexcept -> Aesi {
            Aesi result = *this; result /= divisor; return result;
        }

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
            using enum Sign;
            if(divisor == 0) {
                sign = Zero;
            } else {
                if(divisor < 0) {
                    this->inverse();
                    divisor *= -1;
                }
                base /= static_cast<unsigned long long>(divisor);
                if(base.isZero()) sign = Zero;
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
            using enum Sign;
            if(divisor.isZero()) {
                sign = Zero;
            } else {
                if(divisor.isNegative())
                    this->inverse();
                base /= divisor.base;
                if(base.isZero()) sign = Zero;
            }
            return *this;
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------- @name Modulo operators. ------------------------- */
        /**
         * @brief Modulo operator for built-in types
         * @param modulo Integral
         * @return Aesi
         * @note Returns zero for the modulo of zero
         */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator%(Integral modulo) const noexcept -> Aesi { Aesi result = *this; result %= modulo; return result; }

        /**
         * @brief Modulo operator
         * @param modulo Aesi&
         * @return Aesi
         * @details DETAILS
         * @note Returns zero for the modulo of zero
         */
        [[nodiscard]]
        gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi { Aesi result = *this; result %= modulo; return result; }

        /**
         * @brief Modulo assignment operator for built-in types
         * @param modulo Integral
         * @return Aesi&
         * @note Returns zero for the modulo of zero
         */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator%=(Integral modulo) noexcept -> Aesi& {
            using enum Sign;
            if(modulo == 0) {
                sign = Zero;
            } else {
                if(modulo < 0) {
                    this->inverse();
                    modulo *= -1;
                }
                base %= static_cast<unsigned long long>(modulo);
            }
            return *this;
        }

        /**
         * @brief Modulo assignment operator
         * @param modulo Aesi&
         * @return Aesi&
         * @details DETAILS
         * @note Returns zero for the modulo of zero
         */
        gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {
            if(modulo.isZero())
                return *this;

            if(modulo.isNegative())
                this->inverse();
            base %= modulo.base;
            if(base.isZero())
                sign = Sign::Zero;

            return *this;
        }
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */

    /* --------------------- @name Comparison operators. --------------------- */
    /* ------------------------ @name Equality operators. ------------------------ */
        /**
         * @brief Equality operator for built-in types
         * @param integral Integral
         * @return bool
         */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator==(Integral integral) const noexcept -> bool {
            return compareTo(integral) == Comparison::equal;
        }

        /**
         * @brief Equality operator
         * @param other Aesi&
         * @return bool
         */
        gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool {
            return sign == other.sign && base == other.base;
        }

        /**
         * @brief Different precision equlity operator
         * @param other Aesi&
         * @return bool
         */
        template <std::size_t otherBitness> requires (otherBitness != bitness)
        gpu constexpr auto operator==(const Aesi<otherBitness>& other) const noexcept -> bool {
            return precisionCast<otherBitness>() == other;
        }
    /* --------------------------------------------------------------------------- */


    /* ----------------------- @name Comparison operators. ----------------------- */
        /**
         * @brief Comparison operator for built-in types
         * @param integral Integral
         * @return Comparison
         */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto compareTo(Integral integral) const noexcept -> Comparison {
            using enum Sign; using enum Comparison;
            if(integral == 0) {
                switch(sign) {
                    case Positive:
                        return greater;
                    case Negative:
                        return less;
                    default:
                        return equal;
                }
            } else if(integral < 0) {
                if(sign == Negative)
                    switch(base.compareTo(static_cast<unsigned long long>(integral * -1))) {
                        using enum Comparison;
                        case greater:
                            return less;
                        case less:
                            return greater;
                        default:
                            return equal;
                    }
                else return greater;
            } else {
                if(sign == Positive)
                    return base.compareTo(static_cast<unsigned long long>(integral));
                return less;
            }
        }

        /**
         * @brief Different precision comparison operator
         * @param value Aesi&
         * @return Comparison
         */
        template <std::size_t otherBitness = bitness> [[nodiscard]]
        gpu constexpr auto compareTo(const Aesi<otherBitness>& value) const noexcept -> Comparison {
            return precisionCast<otherBitness>().compareTo(value);
        }

        /**
         * @brief Comparison operator
         * @param value Aesi&
         * @return Comparison
         */
        [[nodiscard]]
        gpu constexpr auto compareTo(const Aesi& value) const noexcept -> Comparison {
            using enum Sign; using enum Comparison;
            if(value.isZero()) {
                switch(sign) {
                    case Positive:
                        return greater;
                    case Negative:
                        return less;
                    default:
                        return equal;
                }
            }

            if(value.isNegative()) {
                if(sign == Negative)
                    switch(base.compareTo(value.base)) {
                        case greater:
                            return less;
                        case less:
                            return greater;
                        default:
                            return equal;
                    }
                return greater;
            }

            if(sign == Positive)
                return base.compareTo(value.base);
            return less;
        }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Spaceship operators. ----------------------- */
#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (PRE_CPP_20)) && !defined DOXYGEN_SKIP
        /**
         * @brief Oldstyle comparison operator(s). Used inside CUDA cause it does not support <=> operator.
         */
        gpu constexpr auto operator!=(const Aeu& value) const noexcept -> bool {
            return !this->operator==(value);
        }
        gpu constexpr auto operator<(const Aeu& value) const noexcept -> bool {
            return this->compareTo(value) == Comparison::less;
        }
        gpu constexpr auto operator<=(const Aeu& value) const noexcept -> bool {
            return !this->operator>(value);
        }
        gpu constexpr auto operator>(const Aeu& value) const noexcept -> bool {
            return this->compareTo(value) == Comparison::greater;
        }
        gpu constexpr auto operator>=(const Aeu& value) const noexcept -> bool {
            return !this->operator<(value);
        }
#else
        /**
         * @brief Three-way comparison operator
         * @param other Aesi
         * @return Std::Strong_ordering
         * @note Available from C++20 standard and further. Should almost never return Strong_ordering::Equivalent
         */
        gpu constexpr auto operator<=>(const Aesi& other) const noexcept -> std::strong_ordering {
            switch(this->compareTo(other)) {
                using enum Comparison;
                case less:
                    return std::strong_ordering::less;
                case greater:
                    return std::strong_ordering::greater;
                case equal:
                    return std::strong_ordering::equal;
                default:
                    return std::strong_ordering::equivalent;
            }
        }

        /**
         * @brief Three-way comparison operator for numbers of different precision and built-in integral types
         * @param other Unsigned
         * @return Std::Strong_ordering
         * @note Available from C++20 standard and further. Should almost never return Strong_ordering::Equivalent
         */
        template <typename Object>
        gpu constexpr auto operator<=>(const Object& other) const noexcept -> std::strong_ordering {
            switch(this->compareTo(other)) {
                using enum Comparison;
                case less:
                    return std::strong_ordering::less;
                case greater:
                    return std::strong_ordering::greater;
                case equal:
                    return std::strong_ordering::equal;
                default:
                    return std::strong_ordering::equivalent;
            }
        };
#endif
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */

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
     * @brief Check whether number is positive
     * @return Boolean: true if the number is positive and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isPositive() const noexcept -> bool { return sign == Sign::Positive; }

    /**
     * @brief Check whether number is negative
     * @return Boolean: true if the number is negative and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isNegative() const noexcept -> bool { return sign == Sign::Negative; }

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
    gpu constexpr auto inverse() noexcept -> void {
        using enum Sign;
        sign = (sign == Zero ? Zero : (sign == Negative ? Positive : Negative));
    }
    /* ----------------------------------------------------------------------- */

    /* -------------- @name Public arithmetic and number theory. ------------- */
    /**
     * @brief Integral division
     * @param number Aesi&
     * @param divisor Aesi&
     * @param quotient Aesi& OUT
     * @param remainder Aesi& OUT
     * @details Returns values by references
     */
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor, Aesi& quotient, Aesi& remainder) noexcept -> void {
        using enum Sign;
        if(number.sign == Zero || divisor.sign == Zero) {
            quotient.sign = Zero;
            remainder.sign = Zero;
            return;
        }

        Base::divide(number.base, divisor.base, quotient.base, remainder.base);
        if(number.sign == Positive) {
            if(divisor.sign == Positive) {
                quotient.sign = Positive;
                remainder.sign = Positive;
            } else {
                quotient.sign = Negative;
                remainder.sign = Positive;
            }
        } else {
            if(divisor.sign == Positive) {
                quotient.sign = Negative;
                remainder.sign = Negative;
            } else {
                quotient.sign = Positive;
                remainder.sign = Negative;
            }
        }

        if(quotient.base.isZero())
            quotient.sign = Zero;
        if(remainder.base.isZero())
            remainder.sign = Zero;
    }

    /**
     * @brief Square root
     * @return Aesi
     * @details Returns zero for negative
     */
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {
        using enum Sign;
        if(sign == Positive)
            return Aesi { Positive, base.squareRoot() };
        Aesi result; result.sign = Zero; return result;
    }

    /**
     * @brief Fast exponentiation of 2
     * @return Aesi
     */
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t power) noexcept -> Aesi {
        using enum Sign;
        Aesi result { Positive, Aeu<bitness>::power2(power) };
        if(result.base.isZero())
            result.sign = Zero;
        return result;
    }
    /* ----------------------------------------------------------------------- */

    /**
     * @brief Cast for built-in integral types
     * @return Integral
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        using enum Sign;
        if(sign == Zero)
            return Integral(0);

        if constexpr (std::is_signed_v<Integral>) {
            if(sign == Negative)
                return base.template integralCast<Integral>() * -1;
        }

        return base.template integralCast<Integral>();
    }

    /**
     * @brief Number's precision cast
     * @return Aesi<newBitness>
     */
    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> {
        using enum Sign;
        if(sign != Zero) {
            Aesi<newBitness> result = 1;

            const std::size_t blockBoarder = (newBitness > bitness ? totalBlocksNumber() : Aesi<newBitness>::totalBlocksNumber());
            for(std::size_t blockIdx = 0; blockIdx < blockBoarder; ++blockIdx)
                result.setBlock(blockIdx, getBlock(blockIdx));

            if(sign == Negative)
                result.inverse();

            return result;
        }

        return Aesi<newBitness> {};
    }

    /**
     * @brief Unsigned cast
     * @return Aeu
     */
    gpu constexpr auto unsignedCast() const noexcept -> Aeu<bitness> { return base; }

    /* ----------------- @name Public input-output operators. ---------------- */
    /**
      * @brief Character buffer output operator
      * @param buffer Char*
      * @param bufferSize Size_t
      * @param showBase Boolean
      * @param hexUppercase Boolean
      * @return Size_t - amount of symbols written
      * @details Places the maximum possible amount of number's characters in buffer. Base parameter should be 2, 8, 10, or 16
      * @note Works significantly faster for hexadecimal notation
      */
    template <byte notation, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (notation == 2 || notation == 8 || notation == 10 || notation == 16))
    gpu constexpr auto getString(Char* buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t {
        using enum Sign;

        if(sign != Zero) {
            if(sign == Negative && bufferSize > 0) {
                *buffer++ = [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ();
                --bufferSize;
            }
            return base.template getString<notation, Char>(buffer, bufferSize, showBase, hexUppercase);
        }

        if(showBase) {
            switch(notation) {
                case 2: {
                    if(bufferSize < 3) return 0;
                    buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
                    buffer[1] = [] { if constexpr (std::is_same_v<Char, char>) { return 'b'; } else { return L'b'; } } ();
                    buffer[2] = buffer[0];
                    return 3;
                }
                case 8: {
                    if(bufferSize < 3) return 0;
                    buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
                    buffer[1] = [] { if constexpr (std::is_same_v<Char, char>) { return 'o'; } else { return L'o'; } } ();
                    buffer[2] = buffer[0];
                    return 3;
                }
                case 16: {
                    if(bufferSize < 3) return 0;
                    buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
                    buffer[1] = [] { if constexpr (std::is_same_v<Char, char>) { return 'x'; } else { return L'x'; } } ();
                    buffer[2] = buffer[0];
                    return 3;
                }
                default: {
                    if(bufferSize < 1) return 0;
                    buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
                    return 1;
                }
            }
        }

        buffer[0] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ();
        return 1;
    }

    /**
     * @brief STD stream output operator
     * @param os Ostream
     * @param number Aeu
     * @return Ostream
     * @details Writes number in stream. Accepts STD streams based on char or wchar_t. Supports stream manipulators:
     * - Number's notation (std::hex, std::dec, std::oct);
     * - Number's base (std::showbase);
     * - Hexadecimal letters case (std::uppercase, std::lowercase)
     * @note Works significantly faster for hexadecimal notation
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& os, const Aesi& number) -> std::basic_ostream<Char>& {
        using enum Sign;
        if(number.sign != Zero) {
            if(number.sign == Negative)
                os << [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ();
            return os << number.base;
        }
        return os << '0';
    }
    /* ----------------------------------------------------------------------- */

#if defined __CUDACC__ || defined DOXYGEN_SKIP
    /**
     * @brief Atomicity-oriented object assignment operator
     * @param Aesi assigning
     * @note Method itself is not fully-atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for assigning encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicSet(const Aesi& value) noexcept -> void {
        base.tryAtomicSet(value.base);
        sign = value.sign; /* TODO: Make enum substitution using existing CUDA atomics */
    }

    /**
     * @brief Atomicity-oriented object exchangement operator
     * @param Aesi exchangeable
     * @note Method itself is not fully-atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for exchanging encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicExchange(const Aesi& value) noexcept -> void {
        base.tryAtomicExchange(value.base);
        sign = value.sign; /* TODO: Make enum substitution using existing CUDA atomics */
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
