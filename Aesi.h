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
    gpu constexpr Aesi() noexcept = default;

    gpu constexpr Aesi(const Aesi& copy) noexcept = default;

    gpu constexpr Aesi& operator=(const Aesi& other) noexcept { base = other.base; sign = other.sign; return *this; }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept { /* TODO: Complete. */ }

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept : base(ptr, size) { /* TODO: Complete. */ }

    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&literal)[arrayLength]) noexcept : Aesi(literal, arrayLength) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(const String& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    explicit gpu constexpr Aesi(const Aeu<bitness>& value) : base(value), sign(Sign::Positive) {}

#ifdef AESI_CRYPTOPP_INTEGRATION
    constexpr Aesi(const CryptoPP::Integer& value) : base(value) { if(value.IsNegative()) sign = Sign::Positive; }
#endif

#ifdef AESI_GMP_INTEGRATION
    constexpr Aesi(const mpz_class& value) : base(value) { if(value < 0) sign = Sign::Positive; }
#endif
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /* ------------------------- @name Unary operators. -------------------------- */
        gpu constexpr auto operator+() const noexcept -> Aesi { return *this; }

        [[nodiscard]]
        gpu constexpr auto operator-() const noexcept -> Aesi { return {}; }

        gpu constexpr auto operator++() noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator++(int) & noexcept -> Aesi { return {}; }

        gpu constexpr auto operator--() noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator--(int) & noexcept -> Aesi { return {}; }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Addition operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator+(Integral addendum) const noexcept -> Aesi { return {}; }

        [[nodiscard]]
        gpu constexpr auto operator+(const Aesi& addendum) const noexcept -> Aesi { return {}; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator+=(Integral addendum) noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator+=(const Aesi& addendum) noexcept -> Aesi& { return *this; }
    /* --------------------------------------------------------------------------- */

    /* ----------------------- @name Subtraction operators. ---------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator-(Integral subtrahend) const noexcept -> Aesi { return *this; }

        [[nodiscard]]
        gpu constexpr auto operator-(const Aesi& subtrahend) const noexcept -> Aesi { return *this; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator-=(Integral subtrahend) noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator-=(const Aesi& subtrahend) noexcept -> Aesi& { return *this; }
    /* --------------------------------------------------------------------------- */

    /* --------------------- @name Multiplication operators. --------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator*(Integral factor) const noexcept -> Aesi { return {}; }

        [[nodiscard]]
        gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi { return {}; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator*=(Integral factor) noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& { return *this; }
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Division operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator/(Integral divisor) const noexcept -> Aesi { return {}; }

        [[nodiscard]]
        gpu constexpr auto operator/(const Aesi& divisor) const noexcept -> Aesi { return {}; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator/=(Integral factor) noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& { return *this; }
    /* --------------------------------------------------------------------------- */

    /* ------------------------- @name Modulo operators. ------------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator%(Integral modulo) const noexcept -> Aesi { return {}; }

        [[nodiscard]]
        gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi { return {}; }

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator%=(Integral modulo) noexcept -> Aesi& { return *this; }

        gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& { return *this; }
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */

    /* --------------------- @name Comparison operators. --------------------- */
    /* ------------------------ @name Equality operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator==(Integral value) const noexcept -> bool { return compareTo(value) == Comparison::equal; }

        gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool { return sign == other.sign && base == other.base; }

        template <std::size_t otherBitness> requires (otherBitness != bitness)
        gpu constexpr auto operator==(const Aesi<otherBitness>& other) const noexcept -> bool { return compareTo(other) == Comparison::equal; }
    /* --------------------------------------------------------------------------- */


    /* ----------------------- @name Comparison operators. ----------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto compareTo(Integral value) const noexcept -> Comparison { return Comparison::equal; }

        template <std::size_t otherBitness = bitness> [[nodiscard]]
        gpu constexpr auto compareTo(const Aesi<otherBitness>& other) const noexcept -> Comparison { return Comparison::equal; };
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Spaceship operators. ----------------------- */
#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (PRE_CPP_20)) && !defined DOXYGEN_SKIP
        /**
         * @brief Oldstyle comparison operator(s). Used inside CUDA cause it does not support <=> on preCpp20
         */
        gpu constexpr auto operator!=(const Aeu& value) const noexcept -> bool { return !this->operator==(value); }
        gpu constexpr auto operator<(const Aeu& value) const noexcept -> bool { return this->compareTo(value) == Comparison::less; }
        gpu constexpr auto operator<=(const Aeu& value) const noexcept -> bool { return !this->operator>(value); }
        gpu constexpr auto operator>(const Aeu& value) const noexcept -> bool { return this->compareTo(value) == Comparison::greater; }
        gpu constexpr auto operator>=(const Aeu& value) const noexcept -> bool { return !this->operator<(value); }
#else
        /**
         * @brief Three-way comparison operator
         * @param Aeu other
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

        /**
         * @brief Three-way comparison operator for numbers of different precision and built-in integral types
         * @param Aeu<diffPrec>/Unsigned other
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


    /* ---------------------- @name Supporting methods. ---------------------- */
    gpu constexpr auto setBit(std::size_t index, bool bit) noexcept -> void {}

    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool { return base.getBit(index); }

    gpu constexpr auto setByte(std::size_t index, byte byte) noexcept -> void {}

    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> byte { return base.getByte(index); }

    gpu constexpr auto setBlock(std::size_t index, block block) noexcept -> void {}

    [[nodiscard]]
    gpu constexpr auto getBlock(std::size_t index) const noexcept -> block { return base.getBlock(index); }

    [[nodiscard]]
    gpu constexpr auto byteCount() const noexcept -> std::size_t { return base.byteCount(); }

    [[nodiscard]]
    gpu constexpr auto bitCount() const noexcept -> std::size_t { return base.bitCount(); }

    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return base.isOdd(); }

    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return base.isEven(); }

    [[nodiscard]]
    gpu constexpr auto isZero() const noexcept -> bool { return sign == Sign::Zero; }

    [[nodiscard]]
    gpu static constexpr auto getBitness() noexcept -> std::size_t { return bitness; }

    [[nodiscard]]
    gpu static constexpr auto getBlocksNumber() noexcept -> std::size_t { return Aeu<bitness>::totalBlocksNumber(); }

    gpu constexpr auto swap(Aesi& other) noexcept -> void {}

    gpu constexpr auto inverse() noexcept -> void {}
    /* ----------------------------------------------------------------------- */


    /* -------------- @name Public arithmetic and number theory. ------------- */
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor, Aesi& quotient, Aesi& remainder) noexcept -> void {}

    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi { return Aesi {}; }

    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t power) noexcept -> Aesi { Aesi result { Sign::Positive, Aeu<bitness>::power2(power) }; return result; }
    /* ----------------------------------------------------------------------- */

    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral { return Integral(); }

    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> { return {}; }

    gpu constexpr auto unsignedCast() const noexcept -> Aeu<bitness> { return base; };

    /* ----------------- @name Public input-output operators. ---------------- */
    template <byte notation, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (notation == 2 || notation == 8 || notation == 10 || notation == 16))
    gpu constexpr auto getString(Char* const buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t { return 0; }

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) -> std::basic_ostream<Char>& { return ss; }
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
