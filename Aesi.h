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

    gpu constexpr Aesi(const Aesi& copy) noexcept {}

    gpu constexpr Aesi& operator=(const Aesi& other) noexcept {}

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept {}

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept : base(ptr, size) {}

    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&literal)[arrayLength]) noexcept : Aesi(literal, arrayLength) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()) {}

    explicit gpu constexpr Aesi(const Aeu<bitness>& value) : base(value), sign(Sign::Positive) {}

#ifdef AESI_CRYPTOPP_INTEGRATION
    constexpr Aesi(const CryptoPP::Integer& value) : base(value) { if(value.IsNegative()) sign = Sign::Positive; }
#endif

#ifdef AESI_GMP_INTEGRATION
    constexpr Aesi(const mpz_class& value) : base(value) {}
#endif
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /* ------------------------- @name Unary operators. -------------------------- */
        gpu constexpr auto operator+() const noexcept -> Aesi { return *this; }

        [[nodiscard]]
        gpu constexpr auto operator-() const noexcept -> Aesi {}

        gpu constexpr auto operator++() noexcept -> Aesi& {}

        gpu constexpr auto operator++(int) & noexcept -> Aesi {}

        gpu constexpr auto operator--() noexcept -> Aesi& {}

        gpu constexpr auto operator--(int) & noexcept -> Aesi {}
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Addition operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator+(Integral addendum) const noexcept -> Aesi {}

        [[nodiscard]]
        gpu constexpr auto operator+(const Aesi& addendum) const noexcept -> Aesi {}

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator+=(Integral addendum) noexcept -> Aesi& {}

        gpu constexpr auto operator+=(const Aesi& addendum) noexcept -> Aesi& {}
    /* --------------------------------------------------------------------------- */

    /* ----------------------- @name Subtraction operators. ---------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator-(Integral subtrahend) const noexcept -> Aesi {}

        [[nodiscard]]
        gpu constexpr auto operator-(const Aesi& subtrahend) const noexcept -> Aesi {}

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator-=(Integral subtrahend) noexcept -> Aesi& {}

        gpu constexpr auto operator-=(const Aesi& subtrahend) noexcept -> Aesi& {}
    /* --------------------------------------------------------------------------- */

    /* --------------------- @name Multiplication operators. --------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator*(Integral factor) const noexcept -> Aesi {}

        [[nodiscard]]
        gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi {}

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator*=(Integral factor) noexcept -> Aesi& {}

        gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& {}
    /* --------------------------------------------------------------------------- */

    /* ------------------------ @name Division operators. ------------------------ */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator/(Integral divisor) const noexcept -> Aesi {}

        [[nodiscard]]
        gpu constexpr auto operator/(const Aesi& divisor) const noexcept -> Aesi {}

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator/=(Integral factor) noexcept -> Aesi& {}

        gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& {}
    /* --------------------------------------------------------------------------- */

    /* ------------------------- @name Modulo operators. ------------------------- */
        template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
        gpu constexpr auto operator%(Integral modulo) const noexcept -> Aesi {}

        [[nodiscard]]
        gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi {}

        template <typename Integral> requires (std::is_integral_v<Integral>)
        gpu constexpr auto operator%=(Integral modulo) noexcept -> Aesi& {}

        gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {}
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */


    /* ----------------------- @name Bitwise operators. ---------------------- */
#define BitwiseDisabled "For bitwise operations use unsigned integer class"

    [[nodiscard]]
    gpu constexpr auto operator~() const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    [[nodiscard]]
    gpu constexpr auto operator^(const Aesi& other) const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    gpu constexpr auto operator^=(const Aesi& other) noexcept -> Aesi& { static_assert(false, BitwiseDisabled); }

    [[nodiscard]]
    gpu constexpr auto operator&(const Aesi& other) const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    gpu constexpr auto operator&=(const Aesi& other) noexcept -> Aesi& { static_assert(false, BitwiseDisabled); }

    [[nodiscard]]
    gpu constexpr auto operator|(const Aesi& other) const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    gpu constexpr auto operator|=(const Aesi& other) noexcept -> Aesi& { static_assert(false, BitwiseDisabled); }

    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator<<(Integral bitShift) const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator<<=(Integral bitShift) noexcept -> Aesi& { static_assert(false, BitwiseDisabled); }

    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator>>(Integral bitShift) const noexcept -> Aesi { static_assert(false, BitwiseDisabled); }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator>>=(Integral bitShift) noexcept -> Aesi& { static_assert(false, BitwiseDisabled); }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Comparison operators. --------------------- */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator==(Integral value) const noexcept -> bool {
        if(value < 0)
            return sign == Sign::Negative && base == static_cast<uint64_t>(abs(static_cast<long long>(value)));
        else if(value > 0)
            return sign == Sign::Positive && base == static_cast<uint64_t>(value);
        else return sign == Sign::Zero;
    }

    gpu constexpr auto operator==(const Aesi& other) const noexcept -> bool { return sign == other.sign && base == other.base; };

    [[nodiscard]]
    gpu constexpr auto compareTo(const Aesi& other) const noexcept -> Comparison { return Comparison::equal; };
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
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second, Aesi& bezoutX, Aesi& bezoutY) noexcept -> Aesi {}

    [[nodiscard]]
    gpu static constexpr auto lcm(const Aesi& first, const Aesi& second) noexcept -> Aesi {}

    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aesi {}

    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {}
    /* ----------------------------------------------------------------------- */

    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {}

    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> {}

    gpu constexpr auto unsignedCast() const noexcept -> Aeu<bitness> { return base; };

    /* ----------------- @name Public input-output operators. ---------------- */
    template <byte notation, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (notation == 2 || notation == 8 || notation == 10 || notation == 16))
    gpu constexpr auto getString(Char* const buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t { return 0; }

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) -> std::basic_ostream<Char>& {}
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
