#ifndef AESI_MULTIPRECISION
#define AESI_MULTIPRECISION

/// @cond HIDE_INCLUDES
#include <iostream>
#include <array>

#ifdef __CUDACC__
    #define gpu __host__ __device__
    #include <thrust/pair.h>
#else
    #define gpu
    #include <utility>
#endif
/// @endcond

/**
 * @file AesiMultiprecision.h
 * @brief Long precision integer with arithmetic operations
 */

namespace {
    using byte = uint8_t;
    using block = uint32_t;

    constexpr std::size_t bitsInByte = 8, blockBitLength = sizeof(block) * bitsInByte;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;

    /**
     * @enum AesiCMP
     * @brief Analogue of STD::Strong_ordering since CUDA does not support <=> operator
     */
    enum class AesiCMP { equal = 0, less = 1, greater = 2, equivalent = 3 };
}

/**
 * @class Aesi
 * @brief Long precision integer with arithmetic operations
 * @details May be used to represent positive and negative integers. Number precision is set in template parameter bitness.
 */
template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aesi final {
    static_assert(bitness > sizeof(uint64_t), "Use built-in types for numbers 64-bit or less.");

    static constexpr std::size_t blocksNumber = bitness / blockBitLength;

#ifdef __CUDACC__
    template <typename T1, typename T2>
    using pair = thrust::pair<T1, T2>;
#else
    template <typename T1, typename T2>
    using pair = std::pair<T1, T2>;
#endif

    /* -------------------------- @name Class members. ----------------------- */
    /**
     * @brief Block line of the number
     */
    using blockLine = std::array<block, blocksNumber>;
    blockLine blocks;

    /**
     * @enum Aesi::Sign
     * @brief Specifies sign of the number. Should be Positive, Negative or Zero
     */
    enum Sign { Zero = 0, Positive = 1, Negative = 2 } sign;
    /* ----------------------------------------------------------------------- */


    /* ------------------------ @name Helper functions. ---------------------- */
    /**
     * @brief Makes line addition
     * @param BlockLine dst
     * @param BlockLine src
     * @return Uint64 - carry out from addition
     */
    gpu static constexpr auto addLine(blockLine& dst, const blockLine& src) noexcept -> uint64_t {
        uint64_t carryOut = 0;
        for (std::size_t i = 0; i < blocksNumber; ++i) {
            uint64_t sum = static_cast<uint64_t>(dst[i])
                                     + static_cast<uint64_t>(src[i]) + carryOut;
            carryOut = sum / blockBase;
            dst[i] = sum % blockBase;
        }
        return carryOut;
    }

    /**
     * @brief Makes complement block line for line
     * @param BlockLine line
     * @return BlockLine
     */
    [[nodiscard]]
    gpu static constexpr auto makeComplement(const blockLine& line) noexcept -> blockLine {
        blockLine result {};

        uint64_t carryOut = 1;
        for(std::size_t i = 0; i < blocksNumber; ++i) {
            const uint64_t sum = blockBase - 1ULL - static_cast<uint64_t>(line[i]) + carryOut;
            carryOut = sum / blockBase; result[i] = sum % blockBase;
        }

        return result;
    }

    /**
     * @brief Checks if block line is empty
     * @param BlockLine line
     * @return Bool
     */
    gpu static constexpr auto isLineEmpty(const blockLine& line) noexcept -> bool {
        return lineLength(line) == 0;
    }

    /**
     * @brief Counts the number of non-zero blocks inside block line starting from right
     * @param BlockLine line
     * @return Size_t
     */
    gpu static constexpr auto lineLength(const blockLine& line) noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(line[i]) return i + 1;
        return 0;
    }
    /* ----------------------------------------------------------------------- */

public:
    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aesi() noexcept : blocks{}, sign { Zero } {};

    /**
     * @brief Copy constructor
     */
    gpu constexpr Aesi(const Aesi& copy) noexcept {
        sign = copy.sign; if(copy.sign != Zero) blocks = copy.blocks;
    };

    /**
     * @brief Copy assignment operator
     */
    gpu constexpr Aesi& operator=(const Aesi& other) noexcept {
        blocks = other.blocks; sign = other.sign; return *this;
    }

    /**
     * @brief Integral constructor
     * @param value Integral
     * @details Accepts each integral built-in type signed and unsigned
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi(Integral value) noexcept {
        if(value != 0) {
            uint64_t tValue;
            if (value < 0) {
                sign = Negative;
                tValue = static_cast<uint64_t>(value * -1);
            } else {
                sign = Positive;
                tValue = static_cast<uint64_t>(value);
            }

            for (std::size_t i = 0; i < blocksNumber; ++i) {
                blocks[i] = static_cast<block>(tValue % blockBase);
                tValue /= blockBase;
            }
        } else {
            for (std::size_t i = 0; i < blocksNumber; ++i)
                blocks[i] = 0;
            sign = Zero;
        }
    }

    /**
     * @brief Pointer-based char constructor
     * @param Char* pointer
     * @param Size_t size
     * @details Accepts decimal literals along with binary (starting with 0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept : Aesi() {
        if(size == 0) return;

        constexpr const Char* characters = [] {
            if constexpr (std::is_same_v<char, Char>) {
                return "-09aAfFoObBxX";
            } else {
                return L"-09aAfFoObBxX";
            }
        } ();

        std::size_t position = 0;

        bool negative = false;
        if(ptr[position] == characters[0]) {
            negative = true; ++position;
        }

        const auto base = [&ptr, &size, &position, &characters] {
            if (ptr[position] == characters[1] && size > position + 1) {
                switch (ptr[position + 1]) {
                    case characters[9]:
                    case characters[10]:
                        position += 2; return 2;
                    case characters[7]:
                    case characters[8]:
                        position += 2; return 8;
                    case characters[11]:
                    case characters[12]:
                        position += 2; return 16;
                    default:
                        return 10;
                }
            } else return 10;
        } ();
        for(; position < size; ++position) {
            const auto digit = [&characters] (Char ch) {
                if(characters[1] <= ch && ch <= characters[2])
                    return static_cast<int>(ch) - static_cast<int>(characters[1]);
                if(characters[3] <= ch && ch <= characters[5])
                    return static_cast<int>(ch) - static_cast<int>(characters[3]) + 10;
                if(characters[4] <= ch && ch <= characters[6])
                    return static_cast<int>(ch) - static_cast<int>(characters[4]) + 10;
                return 99;
            } (ptr[position]);

            if(digit < base) {
                this->operator*=(base);
                this->operator+=(digit);
            }
        }

        if(negative) sign = Negative;
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
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()){}

    /**
     * @brief Different precision copy constructor
     * @param Aesi<nPrecision> copy
     * @details Constructs object based on different precision object's block line. Same as precision cast operator
     */
    template<std::size_t rBitness> requires (rBitness != bitness)
    gpu constexpr Aesi(const Aesi<rBitness>& copy) noexcept {
        this->operator=(copy.template precisionCast<bitness>());
    }
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
        if(sign == Zero) return Aesi {};
        Aesi result = *this;
        result.sign = (result.sign == Positive ? Negative : Positive); return result;
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
        Aesi old = *this; operator++(); return old;
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
        Aesi old = *this; operator--(); return old;
    }

    /**
     * @brief Addition operator
     * @param Aesi addendum
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator+(const Aesi& addendum) const noexcept -> Aesi {
        Aesi result = *this; result += addendum; return result;
    }

    /**
     * @brief Assignment addition operator
     * @param Aesi addendum
     * @return Aesi&
     */
    gpu constexpr auto operator+=(const Aesi& addendum) noexcept -> Aesi& {
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
    gpu constexpr auto operator-(const Aesi& subtrahend) const noexcept -> Aesi {
        Aesi result = *this; result -= subtrahend; return result;
    }

    /**
     * @brief Assignment subtraction operator
     * @param Aesi subtrahend
     * @return Aesi&
     */
    gpu constexpr auto operator-=(const Aesi& subtrahend) noexcept -> Aesi& {
        return this->operator+=(-subtrahend);
    }

    /**
     * @brief Multiplication operator
     * @param Aesi factor
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator*(const Aesi& factor) const noexcept -> Aesi {
        Aesi result = *this; result *= factor; return result;
    }

    /**
     * @brief Assignment multiplication operator
     * @param Aesi factor
     * @return Aesi&
     */
    gpu constexpr auto operator*=(const Aesi& factor) noexcept -> Aesi& {
        if(sign == Zero) return *this;
        if(factor.sign == Zero)
            return this->operator=(Aesi {});
        sign = (sign != factor.sign ? Negative : Positive);

        constexpr auto multiplyLines = [] (const blockLine& longerLine, const std::size_t longerLength,
                const blockLine& smallerLine, const std::size_t smallerLength) {
            blockLine buffer {};

            for(std::size_t i = 0; i < longerLength; ++i) {
                uint64_t tBlock = longerLine[i], carryOut = 0;
                for(std::size_t j = 0; j < smallerLength && i + j < buffer.size(); ++j) {
                    const auto product = tBlock * static_cast<uint64_t>(smallerLine[j]) + carryOut;
                    const auto block = static_cast<uint64_t>(buffer[i + j]) + (product % blockBase);
                    carryOut = product / blockBase + block / blockBase;
                    buffer[i + j] = block % blockBase;
                }
                if(smallerLength < blocksNumber)
                    buffer[smallerLength + i] += carryOut;
            }

            return buffer;
        };

        const std::size_t thisLength = lineLength(blocks), valueLength = lineLength(factor.blocks);
        if(thisLength > valueLength)
            blocks = multiplyLines(blocks, thisLength, factor.blocks, valueLength);
        else
            blocks = multiplyLines(factor.blocks, valueLength, blocks, thisLength);

        return *this;
    }

    /**
     * @brief Division operator
     * @param Aesi divisor
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator/(const Aesi& divisor) const noexcept -> Aesi {
        Aesi quotient, _; divide(*this, divisor, quotient, _); return quotient;
    }

    /**
     * @brief Assignment division operator
     * @param Aesi divisor
     * @return Aesi&
     */
    gpu constexpr auto operator/=(const Aesi& divisor) noexcept -> Aesi& {
        return this->operator=(divide(*this, divisor).first);
    }

    /**
     * @brief Modulo operator
     * @param Aesi modulo
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto operator%(const Aesi& modulo) const noexcept -> Aesi {
        Aesi _, remainder; divide(*this, modulo, _, remainder); return remainder;
    }

    /**
     * @brief Assignment modulo operator
     * @param Aesi modulo
     * @return Aesi&
     */
    gpu constexpr auto operator%=(const Aesi& modulo) noexcept -> Aesi& {
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
    gpu constexpr auto operator~() const noexcept -> Aesi {
        Aesi result {};
        for(std::size_t i = 0; i < blocksNumber; ++i)
            result.blocks[i] = ~blocks[i];
        if(isLineEmpty(result.blocks))
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
    gpu constexpr auto operator^(const Aesi& other) const noexcept -> Aesi {
        Aesi result = *this; result ^= other; return result;
    }

    /**
     * @brief Assignment bitwise XOR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator^=(const Aesi& other) noexcept -> Aesi& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] ^= other.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
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
        Aesi result = *this; result &= other; return result;
    }

    /**
     * @brief Assignment bitwise AND operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator&=(const Aesi& other) noexcept -> Aesi& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] &= other.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
        return *this;
    }

    /**
     * @brief Bitwise OR operator
     * @param Aesi other
     * @return Aesi
     * @note Does not affect the sign
     */
    [[nodiscard]]
    gpu constexpr auto operator|(const Aesi& other) const noexcept -> Aesi {
        Aesi result = *this; result |= other; return result;
    }

    /**
     * @brief Assignment bitwise OR operator
     * @param Aesi other
     * @return Aesi&
     * @note Does not affect the sign
     */
    gpu constexpr auto operator|=(const Aesi& other) noexcept -> Aesi& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] |= other.blocks[i];
        if(sign == Zero && !isLineEmpty(blocks)) sign = Positive;
        return *this;
    }

    /**
     * @brief Left shift operator
     * @param Integral bit_shift
     * @return Aesi
     * @note Does right shift (>>) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto operator<<(Integral bitShift) const noexcept -> Aesi {
        Aesi result = *this; result <<= bitShift; return result;
    }

    /**
     * @brief Left shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does right shift (>>=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator<<=(Integral bitShift) noexcept -> Aesi& {
        if(bitShift < 0)
            return this->operator>>=(-bitShift);

        if(bitShift < bitness && bitShift > 0) {
            const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
            const block stamp = (1UL << (blockBitLength - remainder)) - 1;

            for (long long i = blocksNumber - 1; i >= (quotient + (remainder ? 1 : 0)); --i)
                blocks[i] = ((blocks[i - quotient] & stamp) << remainder) | ((blocks[i - quotient - (remainder ? 1 : 0)] & ~stamp) >> (blockBitLength - remainder));

            blocks[quotient] = (blocks[0] & stamp) << remainder;

            for (std::size_t i = 0; i < quotient; ++i)
                blocks[i] = 0;

            if(isLineEmpty(blocks)) sign = Zero;
        }

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
        Aesi result = *this; result >>= bitShift; return result;
    }

    /**
     * @brief Right shift assignment operator
     * @param Integral bit_shift
     * @return Aesi&
     * @note Does left shift (<<=) for negative bit_shift value, and nothing for positive shift greater than precision
     */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr auto operator>>=(Integral bitShift) noexcept -> Aesi& {
        if(bitShift < 0)
            return this->operator<<=(-bitShift);

        if(bitShift < bitness && bitShift > 0) {
            const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
            const block stamp = (1UL << remainder) - 1;

            for(std::size_t i = 0; i < blocksNumber - (quotient + (remainder ? 1 : 0)); ++i)
                blocks[i] = ((blocks[i + quotient + (remainder ? 1 : 0)] & stamp) << (blockBitLength - remainder)) | ((blocks[i + quotient] & ~stamp) >> remainder);

            blocks[blocksNumber - 1 - quotient] = (blocks[blocksNumber - 1] & ~stamp) >> remainder;

            for(long long i = blocksNumber - quotient; i < blocksNumber; ++i)
                blocks[i] = 0;

            if(isLineEmpty(blocks)) sign = Zero;
        }

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
        if(sign != Zero || other.sign != Zero)
            return (sign == other.sign && blocks == other.blocks); else return true;
    };

    /**
     * @brief Internal comparison operator
     * @param Aesi other
     * @return AesiCMP
     * @note Should almost never return AesiCMP::Equivalent
     */
    [[nodiscard]]
    gpu constexpr auto compareTo(const Aesi& other) const noexcept -> AesiCMP {
        switch (sign) {
            case Zero:
                switch (other.sign) {
                    case Zero: return AesiCMP::equal;
                    case Positive: return AesiCMP::less;
                    case Negative: return AesiCMP::greater;
                    default: return AesiCMP::equivalent;
                }
            case Positive:
                switch (other.sign) {
                    case Positive: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(other.blocks);
                        if(thisLength != valueLength)
                            return (thisLength > valueLength) ? AesiCMP::greater : AesiCMP::less;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != other.blocks[i])
                                return (blocks[i] > other.blocks[i]) ? AesiCMP::greater : AesiCMP::less;

                        return AesiCMP::equal;
                    }
                    case Zero:
                    case Negative: return AesiCMP::greater;
                    default: return AesiCMP::equivalent;
                }
            case Negative:
                switch (other.sign) {
                    case Negative: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(other.blocks);
                        if(thisLength != valueLength)
                            return (static_cast<long long>(thisLength) * -1 > static_cast<long long>(valueLength) * -1) ? AesiCMP::greater : AesiCMP::less;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != other.blocks[i])
                                return (static_cast<long>(blocks[i]) * -1 > static_cast<long>(other.blocks[i]) * -1) ? AesiCMP::greater : AesiCMP::less;

                        return AesiCMP::equal;
                    }
                    case Zero:
                    case Positive: return AesiCMP::less;
                    default: return AesiCMP::equivalent;
                }
            default: return AesiCMP::equivalent;
        }
    };

#if (defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)) && !defined DOXYGEN_SHOULD_SKIP_THIS
    /**
     * @brief Oldstyle comparison operator(s). Used inside CUDA cause it does not support <=> on device
     */
    gpu constexpr auto operator!=(const Aesi& value) const noexcept -> bool { return !this->operator==(value); }
    gpu constexpr auto operator<(const Aesi& value) const noexcept -> bool { return this->compareTo(value) == AesiCMP::less; }
    gpu constexpr auto operator<=(const Aesi& value) const noexcept -> bool { return !this->operator>(value); }
    gpu constexpr auto operator>(const Aesi& value) const noexcept -> bool { return this->compareTo(value) == AesiCMP::greater; }
    gpu constexpr auto operator>=(const Aesi& value) const noexcept -> bool { return !this->operator<(value); }
#else
    /**
     * @brief Three-way comparison operator
     * @param Aesi other
     * @return Std::Strong_ordering
     * @note Should almost never return Strong_ordering::Equivalent
     */
    gpu constexpr auto operator<=>(const Aesi& other) const noexcept -> std::strong_ordering {
        switch(this->compareTo(other)) {
            case AesiCMP::less: return std::strong_ordering::less;
            case AesiCMP::greater: return std::strong_ordering::greater;
            case AesiCMP::equal: return std::strong_ordering::equal;
            default: return std::strong_ordering::equivalent;
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
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        if(bit) {
            blocks[blockNumber] |= (1U << bitNumber);
            if(sign == Zero && !isLineEmpty(blocks))
                sign = Positive;
        } else {
            blocks[blockNumber] &= (~(1U << bitNumber));
            if(sign != Zero && isLineEmpty(blocks))
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
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool {
        if(index >= bitness) return false;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        return blocks[blockNumber] & (1U << bitNumber);
    }

    /**
     * @brief Set byte in number by index starting from the right
     * @param Size_t index
     * @param Byte byte
     * @note Does nothing for index out of range
     */
    gpu constexpr auto setByte(std::size_t index, byte byte) noexcept -> void {
        if(index > blocksNumber * sizeof(block)) return;

        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        blocks[blockNumber] &= ~(0xffU << shift); blocks[blockNumber] |= static_cast<block>(byte) << shift;

        if(sign != Zero && isLineEmpty(blocks)) sign = Zero;
        if(sign == Zero && !isLineEmpty(blocks)) sign = Positive;
    }

    /**
     * @brief Get byte in number by index starting from the right
     * @param Size_t index
     * @return Byte
     * @note Returns zero for index out of range
     */
    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> byte {
        if(index > blocksNumber * sizeof(block)) return 0;
        
        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        return (blocks[blockNumber] & (0xffU << shift)) >> shift;
    }

    /**
     * @brief Get amount of non-empty bytes in number right to left
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto byteCount() const noexcept -> std::size_t {
        std::size_t lastBlock = blocksNumber - 1;
        for(; lastBlock > 0 && blocks[lastBlock] == 0; --lastBlock);

        for(int8_t byteN = sizeof(block) - 1; byteN >= 0; --byteN) {
            const auto byte = (blocks[lastBlock] & (0xffU << (byteN * bitsInByte))) >> (byteN * bitsInByte);
            if(byte)
                return lastBlock * sizeof(block) + byteN + 1;
        }
        return lastBlock * sizeof(block);
    }

    /**
     * @brief Get amount of non-empty bits in number right to left
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto bitCount() const noexcept -> std::size_t {
        std::size_t lastBlock = blocksNumber - 1;
        for(; lastBlock > 0 && blocks[lastBlock] == 0; --lastBlock);

        for(int8_t byteN = sizeof(block) - 1; byteN >= 0; --byteN) {
            const auto byte = (blocks[lastBlock] & (0xffU << (byteN * bitsInByte))) >> (byteN * bitsInByte);
            if(!byte) continue;

            for(int8_t bitN = bitsInByte - 1; bitN >= 0; --bitN) {
                const auto bit = (byte & (0x1u << bitN)) >> bitN;
                if(bit)
                    return (lastBlock * sizeof(block) + byteN) * bitsInByte + bitN + 1;
            }
            return ((lastBlock - 1) * sizeof(block) + byteN) * bitsInByte;
        }
        return lastBlock * sizeof(block);
    }

    /**
     * @brief Get number's absolute value
     * @return Aesi
     */
    [[nodiscard]]
    gpu constexpr auto abs() const noexcept -> Aesi {
        if(sign == Zero)
            return *this;
        Aesi result = *this; result.sign = Positive; return result;
    }

    /**
     * @brief Check whether number is odd
     * @return Bool - true is number is odd and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return (0x1 & blocks[0]) == 1; }

    /**
     * @brief Check whether number is even
     * @return Bool - true is number is even and false otherwise
     */
    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return (0x1 & blocks[0]) == 0; }

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
    gpu constexpr auto getBitness() const noexcept -> std::size_t { return bitness; }

    /**
     * @brief Get square root
     * @return Aesi
     * @note Returns zero for negative value or zero
     */
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {
        if(sign != Positive)
            return Aesi {};

        Aesi x, y = power2((bitCount() + 1) / 2);

        do {
            x = y;
            y = (x + this->operator/(x)) >> 1;
        } while (y < x);

        return x;
    }

    /**
     * @brief Make swap between two objects
     * @param Aesi other
     */
    gpu constexpr auto swap(Aesi& other) noexcept -> void {
        Aesi t = other; other.operator=(*this); this->operator=(t);
    }
    /* ----------------------------------------------------------------------- */


    /* -------------- @name Public arithmetic and number theory. ------------- */
    /**
     * @brief Integer division
     * @param Aesi number
     * @param Aesi divisor
     * @param Aesi quotient OUT
     * @param Aesi remainder OUT
     * @return Quotient and remainder by reference
     */
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor, Aesi& quotient, Aesi& remainder) noexcept -> void {
        const Aesi divAbs = divisor.abs();
        const auto ratio = number.abs().compareTo(divAbs);

        if(!quotient.isZero()) quotient = Aesi {};
        if(!remainder.isZero()) remainder = Aesi {};

        if(ratio == AesiCMP::greater) {
            const auto bitsUsed = lineLength(number.blocks) * blockBitLength;
            for(long long i = bitsUsed - 1; i >= 0; --i) {
                remainder <<= 1;
                remainder.setBit(0, number.getBit(i));

                if(remainder >= divAbs) {
                    remainder -= divAbs;
                    quotient.setBit(i, true);
                }
            }

            if(isLineEmpty(quotient.blocks))
                quotient.sign = Zero; else if(number.sign != divisor.sign) quotient = -quotient;
            if(isLineEmpty(remainder.blocks))
                remainder.sign = Zero; else if(number.sign == Negative) remainder = -remainder;
        } else if(ratio == AesiCMP::less)
            remainder = number; else quotient = 1;
    }

    /**
     * @brief Integer division.
     * @param Aesi number
     * @param Aesi divisor
     * @return Pair(Quotient, Remainder)
     */
    [[nodiscard]]
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor) noexcept -> pair<Aesi, Aesi> {
        pair<Aesi, Aesi> results = { 0, 0 }; divide(number, divisor, results.first, results.second); return results;
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
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second, Aesi& bezoutX, Aesi& bezoutY) noexcept -> Aesi {
        Aesi gcd, tGcd, quotient, remainder;

        const auto ratio = first.compareTo(second);
        if(ratio == AesiCMP::greater) {
            gcd = second; tGcd = first;
            divide(first, second, quotient, remainder);
        } else {
            gcd = first; tGcd = second;
            divide(second, first, quotient, remainder);
        }

        bezoutX = 0; bezoutY = 1;
        for(Aesi tX = 1, tY = 0; remainder != 0; ) {
            tGcd = gcd; gcd = remainder;

            Aesi t = bezoutX; bezoutX = tX - quotient * bezoutX; tX = t;
            t = bezoutY; bezoutY = tY - quotient * bezoutY; tY = t;

            divide(tGcd, gcd, quotient, remainder);
        }

        if(ratio != AesiCMP::greater)
            bezoutX.swap(bezoutY);

        return gcd;
    }

    /**
     * @brief Greatest common divisor
     * @param Aesi first
     * @param Aesi second
     * @return Aesi
     */
    [[nodiscard]]
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        Aesi bezoutX, bezoutY; return gcd(first, second, bezoutX, bezoutY);
    }

    /**
     * @brief Least common multiplier
     * @param Aesi first
     * @param Aesi second
     * @return Aesi
     */
    [[nodiscard]]
    gpu static constexpr auto lcm(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        Aesi bezoutX, bezoutY; return first / gcd(first, second, bezoutX, bezoutY) * second;
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
    gpu static constexpr auto powm(const Aesi& base, const Aesi& power, const Aesi& mod) noexcept -> Aesi {
        constexpr auto remainingBlocksEmpty = [] (const Aesi& value, std::size_t offset) {
            for(std::size_t i = offset / blockBitLength; i < value.blocksNumber; ++i)
                if (value.blocks[i] != 0) return false;
            return true;
        };

        Aesi result = 1;
        auto [_, b] = divide(base, mod);

        for(unsigned iteration = 0; !remainingBlocksEmpty(power, iteration); iteration++) {
            if(power.getBit(iteration)) {
                const auto [quotient, remainder] = divide(result * b, mod);
                result = remainder;
            }

            const auto [quotient, remainder] = divide(b * b, mod);
            b = remainder;
        }

        return result;
    }

    /**
     * @brief Fast exponentiation for 2
     * @param Size_t power
     * @return Aesi
     * @details Returns zero for power greater than current bitness
     */
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aesi {
        Aesi result {}; result.setBit(e, true); return result;
    }
    /* ----------------------------------------------------------------------- */


    /**
     * @brief Cast from Aesi to built-in integral types
     * @param Type integral_type TEMPLATE
     * @return Integral
     * @details Takes the lowes part of Aesi for conversion. Accepts signed and unsigned types
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        const uint64_t value = (static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0]);
        if constexpr (std::is_signed_v<Integral>)
            return static_cast<Integral>(value) * (sign == Negative ? -1 : 1); else return static_cast<Integral>(value);
    }

    /**
     * @brief Precision cast for Aesi numbers of different length
     * @param Size_t new_bitness TEMPLATE
     * @return Aesi<new_bitness>
     * @details If required precision greater than current precision, remaining blocks are filled with zeros.
     * Otherwise - number is cropped inside smaller blocks array
     * @note This method is used in all manipulations between numbers of different precision. Using this method is not recommended,
     * cause it leads to redundant copying and may be slow
     */
    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aesi<newBitness> {
        Aesi<newBitness> result = 0;

        long long startBlock = (blocksNumber < (newBitness / blockBitLength) ? blocksNumber - 1 : (newBitness / blockBitLength) - 1);
        for(; startBlock >= 0; --startBlock) {
            result <<= blockBitLength;
            result |= blocks[startBlock];
        }

        if(sign == Negative) result *= -1;
        return result;
    }

    /**
     * @brief Print number inside C-style array buffer
     * @param Byte base TEMPLATE
     * @param Char* buffer
     * @param Size_t buffer_size
     * @param Bool show_number_base
     * @param Bool use_hexadecimal_uppercase
     * @return Size_t - amount of symbols written
     * @details Places the maximum possible amount of number's characters in buffer. Base parameter should be 2, 8, 10,
     * or 16 and should be known at compile time
     * @note Works significantly faster for hexadecimal notation
     */
    template <byte base, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> &&
            (base == 2 || base == 8 || base == 10 || base == 16))
    gpu constexpr auto getString(Char* const buffer, std::size_t bufferSize, bool showBase = false, bool hexUppercase = false) const noexcept -> std::size_t {
        if(bufferSize < 2) return 0;

        std::size_t position = 0;

        if (showBase && bufferSize > 3) {
            if constexpr (base == 2) {
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0b", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0b", 2 * sizeof(Char));
                }
                position += 2;
            } else if constexpr (base == 8) {
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0o", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0o", 2 * sizeof(Char));
                }
                position += 2;
            } else if constexpr (base == 16) {
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0x", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0x", 2 * sizeof(Char));
                }
                position += 2;
            }
        }


        if(sign != Zero) {
            if constexpr (base == 16) {
                long long iter = blocks.size() - 1;
                for (; blocks[iter] == 0 && iter >= 0; --iter);

                if constexpr (std::is_same_v<Char, char>) {
                    position += snprintf(buffer + position, bufferSize - position, (hexUppercase ? "%X" : "%x"), blocks[iter--]);
                    for (; iter >= 0; --iter)
                        position += snprintf(buffer + position, bufferSize - position, (hexUppercase ? "%08X" : "%08x"), blocks[iter]);
                } else {
                    position += swprintf(buffer + position, bufferSize - position, (hexUppercase ? L"%X" : L"%x"), blocks[iter--]);
                    for (; iter >= 0; --iter)
                        position += swprintf(buffer + position, bufferSize - position, (hexUppercase ? L"%08X" : L"%08x"), blocks[iter]);
                }
            } else {
                const auto startPosition = position;

                Aesi copy = *this;
                while (copy != 0 && position < bufferSize) {
                    auto [quotient, remainder] = divide(copy, base);
                    if constexpr (std::is_same_v<Char, char>) {
                        buffer[position++] = '0' + remainder.template integralCast<byte>();
                    } else {
                        buffer[position++] = L'0' + remainder.template integralCast<byte>();
                    }
                    copy = quotient;
                }

                for (std::size_t i = startPosition; i * 2 < position; ++i) {
                    Char t = buffer[i]; buffer[i] = buffer[position - 1 - i]; buffer[position - 1 - i] = t;
                }
            }
        } else
            if constexpr (std::is_same_v<Char, char>) {
                buffer[position++] = '0';
            } else {
                buffer[position++] = L'0';
            }
        buffer[position++] = Char {};
        return position;
    }

    /**
     * @brief Print number inside STD stream
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
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) noexcept -> std::basic_ostream<Char>& {
        auto flags = ss.flags();

        if(value.sign != Zero) {
            if (value.sign == Negative) ss.write([] { if constexpr (std::is_same_v<Char, char>) { return "-"; } else { return L"-"; } } (), 1);

            const auto base = [] (long baseField, std::basic_ostream<Char>& ss, bool showbase) {
                auto base = (baseField == std::ios::hex ? 16 : (baseField == std::ios::oct ? 8 : 10));
                if(showbase && base != 10)
                    ss << [&base] { if constexpr (std::is_same_v<Char, char>) { return base == 8 ? "0o" : "0x"; } else { return base == 8 ? L"0o" : L"0x"; }} () << std::noshowbase ;
                return base;
            } (flags & std::ios::basefield, ss, flags & std::ios::showbase);

            if(base == 16) {
                long long iter = value.blocks.size() - 1;
                for(; value.blocks[iter] == 0 && iter >= 0; --iter);

                ss << value.blocks[iter--];
                for (; iter >= 0; --iter) {
                    ss.fill([] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } ());
                    ss.width(8); ss << std::right << value.blocks[iter];
                }
            } else {
                /* Well, here we use a pre-calculated magic number to ratio the length of numbers in decimal or octal notation according to bitness.
                 * It is 2.95-98 for octal and 3.2 for decimal. */
                constexpr auto bufferSize = static_cast<std::size_t>(static_cast<double>(bitness) / 2.95);
                Char buffer [bufferSize] {}; std::size_t filled = 0;

                Aesi copy = value;
                while(copy != 0 && filled < bufferSize) {
                    const auto [quotient, remainder] = divide(copy, base);
                    buffer[filled++] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } () + remainder.template integralCast<byte>();
                    copy = quotient;
                }

                for(; filled > 0; --filled)
                    ss << buffer[filled - 1];
            }
        } else ss.write([] { if constexpr (std::is_same_v<Char, char>) { return "0"; } else { return L"0"; } } (), 1);

        return ss;
    }

#ifdef __CUDACC__
    /**
     * @brief Object assignation using atomic CUDA operations
     * @param Aesi assigning
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks
     * This method is an interface for assigning encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicSet(const Aesi& value) noexcept -> void {
        sign = value.sign;
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&blocks[i], value.blocks[i]);
    }

    /**
     * @brief Object exchange using atomic CUDA operations
     * @param Aesi exchangeable
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks
     * This method is an interface for exchanging encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicExchange(const Aesi& value) noexcept -> void {
        Sign tSign = sign; sign = value.sign; value.sign = tSign;
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&value.blocks[i], atomicExch(&blocks[i], value.blocks[i]));
    }
#endif
};

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
 * @typedef Aesi2048
 * @brief Number with precision 2048-bit. */
using Aesi2048 = Aesi<2048>;

/**
 * @typedef Aesi4096
 * @brief Number with precision 4096-bit. */
using Aesi4096 = Aesi<4096>;

/**
 * @typedef Aesi8192
 * @brief Number with precision 8192-bit. */
using Aesi8192 = Aesi<8192>;

/// @cond HIDE_INCLUDES
#include "Multiprecision.h"
/// @endcond

#endif //AESI_MULTIPRECISION
