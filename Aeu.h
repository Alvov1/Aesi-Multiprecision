#ifndef AEU_MULTIPRECISION
#define AEU_MULTIPRECISION

/// @cond HIDE_INCLUDES
#include <iostream>
#include <cassert>

#ifdef __CUDACC__
    #define gpu __host__ __device__
    #include <cuda/std/utility>
    #include <cuda/std/array>
#else
    #define gpu
    #include <utility>
    #include <array>
    #include <bit>
#endif
/// @endcond

#if defined AESI_UNSAFE
    #warning Enabled nightly mode for the library. Functions and methods input arguments are not checked for validity. Be really gentle
#endif

/**
 * @file Aeu.h
 * @brief Long precision unsigned integer with arithmetic operations
 */

namespace {
    using byte = uint8_t;
    using block = uint32_t;

    constexpr std::size_t bitsInByte = 8, blockBitLength = sizeof(block) * bitsInByte;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;
    constexpr block blockMax = 0xff'ff'ff'ffu;

    /**
     * @enum Comparison
     * @brief Analogue of STD::Strong_ordering since CUDA does not support <=> operator
     */
    enum Comparison { equal = 0, less = 1, greater = 2, equivalent = 3 };
}

/**
 * @class Aeu
 * @brief Long precision unsigned integer
 * @details May be used to represent only positive integers. For negative use Aesi. Number precision is set in template parameter bitness.
 */
template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aeu final {
    static_assert(bitness > sizeof(uint64_t), "Use built-in types for numbers 64-bit or less.");

    static constexpr std::size_t blocksNumber = bitness / blockBitLength;

#ifdef __CUDACC__
    template <typename T1, typename T2>
    using pair = cuda::std::pair<T1, T2>;
    using blockLine = cuda::std::array<block, blocksNumber>;
#else
    template<typename T1, typename T2>
    using pair = std::pair<T1, T2>;
    using blockLine = std::array<block, blocksNumber>;
#endif

    /* -------------------------- @name Class members. ----------------------- */
    /**
     * @brief Block line of the number
     */
    blockLine blocks;
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
            uint64_t sum = static_cast<uint64_t>(dst[i]) + static_cast<uint64_t>(src[i]) + carryOut;
            carryOut = sum / blockBase; dst[i] = static_cast<block>(sum % blockBase);
        }
        return carryOut;
    }

    /**
     * @brief Makes complement block line for line
     * @param BlockLine line
     * @return BlockLine
     */
    gpu static constexpr auto makeComplement(blockLine& line) noexcept -> uint64_t {
        uint64_t carryOut = 1;
        for(std::size_t i = 0; i < blocksNumber; ++i) {
            const uint64_t sum = blockBase - 1ULL - static_cast<uint64_t>(line[i]) + carryOut;
            carryOut = sum / blockBase; line[i] = static_cast<block>(sum % blockBase);
        }
        return carryOut;
    }
    /* ----------------------------------------------------------------------- */

public:
    /* --------------------- @name Different constructors. ------------------- */
    /**
     * @brief Default constructor
     */
    gpu constexpr Aeu() noexcept = default;

    /**
     * @brief Copy constructor
     */
    gpu constexpr Aeu(const Aeu& copy) noexcept = default;

    /**
     * @brief Copy assignment operator
     * @param Aeu other
     */
    gpu constexpr Aeu& operator=(const Aeu& other) noexcept { blocks = other.blocks; return *this; }

    /**
     * @brief Integral constructor
     * @param value Integral
     * @details Accepts both signed and unsigned built-in integral types. When calling this constructor on negative value, final blocks would be inverted.
     * @note Be aware of calling this constructor explicitly
    */
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aeu(Integral value) noexcept {
        if(value != 0) {
            uint64_t tValue = (value < 0 ? static_cast<uint64_t>(value * -1) : static_cast<uint64_t>(value));
            for (std::size_t i = 0; i < blocksNumber; ++i) {
                blocks[i] = static_cast<block>(tValue % blockBase);
                tValue /= blockBase;
            }
            if(value < 0)
                makeComplement(blocks);
        } else
            blocks = {};
    }

    /**
     * @brief Pointer-based character constructor
     * @param Char* pointer
     * @param Size_t size
     * @details Accepts decimal strings along with binary (starting with 0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aeu(const Char* ptr, std::size_t size) noexcept : Aeu {} {
        if(size == 0) return;

        constexpr const Char* characters = [] {
            if constexpr (std::is_same_v<char, Char>) {
                return "09aAfFoObBxX";
            } else {
                return L"09aAfFoObBxX";
            }
        } ();
        std::size_t position = 0;
        const auto negative = (ptr[0] == [] { if constexpr (std::is_same_v<Char, char>) { return '-'; } else { return L'-'; } } ());
        if(negative) ++position;

        const auto base = [&ptr, &size, &position, &characters] {
            if (ptr[position] == characters[0] && size > position + 1) {
                switch (ptr[position + 1]) {
                    case characters[8]:
                    case characters[9]:
                        position += 2; return 2;
                    case characters[6]:
                    case characters[7]:
                        position += 2; return 8;
                    case characters[10]:
                    case characters[11]:
                        position += 2; return 16;
                    default:
                        return 10;
                }
            } else return 10;
        } ();
        for(; position < size; ++position) {
            const auto digit = [&characters] (Char ch) {
                if(characters[0] <= ch && ch <= characters[1])
                    return static_cast<int>(ch) - static_cast<int>(characters[0]);
                if(characters[2] <= ch && ch <= characters[4])
                    return static_cast<int>(ch) - static_cast<int>(characters[2]) + 10;
                if(characters[3] <= ch && ch <= characters[5])
                    return static_cast<int>(ch) - static_cast<int>(characters[3]) + 10;
                return 99;
            } (ptr[position]);

            if(digit < base) {
                this->operator*=(base);
                this->operator+=(digit);
            }
        }

        if(negative && !isZero())
            makeComplement(blocks);
    }

    /**
     * @brief C-style string literal constructor
     * @param Char[] literal
     * @details Accepts decimal literals along with binary (starting with 0b/0B), octal (0o/0O) and hexadecimal (0x/0X)
     */
    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aeu(const Char (&literal)[arrayLength]) noexcept : Aeu(literal, arrayLength) {}

    /**
     * @brief String or string-view based constructor
     * @param String/String-View sv
     * @details Constructs object from STD::Basic_String or STD::Basic_String_View. Accepts objects based on char or wchar_t
     */
    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aeu(String&& stringView) noexcept : Aeu(stringView.data(), stringView.size()) {}
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Arithmetic operators. --------------------- */
    /* ------------------------- @name Unary operators. -------------------------- */
        /**
         * @brief Unary plus operator
         * @return Aeu
         * @note Does basically nothing
         */
        [[nodiscard]]
        gpu constexpr auto operator+() const noexcept -> Aeu { return *this; }

        /**
         * @brief Unary minus operator
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator-() const noexcept -> Aeu {
            Aeu result = *this; makeComplement(result.blocks); return result;
        }

        /**
         * @brief Prefix increment
         * @return Aeu&
         */
        gpu constexpr auto operator++() noexcept -> Aeu& { return this->operator+=(1); }

        /**
         * @brief Postfix increment
         * @return Aeu
         */
        gpu constexpr auto operator++(int) & noexcept -> Aeu {
            Aeu old = *this; operator++(); return old;
        }

        /**
         * @brief Prefix decrement
         * @return Aeu&
         */
        gpu constexpr auto operator--() noexcept -> Aeu& { return this->operator-=(1); }

        /**
         * @brief Postfix decrement
         * @return Aeu
         */
        gpu constexpr auto operator--(int) & noexcept -> Aeu {
            Aeu old = *this; operator--(); return old;
        }
    /* --------------------------------------------------------------------------- */


    /* ------------------------ @name Addition operators. ------------------------ */
        /**
         * @brief Addition operator for built-in integral types
         * @param Unsigned addendum
         * @return Aeu
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>) [[nodiscard]]
        gpu constexpr auto operator+(Unsigned addendum) const noexcept -> Aeu {
            Aeu result = *this; result->operator+=(addendum); return result;
        }

        /**
         * @brief Addition operator
         * @param Aeu addendum
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator+(const Aeu& addendum) const noexcept -> Aeu {
            Aeu result = *this; result += addendum; return result;
        }

        /**
         * @brief Assignment addition operator for built-in integral types
         * @param Unsigned addendum
         * @return Aeu&
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>)
        gpu constexpr auto operator+=(Unsigned addendum) noexcept -> Aeu& {
            for(std::size_t i = 0; i < blocksNumber; ++i) {
                const auto currentSum = static_cast<uint64_t>(blocks[i]) + static_cast<uint64_t>(addendum);
                addendum = currentSum / blockMax; blocks[i] = currentSum % blockMax;
            }
            return *this;
        }

        /**
         * @brief Assignment addition operator
         * @param Aeu addendum
         * @return Aeu&
         */
        gpu constexpr auto operator+=(const Aeu& addendum) noexcept -> Aeu& {
            addLine(blocks, addendum.blocks); return *this;
        }
    /* --------------------------------------------------------------------------- */


    /* ----------------------- @name Subtraction operators. ---------------------- */
        /**
         * @brief Subtraction operator for built-in integral types
         * @param Unsigned subtrahend
         * @return Aeu
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>) [[nodiscard]]
        gpu constexpr auto operator-(Unsigned subtrahend) const noexcept -> Aeu {
            Aeu result = *this; result->operator-=(subtrahend); return result;
        }

        /**
         * @brief Subtraction operator
         * @param Aeu subtrahend
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator-(const Aeu& subtrahend) const noexcept -> Aeu {
            Aeu result = *this; result -= subtrahend; return result;
        }

        /**
         * @brief Assignment subtraction operator for built-in integral types
         * @param Unsigned subtrahend
         * @return Aeu&
         */ /* TODO: Complete */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>)
        gpu constexpr auto operator-=(Unsigned subtrahend) noexcept -> Aeu& = delete;

        /**
         * @brief Assignment subtraction operator
         * @param Aeu subtrahend
         * @return Aeu&
         */
        gpu constexpr auto operator-=(const Aeu& subtrahend) noexcept -> Aeu& {
            return this->operator+=(-subtrahend);
        }
    /* --------------------------------------------------------------------------- */


    /* --------------------- @name Multiplication operators. --------------------- */
        /**
         * @brief Multiplication operator for built-in integral types
         * @param Unsigned factor
         * @return Aeu
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>) [[nodiscard]]
        gpu constexpr auto operator*(Unsigned factor) const noexcept -> Aeu {
            Aeu result = *this; result->operator*=(factor); return result;
        }

        /**
         * @brief Multiplication operator
         * @param Aeu factor
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator*(const Aeu& factor) const noexcept -> Aeu {
            Aeu result = *this; result *= factor; return result;
        }

        /**
         * @brief Assignment multiplication operator for built-in integral types
         * @param Unsigned factor
         * @return Aeu&
         * @details Works with the greatest performance with types smaller than uint64_t
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>)
        gpu constexpr auto operator*=(Unsigned factor) noexcept -> Aeu& {
            if constexpr (std::is_same_v<Unsigned, uint64_t>) {
                const auto longerLength = filledBlocksNumber(), smallerLength = (factor > blockMax ? 2UL : 1UL);
                blockLine buffer {};

                for(std::size_t i = 0; i < longerLength; ++i) {
                    uint64_t tBlock = static_cast<uint64_t>(blocks[i]), carryOut = 0;

                    for(std::size_t j = 0; j < smallerLength && i + j < buffer.size(); ++j) {
                        const auto product = tBlock * ((factor >> blockBitLength * j) & 0x00'00'00'00'ff'ff'ff'ff) + carryOut;
                        const auto block = static_cast<uint64_t>(buffer[i + j]) + (product % blockBase);
                        carryOut = product / blockBase + block / blockBase;
                        buffer[i + j] = static_cast<::block>(block % blockBase);
                    }

                    if(smallerLength + i < buffer.size())
                        buffer[smallerLength + i] += static_cast<block>(carryOut);
                }

                blocks = buffer;
                return *this;
            } else {
                uint64_t carryOut = 0;
                for (std::size_t i = 0; i < blocksNumber; ++i) {
                    const auto product = static_cast<uint64_t>(factor) * static_cast<uint64_t>(blocks[i]) + carryOut;
                    blocks[i] = static_cast<block>(product % blockBase);
                    carryOut = product / blockBase;
                }
                return *this;
            }
        };

        /**
         * @brief Assignment multiplication operator
         * @param Aeu factor
         * @return Aeu&
         */
        gpu constexpr auto operator*=(const Aeu& factor) noexcept -> Aeu& {
            constexpr auto multiplyLines = [] (const blockLine& longerLine, const std::size_t longerLength,
                                               const blockLine& smallerLine, const std::size_t smallerLength) {
                blockLine buffer {};

                for(std::size_t i = 0; i < longerLength; ++i) {
                    uint64_t tBlock = static_cast<uint64_t>(longerLine[i]), carryOut = 0;

                    for(std::size_t j = 0; j < smallerLength && i + j < buffer.size(); ++j) {
                        const auto product = tBlock * static_cast<uint64_t>(smallerLine[j]) + carryOut;
                        const auto block = static_cast<uint64_t>(buffer[i + j]) + (product % blockBase);
                        carryOut = product / blockBase + block / blockBase;
                        buffer[i + j] = static_cast<::block>(block % blockBase);
                    }

                    if(smallerLength < blocksNumber && smallerLength + i < buffer.size())
                        buffer[smallerLength + i] += static_cast<block>(carryOut);
                }

                return buffer;
            };

            const std::size_t thisLength = this->filledBlocksNumber(), valueLength = factor.filledBlocksNumber();
            if(thisLength > valueLength)
                blocks = multiplyLines(blocks, thisLength, factor.blocks, valueLength);
            else
                blocks = multiplyLines(factor.blocks, valueLength, blocks, thisLength);

            return *this;
        }
    /* --------------------------------------------------------------------------- */


    /* ------------------------ @name Division operators. ------------------------ */
        /**
         * @brief Division operator for built-in integral types
         * @param Unsigned divisor
         * @return Aeu
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>) [[nodiscard]]
        gpu constexpr auto operator/(Unsigned divisor) const noexcept -> Aeu {
            Aeu result = *this; result->operator/=(divisor); return result;
        }

        /**
         * @brief Division operator
         * @param Aeu divisor
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator/(const Aeu& divisor) const noexcept -> Aeu {
            Aeu quotient, _; divide(*this, divisor, quotient, _); return quotient;
        }

        /**
         * @brief Assignment division operator for built-in integral types
         * @param Unsigned divisor
         * @return Aeu&
         */ /* TODO: Complete */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>)
        gpu constexpr auto operator/=(Unsigned divisor) noexcept -> Aeu& = delete;

        /**
         * @brief Assignment division operator
         * @param Aeu divisor
         * @return Aeu&
         */
        gpu constexpr auto operator/=(const Aeu& divisor) noexcept -> Aeu& {
            return this->operator=(this->operator/(divisor));
        }
    /* --------------------------------------------------------------------------- */


    /* ------------------------- @name Modulo operators. ------------------------- */
        /**
         * @brief Modulo operator for built-in integral types
         * @param Unsigned modulo
         * @return Aeu
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>) [[nodiscard]]
        gpu constexpr auto operator%(Unsigned modulo) const noexcept -> Aeu {
            Aeu result = *this; result->operator%=(modulo); return result;
        }

        /**
         * @brief Modulo operator
         * @param Aeu modulo
         * @return Aeu
         */
        [[nodiscard]]
        gpu constexpr auto operator%(const Aeu& modulo) const noexcept -> Aeu {
            Aeu _, remainder; divide(*this, modulo, _, remainder); return remainder;
        }

        /**
         * @brief Assignment modulo operator for built-in integral types
         * @param Unsigned modulo
         * @return Aeu&
         */ /* TODO: Complete */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned>)
        gpu constexpr auto operator%=(Unsigned modulo) noexcept -> Aeu& = delete;

        /**
         * @brief Assignment modulo operator
         * @param Aeu modulo
         * @return Aeu&
         */
        gpu constexpr auto operator%=(const Aeu& modulo) noexcept -> Aeu& {
            return this->operator=(this->operator%(modulo));
        }
    /* --------------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */


    /* ----------------------- @name Bitwise operators. ---------------------- */
    /**
     * @brief Bitwise complement operator
     * @return Aeu
     */
    [[nodiscard]]
    gpu constexpr auto operator~() const noexcept -> Aeu {
        Aeu result;
        for(std::size_t i = 0; i < blocksNumber; ++i)
            result.blocks[i] = ~blocks[i];
        return result;
    }

    /**
     * @brief Bitwise XOR operator
     * @param Aeu other
     * @return Aeu
     */
    [[nodiscard]]
    gpu constexpr auto operator^(const Aeu& other) const noexcept -> Aeu {
        Aeu result = *this; result ^= other; return result;
    }

    /**
     * @brief Assignment bitwise XOR operator
     * @param Aeu other
     * @return Aeu&
     */
    gpu constexpr auto operator^=(const Aeu& other) noexcept -> Aeu& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] ^= other.blocks[i];
        return *this;
    }

    /**
     * @brief Bitwise AND operator
     * @param Aeu other
     * @return Aeu
     */
    [[nodiscard]]
    gpu constexpr auto operator&(const Aeu& other) const noexcept -> Aeu {
        Aeu result = *this; result &= other; return result;
    }

    /**
     * @brief Assignment bitwise AND operator
     * @param Aeu other
     * @return Aeu&
     */
    gpu constexpr auto operator&=(const Aeu& other) noexcept -> Aeu& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] &= other.blocks[i];
        return *this;
    }

    /**
     * @brief Bitwise OR operator
     * @param Aeu other
     * @return Aeu
     */
    [[nodiscard]]
    gpu constexpr auto operator|(const Aeu& other) const noexcept -> Aeu {
        Aeu result = *this; result |= other; return result;
    }

    /**
     * @brief Assignment bitwise OR operator
     * @param Aeu other
     * @return Aeu&
     */
    gpu constexpr auto operator|=(const Aeu& other) noexcept -> Aeu& {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] |= other.blocks[i];
        return *this;
    }

    /**
     * @brief Left shift operator
     * @param Unsigned bit_shift
     * @return Aeu
     * @note Does nothing for shift greater than precision
     */
    template <typename Unsigned> requires (std::is_integral_v<Unsigned> && std::is_unsigned_v<Unsigned>) [[nodiscard]]
    gpu constexpr auto operator<<(Unsigned bitShift) const noexcept -> Aeu {
        Aeu result = *this; result.operator<<=(bitShift); return result;
    }

    /**
     * @brief Left shift assignment operator
     * @param Unsigned bit_shift
     * @return Aeu&
     * @note Does nothing for shift greater than precision
     */
    template <typename Unsigned> requires (std::is_integral_v<Unsigned> && std::is_unsigned_v<Unsigned>)
    gpu constexpr auto operator<<=(Unsigned bitShift) noexcept -> Aeu& {
        if(bitShift >= bitness || bitShift == 0) return *this;

        const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
        const block stamp = (1UL << (blockBitLength - remainder)) - 1;

        for (long long i = blocksNumber - 1; i >= (quotient + (remainder ? 1 : 0)); --i)
            blocks[i] = ((blocks[i - quotient] & stamp) << remainder) | ((blocks[i - quotient - (remainder ? 1 : 0)] & ~stamp) >> ((blockBitLength - remainder) % blockBitLength));

        blocks[quotient] = (blocks[0] & stamp) << remainder;

        for (std::size_t i = 0; i < quotient; ++i)
            blocks[i] = 0;
        return *this;
    }

    /**
     * @brief Right shift operator
     * @param Unsigned bit_shift
     * @return Aeu
     * @note Does nothing for shift greater than precision
     */
    template <typename Unsigned> requires (std::is_integral_v<Unsigned> && std::is_unsigned_v<Unsigned>) [[nodiscard]]
    gpu constexpr auto operator>>(Unsigned bitShift) const noexcept -> Aeu {
        Aeu result = *this; result >>= bitShift; return result;
    }

    /**
     * @brief Right shift assignment operator
     * @param Unsigned bit_shift
     * @return Aeu&
     * @note Does nothing for shift greater than precision
     */
    template <typename Unsigned> requires (std::is_integral_v<Unsigned> && std::is_unsigned_v<Unsigned>)
    gpu constexpr auto operator>>=(Unsigned bitShift) noexcept -> Aeu& {
        if(bitShift >= bitness || bitShift == 0) return *this;

        const std::size_t quotient = bitShift / blockBitLength, remainder = bitShift % blockBitLength;
        const block stamp = (1UL << remainder) - 1;

        for(std::size_t i = 0; i < blocksNumber - (quotient + (remainder ? 1 : 0)); ++i)
            blocks[i] = ((blocks[i + quotient + (remainder ? 1 : 0)] & stamp) << ((blockBitLength - remainder) % blockBitLength)) | ((blocks[i + quotient] & ~stamp) >> remainder);

        blocks[blocksNumber - 1 - quotient] = (blocks[blocksNumber - 1] & ~stamp) >> remainder;

        for(long long i = blocksNumber - quotient; i < blocksNumber; ++i)
            blocks[i] = 0;
        return *this;
    }
    /* ----------------------------------------------------------------------- */


    /* --------------------- @name Comparison operators. --------------------- */
    /* ------------------------ @name Equality operators. ------------------------ */
        /**
         * @brief Equality check operator for built-in types uint8_t, uint16_t, uint32_t
         * @param Unsigned other
         * @return Bool
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned> && sizeof(Unsigned) < 8)
        gpu constexpr auto operator==(Unsigned other) const noexcept -> bool {
            return filledBlocksNumber() <= 1 && static_cast<block>(other) == blocks[0];
        }
        gpu constexpr auto operator==(uint64_t other) const noexcept -> bool {
            return filledBlocksNumber() <= 2 && ((static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0])) == other;
        }

        /**
         * @brief Equality check operator for numbers of the same precision
         * @param Aeu other
         * @return Bool
         */
        gpu constexpr auto operator==(const Aeu& other) const noexcept -> bool { return blocks == other.blocks; };

        /**
         * @brief Templated Equality check operator for numbers of different precision
         * @param Aeu other
         * @return Bool
         */
        template <std::size_t otherBitness> requires (otherBitness != bitness)
        gpu constexpr auto operator==(const Aeu<otherBitness>& other) const noexcept -> bool { return compareTo(other) == Comparison::equal; };
    /* --------------------------------------------------------------------------- */


    /* ----------------------- @name Comparison operators. ----------------------- */
        /**
         * @brief Internal comparison operator for built-in integral types uint8_t, uint16_t, uint32_t
         * @param Unsigned other
         * @return Comparison
         * @note Should almost never return Comparison::Equivalent
         */
        template <typename Unsigned> requires (std::is_unsigned_v<Unsigned> && sizeof(Unsigned) < 8) [[nodiscard]]
        gpu constexpr auto compareTo(Unsigned other) const noexcept -> Comparison {
            if(filledBlocksNumber() > 1) return Comparison::greater;
            const auto cmp = static_cast<block>(other);
            if(blocks[0] > cmp)
                return Comparison::greater;
            else if(blocks[0] < cmp)
                return Comparison::less; else return Comparison::equal;
        }
    
        /**
         * @brief Internal comparison operator for type uint64_t
         * @param uint64_t other
         * @return Comparison
         * @note Should almost never return Comparison::Equivalent
         */
        [[nodiscard]]
        gpu constexpr auto compareTo(uint64_t other) const noexcept -> Comparison {
            if(filledBlocksNumber() > 2) return Comparison::greater;
            const auto base = (static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0]);
            if(base > other)
                return Comparison::greater; else if(base < other) return Comparison::less; else return Comparison::equal;
        }
    
        /**
         * @brief Internal comparison operator
         * @param Aeu other
         * @return Comparison
         * @note Should almost never return Comparison::Equivalent
         */
        template <std::size_t otherBitness = bitness> [[nodiscard]]
        gpu constexpr auto compareTo(const Aeu<otherBitness>& other) const noexcept -> Comparison {
            /* First compare in common precision. */
            const auto lowerBlockBorder = (blocksNumber < other.totalBlocksNumber() ? blocksNumber : other.totalBlocksNumber());
            for(long long i = lowerBlockBorder - 1; i >= 0; --i) {
                const block thisBlock = blocks[i], otherBlock = other.getBlock(i);
                if(thisBlock != otherBlock)
                    return (thisBlock > otherBlock ? Comparison::greater : Comparison::less);
            }
    
            if constexpr (otherBitness != blocksNumber * blockBitLength) {
                /* If larger number contains data out of lower number's range, it's greater. */
                if (other.totalBlocksNumber() > blocksNumber) {
                    for (long long i = other.totalBlocksNumber() - 1; i > lowerBlockBorder - 1; --i)
                        if (other.getBlock(i) != 0)
                            return Comparison::less;
                } else if (blocksNumber > other.totalBlocksNumber()) {
                    for (long long i = blocksNumber - 1; i > lowerBlockBorder - 1; --i)
                        if (blocks[i] != 0)
                            return Comparison::greater;
                }
            }
    
            return Comparison::equal;
        }
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
        gpu constexpr auto operator<=>(const Aeu& other) const noexcept -> std::strong_ordering {
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
    /**
     * @brief Set bit in number by index starting from the right
     * @param Size_t index
     * @param Bool bit
     * @note Does nothing for index out of range. Index check is disabled in unsafe mode
     */
    gpu constexpr auto setBit(std::size_t index, bool bit) noexcept -> void {
#ifndef AESI_UNSAFE
        if(index >= bitness) return;
#endif
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        assert(blockNumber < blocksNumber && bitNumber < blockBitLength);
        if(bit)
            blocks[blockNumber] |= (1U << bitNumber);
        else
            blocks[blockNumber] &= (~(1U << bitNumber));
    }

    /**
     * @brief Get bit in number by index staring from the right
     * @param Size_t index
     * @return Bool
     * @note Returns zero for index out of range. Index check is disabled in unsafe mode
     */
    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool {
#ifndef AESI_UNSAFE
            if(index >= bitness) return false;
#endif
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        assert(blockNumber < blocksNumber && bitNumber < blockBitLength);
        return blocks[blockNumber] & (1U << bitNumber);
    }

    /**
     * @brief Set byte in number by index starting from the right
     * @param Size_t index
     * @param Byte byte
     * @note Does nothing for index out of range. Index check is disabled in unsafe mode.
     */
    gpu constexpr auto setByte(std::size_t index, byte byte) noexcept -> void {
#ifndef AESI_UNSAFE
        if(index > blocksNumber * sizeof(block)) return;
#endif
        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        assert(blockNumber < blocksNumber && byteInBlock < sizeof(block));
        blocks[blockNumber] &= ~(0xffU << shift); blocks[blockNumber] |= static_cast<block>(byte) << shift;
    }

    /**
     * @brief Get byte in number by index starting from the right
     * @param Size_t index
     * @return Byte
     * @note Returns zero for index out of range. Index check is disabled in unsafe mode.
     */
    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> byte {
#ifndef AESI_UNSAFE
        if(index > blocksNumber * sizeof(block)) return 0;
#endif
        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        assert(blockNumber < blocksNumber && byteInBlock < sizeof(block));
        return (blocks[blockNumber] & (0xffU << shift)) >> shift;
    }

    /**
     * @brief Set block in number by index starting from the right
     * @param Size_t index
     * @param Block byte
     * @note Does nothing for index out of range. Index check is disabled in unsafe mode.
     */
    gpu constexpr auto setBlock(std::size_t index, block block) noexcept -> void {
#ifndef AESI_UNSAFE
        if(index >= blocksNumber) return;
#endif
        blocks[index] = block;
    }

    /**
     * @brief Get block in number by index starting from the right
     * @param Size_t index
     * @return Block
     * @note Returns zero for index out of range. Index check is disabled in unsafe mode.
     */
    [[nodiscard]]
    gpu constexpr auto getBlock(std::size_t index) const noexcept -> block {
#ifndef AESI_UNSAFE
        if(index >= blocksNumber) return block();
#endif
        return blocks[index];
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
    constexpr auto bitCount() const noexcept -> std::size_t {
        std::size_t lastBlockIdx = blocksNumber - 1;
        for(; lastBlockIdx > 0 && blocks[lastBlockIdx] == 0; --lastBlockIdx);

        const block lastBlock = blocks[lastBlockIdx], bitsInBlock = (sizeof(block) * 8 - std::countl_zero(lastBlock));
        return lastBlockIdx * sizeof(block) * bitsInByte + bitsInBlock;
    }

#ifdef __CUDACC__
    __device__ constexpr auto bitCount() const noexcept -> std::size_t {
        std::size_t lastBlockIdx = blocksNumber - 1;
        for(; lastBlockIdx > 0 && blocks[lastBlockIdx] == 0; --lastBlockIdx);

        const block lastBlock = blocks[lastBlockIdx], bitsInBlock = (sizeof(block) * 8 - __clz(lastBlock));
        return lastBlockIdx * sizeof(block) * bitsInByte + bitsInBlock;
    }
#endif

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
    gpu constexpr auto isZero() const noexcept -> bool { return filledBlocksNumber() == 0; }

    /**
     * @brief Get number of non-empty blocks inside object starting from the right
     * @return Size_t
     */
    [[nodiscard]]
    gpu constexpr auto filledBlocksNumber() const noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(blocks[i]) return i + 1;
        return 0;
    }

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
    gpu static constexpr auto totalBlocksNumber() noexcept -> std::size_t { return blocksNumber; }

    /**
     * @brief Make swap between two objects
     * @param Aeu other
     */
    gpu constexpr auto swap(Aeu& other) noexcept -> void {
        Aeu t = other; other.operator=(*this); this->operator=(t);
    }
    /* ----------------------------------------------------------------------- */

    
    /* -------------- @name Public arithmetic and number theory. ------------- */
    /**
     * @brief Integer division. Returns results by reference
     * @param Aeu number
     * @param Aeu divisor
     * @param Aeu quotient OUT
     * @param Aeu remainder OUT
     * @return Quotient and remainder by reference
     */
    gpu static constexpr auto divide(const Aeu& number, const Aeu& divisor, Aeu& quotient, Aeu& remainder) noexcept -> void {
        /* TODO Попробовать ускорить divide для работы со встроенными типами. */
        const auto ratio = number.compareTo(divisor);

        quotient = Aeu {}; remainder = Aeu {};

        if(ratio == Comparison::greater) {
            const auto bitsUsed = number.filledBlocksNumber() * blockBitLength;
            for(long long i = bitsUsed - 1; i >= 0; --i) {
                remainder <<= 1u;
                remainder.setBit(0, number.getBit(i));

                if(remainder >= divisor) {
                    remainder -= divisor;
                    quotient.setBit(i, true);
                }
            }
        } else if(ratio == Comparison::less)
            remainder = number; else quotient = 1;
    }

    /**
    * @brief Integer division. Returns results by value
    * @param Aeu number
    * @param Aeu divisor
    * @return Pair(Quotient, Remainder)
    */
    [[nodiscard]]
    gpu static constexpr auto divide(const Aeu& number, const Aeu& divisor) noexcept -> pair<Aeu, Aeu> {
        pair<Aeu, Aeu> results; divide(number, divisor, results.first, results.second); return results;
    }
    
    /**
     * @brief Extended Euclidean algorithm for greatest common divisor
     * @param Aeu first
     * @param Aeu second
     * @param Aeu bezoutX OUT
     * @param Aeu bezoutY OUT
     * @return Aeu
     * @details Counts Bézout coefficients along with the greatest common divisor. Returns coefficients by reference
     */
    gpu static constexpr auto gcd(const Aeu& first, const Aeu& second, Aeu& bezoutX, Aeu& bezoutY) noexcept -> Aeu {
        Aeu gcd, tGcd, quotient, remainder;

        const auto ratio = first.compareTo(second);
        if(ratio == Comparison::greater) {
            gcd = second;
            divide(first, second, quotient, remainder);
        } else {
            gcd = first;
            divide(second, first, quotient, remainder);
        }

        bezoutX = 0; bezoutY = 1;
        for(Aeu tX = 1, tY = 0; remainder != 0; ) {
            tGcd = gcd; gcd = remainder;

            Aeu t = bezoutX; bezoutX = tX - quotient * bezoutX; tX = t;
            t = bezoutY; bezoutY = tY - quotient * bezoutY; tY = t;

            divide(tGcd, gcd, quotient, remainder);
        }

        if(ratio != Comparison::greater)
            bezoutX.swap(bezoutY);
        
        return gcd;
    }

    /**
     * @brief Greatest common divisor
     * @param Aeu first
     * @param Aeu second
     * @return Aeu
     * @details Faster version, ignoring bezout coefficients
     */
    gpu static constexpr auto gcd(const Aeu& first, const Aeu& second) noexcept -> Aeu {
        Aeu gcd, tGcd, quotient, remainder;

        const auto ratio = first.compareTo(second);
        if(ratio == Comparison::greater) {
            gcd = second;
            divide(first, second, quotient, remainder);
        } else {
            gcd = first;
            divide(second, first, quotient, remainder);
        }

        for(Aeu tX = 1, tY = 0; remainder != 0; ) {
            tGcd = gcd; gcd = remainder;
            divide(tGcd, gcd, quotient, remainder);
        }
        
        return gcd;
    }

    /**
     * @brief Least common multiplier
     * @param Aeu first
     * @param Aeu second
     * @return Aeu
     */
    [[nodiscard]]
    gpu static constexpr auto lcm(const Aeu& first, const Aeu& second) noexcept -> Aeu { return first / gcd(first, second) * second; }
    
    /**
     * @brief Exponentiation by modulo
     * @param Aeu base
     * @param Aeu power
     * @param Aeu modulo
     * @return Aeu
     * @note Be aware of overflow
     * @details Accepts power of different precision rather than base and modulo
     */
    template <std::size_t powerBitness = bitness>
    gpu static constexpr auto powm(const Aeu& base, const Aeu<powerBitness>& power, const Aeu& mod) noexcept -> Aeu {
        if(base == 1u) 
            return base;
        if(base == 0u)
            return { 1u };

        Aeu output = 1;
        auto [_, b] = divide(base, mod);

        for(unsigned iteration = 0; power.filledBlocksNumber() * blockBitLength != iteration; iteration++) {
            if(power.getBit(iteration)) {
                const auto [quotient, remainder] = divide(output * b, mod);
                output = remainder;
            }

            const auto [quotient, remainder] = divide(b * b, mod);
            b = remainder;
        }

        return output;
    }
    
    /**
     * @brief Fast exponentiation for powers of 2
     * @param Size_t power
     * @return Aeu
     * @details Returns zero for power greater than current bitness
     */
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aeu { Aeu result {}; result.setBit(e, true); return result; }
    
    /**
     * @brief Get square root
     * @return Aeu
     * @note Returns zero for negative value or zero
     */
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aeu {
        Aeu x, y = power2((bitCount() + 1) / 2);

        do {
            x = y;
            y = (x + this->operator/(x)) >> 1u;
        } while (y < x);

        return x;
    }
    /* ----------------------------------------------------------------------- */


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
    template <byte base, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> && (base == 2 || base == 8 || base == 10 || base == 16))
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

        if(isZero()) {
            buffer[position++] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; }}();
            return position;
        }

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

            Aeu copy = *this;
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

        buffer[position++] = Char {};
        return position;
    }

    /**
     * @brief STD stream output operator
     * @param Ostream stream
     * @param Aeu number
     * @return Ostream
     * @details Writes number in stream. Accepts STD streams based on char or wchar_t. Supports stream manipulators:
     * - Number's notation (std::hex, std::dec, std::oct);
     * - Number's base (std::showbase);
     * - Hexadecimal letters case (std::uppercase, std::lowercase)
     * @note Works significantly faster for hexadecimal notation
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    friend constexpr auto operator<<(std::basic_ostream<Char>& ss, const Aeu& value) -> std::basic_ostream<Char>& {
        auto flags = ss.flags();

        const auto base = [] (long baseField, std::basic_ostream<Char>& ss, bool showbase) {
            auto base = (baseField == std::ios::hex ? 16 : (baseField == std::ios::oct ? 8 : 10));
            if(showbase && base != 10)
                ss << [&base] { if constexpr (std::is_same_v<Char, char>) { return base == 8 ? "0o" : "0x"; } else { return base == 8 ? L"0o" : L"0x"; }} () << std::noshowbase ;
            return base;
        } (flags & std::ios::basefield, ss, flags & std::ios::showbase);

        if(value.isZero())
            return ss << '0';

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
             * * It is 2.95-98 for octal and 3.2 for decimal. */
            constexpr auto bufferSize = static_cast<std::size_t>(static_cast<double>(bitness) / 2.95);
            Char buffer [bufferSize] {}; std::size_t filled = 0;

            Aeu copy = value;
            while(!copy.isZero() && filled < bufferSize) {
                const auto [quotient, remainder] = divide(copy, base);
                buffer[filled++] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } () + remainder.template integralCast<byte>();
                copy = quotient;
            }

            for(; filled > 0; --filled)
                ss << buffer[filled - 1];
        }

        return ss;
    }


    /**
     * @brief STD stream binary reading operator
     * @param Istream stream
     * @param Boolean big_endian
     * @return Aeu
     * @details Reads number from stream using .read method. Accepts STD streams based on char or wchar_t.
     * @note Fills empty bits with 0s on EOF of the stream
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    constexpr auto readBinary(std::basic_istream<Char>& istream, bool bigEndian = true) -> void {
        blocks = {};
        if(bigEndian) {
            for(auto it = blocks.rbegin(); it != blocks.rend(); ++it)
                if(!istream.read(reinterpret_cast<char*>(&*it), sizeof(block))) break;
        } else {
            for(auto& tBlock: blocks)
                if(!istream.read(reinterpret_cast<char*>(&tBlock), sizeof(block))) break;
        }
    }


    /**
     * @brief STD stream binary writing operator
     * @param Ostream stream
     * @param Boolean big_endian
     * @details Writes number in stream using .write method. Accepts STD streams based on char or wchar_t.
     */
    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    constexpr auto writeBinary(std::basic_ostream<Char>& ostream, bool bigEndian = true) const noexcept -> void {
        if(bigEndian) {
            for(auto it = blocks.rbegin(); it != blocks.rend(); ++it)
                if(!ostream.write(reinterpret_cast<const char*>(&*it), sizeof(block))) break;
        } else {
            for(auto& block: blocks)
                if(!ostream.write(reinterpret_cast<const char*>(&block), sizeof(block))) break;
        }
    }
    /* ----------------------------------------------------------------------- */


    /* -------------------- @name Public casting operators. ------------------ */
    /**
     * @brief Built-in integral type cast operator
     * @param Type integral_type TEMPLATE
     * @return Integral
     * @details Takes the lowes part of Aeu for conversion. Accepts signed and unsigned types
     */
    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        const uint64_t value = (static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0]);
        return static_cast<Integral>(value);
    }

    /**
     * @brief Precision cast operator
     * @param Size_t new_bitness TEMPLATE
     * @return Aeu<new_bitness>
     * @details If required precision greater than current precision, remaining blocks are filled with zeros.
     * Otherwise - number is cropped inside smaller blocks array
     * @note Using this method directly is not recommended,
     * cause it leads to redundant copying and may be slow
     */
    template <std::size_t newBitness> requires (newBitness != bitness) [[nodiscard]]
    gpu constexpr auto precisionCast() const noexcept -> Aeu<newBitness> {
        Aeu<newBitness> result {};

        const std::size_t blockBoarder = (newBitness > bitness ? Aeu<bitness>::blocksNumber : Aeu<newBitness>::totalBlocksNumber());
        for(std::size_t blockIdx = 0; blockIdx < blockBoarder; ++blockIdx)
            result.setBlock(blockIdx, blocks[blockIdx]);

        return result;
    }
    /* ----------------------------------------------------------------------- */


#if defined __CUDACC__
    /* ------------------- @name Atomic-like CUDA operators. ----------------- */
    /**
     * @brief Atomicity-oriented object assignment operator
     * @param Aeu assigning
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for assigning encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicSet(const Aeu& value) noexcept -> void {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&blocks[i], value.blocks[i]);
    }

    /**
     * @brief Atomicity-oriented object exchangement operator
     * @param Aeu exchangeable
     * @note Method itself is not atomic. There may be race conditions between two consecutive atomic calls on number blocks.
     * This method is an interface for exchanging encapsulated class members atomically one by one
     */
    __device__ constexpr auto tryAtomicExchange(const Aeu& value) noexcept -> void {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&value.blocks[i], atomicExch(&blocks[i], value.blocks[i]));
    }
    /* ----------------------------------------------------------------------- */
#endif
};

/* -------------------------------------------- @name Type-definitions  ------------------------------------------- */
/**
 * @typedef Aeu128
 * @brief Number with precision 128-bit. */
using Aeu128 = Aeu<128>;

/**
 * @typedef Aeu256
 * @brief Number with precision 128-bit. */
using Aeu256 = Aeu<256>;

/**
 * @typedef Aeu512
 * @brief Number with precision 512-bit. */
using Aeu512 = Aeu<512>;

/**
 * @typedef Aeu768
 * @brief Number with precision 768-bit. */
using Aeu768 = Aeu<768>;

/**
 * @typedef Aeu1024
 * @brief Number with precision 1024-bit. */
using Aeu1024 = Aeu<1024>;

/**
 * @typedef Aeu1536
 * @brief Number with precision 1536-bit. */
using Aeu1536 = Aeu<1536>;

/**
 * @typedef Aeu2048
 * @brief Number with precision 2048-bit. */
using Aeu2048 = Aeu<2048>;

/**
 * @typedef Aeu3072
 * @brief Number with precision 3072-bit. */
using Aeu3072 = Aeu<3072>;

/**
 * @typedef Aeu4096
 * @brief Number with precision 4096-bit. */
using Aeu4096 = Aeu<4096>;

/**
 * @typedef Aeu6144
 * @brief Number with precision 6144-bit. */
using Aeu6144 = Aeu<6144>;

/**
 * @typedef Aeu8192
 * @brief Number with precision 8192-bit. */
using Aeu8192 = Aeu<8192>;
/* ---------------------------------------------------------------------------------------------------------------- */

/* ------------------------------------------ @name Integral conversions  ----------------------------------------- */
/**
 * @brief Integral conversion addition operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) + value; }

/**
 * @brief Integral conversion subtraction operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) - value; }

/**
 * @brief Integral conversion multiplication operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) * value; }

/**
 * @brief Integral conversion division operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) / value; }

/**
 * @brief Integral conversion modulo operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) % value; }

/**
 * @brief Integral conversion bitwise XOR operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) ^ value; }

/**
 * @brief Integral conversion bitwise AND operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) & value; }

/**
 * @brief Integral conversion bitwise OR operator
 * @param Integral number
 * @param Aeu value
 * @return Aeu
 */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aeu<bitness>& value) noexcept { return Aeu<bitness>(number) | value; }
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //AEU_MULTIPRECISION
