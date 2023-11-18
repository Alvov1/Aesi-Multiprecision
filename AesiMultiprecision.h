#ifndef AESI_MULTIPRECISION
#define AESI_MULTIPRECISION

#include <iostream>

#ifdef __CUDACC__
    #define gpu __host__ __device__
    #include <thrust/pair.h>
#else
    #define gpu
    #include <utility>
#endif

namespace {
    using block = uint32_t;
    constexpr auto bitsInByte = 8;
    constexpr auto blockBitLength = sizeof(block) * bitsInByte;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;
}

enum class AesiCMP { equal = 0, less = 1, greater = 2, equivalent = 3 };

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

    template <typename ValueType, std::size_t lineSize>
    struct MyArray final {
        ValueType data [lineSize] {};
        gpu constexpr bool operator==(const MyArray& value) const noexcept {
            for(std::size_t i = 0; i < lineSize; ++i)
                if(data[i] != value.data[i]) return false; return true;
        };
        [[nodiscard]] gpu constexpr auto size() const noexcept -> std::size_t { return lineSize; };
        gpu constexpr auto operator[] (std::size_t index) const noexcept -> const ValueType& { return data[index]; }
        gpu constexpr auto operator[] (std::size_t index) noexcept -> ValueType& { return data[index]; }
    };
    using blockLine = MyArray<block, blocksNumber>;
    enum Sign { Zero = 0, Positive = 1, Negative = 2 };

    /* ---------------------------- Class members. --------------------------- */
    blockLine blocks {};
    Sign sign { Zero };
    /* ----------------------------------------------------------------------- */


    /* --------------------------- Helper functions. ------------------------- */
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
    gpu static constexpr auto makeComplement(const blockLine& line) noexcept -> blockLine {
        blockLine result {};

        uint64_t carryOut = 1;
        for(std::size_t i = 0; i < blocksNumber; ++i) {
            const uint64_t sum = blockBase - 1ULL - static_cast<uint64_t>(line[i]) + carryOut;
            carryOut = sum / blockBase; result[i] = sum % blockBase;
        }

        return result;
    }
    gpu static constexpr auto isLineEmpty(const blockLine& line) noexcept -> bool {
        return lineLength(line) == 0;
    }
    gpu static constexpr auto lineLength(const blockLine& line) noexcept -> std::size_t {
        for(long long i = blocksNumber - 1; i >= 0; --i)
            if(line[i]) return i + 1;
        return 0;
    }
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor) noexcept -> pair<Aesi, Aesi> {
        const Aesi divAbs = divisor.abs();
        const auto ratio = number.abs().compareTo(divAbs);

        Aesi quotient = 0, remainder = 0;
//        pair<Aesi, Aesi> results = {0, 0 };
//        auto& [quotient, remainder] = results;

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

        return { quotient, remainder };
    }
    /* ----------------------------------------------------------------------- */

public:
    /* ----------------------- Different constructors. ----------------------- */
    gpu constexpr Aesi() noexcept {};
    gpu constexpr Aesi(const Aesi& copy) noexcept {
        sign = copy.sign;
        if(copy.sign != Zero) blocks = copy.blocks;
    };
    gpu constexpr Aesi& operator=(const Aesi& other) noexcept {
        blocks = other.blocks; sign = other.sign; return *this;
    }

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
        }
    }

    template <typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>)
    gpu constexpr Aesi(const Char* ptr, std::size_t size) noexcept {
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

    template <typename Char, std::size_t arrayLength> requires (arrayLength > 1 && (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t>))
    gpu constexpr Aesi(const Char (&array)[arrayLength]) noexcept : Aesi(array, arrayLength) {}

    template <typename String, typename Char = typename String::value_type> requires (std::is_same_v<std::basic_string<Char>,
            typename std::decay<String>::type> || std::is_same_v<std::basic_string_view<Char>, typename std::decay<String>::type>)
    gpu constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()){}

    template<std::size_t rBitness> requires (rBitness != bitness)
    gpu constexpr Aesi(const Aesi<rBitness>& copy) noexcept {
        this->operator=(copy.template precisionCast<bitness>());
    }
    /* ----------------------------------------------------------------------- */


    /* ------------------------ Arithmetic operators. ------------------------ */
    gpu constexpr Aesi operator+() const noexcept { return *this; }
    gpu constexpr Aesi operator-() const noexcept {
        if(sign == Zero) return Aesi();
        Aesi result = *this;
        result.sign = (result.sign == Positive ? Negative : Positive); return result;
    }

    gpu constexpr Aesi& operator++() noexcept { return this->operator+=(1); }
    gpu constexpr Aesi operator++(int) & noexcept {
        Aesi old = *this; operator++(); return old;
    }
    gpu constexpr Aesi& operator--() noexcept { return this->operator-=(1); }
    gpu constexpr Aesi operator--(int) & noexcept {
        Aesi old = *this; operator--(); return old;
    }

    gpu constexpr Aesi operator+(const Aesi& value) const noexcept {
        Aesi result = *this; result += value; return result;
    }
    gpu constexpr Aesi& operator+=(const Aesi& value) noexcept {
        if(sign == Zero) return this->operator=(value);
        if(value.sign == Zero) return *this;

        if (sign != value.sign) {
            if (sign == Negative)
                blocks = makeComplement(blocks);
            const uint64_t carryOut = (value.sign != Negative ?
                                       addLine(blocks, value.blocks) : addLine(blocks, makeComplement(value.blocks)));
            if (carryOut == 0) {
                blocks = makeComplement(blocks);
                sign = Negative;
            } else sign = Positive;
        } else
            addLine(blocks, value.blocks);

        if (isLineEmpty(blocks))
            sign = Zero;

        return *this;
    }

    gpu constexpr Aesi operator-(const Aesi& value) const noexcept {
        Aesi result = *this; result -= value; return result;
    }
    gpu constexpr Aesi& operator-=(const Aesi& value) noexcept {
        return this->operator+=(-value);
    }

    gpu constexpr Aesi operator*(const Aesi& value) const noexcept {
        Aesi result = *this; result *= value; return result;
    }
    gpu constexpr Aesi& operator*=(const Aesi& value) noexcept {
        if(sign == Zero) return *this;
        if(value.sign == Zero)
            return this->operator=(Aesi());
        sign = (sign != value.sign ? Negative : Positive);

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

        const std::size_t thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
        if(thisLength > valueLength)
            blocks = multiplyLines(blocks, thisLength, value.blocks, valueLength);
        else
            blocks = multiplyLines(value.blocks, valueLength, blocks, thisLength);

        return *this;
    }

    gpu constexpr Aesi operator/(const Aesi& divisor) const noexcept {
        return divide(*this, divisor).first;
    }
    gpu constexpr Aesi& operator/=(const Aesi& divisor) noexcept {
        return this->operator=(divide(*this, divisor).first);
    }

    gpu constexpr Aesi operator%(const Aesi& divisor) const noexcept {
        return divide(*this, divisor).second;
    }
    gpu constexpr Aesi& operator%=(const Aesi& divisor) noexcept {
        return this->operator=(divide(*this, divisor).second);
    }
    /* ----------------------------------------------------------------------- */


    /* ------------------------- Bitwise operators. -------------------------- */
    gpu constexpr Aesi operator~() const noexcept {
        Aesi result {};
        for(std::size_t i = 0; i < blocksNumber; ++i)
            result.blocks[i] = ~blocks[i];
        if(isLineEmpty(result.blocks))
            result.sign = Zero; else result.sign = sign;
        return result;
    }

    gpu constexpr Aesi operator^(const Aesi& value) const noexcept {
        Aesi result = *this; result ^= value; return result;
    }
    gpu constexpr Aesi& operator^=(const Aesi& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] ^= value.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
        return *this;
    }

    gpu constexpr Aesi operator&(const Aesi& value) const noexcept {
        Aesi result = *this; result &= value; return result;
    }
    gpu constexpr Aesi& operator&=(const Aesi& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] &= value.blocks[i];
        if(isLineEmpty(blocks)) sign = Zero;
        return *this;
    }

    gpu constexpr Aesi operator|(const Aesi& value) const noexcept {
        Aesi result = *this; result |= value; return result;
    }
    gpu constexpr Aesi& operator|=(const Aesi& value) noexcept {
        for(std::size_t i = 0; i < blocksNumber; ++i)
            blocks[i] |= value.blocks[i];
        if(sign == Zero && !isLineEmpty(blocks)) sign = Positive;
        return *this;
    }

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi operator<<(Integral bitShift) const noexcept {
        Aesi result = *this; result <<= bitShift; return result;
    }
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi& operator<<=(Integral bitShift) noexcept {
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

    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi operator>>(Integral bitShift) const noexcept {
        Aesi result = *this; result >>= bitShift; return result;
    }
    template <typename Integral> requires (std::is_integral_v<Integral>)
    gpu constexpr Aesi& operator>>=(Integral bitShift) noexcept {
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


    /* ----------------------- Comparison operators. ------------------------- */
    gpu constexpr auto operator==(const Aesi& value) const noexcept -> bool {
        if(sign != Zero || value.sign != Zero)
            return (sign == value.sign && blocks == value.blocks); else return true;
    };
    gpu constexpr auto compareTo(const Aesi& value) const noexcept -> AesiCMP {
        switch (sign) {
            case Zero:
                switch (value.sign) {
                    case Zero: return AesiCMP::equal;
                    case Positive: return AesiCMP::less;
                    case Negative: return AesiCMP::greater;
                    default: return AesiCMP::equivalent;
                }
            case Positive:
                switch (value.sign) {
                    case Positive: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength)
                            return (thisLength > valueLength) ? AesiCMP::greater : AesiCMP::less;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i])
                                return (blocks[i] > value.blocks[i]) ? AesiCMP::greater : AesiCMP::less;

                        return AesiCMP::equal;
                    }
                    case Zero:
                    case Negative: return AesiCMP::greater;
                    default: return AesiCMP::equivalent;
                }
            case Negative:
                switch (value.sign) {
                    case Negative: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength)
                            return (static_cast<long long>(thisLength) * -1 > static_cast<long long>(valueLength) * -1) ? AesiCMP::greater : AesiCMP::less;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i])
                                return (static_cast<long>(blocks[i]) * -1 > static_cast<long>(value.blocks[i]) * -1) ? AesiCMP::greater : AesiCMP::less;

                        return AesiCMP::equal;
                    }
                    case Zero:
                    case Positive: return AesiCMP::less;
                    default: return AesiCMP::equivalent;
                }
            default: return AesiCMP::equivalent;
        }
    };

#if defined(__CUDACC__) || __cplusplus < 202002L || defined (DEVICE_TESTING)
    gpu constexpr auto operator!=(const Aesi& value) const noexcept -> bool { return !this->operator==(value); }
    gpu constexpr auto operator<(const Aesi& value) const noexcept -> bool { return this->compareTo(value) == AesiCMP::less; }
    gpu constexpr auto operator<=(const Aesi& value) const noexcept -> bool { return !this->operator>(value); }
    gpu constexpr auto operator>(const Aesi& value) const noexcept -> bool { return this->compareTo(value) == AesiCMP::greater; }
    gpu constexpr auto operator>=(const Aesi& value) const noexcept -> bool { return !this->operator<(value); }
#else
    gpu constexpr auto operator<=>(const Aesi& value) const noexcept -> std::strong_ordering {
        switch(this->compareTo(value)) {
            case AesiCMP::less: return std::strong_ordering::less;
            case AesiCMP::greater: return std::strong_ordering::greater;
            case AesiCMP::equal: return std::strong_ordering::equal;
            default: return std::strong_ordering::equivalent;
        }
    };
#endif
    /* ----------------------------------------------------------------------- */


    /* ------------------------ Supporting methods. -------------------------- */
    gpu constexpr auto setBit(std::size_t index, bool value) noexcept -> void {
        if(index >= bitness) return;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        if(value) {
            blocks[blockNumber] |= (1U << bitNumber);
            if(sign == Zero && !isLineEmpty(blocks))
                sign = Positive;
        } else {
            blocks[blockNumber] &= (~(1U << bitNumber));
            if(sign != Zero && isLineEmpty(blocks))
                sign = Zero;
        }
    }
    gpu constexpr auto setByte(std::size_t index, uint8_t byte) noexcept -> void {
        if(index > blocksNumber * sizeof(block)) return;

        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        blocks[blockNumber] &= ~(0xffU << shift); blocks[blockNumber] |= static_cast<block>(byte) << shift;

        if(sign != Zero && isLineEmpty(blocks)) sign = Zero;
        if(sign == Zero && !isLineEmpty(blocks)) sign = Positive;
    }
    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool {
        if(index >= bitness) return false;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        return blocks[blockNumber] & (1U << bitNumber);
    }
    [[nodiscard]]
    gpu constexpr auto getByte(std::size_t index) const noexcept -> uint8_t {
        if(index > blocksNumber * sizeof(block)) return 0;
        
        const std::size_t blockNumber = index / sizeof(block), byteInBlock = index % sizeof(block), shift = byteInBlock * bitsInByte;
        return (blocks[blockNumber] & (0xffU << shift)) >> shift;
    }
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
    [[nodiscard]]
    gpu constexpr auto abs() const noexcept -> Aesi {
        if(sign == Zero)
            return *this;
        Aesi result = *this; result.sign = Positive; return result;
    }
    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return (0x1 & blocks[0]) == 1; }
    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return (0x1 & blocks[0]) == 0; }
    [[nodiscard]]
    gpu constexpr auto isZero() const noexcept -> bool { return sign == Zero; }
    [[nodiscard]]
    gpu constexpr auto getBitness() const noexcept -> std::size_t { return bitness; }
    [[nodiscard]]
    gpu constexpr auto squareRoot() const noexcept -> Aesi {
        if(sign != Positive)
            return Aesi {};

        Aesi x {}, y = power2((bitCount() + 1) / 2);

        do {
            x = y;
            y = (x + this->operator/(x)) >> 1;
        } while (y < x);

        return x;
    }
    /* ----------------------------------------------------------------------- */


    /* -------------------- Public number theory functions. ------------------ */
    [[nodiscard]]
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        auto[greater, smaller] = [&first, &second] {
            const auto ratio = first.compareTo(second);
            return ratio == AesiCMP::greater ? pair<Aesi, Aesi> {first, second } : pair<Aesi, Aesi> {second, first };
        } ();
        while(!isLineEmpty(smaller.blocks)) {
            auto [quotient, remainder] = divide(greater, smaller);
            greater = smaller; smaller = remainder;
        }
        return greater;
    }
    [[nodiscard]]
    gpu static constexpr auto lcm(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        return first / gcd(first, second) * second;
    }
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
    [[nodiscard]]
    gpu static constexpr auto power2(std::size_t e) noexcept -> Aesi {
        Aesi result {}; result.setBit(e, true); return result;
    }
    /* ----------------------------------------------------------------------- */


    template <typename Integral> requires (std::is_integral_v<Integral>) [[nodiscard]]
    gpu constexpr auto integralCast() const noexcept -> Integral {
        const uint64_t value = (static_cast<uint64_t>(blocks[1]) << blockBitLength) | static_cast<uint64_t>(blocks[0]);
        if constexpr (std::is_signed_v<Integral>)
            return static_cast<Integral>(value) * (sign == Negative ? -1 : 1); else return static_cast<Integral>(value);
    }

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

    gpu constexpr auto introspect() const noexcept -> void {
        printf("Sign: %d\n", static_cast<int>(sign));
        for(std::size_t i = 0; i < blocksNumber; ++i)
            printf("%u ", blocks[i]); printf("\n");
    }

    template <uint8_t base, typename Char> requires (std::is_same_v<Char, char> || std::is_same_v<Char, wchar_t> &&
            (base == 2 || base == 8 || base == 10 || base == 16))
    gpu constexpr auto getString(Char* const buffer, std::size_t bufferSize, bool showBase = false) const noexcept -> std::size_t {
        if(bufferSize < 2) return 0;

        std::size_t position = 0;

        if (showBase && bufferSize > 3) {
            if constexpr (base == 2) {              // Binary
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0b", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0b", 2 * sizeof(Char));
                }
                position += 2;
            } else if constexpr (base == 8) {       // Octal
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0o", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0o", 2 * sizeof(Char));
                }
                position += 2;
            } else if constexpr (base == 16) {      // Hexadecimal
                if constexpr (std::is_same_v<Char, char>) {
                    memcpy(buffer, "0x", 2 * sizeof(Char));
                } else {
                    memcpy(buffer, L"0x", 2 * sizeof(Char));
                }
                position += 2;
            }                                       // Without base in decimal
        }


        if(sign != Zero) {
            if constexpr (base == 16) {
                long long iter = blocks.size() - 1;
                for (; blocks[iter] == 0 && iter >= 0; --iter);

                if constexpr (std::is_same_v<Char, char>) {
                    position += snprintf(buffer + position, bufferSize - position, "%x", blocks[iter--]);
                    for (; iter >= 0; --iter)
                        position += snprintf(buffer + position, bufferSize - position, "%08x", blocks[iter]);
                } else {
                    position += swprintf(buffer + position, bufferSize - position, L"%x", blocks[iter--]);
                    for (; iter >= 0; --iter)
                        position += swprintf(buffer + position, bufferSize - position, L"%08x", blocks[iter]);
                }
            } else {
                const auto startPosition = position;

                Aesi copy = *this;
                while (copy != 0 && position < bufferSize) {
                    auto [quotient, remainder] = divide(copy, base);
                    if constexpr (std::is_same_v<Char, char>) {
                        buffer[position++] = '0' + remainder.template integralCast<uint8_t>();
                    } else {
                        buffer[position++] = L'0' + remainder.template integralCast<uint8_t>();
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
        buffer[position++] = Char();
        return position;
    }

    template <typename Char>
    constexpr friend auto operator<<(std::basic_ostream<Char>& ss, const Aesi& value) noexcept -> std::basic_ostream<Char>& {
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
                while(copy != 0) {
                    const auto [quotient, remainder] = divide(copy, base);
                    buffer[filled++] = [] { if constexpr (std::is_same_v<Char, char>) { return '0'; } else { return L'0'; } } () + remainder.template integralCast<uint8_t>();
                    copy = quotient;
                }

                for(; filled > 0; --filled)
                    ss << buffer[filled - 1];
            }
        } else ss.write([] { if constexpr (std::is_same_v<Char, char>) { return "0"; } else { return L"0"; } } (), 1);

        return ss;
    }

#ifdef __CUDACC__
    __device__ constexpr auto atomicSet(const Aesi& value) noexcept -> void {
        sign = value.sign;
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&blocks[i], value.blocks[i]);
    }

    __device__ constexpr auto atomicExchange(const Aesi& value) noexcept -> void {
        Sign tSign = sign; sign = value.sign; value.sign = tSign;
        for(std::size_t i = 0; i < blocksNumber; ++i)
            atomicExch(&value.blocks[i], atomicExch(&blocks[i], value.blocks[i]));
    }
#endif
};

using Aesi128 = Aesi<128>;
using Aesi256 = Aesi<256>;
using Aesi512 = Aesi<512>;
using Aesi1024 = Aesi<1024>;
using Aesi2048 = Aesi<2048>;
using Aesi4096 = Aesi<4096>;
using Aesi8192 = Aesi<8192>;

#include "PrecisionCast.h"

#endif //AESI_MULTIPRECISION
