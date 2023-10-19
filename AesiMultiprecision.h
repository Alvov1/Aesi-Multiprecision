#ifndef MULTIPRECISION_GPU_H
#define MULTIPRECISION_GPU_H

#include <iostream>

#ifdef __CUDACC__
    #define gpu __host__ __device__
    #include <cuda/std/utility>
#else
    #define gpu
    #include <utility>
#endif

namespace {
    using block = uint32_t;
    constexpr auto blockBitLength = sizeof(block) * 8;
    constexpr uint64_t blockBase = 1ULL << blockBitLength;
}

template <std::size_t bitness = 512> requires (bitness % blockBitLength == 0)
class Aesi final {
    static_assert(bitness > sizeof(uint64_t), "Use built-in types for numbers 64-bit or less.");

    static constexpr std::size_t blocksNumber = bitness / blockBitLength;

    template <typename ValueType, std::size_t lineSize>
    struct MyArray final {
        ValueType data [lineSize] {};
        gpu constexpr bool operator==(const MyArray& value) const noexcept {
            for(std::size_t i = 0; i < lineSize; ++i)
                if(data[i] != value.data[i]) return false; return true;
        };
        [[nodiscard]] gpu constexpr auto size() const noexcept -> std::size_t { return lineSize; };
        gpu constexpr auto operator[](std::size_t index) const noexcept -> const ValueType& { return data[index]; }
        gpu constexpr auto operator[](std::size_t index) noexcept -> ValueType& { return data[index]; }
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
    gpu static constexpr auto divide(const Aesi& number, const Aesi& divisor) noexcept -> std::pair<Aesi, Aesi> {
        const Aesi divAbs = divisor.abs();
        const auto ratio = number.abs().operator<=>(divAbs);

        std::pair<Aesi, Aesi> results = {0, 0 };
        auto& [quotient, remainder] = results;

        if(ratio == std::strong_ordering::greater) {
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
        } else if(ratio == std::strong_ordering::less)
            remainder = number; else quotient = 1;

        return results;
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
    constexpr Aesi(String&& stringView) noexcept : Aesi(stringView.data(), stringView.size()){}

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
    gpu constexpr bool operator==(const Aesi& value) const noexcept {
        if(sign != Zero || value.sign != Zero)
            return (sign == value.sign && blocks == value.blocks); else return true;
    };
    gpu constexpr std::strong_ordering operator<=>(const Aesi& value) const noexcept {
        switch (sign) {
            case Zero:
                switch (value.sign) {
                    case Zero: return std::strong_ordering::equal;
                    case Positive: return std::strong_ordering::less;
                    case Negative: return std::strong_ordering::greater;
                    default: return std::strong_ordering::equivalent;
                }
            case Positive:
                switch (value.sign) {
                    case Positive: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength) return thisLength <=> valueLength;

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i]) return blocks[i] <=> value.blocks[i];

                        return std::strong_ordering::equal;
                    }
                    case Zero:
                    case Negative: return std::strong_ordering::greater;
                    default: return std::strong_ordering::equivalent;
                }
            case Negative:
                switch (value.sign) {
                    case Negative: {
                        const auto thisLength = lineLength(blocks), valueLength = lineLength(value.blocks);
                        if(thisLength != valueLength) return (static_cast<long long>(thisLength) * -1) <=> (static_cast<long long>(valueLength) * -1);

                        for(long long i = thisLength; i >= 0; --i)
                            if(blocks[i] != value.blocks[i]) return (static_cast<long>(blocks[i]) * -1) <=> (static_cast<long>(value.blocks[i]) * -1);

                        return std::strong_ordering::equal;
                    }
                    case Zero:
                    case Positive: return std::strong_ordering::less;
                    default: return std::strong_ordering::equivalent;
                }
            default: return std::strong_ordering::equivalent;
        }
    };
    /* ----------------------------------------------------------------------- */


    /* ------------------------ Arithmetic methods. -------------------------- */
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
    [[nodiscard]]
    gpu constexpr auto getBit(std::size_t index) const noexcept -> bool {
        if(index >= bitness) return false;
        const std::size_t blockNumber = index / blockBitLength, bitNumber = index % blockBitLength;
        return blocks[blockNumber] & (1U << bitNumber);
    }
    [[nodiscard]]
    gpu constexpr auto isOdd() const noexcept -> bool { return (0x1 & blocks[0]) == 1; }
    [[nodiscard]]
    gpu constexpr auto isEven() const noexcept -> bool { return (0x1 & blocks[0]) == 0; }
    [[nodiscard]]
    gpu constexpr auto abs() const noexcept -> Aesi {
        if(sign == Zero)
            return *this;
        Aesi result = *this; result.sign = Positive; return result;
    }
    /* ----------------------------------------------------------------------- */


    /* -------------------- Public number theory functions. ------------------ */
    gpu static constexpr auto gcd(const Aesi& first, const Aesi& second) noexcept -> Aesi {
        auto[greater, smaller] = [&first, &second] {
            const auto ratio = first.operator<=>(second);
            return ratio == std::strong_ordering::greater ?
                   std::pair { first, second }
                                                          :
                   std::pair { second, first };
        } ();
        while(!isLineEmpty(smaller.blocks)) {
            auto [quotient, remainder] = divide(greater, smaller);
            greater = smaller; smaller = remainder;
        }
        return greater;
    }
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

    constexpr friend std::ostream& operator<<(std::ostream& ss, const Aesi& value) noexcept {
        auto flags = ss.flags();

        if(value.sign != Zero) {
            if (value.sign == Negative) ss.write("-", 1);

            const auto base = [] (long baseField, std::ostream& ss, bool showbase) {
                auto base = (baseField == std::ios::hex ? 16 : (baseField == std::ios::oct ? 8 : 10));
                if(showbase && base != 10) ss.write(base == 8 ? "0o" : "0x", 2);
                return base;
            } (flags & std::ios::basefield, ss, flags & std::ios::showbase);

            long long iter = value.blocks.size() - 1;
            for(; value.blocks[iter] == 0 && iter >= 0; --iter);

            if(base == 16) {
                ss << value.blocks[iter--];
                for (; iter >= 0; --iter) {
                    ss.fill('0'); ss.width(8); ss << std::right << value.blocks[iter];
                }
            } else {
                /* Well, here we use a pre-calculated magic number to ratio the lengths of numbers in decimal or octal notation according to bitness.
                 * It is 2.95-98 for octal and 3.2 for decimal. */
                constexpr auto bufferSize = static_cast<std::size_t>(static_cast<double>(bitness) / 2.95);
                MyArray<char, bufferSize> buffer {}; std::size_t filled = 0;

                Aesi copy = value;
                while(copy != 0) {
                    auto [quotient, remainder] = divide(copy, base);
                    buffer[filled++] = '0' + remainder.template integralCast<unsigned long>();
                    copy = quotient;
                }

                for(; filled > 0; --filled)
                    ss << buffer[filled - 1];
            }
        } else ss.write("0", 1);

        return ss;
    }
};

/* ---------------------------------------- Different precision comparison ---------------------------------------- */
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr bool operator==(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        Aesi<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return (first == reducedSecond);
    } else {
        Aesi<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return (reducedFirst == second);
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr std::strong_ordering operator<=>(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept {
    if constexpr (bFirst > bSecond) {
        Aesi<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return (first <=> reducedSecond);
    } else {
        Aesi<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return (reducedFirst <=> second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision addition ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator+(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) + value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator+(const Aesi<bFirst>& first, const Aesi<bSecond>& second) noexcept
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        Aesi<bFirst> reducedSecond = second.template precisionCast<bFirst>();
        return first + reducedSecond;
    } else {
        Aesi<bSecond> reducedFirst = first.template precisionCast<bSecond>();
        return reducedFirst + second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator+=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator+=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* --------------------------------------- Different precision subtraction ---------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator-(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) - value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator-(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first - second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() - second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator-=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator-=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------- Different precision multiplication -------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator*(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) * value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator*(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first * second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() * second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator*=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator*=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------- Different precision division ----------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator/(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) / value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator/(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first / second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() / second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator/=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator/=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------ Different precision modulo ------------------------------------------ */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator%(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) % value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator%(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first % second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() % second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator%=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator%=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Different precision XOR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator^(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) ^ value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator^(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first ^ second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() ^ second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator^=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator^=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision AND ------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator&(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) & value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator&(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first & second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() & second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator&=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator&=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* -------------------------------------------- Different precision OR -------------------------------------------- */
template <std::size_t bitness, typename Integral> requires (std::is_integral_v<Integral>)
gpu constexpr auto operator|(Integral number, const Aesi<bitness>& value) noexcept {
    return Aesi<bitness>(number) | value;
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst != bSecond)
gpu constexpr auto operator|(const Aesi<bFirst>& first, const Aesi<bSecond>& second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return first | second.template precisionCast<bFirst>();
    } else {
        return first.template precisionCast<bSecond>() | second;
    }
}
template <std::size_t bFirst, std::size_t bSecond> requires (bFirst > bSecond)
gpu constexpr auto operator|=(Aesi<bFirst>& first, const Aesi<bSecond>& second) -> Aesi<bFirst>& {
    return first.operator|=(second.template precisionCast<bFirst>());
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ------------------------------------------- Greatest common divisor -------------------------------------------- */
template<std::size_t bFirst, std::size_t bSecond>
gpu constexpr auto gcd(const Aesi<bFirst> &first, const Aesi<bSecond> &second)
-> typename std::conditional<(bFirst > bSecond), Aesi<bFirst>, Aesi<bSecond>>::type {
    if constexpr (bFirst > bSecond) {
        return Aesi<bFirst>::gcd(first, second.template precisionCast<bFirst>());
    } else {
        return Aesi<bSecond>::gcd(first.template precisionCast<bSecond>(), second);
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */


/* ----------------------------------------------- Power by modulo ------------------------------------------------ */
template<std::size_t bBase, std::size_t bPow, std::size_t bMod>
gpu constexpr auto powm(const Aesi<bBase> &base, const Aesi<bPow> &power, const Aesi<bMod> &mod)
-> typename std::conditional<(bBase > bPow),
        typename std::conditional<(bBase > bMod), Aesi<bBase>, Aesi<bMod>>::value,
        typename std::conditional<(bPow > bMod), Aesi<bPow>, Aesi<bMod>>::value>::value {
    if constexpr (bBase > bPow) {
        return powm(base, power.template precisionCast<bBase>(), mod);
    } else {
        return powm(base.template precisionCast<bPow>(), power, mod);
    }
}

namespace {
    template<std::size_t bCommon, std::size_t bDiffer>
    gpu constexpr auto powm(const Aesi<bCommon> &base, const Aesi<bCommon> &power, const Aesi<bDiffer> &mod)
    -> typename std::conditional<(bCommon > bDiffer), Aesi<bCommon>, Aesi<bDiffer>>::type {
        if constexpr (bCommon > bDiffer) {
            return Aesi<bCommon>::powm(base, power, mod.template precisionCast<bCommon>());
        } else {
            return Aesi<bDiffer>::powm(base.template precisionCast<bDiffer>(),
                                       power.template precisionCast<bDiffer>(), mod);
        }
    }
}
/* ---------------------------------------------------------------------------------------------------------------- */

#endif //MULTIPRECISION_GPU_H
