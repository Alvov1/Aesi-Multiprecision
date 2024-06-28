#include <gtest/gtest.h>
#include "../../../Aeu.h"
#include "../../generation.h"

TEST(Unsigned_Bitwise, GetSetBit) {
    constexpr auto testsAmount = 2048, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(bitness - 20);
        Aeu<bitness> aeu {};
        for (std::size_t j = 0; j < value.BitCount(); ++j)
            aeu.setBit(j, value.GetBit(j));
        EXPECT_EQ(aeu, value);

        aeu = value;
        for (std::size_t j = 0; j < value.BitCount(); ++j)
            EXPECT_EQ(aeu.getBit(j), value.GetBit(j));
    }
}

TEST(Unsigned_Bitwise, GetSetByte) {
    constexpr auto testsAmount = 2048, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(bitness - 20);
        Aeu<bitness> aeu {};
        for (std::size_t j = 0; j < value.ByteCount(); ++j)
            aeu.setByte(j, value.GetByte(j));
        EXPECT_EQ(aeu, value);

        aeu = value;
        for (std::size_t j = 0; j < value.ByteCount(); ++j)
            EXPECT_EQ(aeu.getByte(j), value.GetByte(j));
    }
}

TEST(Unsigned_Bitwise, GetSetBlock) {
    constexpr auto testsAmount = 2048, bitness = 256;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(bitness - 20);
        Aeu<bitness> aeu {};

        const auto totalBlocks = value.ByteCount() / 4;
        for (std::size_t j = 0; j < totalBlocks; ++j) {
            uint32_t block = 0;
            for (std::size_t k = 1; k < 5; ++k) {
                const auto byte = static_cast<uint32_t>(value.GetByte((j + 1) * 4 - k));
                block <<= 8;
                block |= byte;
            }
            aeu.setBlock(j, block);
        }

        uint32_t block = 0;
        for (std::size_t j = value.ByteCount() - 1; j >= totalBlocks * 4; --j) {
            const auto byte = static_cast<uint32_t>(value.GetByte(j));
            block <<= 8;
            block |= byte;
        }
        aeu.setBlock(totalBlocks, block);

        EXPECT_EQ(aeu, value);
    }
}

TEST(Unsigned_Bitwise, CountBitsBytes) {
    constexpr auto testsAmount = 2048, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = Generation::getRandomWithBits(bitness - 20);
        Aeu<bitness> aeu = value;
        EXPECT_EQ(value.BitCount(), aeu.bitCount());
    }
}