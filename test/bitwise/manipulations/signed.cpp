#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Bitwise, GetSetBit) {
    constexpr auto testsAmount = 2, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(bitness - 20);
        Aesi<bitness> aeu {};
        for (std::size_t j = 0; j < value.BitCount(); ++j)
            aeu.setBit(j, value.GetBit(j));
        if(i % 2 == 0) aeu.inverse();
        EXPECT_EQ(aeu, value);

        aeu = value;
        for (std::size_t j = 0; j < value.BitCount(); ++j)
            EXPECT_EQ(aeu.getBit(j), value.GetBit(j));
    }
}

TEST(Signed_Bitwise, GetSetByte) {
    constexpr auto testsAmount = 2, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(bitness - 20);
        Aesi<bitness> aeu {};
        for (std::size_t j = 0; j < value.ByteCount(); ++j)
            aeu.setByte(j, value.GetByte(j));
        if(i % 2 == 0) aeu.inverse();
        EXPECT_EQ(aeu, value);

        aeu = value;
        for (std::size_t j = 0; j < value.ByteCount(); ++j)
            EXPECT_EQ(aeu.getByte(j), value.GetByte(j));
    }
}

TEST(Signed_Bitwise, GetSetBlock) {
    constexpr auto testsAmount = 2, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(bitness - 20);
        Aesi<bitness> aeu {};

        const auto totalBlocks = value.ByteCount() / 4,
                remainingBytes = value.ByteCount() % 4;
        for (std::size_t j = 0; j < totalBlocks; ++j) {
            uint32_t block = 0;
            for (std::size_t k = 0; k < 4; ++k) {
                block |= value.GetByte(j * 4 + k);
                block <<= 8;
            }
            aeu.setBlock(j, block);
        }

        uint32_t block = 0;
        for (std::size_t j = 0; j < remainingBytes; ++j) {
            block |= value.GetByte(totalBlocks * 4 + j);
            block <<= 8;
        }
        aeu.setBlock(totalBlocks, block);

        if(i % 2 == 0) aeu.inverse();

        EXPECT_EQ(aeu, value);
    }
}

TEST(Signed_Bitwise, CountBitsBytes) {
    constexpr auto testsAmount = 2, bitness = 2048;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(bitness - 20);
        Aesi<bitness> aeu = value;
        EXPECT_EQ(value.BitCount(), aeu.bitCount());
    }
}