#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_Bitwise, GetSetBit) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? -1 : 1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aesi<blocksNumber * 32> aeu = 1;
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
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? -1 : 1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aesi<blocksNumber * 32> aeu = 1;
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
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? -1 : 1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aesi<blocksNumber * 32> aesi = 1;

        const auto totalBlocks = value.ByteCount() / 4,
                remainingBytes = value.ByteCount() % 4;
        for (std::size_t j = 0; j < totalBlocks; ++j) {
            uint32_t block = 0;
            for (std::size_t k = 0; k < 5; ++k) {
                const auto byte = static_cast<uint32_t>(value.GetByte((j + 1) * 4 - k));
                block <<= 8;
                block |= byte;
            }
            aesi.setBlock(j, block);
        }

        uint32_t block = 0;
        for (std::size_t j = value.ByteCount() - 1; j >= totalBlocks * 4; --j) {
            const auto byte = static_cast<uint32_t>(value.GetByte(j));
            block <<= 8;
            block |= byte;
        }
        aesi.setBlock(totalBlocks, block);

        if(i % 2 == 0) aesi.inverse();

        EXPECT_EQ(aesi, value);
    }
}

TEST(Signed_Bitwise, CountBitsBytes) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto value = (i % 2 == 0 ? 1 : -1) * Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aesi<blocksNumber * 32> aeu = value;
        EXPECT_EQ(value.BitCount(), aeu.bitCount());
    }
}