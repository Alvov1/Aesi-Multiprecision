#include <gtest/gtest.h>
#include <format>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, NOT) {
    constexpr auto testsAmount = 2048, blocksNumber = 8;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandomWithBits(blocksNumber * 32);
        if(value.BitCount() < blocksNumber * 32)
            value <<= (blocksNumber * 32 - value.BitCount());

        std::stringstream ss {};
        ss << std::format("0b{:b}", static_cast<uint8_t>(~value.GetByte(value.ByteCount() - 1)));
        for(long long j = value.ByteCount() - 2; j >= 0; --j)
            ss << std::bitset<8>(~value.GetByte(j));

        Aeu<blocksNumber * 32> aeu = value, notted = ss.str();
        EXPECT_EQ(~aeu, notted);
    }
}
