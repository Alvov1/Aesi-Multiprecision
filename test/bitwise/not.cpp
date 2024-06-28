#include <gtest/gtest.h>
#include <format>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Bitwise, NOT) {
    Aeu256 m0 = "56061994118377870519623994827540022144637583439516706780608137993940235344250";
    EXPECT_EQ(~m0, "0b1000010000001110000010010010100111110110100011100011100001100101110011010110001000110101010100001111001010110101000101111100001010101001100101111000000110100011100011011100101100010100001010101110111011111110111000000100101011010100000111011010011010000101");

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
