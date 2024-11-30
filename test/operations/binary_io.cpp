#include <gtest/gtest.h>
#include <bitset>
#include <iomanip>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_BinaryIO, BinaryRead) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (unsigned i = 0; i < testsAmount; i += 2) {
        std::array<uint32_t, blocksNumber> blocks {};
        std::stringstream ss, ss2; ss2 << "0x" << std::hex;

        Aeu<blocksNumber * 32> a {};
        if(i % 2 == 0) {
            for(auto& block: blocks) {
                block = Generation::getRandom<uint32_t>();
                ss.write(reinterpret_cast<const char*>(&block), sizeof(block));
                ss2 << std::setw(8) << std::setfill('0') << block;
            }

            a.readBinary(ss, true);
        } else {
            for(auto& block: blocks) {
                block = Generation::getRandom<uint32_t>();
                ss2 << std::hex << block;
            }

            for(long long j = blocks.size() - 1; j >= 0; --j)
                ss.write(reinterpret_cast<const char*>(&blocks[j]), sizeof(uint32_t));

            a.readBinary(ss, false);
        }

        EXPECT_EQ(a, ss2.str());
    }
}

TEST(Unsigned_BinaryIO, BinaryWrite) {
    constexpr auto testsAmount = 1024, blocksNumber = 32;
    for (unsigned i = 0; i < testsAmount; ++i) {
        std::stringstream ss, ss2; ss << std::hex << "0x";
        for (std::size_t j = 0; j < blocksNumber; ++j)
            ss << Generation::getRandom<uint32_t>();

        Aeu<blocksNumber * 32> l = ss.str(), r {};
        uint32_t temp;

        if(i % 2 == 0) {
            l.writeBinary(ss2, false);
            for(std::size_t j = 0; j < blocksNumber; ++j) {
                ss2.read(reinterpret_cast<char*>(&temp), sizeof(uint32_t));
                r.setBlock(j, temp);
            }
        } else {
            l.writeBinary(ss2, true);
            for (long long j = blocksNumber - 1; j >= 0; --j) {
                ss2.read(reinterpret_cast<char*>(&temp), sizeof(uint32_t));
                r.setBlock(j, temp);
            }
        }

        EXPECT_EQ(l, r);
    }
}
