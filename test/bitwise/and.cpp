#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../generation.h"

TEST(Unsigned_Bitwise, AND) {
    {
        Aeu256 l = "0x35be68303d81dc1f00214fe48c3ae43ac6916469b38ec7b9c84d79b7917c32f", r = "0xd665ef66d30ee8c97dda704aeb80133a5ba02aaa00da0335ef91e24ff54909"; EXPECT_EQ(l & r, "0x52648302d00cc0c000107048c38003284900028a00c8031484918249154109");
        l = "0x294ffdeb5690a1c9ff8797a2522fbfde22377d73da54a402a8534493076a272", r = "0x24c01426477a48f6b5c457831014d61ded49703735af04057fa21d4f9e5d4d1"; l &= r; EXPECT_EQ(l, "0x20401422461000c0b58417821004961c2001703310040400280204030648050");
    }
    constexpr auto testsAmount = 2048, blocksNumber = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto l = Generation::getRandomWithBits(blocksNumber * 32 - 20),
                r = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        Aeu<blocksNumber * 32> left = l, right = r;
        EXPECT_EQ(left & right, l & r);

        l = Generation::getRandomWithBits(blocksNumber * 32 - 20),
                r = Generation::getRandomWithBits(blocksNumber * 32 - 20);
        left = l, right = r; left &= right;
        EXPECT_EQ(left, l & r);
    }
}