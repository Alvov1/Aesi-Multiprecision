#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include "../generation.h"

TEST(Signed_SquareRoot, SquareRoot) { EXPECT_TRUE(false); }

TEST(Unsigned_SquareRoot, SquareRoot) {
    for (unsigned i = 0; i < 512; ++i) {
        const auto value = Generation::getRandomWithBits(500);

        std::stringstream ss1 {}, ss2 {};
        ss1 << std::hex << "0x" << value;
        ss2 << std::hex << "0x" << value.SquareRoot();

        Aeu512 m = ss1.str();
        EXPECT_EQ(m.squareRoot(), Aeu512(ss2.str()));
    }
}