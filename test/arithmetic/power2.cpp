#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include <cryptopp/integer.h>

TEST(Unsigned, Power2) {
    for (std::size_t i = 0; i < 511; ++i) {
        std::stringstream ss; ss << "0x" << std::hex << CryptoPP::Integer::Power2(i);
        EXPECT_EQ(Aeu512::power2(i), ss.str());
    }
    EXPECT_EQ(Aeu512::power2(512), 0u);
}