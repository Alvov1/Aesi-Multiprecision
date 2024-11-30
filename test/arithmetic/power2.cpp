#include <gtest/gtest.h>
#include "../../Aeu.h"
#include "../../Aesi.h"
#include <cryptopp/integer.h>

TEST(Signed, Power2) {
    constexpr std::size_t N = 512;
    for (std::size_t i = 0; i < N; ++i) {
        std::stringstream ss; ss << "0x" << std::hex << CryptoPP::Integer::Power2(i);
        EXPECT_EQ(Aesi<N>::power2(i), ss.str());
    }
    EXPECT_EQ(Aesi<N>::power2(512), 0u);
}

TEST(Unsigned, Power2) {
    constexpr std::size_t N = 512;
    for (std::size_t i = 0; i < N; ++i) {
        std::stringstream ss; ss << "0x" << std::hex << CryptoPP::Integer::Power2(i);
        EXPECT_EQ(Aeu<N>::power2(i), ss.str());
    }
    EXPECT_EQ(Aeu<N>::power2(512), 0u);
}
