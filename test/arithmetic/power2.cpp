#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>

TEST(Signed, Power2) {
    constexpr std::size_t N = 512;
    for (std::size_t i = 0; i < N; ++i) {
        mpz_class power2 {}, two = 2;
        mpz_pow_ui(power2.get_mpz_t(), two.get_mpz_t(), i);

        std::stringstream ss; ss << "0x" << std::hex << power2;
        EXPECT_EQ(Aesi<N>::power2(i), ss.str());
    }
    EXPECT_EQ(Aesi<N>::power2(512), 0u);
}

TEST(Unsigned, Power2) {
    constexpr std::size_t N = 512;
    for (std::size_t i = 0; i < N; ++i) {
        mpz_class power2 {}, two = 2;
        mpz_pow_ui(power2.get_mpz_t(), two.get_mpz_t(), i);

        std::stringstream ss; ss << "0x" << std::hex << power2;
        EXPECT_EQ(Aeu<N>::power2(i), ss.str());
    }
    EXPECT_EQ(Aeu<N>::power2(512), 0u);
}
