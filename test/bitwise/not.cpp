#include <gtest/gtest.h>
#include <bitset>
#include <AesiMultiprecision/Aeu.h>
#include "../generation.h"

TEST(Unsigned_Bitwise, NOT) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            mpz_class value = Generation::getRandom(N);
            const std::size_t bitCount = mpz_sizeinbase(value.get_mpz_t(), 2);
            if(bitCount < N)
                value <<= (N - bitCount);

            const std::size_t byteCount = (mpz_sizeinbase(value.get_mpz_t(), 2) + 7) / 8;
            auto getByteGmp = [&](std::size_t k) -> unsigned char {
                return static_cast<unsigned char>(mpz_class(value >> (8 * k)).get_ui() & 0xFF);
            };

            std::stringstream ss {};
            std::string binary {};
            for (auto byte = static_cast<uint8_t>(~getByteGmp(byteCount - 1)); byte; byte >>= 1)
                binary += (byte & 1 ? '1' : '0');
            ss << "0b" << std::string(binary.rbegin(), binary.rend());
            for(std::size_t j = byteCount - 1; j-- > 0;)
                ss << std::bitset<8>(static_cast<uint8_t>(~getByteGmp(j)));

            Aeu<N> aeu = value, notted = ss.str();
            EXPECT_EQ(~aeu, notted);
        }
    });
}