#include <gtest/gtest.h>
#include <AesiMultiprecision/Aeu.h>
#include <AesiMultiprecision/Aesi.h>
#include "../../generation.h"

TEST(Unsigned_Bitwise, GetSetBit) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 20);
            const std::size_t bitCount = mpz_sizeinbase(value.get_mpz_t(), 2);
            Aeu<N> aeu {};
            for (std::size_t j = 0; j < bitCount; ++j)
                aeu.setBit(j, mpz_tstbit(value.get_mpz_t(), j));
            EXPECT_EQ(aeu, value);

            aeu = value;
            for (std::size_t j = 0; j < bitCount; ++j)
                EXPECT_EQ(aeu.getBit(j), (bool)mpz_tstbit(value.get_mpz_t(), j));
        }
    });
}

TEST(Unsigned_Bitwise, GetSetByte) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 20);
            const std::size_t byteCount = (mpz_sizeinbase(value.get_mpz_t(), 2) + 7) / 8;
            auto getByteGmp = [&](std::size_t k) -> unsigned char {
                return (unsigned char)(mpz_class(value >> (8 * k)).get_ui() & 0xFF);
            };
            Aeu<N> aeu {};
            for (std::size_t j = 0; j < byteCount; ++j)
                aeu.setByte(j, getByteGmp(j));
            EXPECT_EQ(aeu, value);

            aeu = value;
            for (std::size_t j = 0; j < byteCount; ++j)
                EXPECT_EQ(aeu.getByte(j), getByteGmp(j));
        }
    });
}

TEST(Unsigned_Bitwise, GetSetBlock) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 20);
            const std::size_t byteCount = (mpz_sizeinbase(value.get_mpz_t(), 2) + 7) / 8;
            auto getByteGmp = [&](std::size_t k) -> unsigned char {
                return (unsigned char)(mpz_class(value >> (8 * k)).get_ui() & 0xFF);
            };
            Aeu<N> aeu {};

            const auto totalBlocks = byteCount / 4;
            for (std::size_t j = 0; j < totalBlocks; ++j) {
                uint32_t block = 0;
                for (std::size_t k = 1; k < 5; ++k) {
                    const auto byte = static_cast<uint32_t>(getByteGmp((j + 1) * 4 - k));
                    block <<= 8;
                    block |= byte;
                }
                aeu.setBlock(j, block);
            }

            uint32_t block = 0;
            for (std::size_t j = byteCount - 1; j >= totalBlocks * 4; --j) {
                const auto byte = static_cast<uint32_t>(getByteGmp(j));
                block <<= 8;
                block |= byte;
            }
            aeu.setBlock(totalBlocks, block);

            EXPECT_EQ(aeu, value);
        }
    });
}

TEST(Unsigned_Bitwise, CountBitsBytes) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const auto value = Generation::getRandom(N - 20);
            Aeu<N> aeu = value;
            EXPECT_EQ(mpz_sizeinbase(value.get_mpz_t(), 2), aeu.bitCount());
        }
    });
}

TEST(Signed_Bitwise, GetSetBit) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class value = (i % 2 == 0 ? -1 : 1) * Generation::getRandom(N - 20);
            mpz_class absVal; mpz_abs(absVal.get_mpz_t(), value.get_mpz_t());
            const std::size_t bitCount = mpz_sizeinbase(absVal.get_mpz_t(), 2);

            Aesi<N> aeu = 1;
            for (std::size_t j = 0; j < bitCount; ++j)
                aeu.setBit(j, mpz_tstbit(absVal.get_mpz_t(), j));
            if(i % 2 == 0) aeu.inverse();
            EXPECT_EQ(aeu, value);

            aeu = value;
            for (std::size_t j = 0; j < bitCount; ++j)
                EXPECT_EQ(aeu.getBit(j), (bool)mpz_tstbit(absVal.get_mpz_t(), j));
        }
    });
}

TEST(Signed_Bitwise, GetSetByte) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class value = (i % 2 == 0 ? -1 : 1) * Generation::getRandom(N - 20);
            mpz_class absVal; mpz_abs(absVal.get_mpz_t(), value.get_mpz_t());
            const std::size_t byteCount = (mpz_sizeinbase(absVal.get_mpz_t(), 2) + 7) / 8;
            auto getByteGmp = [&](std::size_t k) -> unsigned char {
                return (unsigned char)(mpz_class(absVal >> (8 * k)).get_ui() & 0xFF);
            };

            Aesi<N> aeu = 1;
            for (std::size_t j = 0; j < byteCount; ++j)
                aeu.setByte(j, getByteGmp(j));
            if(i % 2 == 0) aeu.inverse();
            EXPECT_EQ(aeu, value);

            aeu = value;
            for (std::size_t j = 0; j < byteCount; ++j)
                EXPECT_EQ(aeu.getByte(j), getByteGmp(j));
        }
    });
}

TEST(Signed_Bitwise, GetSetBlock) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class value = (i % 2 == 0 ? -1 : 1) * Generation::getRandom(N - 20);
            mpz_class absVal; mpz_abs(absVal.get_mpz_t(), value.get_mpz_t());
            const std::size_t byteCount = (mpz_sizeinbase(absVal.get_mpz_t(), 2) + 7) / 8;
            auto getByteGmp = [&](std::size_t k) -> unsigned char {
                return (unsigned char)(mpz_class(absVal >> (8 * k)).get_ui() & 0xFF);
            };
            Aesi<N> aesi = 1;

            const auto totalBlocks = byteCount / 4;
            for (std::size_t j = 0; j < totalBlocks; ++j) {
                uint32_t block = 0;
                for (std::size_t k = 0; k < 5; ++k) {
                    const auto byte = static_cast<uint32_t>(getByteGmp((j + 1) * 4 - k));
                    block <<= 8;
                    block |= byte;
                }
                aesi.setBlock(j, block);
            }

            uint32_t block = 0;
            for (std::size_t j = byteCount - 1; j >= totalBlocks * 4; --j) {
                const auto byte = static_cast<uint32_t>(getByteGmp(j));
                block <<= 8;
                block |= byte;
            }
            aesi.setBlock(totalBlocks, block);

            if(i % 2 == 0) aesi.inverse();

            EXPECT_EQ(aesi, value);
        }
    });
}

TEST(Signed_Bitwise, CountBitsBytes) {
    Generation::forEachPrecision([]<std::size_t N>() {
        constexpr auto testsAmount = 256;
        for (std::size_t i = 0; i < testsAmount; ++i) {
            const mpz_class value = (i % 2 == 0 ? 1 : -1) * Generation::getRandom(N - 20);
            Aesi<N> aeu = value;
            EXPECT_EQ(mpz_sizeinbase(value.get_mpz_t(), 2), aeu.bitCount());
        }
    });
}