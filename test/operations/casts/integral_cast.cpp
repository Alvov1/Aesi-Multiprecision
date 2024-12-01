#include <gtest/gtest.h>
#include <iomanip>
#include "../../../Aeu.h"
#include "../../../Aesi.h"
#include "../../generation.h"

TEST(Signed_IntegralCast, IntegralCast) {
    constexpr auto testsAmount = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandom<int64_t>();
        Aesi<128> aesi = value;
        EXPECT_EQ(aesi.integralCast<int64_t>(), value);

        value = Generation::getRandom<int32_t>();
        aesi = value;
        EXPECT_EQ(aesi.integralCast<int32_t>(), value);
    }
}

TEST(Unsigned_IntegralCast, IntegralCast) {
    constexpr auto testsAmount = 64;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        auto value = Generation::getRandom<uint64_t>();
        Aeu<128> aeu = value;
        EXPECT_EQ(aeu.integralCast<uint64_t>(), value);

        value = Generation::getRandom<uint32_t>();
        aeu = value;
        EXPECT_EQ(aeu.integralCast<uint32_t>(), value);
    }

    constexpr auto blocksNumber = 32;
    for (std::size_t i = 0; i < testsAmount; ++i) {
        const auto baseValue = Generation::getRandom<uint64_t>();
        std::stringstream ss; ss << "0x" << std::hex << Generation::getRandomWithBits(blocksNumber * 32 - 96)
            << std::setw(16) << std::setfill('0') << baseValue;
        const auto& ref = ss.str();
        Aeu<blocksNumber * 32> aeu = ss.str();
        EXPECT_EQ(aeu.integralCast<uint64_t>(), baseValue);
    }
}