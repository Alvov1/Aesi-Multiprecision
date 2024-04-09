#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../benchmarks/benchmarks.h"

TEST(Initialization, Binary) {
    Aesi512 m0 = 0b1111111111111111111111111111111111111111111111111111111111111111;
    EXPECT_EQ(m0, 18446744073709551615ULL);

    Aesi512 m1 = -0b100001100011011110111101000001011010111101101;
    EXPECT_EQ(m1, -18446744073709);

    Aesi512 m2 = "0b11011001001110000000010000100010101011011101010101000011111";
    EXPECT_EQ(m2, 489133282872437279);

    Aesi512 m3 = "-0b111010101100001010101111001110111001";
    EXPECT_EQ(m3, -63018038201);

    Aesi512 m4 = "0b1010101010101010101010101";
    EXPECT_EQ(m4, 22369621);

    Aesi512 m5 = "-0b10101001100010101001";
    EXPECT_EQ(m5, -694441);
}

TEST(Initialization, Decimal) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m0 = 99194853094755497;
    EXPECT_EQ(m0, 99194853094755497);

    Aesi512 m1 = -2971215073;
    EXPECT_EQ(m1, -2971215073);

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());

    EXPECT_TRUE(false); // TODO: Previous version was without sign check
}

TEST(Initialization, Octal) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m0 = 05403223057620506251;
    EXPECT_EQ(m0, 99194853094755497);

    Aesi512 m1 = 026106222341;
    EXPECT_EQ(m1, 2971215073);

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());

    EXPECT_TRUE(false); // TODO: Previous version was without sign check
}

TEST(Initialization, Hexadecimal) {
    const auto timeStart = std::chrono::system_clock::now();

    Aesi512 m0 = 0xFFFFFFFFFFFFFFFF;
    EXPECT_EQ(m0, 18446744073709551615ULL);

    Aesi512 m1 = 0x191347024000932;
    EXPECT_EQ(m1, 112929121905936690);

#ifdef NDEBUG
    Logging::addRecord("Hexadecimal_initialization",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */

    EXPECT_TRUE(false); // TODO: Previous version was without sign check
}