#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../benchmarks/benchmarks.h"

TEST(Bitwise, Unsigned_GetSetBit) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(),
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Bitwise, Unsigned_GetSetByte) {
    const auto timeStart = std::chrono::system_clock::now();

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(), std::chrono::system_clock::to_time_t(timeStart), (std::chrono::system_clock::now() - timeStart).count());;
}

TEST(Bitwise, Unsigned_GetSetBlock) {

}

TEST(Bitwise, Unsigned_CountBitsBytes) {
    const auto timeStart = std::chrono::system_clock::now();

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(), std::chrono::system_clock::to_time_t(timeStart), (std::chrono::system_clock::now() - timeStart).count());;
}