#include <gtest/gtest.h>
#include "../../Aesi.h"
#include "../../Aesi-Multiprecision.h"
#include "../benchmarks/benchmarks.h"

TEST(Boolean, Unsigned_ThreeWayComparasion) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(), std::chrono::system_clock::to_time_t(timeStart), (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Boolean, Unsigned_ThreeWayEquallComparasion) {
    const auto timeStart = std::chrono::system_clock::now();

    Logging::addRecord(testing::UnitTest::GetInstance()->current_test_info()->name(), std::chrono::system_clock::to_time_t(timeStart), (std::chrono::system_clock::now() - timeStart).count());;
}

TEST(Boolean, Unsigned_DifferentPrecisions) {

}