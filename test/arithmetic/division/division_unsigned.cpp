#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Division, Unsigned_SmallPositive) {
    
}

TEST(Division, Unsigned_SmallNegative) {

}

TEST(Division, Unsigned_MixedDivision) {

}

TEST(Division, Unsigned_MixedDivisionAssignment) {
}

TEST(Division, Unsigned_DifferentPrecision) {

}

TEST(Division, Unsigned_Huge) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord("Division",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}