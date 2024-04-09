#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Addition, Unsigned_Zero) {
    
}

TEST(Addition, Unsigned_SmallPositive) {
    
}

TEST(Addition, Unsigned_SmallNegative) {
    
}

TEST(Addition, Unsigned_Increment) {
    
}

TEST(Addition, Unsigned_MixedAddition) {
    
}

TEST(Addition, Unsigned_MixedAdditionAssignment) {
    
}

TEST(Addition, Unsigned_DifferentPrecision) {
    
}

TEST(Addition, Unsigned_Huge) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord("Addition",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */

}

TEST(Addition, Unsigned_HugeAssignment) {
    
}