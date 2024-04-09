#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Subtraction, Unsigned_Zero) {
    
}

TEST(Subtraction, Unsigned_SmallPositive) {
    
}

TEST(Subtraction, Unsigned_SmallNegative) {
    
}

TEST(Subtraction, Unsigned_Decrement) {
    
}

TEST(Subtraction, Unsigned_MixedSubtraction) {
    
}

TEST(Subtraction, Unsigned_MixedSubtractionAssignment) {
    
}

TEST(Subtraction, Unsigned_DifferentPrecision) {
    
}

TEST(Subtraction, Unsigned_Huge) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord("Subtraction",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Subtraction, Unsigned_HugeAssignment) {
    
}