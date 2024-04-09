#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Multiplication, Unsigned_ZeroOne) {
    
}

TEST(Multiplication, Unsigned_SmallPositive) {
    
}

TEST(Multiplication, Unsigned_SmallNegative) {
    
}

TEST(Multiplication, Unsigned_MixedMultiplication) {
    
}

TEST(Multiplication, Unsigned_MixedMultiplicationAssignment) {
    
}

TEST(Multiplication, Unsigned_DifferentPrecision) {
    
}

TEST(Multiplication, Unsigned_Huge) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord("Multiplication",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Multiplication, Unsigned_HugeAssignment) {
    
}