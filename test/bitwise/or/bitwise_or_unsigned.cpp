#include <gtest/gtest.h>
#include "../../../Aesi.h"
#include "../../../Aesi-Multiprecision.h"
#include "../../benchmarks/benchmarks.h"

TEST(Bitwise, Unsigned_OR) {
    const auto timeStart = std::chrono::system_clock::now();

#ifdef NDEBUG
    Logging::addRecord("Bitwise_OR",
                       std::chrono::system_clock::to_time_t(timeStart),
                       (std::chrono::system_clock::now() - timeStart).count());
#else
    std::cout << "Time estimated: " << (std::chrono::system_clock::now() - timeStart).count() << " ms." << std::endl;
#endif /* NDEBUG */
}

TEST(Bitwise, Unsigned_DifferentPrecisionOR) {

}